import argparse

from common_voice_corpus import (
    load_durations, load_speakers, filter_speakers, write_subset,
)
from common_voice_sampling import (
    group_by_stratum, 
    sample_speakers, 
    select_clips,

)
from common_voice_strata import (
    ACCENT_TO_STRATUM,
    SIN_DATO,
)

SEED = 42
K = 50
CLIPS_PER_SPEAKER = 5
MIN_DURATION_MS = 3000
STRATA = [a for a in ACCENT_TO_STRATUM.values()] + [SIN_DATO]   


def select_subset(speakers, strata, k, n, seed):
    """
    Elige los hablantes y clips del subconjunto de evaluación.

    Agrupa los hablantes por estrato, elige `k` de cada estrato de
    `strata` con `sample_speakers` y, de cada uno, `n` clips con
    `select_clips`. Todo es reproducible con `seed`. No modifica
    `speakers`.

    Parámetros:
        speakers: diccionario {client_id: lista de clips}, ya filtrado
            (cada hablante con al menos `n` clips).
        strata: lista de estratos a muestrear.
        k: hablantes por estrato.
        n: clips por hablante.
        seed: entero que siembra el muestreo.

    Retorna:
        Diccionario {estrato: {client_id: lista de `n` clips}}, en el
        formato que espera `write_subset`.
    """
    by_stratum = group_by_stratum(speakers)
    sampled = sample_speakers(by_stratum, strata, k, seed)
    selected = {}
    for stratum, client_ids in sampled.items():
        selected[stratum] = {}
        for client_id in client_ids:
            selected[stratum][client_id] = select_clips(speakers[client_id], n, seed, client_id)
    return selected


def parse_args():
    parser = argparse.ArgumentParser(description="Construye un subconjunto de clips de voz validados")
    parser.add_argument("--validated", required=True, help="Archivo con los datos validados")
    parser.add_argument("--durations", required=True, help="Archivo con las duraciones de los clips")
    parser.add_argument("--output", required=True, help="Ruta del archivo de salida")
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--clips", type=int, default=CLIPS_PER_SPEAKER)
    parser.add_argument("--seed",type=int, default=SEED, help="Semilla para el muestreo reproducible")
    return parser.parse_args()


def main():
    args = parse_args()

    durations = load_durations(args.durations)
    speakers, discarded = load_speakers(args.validated, durations, MIN_DURATION_MS)
    eligible = filter_speakers(speakers, args.clips)
    selected = select_subset(eligible, STRATA, args.k, args.clips, args.seed)
    write_subset(selected, args.output)


    print(f"Clips conservados: {sum(len(clips) for clips in speakers.values())}")
    print(f"Clips descartados: {', '.join(f'{razon}: {count}' for razon, count in discarded.items())}")
    print(f"Hablantes antes del filtro: {len(speakers)}")
    print(f"Hablantes después del filtro: {len(eligible)}")
    for stratum, client_ids in selected.items():
        n_clips = sum(len(clips) for clips in client_ids.values())
        print(f"Estrato {stratum}: {len(client_ids)} hablantes, {n_clips} clips")


if __name__ == "__main__":
    main()