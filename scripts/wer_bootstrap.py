"""
Intervalo de confianza del WER por bootstrap sobre hablantes.

Lee el TSV de `transcribe.py` (una fila por clip y versión), cuenta los
errores y las palabras de referencia de cada fila y los suma por hablante y
versión. El WER es de corpus: errores totales entre palabras totales. Los
intervalos salen del mismo bootstrap estratificado de `eer_bootstrap`: en
cada réplica se sortean hablantes con reemplazo dentro de cada estrato y se
usa la MISMA muestra para la versión original y la anonimizada (bootstrap
pareado), de modo que la diferencia entre ambas también tiene intervalo. El
valor puntual es el de todos los datos.
"""
import argparse
import json
from collections import Counter

import numpy as np

from common_voice_corpus import read_tsv_rows
from eer_bootstrap import percentile_ci, stratified_resample
from transcribe import VERSIONS, edit_distance, normalize_final, normalize_provisional

NORMALIZATIONS = {"provisional": normalize_provisional, "final": normalize_final}


def row_counts(reference, hypothesis, normalize=normalize_final):
    """
    Cuenta los errores y las palabras de referencia de una transcripción.

    Parámetros:
        reference: oración de referencia.
        hypothesis: texto transcrito.
        normalize: función de normalización que se aplica a ambos.

    Retorna:
        Una tupla (errores, palabras de referencia), ambos enteros.
    """
    ref_words = normalize(reference).split()
    return edit_distance(ref_words, normalize(hypothesis).split()), len(ref_words)


def speaker_counts(rows, normalize=normalize_final):
    """
    Suma errores y palabras por hablante y versión.

    Verifica que no haya filas repetidas (mismo clip y versión) y que cada
    hablante tenga los mismos clips en las dos versiones; si no, lanza
    `ValueError` con los hablantes o clips afectados.

    Parámetros:
        rows: filas del TSV de `transcribe.py`.
        normalize: función de normalización.

    Retorna:
        Diccionario con:
        - "client_ids": lista de hablantes, ordenada.
        - "strata": estrato de cada hablante, en el mismo orden.
        - "errors", "words", "clips": {versión: arreglo de enteros}, una
          posición por hablante.
    """
    seen, sessions, strata = set(), {}, {}
    totals = {}
    for row in rows:
        key = (row["session_id"], row["version"])
        if row["version"] not in VERSIONS:
            raise ValueError(f"versión desconocida: {row['version']}")
        if key in seen:
            raise ValueError(f"fila repetida: clip {key[0]}, versión {key[1]}")
        seen.add(key)
        client_id = row["client_id"]
        if strata.setdefault(client_id, row["stratum"]) != row["stratum"]:
            raise ValueError(f"el hablante {client_id} aparece en dos estratos")
        sessions.setdefault(client_id, {v: set() for v in VERSIONS})[row["version"]].add(row["session_id"])
        errors, words = row_counts(row["reference"], row["hypothesis"], normalize)
        t = totals.setdefault((client_id, row["version"]), [0, 0, 0])
        t[0] += errors
        t[1] += words
        t[2] += 1

    missing = sorted(c for c, s in sessions.items() if not all(s[v] for v in VERSIONS))
    if missing:
        raise ValueError(f"{len(missing)} hablantes no tienen las dos versiones, p. ej. {missing[:5]}")
    unpaired = sorted(c for c, s in sessions.items() if s["original"] != s["anonymized"])
    if unpaired:
        raise ValueError(f"{len(unpaired)} hablantes tienen clips distintos en cada versión, p. ej. {unpaired[:5]}")

    client_ids = sorted(sessions)
    return {
        "client_ids": client_ids,
        "strata": [strata[c] for c in client_ids],
        **{name: {v: np.array([totals[(c, v)][i] for c in client_ids]) for v in VERSIONS}
           for i, name in enumerate(("errors", "words", "clips"))},
    }


def corpus_wer(errors, words):
    """
    WER de corpus: suma de errores entre suma de palabras de referencia.

    No es el promedio de los WER de cada hablante. Lanza `ValueError` si no
    hay palabras de referencia.
    """
    total = np.sum(words)
    if total == 0:
        raise ValueError("no hay palabras de referencia")
    return float(np.sum(errors) / total)


def wer_values(counts, sample=None):
    """
    WER original, anonimizado y su diferencia para una muestra de hablantes.

    Parámetros:
        counts: salida de `speaker_counts`.
        sample: índices de hablantes (puede haber repetidos); None usa todos
            una vez.

    Retorna:
        Diccionario {"original", "anonymized", "difference"}, donde la
        diferencia es anonimizado menos original.
    """
    index = slice(None) if sample is None else np.asarray(sample)
    values = {v: corpus_wer(counts["errors"][v][index], counts["words"][v][index]) for v in VERSIONS}
    values["difference"] = values["anonymized"] - values["original"]
    return values


def bootstrap_wer(counts, n_boot, seed, progress_every=100):
    """
    Repite el cálculo del WER en `n_boot` réplicas bootstrap.

    En cada réplica se sortea una muestra de hablantes con
    `stratified_resample` y se usa la misma para las dos versiones.

    Parámetros:
        counts: salida de `speaker_counts`.
        n_boot: número de réplicas.
        seed: semilla de `np.random.default_rng`.
        progress_every: cada cuántas réplicas se imprime el avance; 0 o
            None para no imprimir.

    Retorna:
        Diccionario {"original", "anonymized", "difference": arreglo (n_boot,)}.
    """
    rng = np.random.default_rng(seed)
    out = {key: np.empty(n_boot) for key in ("original", "anonymized", "difference")}
    for b in range(n_boot):
        values = wer_values(counts, stratified_resample(counts["strata"], rng))
        for key in out:
            out[key][b] = values[key]
        if progress_every and (b + 1) % progress_every == 0:
            print(f"Réplica {b + 1}/{n_boot}", flush=True)
    return out


def comma(value, decimals=2):
    """Formatea un número con coma decimal."""
    return f"{value:.{decimals}f}".replace(".", ",")


def parse_args():
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description="WER con intervalo de confianza por bootstrap sobre hablantes.")
    parser.add_argument("--transcripts", required=True, help="TSV generado por transcribe.py.")
    parser.add_argument("--output", required=True, help="Archivo JSON de salida.")
    parser.add_argument("--n-boot", type=int, default=1000, help="Número de réplicas bootstrap.")
    parser.add_argument("--boot-seed", type=int, default=2026, help="Semilla del bootstrap.")
    parser.add_argument("--level", type=float, default=0.95, help="Nivel de confianza del intervalo.")
    parser.add_argument("--model-label", default="", help="Nombre del modelo de Whisper, para el registro.")
    parser.add_argument("--normalization", choices=sorted(NORMALIZATIONS), default="final",
                        help="Normalización del texto antes de contar errores.")
    return parser.parse_args()


def main():
    """Calcula el WER y sus intervalos, guarda el JSON e imprime un resumen."""
    args = parse_args()
    counts = speaker_counts(read_tsv_rows(args.transcripts), NORMALIZATIONS[args.normalization])
    point = wer_values(counts)
    replicates = bootstrap_wer(counts, args.n_boot, args.boot_seed)

    wer = {}
    for key in ("original", "anonymized", "difference"):
        low, high = percentile_ci(replicates[key], args.level)
        wer[key] = {"value": point[key], "ci_low": low, "ci_high": high}

    summary = {
        "transcripts": args.transcripts,
        "model_label": args.model_label,
        "normalization": args.normalization,
        "n_boot": args.n_boot,
        "boot_seed": args.boot_seed,
        "level": args.level,
        "n_speakers": len(counts["client_ids"]),
        "speakers_per_stratum": {k: int(v) for k, v in sorted(Counter(counts["strata"]).items())},
        "n_clips": {v: int(counts["clips"][v].sum()) for v in VERSIONS},
        "n_words": {v: int(counts["words"][v].sum()) for v in VERSIONS},
        "n_errors": {v: int(counts["errors"][v].sum()) for v in VERSIONS},
        "wer": wer,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    level = f"{100 * args.level:g}".replace(".", ",")
    label = f" ({args.model_label})" if args.model_label else ""
    print(f"Hablantes: {summary['n_speakers']}, palabras de referencia: {summary['n_words']['original']}, "
          f"réplicas: {args.n_boot}, normalización: {args.normalization}{label}")
    for key, name, unit in (("original", "WER original", "%"), ("anonymized", "WER anonimizado", "%"),
                            ("difference", "Diferencia (anonimizado menos original)", "pp")):
        w = wer[key]
        print(f"{name}: {comma(100 * w['value'])} {unit} "
              f"(IC {level} %: {comma(100 * w['ci_low'])}–{comma(100 * w['ci_high'])} {unit})")
    print(f"Guardado en: {args.output}")


if __name__ == "__main__":
    main()
