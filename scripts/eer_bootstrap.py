"""
Intervalos de confianza del EER por bootstrap sobre hablantes.

En cada réplica se remuestrean hablantes con reemplazo dentro de cada
estrato, se arma la submatriz de puntuaciones de la muestra y se recalcula
el EER con `compute_eer`. La misma muestra se usa en los tres escenarios
(bootstrap pareado), lo que permite dar también intervalos para la
diferencia de EER entre escenarios. Los intervalos son de percentiles; el
EER puntual que se reporta es el de todos los datos, no el promedio de las
réplicas.
"""
import argparse
import json
from collections import Counter

import numpy as np

from eer_scoring import compute_eer
from run_eer import load_and_score

# Diferencias pareadas que se reportan: (nombre, minuendo, sustraendo).
DIFFERENCES = [
    ("o-a menos a-a", "o-a", "a-a"),
    ("a-a menos o-o", "a-a", "o-o"),
    ("o-a menos o-o", "o-a", "o-o"),
]


def stratified_resample(strata, rng):
    """
    Remuestrea hablantes con reemplazo dentro de cada estrato.

    De cada estrato se sacan, con reemplazo, tantos hablantes como tiene, de
    modo que la muestra conserva la composición por estratos del diseño. Los
    índices sorteados de un estrato ocupan las posiciones de ese estrato.

    Parámetros:
        strata: secuencia con el estrato de cada hablante, en el orden de
            `splits` (el de las columnas de `scores`).
        rng: generador `np.random.Generator`.

    Retorna:
        Arreglo de enteros del mismo largo que `strata`, con índices de
        hablantes (puede haber repeticiones).
    """
    strata = np.asarray(strata)
    sample = np.empty(strata.size, dtype=np.intp)
    for stratum in np.unique(strata):
        members = np.flatnonzero(strata == stratum)
        sample[members] = rng.choice(members, size=members.size, replace=True)
    return sample


def copy_trial_rows(owner, n_speakers, sample):
    """
    Filas de prueba de cada copia de una muestra bootstrap.

    Cada posición k de `sample` es una copia del hablante `sample[k]` y se
    lleva todas sus pruebas. Las filas de cada hablante salen de un
    ordenamiento estable de `owner`, así que conservan su orden original.

    Parámetros:
        owner: arreglo con el hablante dueño de cada prueba.
        n_speakers: número de hablantes (columnas de la matriz de puntuaciones).
        sample: índices de hablantes, de `stratified_resample`.

    Retorna:
        Una tupla (rows, row_copy): `rows[r]` es la fila de prueba original
        de la posición r y `row_copy[r]` la copia (posición en `sample`) a
        la que pertenece. Las filas de cada copia van seguidas, en el orden
        de `sample`.
    """
    owner = np.asarray(owner)
    sample = np.asarray(sample)
    order = np.argsort(owner, kind="stable")
    counts = np.bincount(owner, minlength=n_speakers)
    starts = np.cumsum(counts) - counts
    copy_counts = counts[sample]
    row_copy = np.repeat(np.arange(sample.size), copy_counts)
    offset = np.arange(row_copy.size) - np.repeat(np.cumsum(copy_counts) - copy_counts, copy_counts)
    return order[starts[sample][row_copy] + offset], row_copy


def resample_pairs(scores, owner, sample):
    """
    Separa los pares objetivo e impostor de una muestra bootstrap.

    Cada posición k de `sample` es una copia del hablante original
    `sample[k]`, con sus pruebas y su registro. Un par es objetivo si la
    prueba y el registro son de la misma copia, e impostor si son de
    hablantes originales distintos. Los pares entre copias distintas del
    mismo hablante original se excluyen: no son el mismo "hablante" de la
    muestra, pero tampoco son un impostor real. Con `sample` igual a
    `np.arange(número de hablantes)` se obtienen los mismos pares que con
    `split_target_impostor`.

    Parámetros:
        scores: matriz (pruebas, hablantes) de `run_scenarios`.
        owner: arreglo con el hablante dueño de cada prueba.
        sample: índices de hablantes, de `stratified_resample`.

    Retorna:
        Una tupla (target, impostor) de arreglos 1D.
    """
    scores = np.asarray(scores)
    owner = np.asarray(owner)
    sample = np.asarray(sample)

    rows, row_copy = copy_trial_rows(owner, scores.shape[1], sample)
    sub = scores[rows][:, sample]
    is_target = row_copy[:, None] == np.arange(sample.size)[None, :]
    is_impostor = owner[rows][:, None] != sample[None, :]
    return sub[is_target], sub[is_impostor]


def bootstrap_eers(results, strata, n_boot, seed, progress_every=100):
    """
    Calcula el EER de cada escenario en `n_boot` réplicas bootstrap.

    En cada réplica se sortea una sola muestra de hablantes con
    `stratified_resample` y se usa para todos los escenarios (bootstrap
    pareado).

    Parámetros:
        results: salida de `run_scenarios`.
        strata: estrato de cada hablante, en el orden de las columnas.
        n_boot: número de réplicas.
        seed: semilla de `np.random.default_rng`.
        progress_every: cada cuántas réplicas se imprime el avance; 0 o
            None para no imprimir.

    Retorna:
        Diccionario {escenario: arreglo (n_boot,) de EER}.
    """
    for name, r in results.items():
        if r["scores"].shape[1] != len(strata):
            raise ValueError(f"{name}: strata no coincide con las columnas de scores")
    rng = np.random.default_rng(seed)
    eers = {name: np.empty(n_boot) for name in results}
    for b in range(n_boot):
        sample = stratified_resample(strata, rng)
        for name, r in results.items():
            target, impostor = resample_pairs(r["scores"], r["owner"], sample)
            eers[name][b] = compute_eer(target, impostor)[0]
        if progress_every and (b + 1) % progress_every == 0:
            print(f"Réplica {b + 1}/{n_boot}", flush=True)
    return eers


def percentile_ci(values, level=0.95):
    """
    Calcula el intervalo de confianza por percentiles.

    Toma los percentiles (1 - level) / 2 y (1 + level) / 2 de `values`, con
    la interpolación lineal de `np.percentile`.

    Retorna:
        Una tupla (low, high) de floats.
    """
    if not 0 < level < 1:
        raise ValueError("level debe estar entre 0 y 1")
    alpha = 1 - level
    low, high = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def difference_cis(point_eers, replicate_eers, level=0.95):
    """
    Calcula la diferencia de EER entre escenarios y su intervalo.

    La diferencia puntual sale de los EER de todos los datos; el intervalo,
    de las diferencias réplica por réplica (posible porque el bootstrap es
    pareado).

    Parámetros:
        point_eers: diccionario {escenario: EER de todos los datos}.
        replicate_eers: salida de `bootstrap_eers`.
        level: nivel de confianza.

    Retorna:
        Diccionario {nombre de la diferencia: {"diff", "ci_low", "ci_high"}},
        con las diferencias de `DIFFERENCES`.
    """
    out = {}
    for label, a, b in DIFFERENCES:
        low, high = percentile_ci(replicate_eers[a] - replicate_eers[b], level)
        out[label] = {"diff": float(point_eers[a] - point_eers[b]), "ci_low": low, "ci_high": high}
    return out


def percent(value):
    """Formatea una fracción como porcentaje con dos decimales y coma decimal."""
    return f"{100 * value:.2f}".replace(".", ",")


def parse_args():
    """Lee los argumentos de la línea de comandos (valores por defecto de D2)."""
    parser = argparse.ArgumentParser(description="Intervalos de confianza del EER por bootstrap sobre hablantes.")
    parser.add_argument("--embeddings", required=True, help="Archivo .npz generado por extract_embeddings.")
    parser.add_argument("--manifest", required=True, help="Manifiesto TSV generado por anonymize_subset.")
    parser.add_argument("--output", required=True, help="Archivo JSON de salida.")
    parser.add_argument("--replicates-output", required=True, help="Archivo .npz con el EER de cada réplica.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de la partición registro/prueba.")
    parser.add_argument("--n-enroll", type=int, default=3, help="Clips de registro por hablante.")
    parser.add_argument("--n-trial", type=int, default=2, help="Clips de prueba por hablante.")
    parser.add_argument("--n-boot", type=int, default=1000, help="Número de réplicas bootstrap.")
    parser.add_argument("--boot-seed", type=int, default=2026, help="Semilla del bootstrap.")
    parser.add_argument("--level", type=float, default=0.95, help="Nivel de confianza del intervalo.")
    return parser.parse_args()


def main():
    """Corre el bootstrap y guarda el resumen (JSON) y las réplicas (.npz)."""
    args = parse_args()
    splits, excluded, results = load_and_score(
        args.embeddings, args.manifest, args.seed, args.n_enroll, args.n_trial
    )
    strata = [speaker["stratum"] for speaker in splits.values()]

    replicates = bootstrap_eers(results, strata, args.n_boot, args.boot_seed)
    point = {name: float(r["eer"]) for name, r in results.items()}

    scenarios = {}
    for name, r in results.items():
        low, high = percentile_ci(replicates[name], args.level)
        scenarios[name] = {
            "eer": point[name],
            "ci_low": low,
            "ci_high": high,
            "n_target": int(r["n_target"]),
            "n_impostor": int(r["n_impostor"]),
        }

    summary = {
        "embeddings": args.embeddings,
        "manifest": args.manifest,
        "seed": args.seed,
        "n_enroll": args.n_enroll,
        "n_trial": args.n_trial,
        "n_boot": args.n_boot,
        "boot_seed": args.boot_seed,
        "level": args.level,
        "n_speakers": len(splits),
        "speakers_per_stratum": {k: int(v) for k, v in sorted(Counter(strata).items())},
        "excluded": {k: int(v) for k, v in excluded.items()},
        "scenarios": scenarios,
        "differences": difference_cis(point, replicates, args.level),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    np.savez(args.replicates_output, **replicates)

    level = f"{100 * args.level:g}".replace(".", ",")
    print(f"Hablantes: {len(splits)} (excluidos: {len(excluded)}), réplicas: {args.n_boot}")
    for name, s in scenarios.items():
        print(f"{name}: EER = {percent(s['eer'])} % (IC {level} %: {percent(s['ci_low'])}–{percent(s['ci_high'])} %)")
    for label, d in summary["differences"].items():
        print(f"{label}: {percent(d['diff'])} pp (IC {level} %: {percent(d['ci_low'])}–{percent(d['ci_high'])} pp)")
    print(f"Guardado en: {args.output} y {args.replicates_output}")


if __name__ == "__main__":
    main()
