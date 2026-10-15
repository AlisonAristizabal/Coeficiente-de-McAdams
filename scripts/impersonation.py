"""
Tasa de suplantación con grabaciones anonimizadas.

Escenario: un sistema de verificación ajusta su umbral con voces originales
(o-o). Un atacante presenta una grabación anonimizada de la persona (pares
objetivo de o-a). La tasa de suplantación es la fracción de esas grabaciones
que el sistema todavía acepta como la persona. Se reporta en dos puntos de
operación del sistema: el umbral del EER de o-o y el umbral con FAR <=
`max_far` en o-o. Como referencia se da la tasa de aceptación de la persona
real con su voz original (pares objetivo de o-o) con el mismo umbral.

Los intervalos salen del mismo bootstrap sobre hablantes de `eer_bootstrap`:
en cada réplica se usa una sola muestra para o-o y o-a y el umbral se vuelve
a ajustar con el o-o de esa réplica, porque el umbral también es una
estimación. El valor puntual es el de todos los datos.
"""
import argparse
import json

import numpy as np

from eer_bootstrap import percentile_ci, resample_pairs, stratified_resample
from eer_scoring import compute_eer, split_target_impostor
from run_eer import load_and_score

# Puntos de operación: el umbral del EER de o-o y el de FAR <= max_far en o-o.
POINTS = ("eer", "far")
RATES = ("genuine_rate", "impersonation_rate")


def eer_threshold(target_oo, impostor_oo):
    """
    Umbral del EER de o-o, tal como lo devuelve `compute_eer`.

    Parámetros:
        target_oo: puntuaciones de los pares objetivo de o-o.
        impostor_oo: puntuaciones de los pares impostor de o-o.

    Retorna:
        El umbral (float).
    """
    return float(compute_eer(target_oo, impostor_oo)[1])


def far_threshold(target_oo, impostor_oo, max_far):
    """
    Menor umbral observado de o-o cuyo FAR en o-o es <= `max_far`.

    Los candidatos son todas las puntuaciones observadas de o-o (objetivo e
    impostor). Se acepta un par si su puntuación es >= umbral, como en
    `compute_eer`, así que el FAR de un umbral es la fracción de impostores
    con puntuación >= umbral. Entre los candidatos que cumplen se elige el
    menor, que es el que acepta más pares objetivo. Lanza `ValueError` si
    `max_far` no está entre 0 y 1, si algún conjunto está vacío o si ningún
    umbral observado cumple (pasa con `max_far` = 0 cuando la puntuación
    más alta es de un impostor).

    Parámetros:
        target_oo: puntuaciones de los pares objetivo de o-o.
        impostor_oo: puntuaciones de los pares impostor de o-o.
        max_far: FAR máximo permitido, como fracción.

    Retorna:
        El umbral (float).
    """
    if not 0 <= max_far <= 1:
        raise ValueError("max_far debe estar entre 0 y 1")
    target = np.asarray(target_oo, dtype=np.float64)
    impostor = np.sort(np.asarray(impostor_oo, dtype=np.float64))
    if target.size == 0 or impostor.size == 0:
        raise ValueError("target e impostor no pueden estar vacíos")
    thresholds = np.unique(np.concatenate([target, impostor]))
    far = (impostor.size - np.searchsorted(impostor, thresholds, side="left")) / impostor.size
    # El FAR no crece con el umbral: el primero que cumple es el menor.
    meets = far <= max_far
    if not meets.any():
        raise ValueError(f"ningún umbral observado tiene FAR <= {max_far}")
    return float(thresholds[np.argmax(meets)])


def acceptance_rate(target_scores, threshold):
    """
    Fracción de puntuaciones que el sistema acepta (puntuación >= umbral).

    Parámetros:
        target_scores: puntuaciones (arreglo 1D, no vacío).
        threshold: umbral de decisión.

    Retorna:
        La fracción aceptada (float entre 0 y 1).
    """
    scores = np.asarray(target_scores, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("no hay puntuaciones")
    return float(np.mean(scores >= threshold))


def operating_point_rates(target_oo, impostor_oo, target_oa, max_far):
    """
    Calcula umbral y tasas de aceptación en los dos puntos de operación.

    Los umbrales se ajustan solo con o-o, que es lo que conoce el sistema;
    con ese mismo umbral se mide cuántos pares objetivo de o-o (la persona
    real) y de o-a (la grabación anonimizada) se aceptan.

    Parámetros:
        target_oo: pares objetivo de o-o.
        impostor_oo: pares impostor de o-o.
        target_oa: pares objetivo de o-a.
        max_far: FAR máximo del segundo punto de operación.

    Retorna:
        Diccionario {"eer" | "far": {"threshold", "far_oo", "genuine_rate",
        "impersonation_rate"}}. `far_oo` es el FAR de o-o con ese umbral.
    """
    thresholds = {
        "eer": eer_threshold(target_oo, impostor_oo),
        "far": far_threshold(target_oo, impostor_oo, max_far),
    }
    return {
        point: {
            "threshold": threshold,
            "far_oo": acceptance_rate(impostor_oo, threshold),
            "genuine_rate": acceptance_rate(target_oo, threshold),
            "impersonation_rate": acceptance_rate(target_oa, threshold),
        }
        for point, threshold in thresholds.items()
    }


def point_rates(results, max_far):
    """
    Tasas en los dos puntos de operación con todos los datos.

    Parámetros:
        results: salida de `run_scenarios`, con los escenarios "o-o" y "o-a".
        max_far: FAR máximo del segundo punto de operación.

    Retorna:
        La salida de `operating_point_rates`.
    """
    target_oo, impostor_oo = split_target_impostor(results["o-o"]["scores"], results["o-o"]["owner"])
    target_oa, _ = split_target_impostor(results["o-a"]["scores"], results["o-a"]["owner"])
    return operating_point_rates(target_oo, impostor_oo, target_oa, max_far)


def bootstrap_rates(results, strata, n_boot, seed, max_far, progress_every=100):
    """
    Repite el cálculo de las tasas en `n_boot` réplicas bootstrap.

    En cada réplica se sortea una muestra de hablantes con
    `stratified_resample` y se usa la MISMA para o-o y o-a (bootstrap
    pareado). Los pares salen de `resample_pairs` y los umbrales se vuelven a
    ajustar con el o-o de la réplica.

    Parámetros:
        results: salida de `run_scenarios`, con "o-o" y "o-a".
        strata: estrato de cada hablante, en el orden de las columnas.
        n_boot: número de réplicas.
        seed: semilla de `np.random.default_rng`.
        max_far: FAR máximo del segundo punto de operación.
        progress_every: cada cuántas réplicas se imprime el avance; 0 o
            None para no imprimir.

    Retorna:
        Diccionario {"eer" | "far": {"threshold", "genuine_rate",
        "impersonation_rate": arreglo (n_boot,)}}.
    """
    oo, oa = results["o-o"], results["o-a"]
    for name in ("o-o", "o-a"):
        if results[name]["scores"].shape[1] != len(strata):
            raise ValueError(f"{name}: strata no coincide con las columnas de scores")
    rng = np.random.default_rng(seed)
    out = {point: {key: np.empty(n_boot) for key in ("threshold",) + RATES} for point in POINTS}
    for b in range(n_boot):
        sample = stratified_resample(strata, rng)
        target_oo, impostor_oo = resample_pairs(oo["scores"], oo["owner"], sample)
        target_oa, _ = resample_pairs(oa["scores"], oa["owner"], sample)
        rates = operating_point_rates(target_oo, impostor_oo, target_oa, max_far)
        for point in POINTS:
            for key in out[point]:
                out[point][key][b] = rates[point][key]
        if progress_every and (b + 1) % progress_every == 0:
            print(f"Réplica {b + 1}/{n_boot}", flush=True)
    return out


def comma(value, decimals=2):
    """Formatea un número con coma decimal."""
    return f"{value:.{decimals}f}".replace(".", ",")


def parse_args():
    """Lee los argumentos de la línea de comandos (valores por defecto de D2)."""
    parser = argparse.ArgumentParser(description="Tasa de suplantación con grabaciones anonimizadas.")
    parser.add_argument("--embeddings", required=True, help="Archivo .npz generado por extract_embeddings.")
    parser.add_argument("--manifest", required=True, help="Manifiesto TSV generado por anonymize_subset.")
    parser.add_argument("--output", required=True, help="Archivo JSON de salida.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de la partición registro/prueba.")
    parser.add_argument("--n-enroll", type=int, default=3, help="Clips de registro por hablante.")
    parser.add_argument("--n-trial", type=int, default=2, help="Clips de prueba por hablante.")
    parser.add_argument("--n-boot", type=int, default=1000, help="Número de réplicas bootstrap.")
    parser.add_argument("--boot-seed", type=int, default=2026, help="Semilla del bootstrap.")
    parser.add_argument("--level", type=float, default=0.95, help="Nivel de confianza del intervalo.")
    parser.add_argument("--max-far", type=float, default=0.01, help="FAR máximo en o-o del segundo umbral.")
    return parser.parse_args()


def main():
    """Calcula las tasas y sus intervalos, guarda el JSON e imprime un resumen."""
    args = parse_args()
    splits, excluded, results = load_and_score(
        args.embeddings, args.manifest, args.seed, args.n_enroll, args.n_trial
    )
    strata = [speaker["stratum"] for speaker in splits.values()]

    point = point_rates(results, args.max_far)
    replicates = bootstrap_rates(results, strata, args.n_boot, args.boot_seed, args.max_far)

    operating_points = {}
    for name in POINTS:
        entry = {"threshold": point[name]["threshold"], "far_oo": point[name]["far_oo"]}
        for rate in RATES:
            low, high = percentile_ci(replicates[name][rate], args.level)
            entry[rate] = {"value": point[name][rate], "ci_low": low, "ci_high": high}
        operating_points[name] = entry
    operating_points["far"]["max_far"] = args.max_far

    summary = {
        "embeddings": args.embeddings,
        "manifest": args.manifest,
        "seed": args.seed,
        "n_enroll": args.n_enroll,
        "n_trial": args.n_trial,
        "n_boot": args.n_boot,
        "boot_seed": args.boot_seed,
        "level": args.level,
        "max_far": args.max_far,
        "n_speakers": len(splits),
        "excluded": {k: int(v) for k, v in excluded.items()},
        "n_target_oo": int(results["o-o"]["n_target"]),
        "n_impostor_oo": int(results["o-o"]["n_impostor"]),
        "n_target_oa": int(results["o-a"]["n_target"]),
        "operating_points": operating_points,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    level = f"{100 * args.level:g}".replace(".", ",")
    titles = {
        "eer": "Umbral del EER de o-o",
        "far": f"Umbral con FAR <= {100 * args.max_far:g} % en o-o".replace(".", ","),
    }
    print(f"Hablantes: {len(splits)} (excluidos: {len(excluded)}), réplicas: {args.n_boot}")
    for name in POINTS:
        p = operating_points[name]
        print(f"{titles[name]} (umbral {comma(p['threshold'], 4)}, FAR o-o {comma(100 * p['far_oo'])} %):")
        for rate, label in (("genuine_rate", "persona real (o-o)"), ("impersonation_rate", "grabación anonimizada (o-a)")):
            r = p[rate]
            print(f"  {label}: {comma(100 * r['value'])} % aceptada "
                  f"(IC {level} %: {comma(100 * r['ci_low'])}–{comma(100 * r['ci_high'])} %)")
    print(f"Guardado en: {args.output}")


if __name__ == "__main__":
    main()
