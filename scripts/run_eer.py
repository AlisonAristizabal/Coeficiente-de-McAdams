"""
Calcula el EER de los escenarios o-o, o-a y a-a.

Lee los embeddings de `extract_embeddings` y el manifiesto de
`anonymize_subset`, separa los clips de cada hablante en registro y prueba,
puntúa cada prueba contra cada registro con similitud coseno y guarda en un
JSON el EER, el umbral y el número de pares de cada escenario.
"""
import argparse
import json

import numpy as np

from common_voice_corpus import read_tsv_rows
from eer_protocol import split_enrollment_trial
from eer_scoring import (
    build_enrollment,
    build_trials,
    compute_eer,
    score_matrix,
    split_target_impostor,
)

# Cada escenario: (fuente del registro, fuente de la prueba), con las llaves del .npz.
SCENARIOS = {
    "o-o": ("original", "original"),
    "o-a": ("original", "anonymized"),
    "a-a": ("anonymized", "anonymized"),
}


def build_row_index(session_ids):
    """
    Construye el índice de fila de cada clip en la matriz de embeddings.

    Lanza `ValueError` si un `session_id` aparece dos veces, porque entonces
    no se sabría cuál de las dos filas usar.

    Parámetros:
        session_ids: iterable con el `session_id` de cada fila del .npz, en
            orden.

    Retorna:
        Diccionario {session_id: índice de fila}.
    """
    row_of = {}
    for index, session_id in enumerate(session_ids):
        if session_id in row_of:
            raise ValueError(f"session_id duplicado en los embeddings: {session_id}")
        row_of[session_id] = index
    return row_of


def check_splits(splits, row_of):
    """
    Verifica que todos los clips de `splits` tengan embedding.

    Revisa los clips de registro y de prueba de todos los hablantes antes de
    puntuar, para fallar con un mensaje claro y no con un `KeyError` a mitad
    del cálculo. Los clips de `row_of` que no están en `splits` se ignoran.
    No modifica las entradas.

    Parámetros:
        splits: salida de `split_enrollment_trial`.
        row_of: diccionario {session_id: índice de fila}.

    Retorna:
        None. Lanza `ValueError` con el número de clips faltantes y algunos
        ejemplos si falta alguno.
    """
    missing = [
        session_id
        for speaker in splits.values()
        for session_id in speaker["enroll"] + speaker["trial"]
        if session_id not in row_of
    ]
    if missing:
        raise ValueError(f"{len(missing)} clips de splits no están en los embeddings, p. ej. {missing[:5]}")


def score_scenario(enroll_embeddings, trial_embeddings, row_of, splits):
    """
    Puntúa un escenario: pruebas de una fuente contra registros de otra.

    El registro de cada hablante sale de `enroll_embeddings` y las pruebas
    de `trial_embeddings`; así, por ejemplo, o-a usa los originales para el
    registro y los anonimizados para la prueba.

    Parámetros:
        enroll_embeddings: matriz (N, D) de donde salen los registros.
        trial_embeddings: matriz (N, D) de donde salen las pruebas.
        row_of: diccionario {session_id: índice de fila}, común a ambas.
        splits: salida de `split_enrollment_trial`.

    Retorna:
        Una tupla (scores, owner). `scores` es la matriz (pruebas,
        hablantes) de similitud coseno y `owner[i]` es el índice del hablante
        dueño de la prueba i, en el orden de `splits`.
    """
    enrollment =build_enrollment(enroll_embeddings, row_of, splits)
    trials, owner = build_trials(trial_embeddings, row_of, splits)
    scores = score_matrix(enrollment, trials)
    return scores, owner


def run_scenarios(embeddings, row_of, splits, scenarios=SCENARIOS):
    """
    Calcula las puntuaciones y el EER de cada escenario.

    Primero verifica con `check_splits` que todos los clips tengan
    embedding. Luego, para cada escenario, puntúa con `score_scenario`,
    separa los pares objetivo e impostor y calcula el EER con `compute_eer`.
    Devuelve también la matriz de puntuaciones y `owner` para que las
    figuras y el bootstrap no tengan que recalcularlas.

    Parámetros:
        embeddings: diccionario {"original": matriz, "anonymized": matriz},
            ambas con las filas en el orden de `row_of`.
        row_of: diccionario {session_id: índice de fila}.
        splits: salida de `split_enrollment_trial`.
        scenarios: diccionario {escenario: (fuente del registro, fuente de
            la prueba)}; por defecto `SCENARIOS`.

    Retorna:
        Diccionario {escenario: {"scores", "owner", "eer", "threshold",
        "n_target", "n_impostor"}}. Los conteos son enteros de Python.
    """
    check_splits(splits, row_of)
    results = {}
    for name, (enroll_key, trial_key) in scenarios.items():
        scores, owner = score_scenario(embeddings[enroll_key], embeddings[trial_key], row_of, splits)
        target, impostor = split_target_impostor(scores, owner)
        eer, threshold = compute_eer(target, impostor)
        results[name] = {
            "scores": scores,
            "owner": owner,
            "eer": eer,
            "threshold": threshold,
            "n_target": target.shape[0],
            "n_impostor": impostor.shape[0],
        }
    return results


def load_and_score(embeddings_path, manifest_path, seed, n_enroll, n_trial):
    """
    Carga los embeddings y el manifiesto y puntúa los tres escenarios.

    Reúne los pasos comunes a `run_eer`, `plot_eer` y `eer_bootstrap`, para
    que los tres trabajen exactamente con la misma partición y las mismas
    puntuaciones. Los arreglos del .npz se copian dentro del `with`, así el
    archivo queda cerrado al salir.

    Parámetros:
        embeddings_path: archivo .npz de `extract_embeddings`, con las llaves
            `session_ids`, `original` y `anonymized`.
        manifest_path: manifiesto TSV de `anonymize_subset`.
        seed: semilla de la partición registro/prueba.
        n_enroll: clips de registro por hablante.
        n_trial: clips de prueba por hablante.

    Retorna:
        Una tupla (splits, excluded, results): la partición y los hablantes
        excluidos de `split_enrollment_trial`, y la salida de `run_scenarios`.
    """
    with np.load(embeddings_path) as data:
        session_ids = [str(s) for s in data["session_ids"]]
        embeddings = {key: data[key] for key in ("original", "anonymized")}
    row_of = build_row_index(session_ids)

    rows = list(read_tsv_rows(manifest_path))
    splits, excluded = split_enrollment_trial(rows, n_enroll=n_enroll, n_trial=n_trial, seed=seed)

    results = run_scenarios(embeddings, row_of, splits)
    return splits, excluded, results


def parse_args():
    """Lee los argumentos de la línea de comandos (valores por defecto de D2)."""
    parser = argparse.ArgumentParser(description="Calcula el EER de los escenarios o-o, o-a y a-a.")
    parser.add_argument("--embeddings", required=True, help="Archivo .npz generado por extract_embeddings.")
    parser.add_argument("--manifest", required=True, help="Manifiesto TSV generado por anonymize_subset.")
    parser.add_argument("--output", required=True, help="Archivo JSON de salida.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de la partición registro/prueba.")
    parser.add_argument("--n-enroll", type=int, default=3, help="Clips de registro por hablante.")
    parser.add_argument("--n-trial", type=int, default=2, help="Clips de prueba por hablante.")
    return parser.parse_args()


def main():
    """Calcula los tres escenarios, guarda el resumen en JSON y lo imprime."""
    args = parse_args()
    splits, excluded, results = load_and_score(
        args.embeddings, args.manifest, args.seed, args.n_enroll, args.n_trial
    )

    summary = {
        "embeddings": args.embeddings,
        "manifest": args.manifest,
        "seed": args.seed,
        "n_enroll": args.n_enroll,
        "n_trial": args.n_trial,
        "n_speakers": len(splits),
        "excluded": excluded,
        "scenarios": {
            name: {
                "eer": float(r["eer"]),
                "threshold": float(r["threshold"]),
                "n_target": r["n_target"],
                "n_impostor": r["n_impostor"],
            }
            for name, r in results.items()
        },
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Hablantes: {len(splits)} (excluidos: {len(excluded)})")
    for name, s in summary["scenarios"].items():
        print(f"{name}: EER = {100 * s['eer']:.2f} %  objetivo = {s['n_target']}  impostor = {s['n_impostor']}")
    print(f"Guardado en: {args.output}")


if __name__ == "__main__":
    main()