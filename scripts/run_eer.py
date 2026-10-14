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
    row_of = {}
    for index, session_id in enumerate(session_ids):
        if session_id in row_of:
            raise ValueError(f"session_id duplicado en los embeddings: {session_id}")
        row_of[session_id] = index
    return row_of


def check_splits(splits, row_of):
    missing = [
        session_id
        for speaker in splits.values()
        for session_id in speaker["enroll"] + speaker["trial"]
        if session_id not in row_of
    ]
    if missing:
        raise ValueError(f"{len(missing)} clips de splits no están en los embeddings, p. ej. {missing[:5]}")


def score_scenario(enroll_embeddings, trial_embeddings, row_of, splits):
    enrollment = build_enrollment(enroll_embeddings, row_of, splits)
    trials, owner = build_trials(trial_embeddings, row_of, splits)
    scores = score_matrix(enrollment, trials)
    return scores, owner


def run_scenarios(embeddings, row_of, splits, scenarios=SCENARIOS):
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


def parse_args():
    parser = argparse.ArgumentParser(description="Calcula el EER de los escenarios o-o, o-a y a-a.")
    parser.add_argument("--embeddings", required=True, help="Archivo .npz generado por extract_embeddings.")
    parser.add_argument("--manifest", required=True, help="Manifiesto TSV generado por anonymize_subset.")
    parser.add_argument("--output", required=True, help="Archivo JSON de salida.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de la partición registro/prueba.")
    parser.add_argument("--n-enroll", type=int, default=3, help="Clips de registro por hablante.")
    parser.add_argument("--n-trial", type=int, default=2, help="Clips de prueba por hablante.")
    return parser.parse_args()


def main():
    args = parse_args()

    with np.load(args.embeddings) as data:
        session_ids = [str(s) for s in data["session_ids"]]
        embeddings = {key: data[key] for key in ("original", "anonymized")}
    row_of = build_row_index(session_ids)

    rows = list(read_tsv_rows(args.manifest))
    splits, excluded = split_enrollment_trial(
        rows, n_enroll=args.n_enroll, n_trial=args.n_trial, seed=args.seed
    )

    results = run_scenarios(embeddings, row_of, splits)

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