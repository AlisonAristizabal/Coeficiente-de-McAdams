"""
Sensibilidad de la WER por acento a las transcripciones con más del doble de palabras que su referencia.

Repite el análisis de WER por acento de group_results.py dos veces: con
todas las transcripciones y sin las anonimizadas que tienen más del doble
de palabras que su referencia (criterio de range_comparison, normalización
final). Cada hablante tiene un solo clip de transcripción, así que quitar
ese clip (en sus dos versiones, para que la diferencia compare los mismos
clips) saca al hablante del análisis de WER.

Para cada análisis: WER original, anonimizada y su diferencia por acento y
total, con IC por bootstrap estratificado por acento, y la permutación de la
WER anonimizada entre los 9 acentos reales (máximo − mínimo), con las
mismas semillas que group_results.py. No modifica las salidas de
group_results.py: escribe un CSV aparte.

Uso, desde la raíz del repositorio y con venv-eval:

    & $py scripts\\wer_sensitivity.py
"""
import argparse
from pathlib import Path

import numpy as np

from common_voice_corpus import read_tsv_rows
from figuras_resultados import SELECTED_TAG
from group_analysis import TOTAL, ci, config_files, group_codes, shared_samples, test_codes, wer_by_group
from group_results import range_permutation_test, write_csv
from range_comparison import HALLUCINATION_RATIO, manifest_coefficients
from run_alpha_sweep import REPO
from transcribe import normalize_final
from wer_bootstrap import row_counts, speaker_counts

ANALYSES = ("con_todas", "sin_mas_del_doble")
WER_KEYS = {"wer_original": "original", "wer_anonymized": "anonymized", "wer_difference": "difference"}
COLUMNS = (["analysis", "row_type", "group", "in_test", "n_speakers", "n_wer_clips", "n_words", "n_errors_original",
            "n_errors_anonymized"]
           + [f"{m}{s}" for m in WER_KEYS for s in ("", "_ci_low", "_ci_high")]
           + ["metric", "groups", "group_a", "value_a", "group_b", "value_b", "estimate", "n_resamples", "seed",
              "p_value", "clip_alpha", "reference_words", "hypothesis_words"])


def over_double_sessions(rows, normalize=normalize_final, ratio=HALLUCINATION_RATIO):
    """Clips cuya transcripción anonimizada tiene más de `ratio` veces las palabras de su referencia."""
    return {r["session_id"] for r in rows if r["version"] == "anonymized"
            and len(normalize(r["hypothesis"]).split()) > ratio * len(normalize(r["reference"]).split())}


def wer_by_accent(rows, n_boot, boot_seed, n_perm, perm_seed, level):
    """
    WER por acento y total con IC, y permutación de la WER anonimizada entre los acentos reales.

    Los hablantes, los estratos y las muestras bootstrap salen de las filas
    que se reciban (`speaker_counts` ordena por client_id, como la partición
    del EER), así que con todas las filas se obtienen las mismas réplicas que
    en group_results.py.

    Retorna:
        Una tupla (filas por grupo, fila de la permutación).
    """
    counts = speaker_counts(rows, normalize_final)
    groups, codes = group_codes(counts["strata"])
    tested, perm_codes = test_codes(groups, codes, "stratum")
    samples = shared_samples(counts["strata"], n_boot, boot_seed)
    out = []
    for names, group_codes_, n in ((groups, codes, len(groups)), ([TOTAL], np.zeros(codes.size, dtype=int), 1)):
        point = wer_by_group(counts, group_codes_, n)
        reps = [wer_by_group(counts, group_codes_, n, s) for s in samples]
        for g, group in enumerate(names):
            members = group_codes_ == g
            row = {"row_type": "grupo", "group": group, "in_test": group in tested,
                   "n_speakers": int(members.sum()), "n_wer_clips": int(counts["clips"]["anonymized"][members].sum()),
                   "n_words": int(point["words"][g]), "n_errors_original": int(point["errors_original"][g]),
                   "n_errors_anonymized": int(point["errors_anonymized"][g])}
            for column, key in WER_KEYS.items():
                row[column] = float(point[key][g])
                row[f"{column}_ci_low"], row[f"{column}_ci_high"] = ci([r[key][g] for r in reps], level)
            out.append(row)
    test = range_permutation_test(
        lambda c: {"wer_anonymized": wer_by_group(counts, c, len(tested))["anonymized"]},
        perm_codes, n_perm, perm_seed)["wer_anonymized"]
    high, low = int(np.nanargmax(test["values"])), int(np.nanargmin(test["values"]))
    permutation = {"row_type": "permutacion", "metric": "wer_anonymized", "groups": ",".join(tested),
                   "n_speakers": int(np.sum(perm_codes >= 0)), "group_a": tested[high], "value_a": test["values"][high],
                   "group_b": tested[low], "value_b": test["values"][low], "estimate": test["observed"],
                   "n_resamples": n_perm, "seed": perm_seed, "p_value": test["p_value"]}
    return out, permutation


def parse_args(argv=None):
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description="Sensibilidad de la WER por acento a las transcripciones con más "
                                                 "del doble de palabras que su referencia.")
    parser.add_argument("--data-dir", default=str(REPO.parent / "Datos"))
    parser.add_argument("--tag", default=SELECTED_TAG)
    parser.add_argument("--output", default=None,
                        help="Por defecto <data-dir>\\por_grupo_<tag>\\sensibilidad_wer.csv.")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--boot-seed", type=int, default=2026)
    parser.add_argument("--level", type=float, default=0.95)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--perm-seed", type=int, default=2026)
    args = parser.parse_args(argv)
    args.output = args.output or str(Path(args.data_dir) / f"por_grupo_{args.tag}" / "sensibilidad_wer.csv")
    return args


def main(argv=None, log=print):
    """Corre los dos análisis y escribe el CSV."""
    args = parse_args(argv)
    files = config_files(args.tag, args.data_dir)
    rows = list(read_tsv_rows(files["transcripts"]))
    flagged = over_double_sessions(rows)
    kept = [r for r in rows if r["session_id"] not in flagged]
    log(f"Transcripciones con más del doble de palabras: {len(flagged)}; filas: {len(rows)} → {len(kept)}")

    table = []
    for name, subset in zip(ANALYSES, (rows, kept)):
        groups, permutation = wer_by_accent(subset, args.n_boot, args.boot_seed, args.n_perm, args.perm_seed,
                                            args.level)
        table += [{"analysis": name, **r} for r in groups] + [{"analysis": name, **permutation}]

    coefficients = manifest_coefficients(files["manifest"])
    for session in sorted(flagged):
        by_version = {r["version"]: r for r in rows if r["session_id"] == session}
        anon = by_version["anonymized"]
        table.append({"analysis": ANALYSES[1], "row_type": "transcripcion_excluida", "group": anon["stratum"],
                      "n_errors_original": row_counts(anon["reference"], by_version["original"]["hypothesis"])[0],
                      "n_errors_anonymized": row_counts(anon["reference"], anon["hypothesis"])[0],
                      "clip_alpha": coefficients.get(session),
                      "reference_words": len(normalize_final(anon["reference"]).split()),
                      "hypothesis_words": len(normalize_final(anon["hypothesis"]).split())})
    log(f"Tabla: {write_csv(table, COLUMNS, args.output)}")


if __name__ == "__main__":
    main()
