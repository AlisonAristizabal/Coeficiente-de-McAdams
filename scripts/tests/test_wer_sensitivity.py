"""Pruebas del análisis de sensibilidad de la WER por acento, con transcripciones de juguete."""
import csv

import numpy as np
import pytest

pytest.importorskip("matplotlib")

import group_analysis as ga  # noqa: E402
import wer_sensitivity as ws  # noqa: E402
from range_comparison import hallucination_diagnostic  # noqa: E402

STRATA = ("caribe", "chileno", "mexico", "sin_dato")


def transcripts(n_speakers=16, flagged=(2, 9)):
    rng = np.random.default_rng(1)
    words = "el gato come pescado en la casa de mi abuela".split()
    rows = []
    for i in range(n_speakers):
        ref = rng.choice(words, size=6).tolist()
        for version, p in (("original", 0.1), ("anonymized", 0.4)):
            hyp = [w if rng.random() > p else "x" for w in ref]
            if version == "anonymized" and i in flagged:
                hyp = hyp * 3
            rows.append({"session_id": f"t{i}", "client_id": f"spk{i:02d}", "stratum": STRATA[i % 4],
                         "version": version, "reference": " ".join(ref), "hypothesis": " ".join(hyp), "seconds": "1"})
    return rows


def test_over_double_sessions_mismo_criterio_que_range_comparison():
    rows = transcripts()
    assert ws.over_double_sessions(rows) == {"t2", "t9"}
    assert hallucination_diagnostic(rows, {})[0]["n_hallucinated"] == 2


def test_wer_by_accent_con_todas_reproduce_wer_by_group_y_sin_saca_al_hablante():
    rows = transcripts()
    groups, permutation = ws.wer_by_accent(rows, 20, 2026, 30, 2026, 0.95)
    counts = ga.aligned_wer_counts_from_rows(rows, sorted({r["client_id"] for r in rows}))
    names, codes = ga.group_codes(counts["strata"])
    point = ga.wer_by_group(counts, codes, len(names))
    samples = ga.shared_samples(counts["strata"], 20, 2026)
    caribe = groups[0]
    assert (caribe["group"], caribe["wer_anonymized"]) == ("caribe", point["anonymized"][0])
    expected_ci = ga.ci([ga.wer_by_group(counts, codes, len(names), s)["anonymized"][0] for s in samples], 0.95)
    assert (caribe["wer_anonymized_ci_low"], caribe["wer_anonymized_ci_high"]) == expected_ci
    assert permutation["groups"] == "caribe,chileno,mexico" and permutation["n_speakers"] == 12
    assert [g["group"] for g in groups] == list(STRATA) + ["total"]

    kept = [r for r in rows if r["session_id"] not in ws.over_double_sessions(rows)]
    without, permutation = ws.wer_by_accent(kept, 20, 2026, 30, 2026, 0.95)
    by = {g["group"]: g for g in without}
    # t2 (mexico) y t9 (chileno) salen en sus dos versiones; los demás grupos no cambian.
    assert (by["mexico"]["n_speakers"], by["chileno"]["n_speakers"], by["total"]["n_speakers"]) == (3, 3, 14)
    assert by["caribe"]["wer_anonymized"] == caribe["wer_anonymized"]
    assert by["total"]["n_wer_clips"] == 14 and permutation["n_speakers"] == 10


def test_main_escribe_un_csv_aparte(tmp_path):
    f = ga.config_files("u0685_0865", tmp_path)
    rows = transcripts()
    for path, data in ((f["transcripts"], rows),
                       (f["manifest"], [{"client_id": r["client_id"], "stratum": r["stratum"], "path": "x.mp3",
                                         "session_id": r["session_id"], "coefficient": "0.7", "status": "ok",
                                         "error": ""} for r in rows if r["version"] == "original"])):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=list(data[0]), delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
    ws.main(["--data-dir", str(tmp_path), "--n-boot", "5", "--n-perm", "10"], log=lambda *_: None)
    with open(tmp_path / "por_grupo_u0685_0865" / "sensibilidad_wer.csv", encoding="utf-8", newline="") as out:
        table = list(csv.DictReader(out))
    assert [(r["analysis"], r["row_type"]) for r in table].count(("sin_mas_del_doble", "transcripcion_excluida")) == 2
    assert sum(r["row_type"] == "permutacion" for r in table) == 2 and list(table[0]) == ws.COLUMNS
    excluded = [r for r in table if r["row_type"] == "transcripcion_excluida"]
    assert all(int(r["hypothesis_words"]) > 2 * int(r["reference_words"]) for r in excluded)
    assert not any("t2" in v or "spk" in v for r in excluded for v in r.values())  # sin identificadores
