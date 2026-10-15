import csv
import json
import sys

import numpy as np
import pytest

from eer_bootstrap import percentile_ci, stratified_resample
from transcribe import OUTPUT_COLUMNS, normalize_final, normalize_provisional
from wer_bootstrap import (
    NORMALIZATIONS,
    bootstrap_wer,
    corpus_wer,
    main,
    parse_args,
    row_counts,
    speaker_counts,
    wer_values,
)


# ---------------------------------------------------------------------------
# Datos sintéticos
# ---------------------------------------------------------------------------

def row(session_id, client_id, version, reference, hypothesis, stratum="s1"):
    return {"session_id": session_id, "client_id": client_id, "stratum": stratum, "version": version,
            "reference": reference, "hypothesis": hypothesis, "seconds": "1.0"}


WORDS = "el gato come pescado en la casa de mi abuela todos los días por la tarde".split()


def corpus_rows(n_speakers=12, n_strata=3, seed=0):
    """Un clip por hablante; la versión anonimizada tiene más errores que la original."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_speakers):
        reference = rng.choice(WORDS, size=rng.integers(4, 12)).tolist()
        for version, p in (("original", 0.1), ("anonymized", 0.4)):
            hypothesis = [w if rng.random() > p else "x" for w in reference]
            rows.append(row(f"cv_{s}", f"spk{s:02d}", version, " ".join(reference).capitalize() + ".",
                            " ".join(hypothesis), stratum=f"st{s % n_strata}"))
    return rows


# ---------------------------------------------------------------------------
# row_counts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reference, hypothesis, expected", [
    ("El gato come.", "el perro come", (1, 3)),         # 1 sustitución en 3 palabras
    ("El gato come pescado.", "el gato", (2, 4)),       # 2 borrados
    ("Hola.", "hola que tal", (2, 1)),                  # 2 inserciones en 1 palabra
    ("¿Cómo estás?", "Cómo estás", (0, 2)),             # la puntuación no cuenta
    ("Capítulo cuatro.", "Capítulo 4.", (0, 2)),        # normalización final: 4 → cuatro
])
def test_row_counts_a_mano(reference, hypothesis, expected):
    assert row_counts(reference, hypothesis) == expected


def test_row_counts_usa_la_normalizacion_dada():
    assert row_counts("Capítulo cuatro.", "Capítulo 4.", normalize=normalize_provisional) == (1, 2)
    assert row_counts("Hola", "hola", normalize=lambda s: s) == (1, 1)


def test_normalizations_disponibles():
    assert NORMALIZATIONS == {"provisional": normalize_provisional, "final": normalize_final}


# ---------------------------------------------------------------------------
# speaker_counts
# ---------------------------------------------------------------------------

def test_speaker_counts_suma_por_hablante_y_version():
    rows = [
        row("a1", "A", "original", "el gato come", "el gato come"),
        row("a2", "A", "original", "hola", ""),
        row("a1", "A", "anonymized", "el gato come", "el perro"),
        row("a2", "A", "anonymized", "hola", "hola"),
        row("b1", "B", "original", "uno dos", "uno dos", stratum="s2"),
        row("b1", "B", "anonymized", "uno dos", "tres", stratum="s2"),
    ]
    counts = speaker_counts(rows)
    assert counts["client_ids"] == ["A", "B"] and counts["strata"] == ["s1", "s2"]
    assert counts["errors"]["original"].tolist() == [1, 0]
    assert counts["errors"]["anonymized"].tolist() == [2, 2]
    assert counts["words"]["original"].tolist() == [4, 2]
    assert counts["words"]["anonymized"].tolist() == [4, 2]
    assert counts["clips"]["original"].tolist() == [2, 1]


def test_speaker_counts_ordena_por_hablante():
    rows = corpus_rows(n_speakers=5)
    assert speaker_counts(list(reversed(rows)))["client_ids"] == [f"spk{s:02d}" for s in range(5)]


def test_speaker_counts_hablante_sin_una_version_error_claro():
    rows = [row("a1", "A", "original", "hola", "hola"), row("a1", "A", "anonymized", "hola", "hola"),
            row("b1", "B", "original", "hola", "hola")]
    with pytest.raises(ValueError, match=r"1 hablantes no tienen las dos versiones.*'B'"):
        speaker_counts(rows)


def test_speaker_counts_clips_distintos_en_cada_version():
    rows = [row("a1", "A", "original", "hola", "hola"), row("a2", "A", "anonymized", "hola", "hola")]
    with pytest.raises(ValueError, match="clips distintos"):
        speaker_counts(rows)


def test_speaker_counts_fila_repetida():
    rows = [row("a1", "A", "original", "hola", "hola")] * 2
    with pytest.raises(ValueError, match="repetida.*a1"):
        speaker_counts(rows)


def test_speaker_counts_version_desconocida():
    with pytest.raises(ValueError, match="versión desconocida"):
        speaker_counts([row("a1", "A", "otra", "hola", "hola")])


def test_speaker_counts_hablante_en_dos_estratos():
    rows = [row("a1", "A", "original", "hola", "hola", stratum="s1"),
            row("a1", "A", "anonymized", "hola", "hola", stratum="s2")]
    with pytest.raises(ValueError, match="dos estratos"):
        speaker_counts(rows)


def test_speaker_counts_usa_la_normalizacion_dada():
    rows = [row("a1", "A", "original", "capítulo cuatro", "capítulo 4"),
            row("a1", "A", "anonymized", "capítulo cuatro", "capítulo 4")]
    assert speaker_counts(rows)["errors"]["original"].tolist() == [0]
    assert speaker_counts(rows, normalize_provisional)["errors"]["original"].tolist() == [1]


# ---------------------------------------------------------------------------
# corpus_wer y wer_values
# ---------------------------------------------------------------------------

def test_corpus_wer_no_es_el_promedio_por_hablante():
    # Hablante A: 1 error en 1 palabra (100 %); B: 0 en 9 (0 %).
    # Corpus: 1 / 10 = 10 %; promedio por hablante: 50 %.
    errors, words = np.array([1, 0]), np.array([1, 9])
    assert corpus_wer(errors, words) == pytest.approx(0.1)
    assert corpus_wer(errors, words) != pytest.approx(np.mean(errors / words))


def test_corpus_wer_sin_palabras_lanza():
    with pytest.raises(ValueError, match="palabras"):
        corpus_wer(np.array([0]), np.array([0]))


def test_wer_values_todos_los_datos_y_diferencia():
    rows = [row("a1", "A", "original", "hola", ""), row("a1", "A", "anonymized", "hola", "chao"),
            row("b1", "B", "original", "uno dos tres", "uno dos tres"),
            row("b1", "B", "anonymized", "uno dos tres", "uno tres")]
    values = wer_values(speaker_counts(rows))
    assert values["original"] == pytest.approx(1 / 4)
    assert values["anonymized"] == pytest.approx(2 / 4)
    assert values["difference"] == pytest.approx(values["anonymized"] - values["original"])


def test_wer_values_muestra_con_repetidos_pesa_cada_copia():
    rows = [row("a1", "A", "original", "hola", ""), row("a1", "A", "anonymized", "hola", ""),
            row("b1", "B", "original", "uno dos tres", "uno dos tres"),
            row("b1", "B", "anonymized", "uno dos tres", "uno dos tres")]
    counts = speaker_counts(rows)
    # Muestra [A, A, B]: 2 errores en 1 + 1 + 3 palabras.
    assert wer_values(counts, np.array([0, 0, 1]))["original"] == pytest.approx(2 / 5)


# ---------------------------------------------------------------------------
# bootstrap_wer
# ---------------------------------------------------------------------------

def test_bootstrap_wer_forma_y_diferencia():
    counts = speaker_counts(corpus_rows())
    out = bootstrap_wer(counts, 25, seed=0, progress_every=0)
    assert set(out) == {"original", "anonymized", "difference"}
    assert all(v.shape == (25,) for v in out.values())
    np.testing.assert_allclose(out["difference"], out["anonymized"] - out["original"])
    assert out["anonymized"].std() > 0


def test_bootstrap_wer_pareado_versiones_iguales_diferencia_cero():
    rows = [r for r in corpus_rows() if r["version"] == "original"]
    rows += [{**r, "version": "anonymized"} for r in rows]
    out = bootstrap_wer(speaker_counts(rows), 30, seed=0, progress_every=0)
    np.testing.assert_array_equal(out["original"], out["anonymized"])
    assert np.all(out["difference"] == 0)


def test_bootstrap_wer_coincide_replica_por_replica():
    counts = speaker_counts(corpus_rows())
    out = bootstrap_wer(counts, 6, seed=9, progress_every=0)
    rng = np.random.default_rng(9)
    for b in range(6):
        expected = wer_values(counts, stratified_resample(counts["strata"], rng))
        for key in out:
            assert out[key][b] == expected[key]


def test_bootstrap_wer_reproducible_y_depende_de_la_semilla():
    counts = speaker_counts(corpus_rows())
    a = bootstrap_wer(counts, 10, seed=3, progress_every=0)
    b = bootstrap_wer(counts, 10, seed=3, progress_every=0)
    c = bootstrap_wer(counts, 10, seed=4, progress_every=0)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert not np.array_equal(a["anonymized"], c["anonymized"])


def test_bootstrap_wer_imprime_el_avance(capsys):
    bootstrap_wer(speaker_counts(corpus_rows()), 5, seed=0, progress_every=2)
    printed = capsys.readouterr().out
    assert "Réplica 2/5" in printed and "Réplica 4/5" in printed and "Réplica 5/5" not in printed


# ---------------------------------------------------------------------------
# main (de punta a punta)
# ---------------------------------------------------------------------------

def write_tsv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def transcripts(tmp_path):
    rows = corpus_rows()
    # Un clip con cifras para que las dos normalizaciones den resultados distintos.
    rows += [row("cv_n", "spk_n", "original", "Capítulo cuatro.", "Capítulo 4.", stratum="st0"),
             row("cv_n", "spk_n", "anonymized", "Capítulo cuatro.", "Capítulo 4.", stratum="st0")]
    return write_tsv(tmp_path / "t.tsv", rows), rows


def run_main(monkeypatch, tmp_path, path, *extra):
    output = tmp_path / "wer.json"
    monkeypatch.setattr(sys, "argv", ["wer_bootstrap.py", "--transcripts", str(path), "--output", str(output),
                                      "--n-boot", "40", *extra])
    main()
    return json.loads(output.read_text(encoding="utf-8"))


def test_parse_args_valores_por_defecto(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["wer_bootstrap.py", "--transcripts", "t.tsv", "--output", "o.json"])
    args = parse_args()
    assert (args.n_boot, args.boot_seed, args.level, args.model_label, args.normalization) == (
        1000, 2026, 0.95, "", "final")


def test_main_json_de_punta_a_punta(monkeypatch, tmp_path, transcripts):
    path, rows = transcripts
    out = run_main(monkeypatch, tmp_path, path, "--model-label", "turbo")
    assert set(out) == {"transcripts", "model_label", "normalization", "n_boot", "boot_seed", "level", "n_speakers",
                        "speakers_per_stratum", "n_clips", "n_words", "n_errors", "wer"}
    assert (out["model_label"], out["normalization"], out["n_boot"], out["boot_seed"], out["level"]) == (
        "turbo", "final", 40, 2026, 0.95)
    assert out["n_speakers"] == 13 and out["speakers_per_stratum"] == {"st0": 5, "st1": 4, "st2": 4}
    assert out["n_clips"] == {"original": 13, "anonymized": 13}
    counts = speaker_counts(rows)
    assert out["n_words"]["original"] == int(counts["words"]["original"].sum())
    assert out["n_errors"]["anonymized"] == int(counts["errors"]["anonymized"].sum())
    assert set(out["wer"]) == {"original", "anonymized", "difference"}
    for w in out["wer"].values():
        assert set(w) == {"value", "ci_low", "ci_high"} and w["ci_low"] <= w["ci_high"]


def test_main_valor_puntual_de_todos_los_datos(monkeypatch, tmp_path, transcripts):
    path, rows = transcripts
    out = run_main(monkeypatch, tmp_path, path)
    counts = speaker_counts(rows)
    point = wer_values(counts)
    replicates = bootstrap_wer(counts, 40, 2026, progress_every=0)
    for key in point:
        assert out["wer"][key]["value"] == point[key]
        assert (out["wer"][key]["ci_low"], out["wer"][key]["ci_high"]) == pytest.approx(
            percentile_ci(replicates[key], 0.95))
    assert replicates["anonymized"].mean() != pytest.approx(point["anonymized"], abs=1e-9)


def test_main_normalizacion_provisional_y_final(monkeypatch, tmp_path, transcripts):
    path, rows = transcripts
    final = run_main(monkeypatch, tmp_path, path)
    provisional = run_main(monkeypatch, tmp_path, path, "--normalization", "provisional")
    assert provisional["normalization"] == "provisional"
    assert provisional["wer"]["original"]["value"] == wer_values(speaker_counts(rows, normalize_provisional))["original"]
    assert provisional["n_errors"]["original"] == final["n_errors"]["original"] + 1


def test_main_respeta_boot_seed_y_level(monkeypatch, tmp_path, transcripts):
    path, rows = transcripts
    out = run_main(monkeypatch, tmp_path, path, "--boot-seed", "7", "--level", "0.8")
    replicates = bootstrap_wer(speaker_counts(rows), 40, 7, progress_every=0)
    assert (out["boot_seed"], out["level"]) == (7, 0.8)
    w = out["wer"]["difference"]
    assert (w["ci_low"], w["ci_high"]) == pytest.approx(percentile_ci(replicates["difference"], 0.8))


def test_main_hablante_sin_una_version(monkeypatch, tmp_path):
    rows = [r for r in corpus_rows() if not (r["client_id"] == "spk03" and r["version"] == "anonymized")]
    with pytest.raises(ValueError, match="spk03"):
        run_main(monkeypatch, tmp_path, write_tsv(tmp_path / "t.tsv", rows))


def test_main_imprime_resumen_con_coma_decimal(monkeypatch, tmp_path, transcripts, capsys):
    path, _ = transcripts
    out = run_main(monkeypatch, tmp_path, path, "--model-label", "turbo")
    printed = capsys.readouterr().out
    pct = lambda v: f"{100 * v:.2f}".replace(".", ",")
    w = out["wer"]["anonymized"]
    assert f"WER anonimizado: {pct(w['value'])} % (IC 95 %: {pct(w['ci_low'])}–{pct(w['ci_high'])} %)" in printed
    assert "Diferencia (anonimizado menos original)" in printed and " pp" in printed
    assert "normalización: final (turbo)" in printed and "wer.json" in printed
