"""Pruebas del análisis por grupo (estrato y género) con datos de juguete."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")

import group_analysis as ga  # noqa: E402
from eer_bootstrap import copy_trial_rows, resample_pairs, stratified_resample  # noqa: E402
from eer_protocol import split_enrollment_trial  # noqa: E402
from eer_scoring import compute_eer  # noqa: E402
from impersonation import bootstrap_rates, point_rates  # noqa: E402
from run_alpha_sweep import CLIPS_SUBDIR  # noqa: E402
from run_eer import build_row_index, run_scenarios  # noqa: E402
from wer_bootstrap import bootstrap_wer, speaker_counts, wer_values  # noqa: E402

DIM = 8


# ---------------------------------------------------------------------------
# Regla de género y asignación de grupos
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("values, expected", [
    (["male_masculine", "male_masculine", ""], "masculino"),
    (["", "female_feminine"], "femenino"),
    (["", ""], "sin_dato"),
    ([], "sin_dato"),
    (["male_masculine", "female_feminine"], "conflicto"),
    (["male_masculine", "", "intersex"], "conflicto"),
    (["intersex"], "intersex"),
])
def test_speaker_gender_regla(values, expected):
    assert ga.speaker_gender(values) == expected


def test_load_genders_une_por_client_id(tmp_path):
    path = tmp_path / "validated.tsv"
    path.write_text("client_id\tpath\tgender\n" + "".join(f"{c}\tx.mp3\t{g}\n" for c, g in [
        ("a", "male_masculine"), ("a", ""), ("b", "female_feminine"), ("c", ""), ("d", "male_masculine"),
        ("d", "female_feminine"), ("otro", "male_masculine")]), encoding="utf-8")
    assert ga.load_genders(path, ["a", "b", "c", "d", "e"]) == {
        "a": "masculino", "b": "femenino", "c": "sin_dato", "d": "conflicto", "e": "sin_dato"}


STRATA = ["caribe", "caribe", "mexico", "sin_dato", "mexico"]
GENDERS = ["masculino", "femenino", "masculino", "sin_dato", "femenino"]


def test_assign_groups_por_estrato_genero_y_total():
    assert ga.assign_groups(STRATA, GENDERS, "stratum") == STRATA
    assert ga.assign_groups(STRATA, GENDERS, "gender") == GENDERS
    assert ga.assign_groups(STRATA, GENDERS, "all") == ["total"] * 5


def test_assign_groups_solo_un_genero():
    assert ga.assign_groups(STRATA, GENDERS, "stratum", only_gender="masculino") == [
        "caribe", None, "mexico", None, None]


def test_group_codes_orden_y_excluidos():
    groups, codes = ga.group_codes(["mexico", None, "sin_dato", "caribe", "mexico", "conflicto"])
    assert groups == ["caribe", "mexico", "conflicto", "sin_dato"]
    assert codes.tolist() == [1, -1, 3, 0, 1, 2]


def test_test_codes_estrato_excluye_sin_dato():
    groups, codes = ga.group_codes(STRATA)
    keep, recoded = ga.test_codes(groups, codes, "stratum")
    assert keep == ["caribe", "mexico"] and recoded.tolist() == [0, 0, 1, -1, 1]


def test_test_codes_genero_solo_masculino_y_femenino():
    groups, codes = ga.group_codes(GENDERS + ["conflicto", "intersex"])
    keep, recoded = ga.test_codes(groups, codes, "gender")
    assert keep == ["masculino", "femenino"]
    assert recoded.tolist() == [0, 1, 0, -1, 1, -1, -1]


# ---------------------------------------------------------------------------
# Configuraciones y filtro α >= 0,6
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tag, expected", [("a020", (0.2, 0.2)), ("a100", (1.0, 1.0)), ("a072", (0.72, 0.72)),
                                           ("u07_09", (0.7, 0.9)), ("u065_085", (0.65, 0.85)), ("u07_1", (0.7, 1.0))])
def test_parse_tag(tag, expected):
    assert ga.parse_tag(tag) == pytest.approx(expected)


def test_parse_tag_desconocida():
    with pytest.raises(ValueError):
        ga.parse_tag("prueba")


@pytest.mark.parametrize("tag, reliable", [("a020", False), ("a050", False), ("a059", False), ("a060", True),
                                           ("a072", True), ("a100", True), ("u07_09", True), ("u05_08", False)])
def test_wer_reliable_alpha_desde_0_6(tag, reliable):
    alpha_min, alpha_max = ga.parse_tag(tag)
    assert ga.wer_reliable({"alpha_min": alpha_min, "alpha_max": alpha_max}) is reliable


def test_config_files_u07_09_usa_los_nombres_originales(tmp_path):
    f = ga.config_files("u07_09", tmp_path)
    assert (f["embeddings"].name, f["transcripts"].name, f["eer_json"].name, f["impersonation_json"].name) == (
        "emb_k134.npz", "whisper_turbo_k134.tsv", "eer_bootstrap_k134.json", "impersonation_k134.json")
    assert f["manifest"] == tmp_path / "anon_k134" / "anon_manifest_u07_09.tsv"
    g = ga.config_files("a070", tmp_path)
    assert (g["embeddings"].name, g["transcripts"].name) == ("emb_k134_a070.npz", "whisper_turbo_k134_a070.tsv")


def test_discover_configs_entran_solas_y_ordenadas(tmp_path):
    (tmp_path / "anon_k134").mkdir()
    for tag in ("u07_09", "a100", "a072", "a020", "a080"):
        (tmp_path / "anon_k134" / f"anon_manifest_{tag}.tsv").write_text("x")
        f = ga.config_files(tag, tmp_path)
        if tag != "a080":  # sin embeddings: no entra
            f["embeddings"].write_text("x")
    (tmp_path / "anon_k134" / "anon_manifest_prueba.tsv").write_text("x")
    assert [c["tag"] for c in ga.discover_configs(tmp_path)] == ["a020", "a072", "a100", "u07_09"]


# ---------------------------------------------------------------------------
# Prueba de permutación y Holm
# ---------------------------------------------------------------------------

def test_permutation_test_sin_diferencia_real_p_alta():
    rng = np.random.default_rng(0)
    codes = np.repeat(np.arange(4), 50)
    accepted = rng.binomial(2, 0.3, size=200).astype(float)  # misma tasa en todos los grupos
    _, p = ga.permutation_test(accepted, np.full(200, 2.0), codes, 2000, seed=1)
    assert p > 0.2


def test_permutation_test_diferencia_fuerte_p_baja():
    rng = np.random.default_rng(0)
    codes = np.repeat(np.arange(3), 40)
    rates = np.repeat([0.1, 0.5, 0.9], 40)
    accepted = rng.binomial(2, rates).astype(float)
    observed, p = ga.permutation_test(accepted, np.full(120, 2.0), codes, 2000, seed=1)
    assert p == pytest.approx(1 / 2001) and observed > 0.25


def test_permutation_test_p_con_la_formula_y_reproducible():
    codes = np.repeat(np.arange(2), 3)
    num, den = np.array([1, 1, 0, 0, 1, 0.0]), np.ones(6)
    observed, p = ga.permutation_test(num, den, codes, 99, seed=3)
    assert observed == pytest.approx(np.std([2 / 3, 1 / 3]))
    # p = (1 + k) / (1 + N): con N = 99, p * 100 es un entero entre 1 y 100.
    assert p * 100 == pytest.approx(round(p * 100)) and 1 <= round(p * 100) <= 100
    assert ga.permutation_test(num, den, codes, 99, seed=3) == (observed, p)
    # Recuento a mano de las permutaciones con estadístico >= observado.
    rng = np.random.default_rng(3)
    exceed = sum(ga.group_rate_sd(num, den, rng.permutation(codes), 2) >= observed - 1e-12 for _ in range(99))
    assert p == (1 + exceed) / 100


def test_permutation_test_estadistico_es_la_desviacion_entre_grupos():
    codes = np.array([0, 0, 1, 1, 2, 2])
    num = np.array([1, 1, 0, 1, 0, 0.0])
    den = np.array([1, 1, 1, 1, 1, 1.0])
    observed, _ = ga.permutation_test(num, den, codes, 10, seed=0)
    assert observed == pytest.approx(np.std([1.0, 0.5, 0.0]))


def test_permutation_test_conserva_el_tamano_de_cada_grupo(monkeypatch):
    seen = []
    real = ga.group_rate_sd

    def spy(num, den, codes, n_groups):
        seen.append(np.bincount(codes, minlength=n_groups).tolist())
        return real(num, den, codes, n_groups)

    monkeypatch.setattr(ga, "group_rate_sd", spy)
    ga.permutation_test(np.ones(7), np.ones(7), np.array([0, 0, 0, 1, 1, 2, 2]), 20, seed=0)
    assert all(s == [3, 2, 2] for s in seen)


def test_holm_a_mano():
    # Ordenadas: 0,01 (×4), 0,02 (×3), 0,03 (×2), 0,04 (×1) → 0,04; 0,06; 0,06 (monótona); 0,06.
    np.testing.assert_allclose(ga.holm([0.03, 0.01, 0.04, 0.02]), [0.06, 0.04, 0.06, 0.06])


def test_holm_no_pasa_de_uno_y_es_monotona():
    adjusted = ga.holm([0.5, 0.4, 0.01])
    np.testing.assert_allclose(adjusted, [0.8, 0.8, 0.03])
    assert ga.holm([0.9, 0.95]).max() == 1.0


# ---------------------------------------------------------------------------
# Métricas con datos de juguete
# ---------------------------------------------------------------------------

def make_case(n_speakers=12, seed=0):
    rng = np.random.default_rng(seed)
    rows, session_ids, original, anonymized = [], [], [], []
    for s in range(n_speakers):
        center = rng.normal(size=DIM)
        for c in range(5):
            sid = f"cv_{s:02d}_{c}"
            session_ids.append(sid)
            original.append(center + 0.6 * rng.normal(size=DIM))
            anonymized.append(center + 1.5 * rng.normal(size=DIM))
            rows.append({"client_id": f"spk{s:02d}", "stratum": ["caribe", "mexico", "sin_dato"][s % 3],
                         "path": f"{sid}.mp3", "session_id": sid, "coefficient": "0.7", "status": "ok", "error": ""})
    return rows, session_ids, np.array(original, np.float32), np.array(anonymized, np.float32)


@pytest.fixture(scope="module")
def case():
    rows, session_ids, original, anonymized = make_case()
    splits, _ = split_enrollment_trial(rows)
    results = run_scenarios({"original": original, "anonymized": anonymized}, build_row_index(session_ids), splits)
    strata = [s["stratum"] for s in splits.values()]
    transcript_rows = []
    rng = np.random.default_rng(1)
    words = "el gato come pescado en la casa de mi abuela".split()
    for i, cid in enumerate(splits):
        ref = rng.choice(words, size=6).tolist()
        for version, p in (("original", 0.1), ("anonymized", 0.4)):
            hyp = [w if rng.random() > p else "x" for w in ref]
            transcript_rows.append({"session_id": f"t{i}", "client_id": cid, "stratum": strata[i], "version": version,
                                    "reference": " ".join(ref), "hypothesis": " ".join(hyp), "seconds": "1"})
    return {"results": results, "strata": strata, "client_ids": list(splits), "transcripts": transcript_rows}


def test_copy_trial_rows_coincide_con_resample_pairs(case):
    oo = case["results"]["o-o"]
    sample = stratified_resample(case["strata"], np.random.default_rng(3))
    rows, row_copy = copy_trial_rows(oo["owner"], oo["scores"].shape[1], sample)
    target, _ = resample_pairs(oo["scores"], oo["owner"], sample)
    np.testing.assert_array_equal(ga.row_targets(oo)[rows], target)
    np.testing.assert_array_equal(oo["owner"][rows], np.asarray(sample)[row_copy])


def test_un_solo_grupo_reproduce_impersonation(case):
    results, total = case["results"], np.zeros(len(case["strata"]), dtype=int)
    expected = point_rates(results, 0.05)
    thresholds = ga.operating_thresholds(results["o-o"], 0.05)
    counts = ga.impersonation_counts(results, total, 1, thresholds)
    for point in ("eer", "far"):
        assert thresholds[point] == expected[point]["threshold"]
        for rate in ("genuine", "impersonation"):
            assert ga.ratio(*counts[f"{rate}_{point}"])[0] == expected[point][f"{rate}_rate"]


def test_un_solo_grupo_reproduce_el_bootstrap_de_impersonation(case):
    results, strata = case["results"], case["strata"]
    total = np.zeros(len(strata), dtype=int)
    samples = ga.shared_samples(strata, 15, 7)
    reps = [ga.operating_thresholds(results["o-o"], 0.05, s) for s in samples]
    expected = bootstrap_rates(results, strata, 15, 7, 0.05, progress_every=0)
    for b, s in enumerate(samples):
        counts = ga.impersonation_counts(results, total, 1, reps[b], s)
        for point in ("eer", "far"):
            assert reps[b][point] == expected[point]["threshold"][b]
            for rate in ("genuine", "impersonation"):
                assert ga.ratio(*counts[f"{rate}_{point}"])[0] == expected[point][f"{rate}_rate"][b]


def test_un_solo_grupo_reproduce_el_eer(case):
    total = np.zeros(len(case["strata"]), dtype=int)
    for scenario in ga.SCENARIOS:
        assert ga.eer_by_group(case["results"][scenario], total, 1)[0] == case["results"][scenario]["eer"]


def test_eer_dentro_del_grupo_usa_solo_sus_hablantes(case):
    groups, codes = ga.group_codes(case["strata"])
    result = case["results"]["o-a"]
    eers = ga.eer_by_group(result, codes, len(groups))
    for g in range(len(groups)):
        members = np.flatnonzero(codes == g)
        assert eers[g] == compute_eer(*resample_pairs(result["scores"], result["owner"], members))[0]


def test_eer_grupo_de_un_hablante_es_nan(case):
    codes = np.full(len(case["strata"]), -1)
    codes[0] = 0
    assert np.isnan(ga.eer_by_group(case["results"]["o-o"], codes, 1)[0])


def test_grupos_suman_el_total_en_suplantacion(case):
    results = case["results"]
    groups, codes = ga.group_codes(case["strata"])
    thresholds = ga.operating_thresholds(results["o-o"], 0.05)
    total = ga.impersonation_counts(results, np.zeros(codes.size, dtype=int), 1, thresholds)
    by_group = ga.impersonation_counts(results, codes, len(groups), thresholds)
    for key, (accepted, pairs) in by_group.items():
        assert accepted.sum() == total[key][0][0] and pairs.sum() == total[key][1][0]
        assert np.sum(ga.ratio(accepted, pairs) * pairs) / pairs.sum() == pytest.approx(ga.ratio(*total[key])[0])


def test_wer_por_grupo_suma_el_total_y_reproduce_wer_bootstrap(case):
    counts = ga.aligned_wer_counts_from_rows(case["transcripts"], case["client_ids"])
    groups, codes = ga.group_codes(case["strata"])
    by_group = ga.wer_by_group(counts, codes, len(groups))
    total = ga.wer_by_group(counts, np.zeros(codes.size, dtype=int), 1)
    expected = wer_values(counts)
    for key in ("original", "anonymized", "difference"):
        assert total[key][0] == expected[key]
    assert by_group["words"].sum() == total["words"][0]
    assert by_group["errors_anonymized"].sum() == total["errors_anonymized"][0]
    np.testing.assert_allclose(by_group["difference"], by_group["anonymized"] - by_group["original"])


def test_wer_total_reproduce_el_bootstrap_de_wer(case):
    counts = ga.aligned_wer_counts_from_rows(case["transcripts"], case["client_ids"])
    total = np.zeros(len(case["strata"]), dtype=int)
    samples = ga.shared_samples(counts["strata"], 10, 5)
    expected = bootstrap_wer(counts, 10, 5, progress_every=0)
    for b, s in enumerate(samples):
        assert ga.wer_by_group(counts, total, 1, s)["difference"][0] == expected["difference"][b]


def test_aligned_wer_counts_reordena_o_rechaza(case):
    reversed_ids = list(reversed(case["client_ids"]))
    counts = ga.aligned_wer_counts_from_rows(case["transcripts"], reversed_ids)
    plain = speaker_counts(case["transcripts"])
    assert counts["client_ids"] == reversed_ids
    np.testing.assert_array_equal(counts["words"]["original"], plain["words"]["original"][::-1])
    with pytest.raises(ValueError, match="no coinciden"):
        ga.aligned_wer_counts_from_rows(case["transcripts"], case["client_ids"][:-1])


def test_speaker_level_para_las_permutaciones(case):
    results = case["results"]
    thresholds = ga.operating_thresholds(results["o-o"], 0.05)
    counts = ga.aligned_wer_counts_from_rows(case["transcripts"], case["client_ids"])
    values = ga.speaker_level(results, counts, thresholds, len(case["strata"]))
    accepted, pairs = values["impersonation_far"]
    total = ga.impersonation_counts(results, np.zeros(len(case["strata"]), dtype=int), 1, thresholds)
    assert accepted.sum() == total["impersonation_far"][0][0] and pairs.tolist() == [2.0] * len(case["strata"])
    diff, words = values["wer_difference"]
    assert diff.sum() / words.sum() == pytest.approx(wer_values(counts)["difference"])


def test_summary_rows_marca_alpha_bajo(case):
    groups, codes = ga.group_codes(case["strata"])
    counts = ga.aligned_wer_counts_from_rows(case["transcripts"], case["client_ids"])
    samples = ga.shared_samples(case["strata"], 3, 1)
    reps = [ga.operating_thresholds(case["results"]["o-o"], 0.05, s) for s in samples]
    analysis = ga.analyze_config(case["results"], counts, codes, len(groups), samples, reps, 0.05)
    low = ga.summary_rows({"tag": "a050", "alpha_min": 0.5, "alpha_max": 0.5}, groups, codes, analysis, 0.95)
    high = ga.summary_rows({"tag": "a060", "alpha_min": 0.6, "alpha_max": 0.6}, groups, codes, analysis, 0.95)
    assert {r["wer_note"] for r in low} == {"dominado por alucinaciones"} and {r["wer_note"] for r in high} == {""}
    assert [r["n_speakers"] for r in low] == [4, 4, 4]
    assert low[0]["impersonation_far_ci_low"] <= low[0]["impersonation_far_ci_high"]


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def cfg(tag):
    alpha_min, alpha_max = ga.parse_tag(tag)
    return {"tag": tag, "alpha_min": alpha_min, "alpha_max": alpha_max}


def test_heatmap_layout_sin_dato_abajo_y_rangos_a_la_derecha():
    rows, columns = ga.heatmap_layout(["caribe", "mexico", "sin_dato"], [cfg("a060"), cfg("a070"), cfg("u07_09")])
    assert rows == ["caribe", "mexico", None, "sin_dato"]
    assert [c["tag"] if c else None for c in columns] == ["a060", "a070", None, "u07_09"]


def summary_toy():
    rows = []
    for tag in ("a050", "a060", "a070", "u07_09"):
        for k, group in enumerate(("caribe", "mexico", "sin_dato", "total")):
            rows.append({"config": tag, "group": group, "impersonation_far": 0.1 * k + 0.01,
                         "impersonation_far_ci_low": 0.1 * k, "impersonation_far_ci_high": 0.1 * k + 0.03,
                         "wer_difference": 0.02 * k, "wer_difference_ci_low": 0.02 * k - 0.01,
                         "wer_difference_ci_high": 0.02 * k + 0.015})
    return rows


def test_plot_heatmap_valores_en_cada_celda_y_huecos():
    configs = [cfg("a060"), cfg("a070"), cfg("u07_09")]
    fig = ga.plot_heatmap(summary_toy(), ["caribe", "mexico", "sin_dato"], configs, "impersonation_far", "x")
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    assert len(texts) == 3 * 3 and texts.count("21,0") == 3  # sin_dato (k = 2): 21 %
    matrix = ax.images[0].get_array()
    assert matrix.mask[2].all() and matrix.mask[:, 2].all()  # fila y columna de separación
    assert [t.get_text() for t in ax.get_yticklabels()] == ["Caribe", "México", "Sin dato"]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["0,6", "0,7", "U(0,7; 0,9)"]
    assert ax.images[0].get_cmap().name.startswith("Blues")


def test_plot_groups_vs_alpha_solo_alpha_fijo_y_wer_desde_0_6():
    configs = [cfg("a050"), cfg("a060"), cfg("a070"), cfg("u07_09")]
    fig = ga.plot_groups_vs_alpha(summary_toy(), configs, [("caribe", "Caribe", "#0072B2", "-"),
                                                           ("total", "Total", "#000000", ":")])
    top, bottom = fig.axes
    lines = [l for l in top.get_lines() if l.get_marker() == "o"]
    assert sorted(round(x, 2) for x in np.concatenate([l.get_xdata() for l in lines])) == [
        0.49, 0.51, 0.59, 0.61, 0.69, 0.71]  # α 0,5–0,7 desplazados ±0,012, sin el rango
    lines = [l for l in bottom.get_lines() if l.get_marker() == "o"]
    assert all(min(l.get_xdata()) > 0.55 for l in lines)
    assert [t.get_text() for t in fig.legends[0].get_texts()] == ["Caribe", "Total"]


# ---------------------------------------------------------------------------
# main de punta a punta con un ..\Datos de juguete
# ---------------------------------------------------------------------------

def write_tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def toy_data(tmp_path, case):
    rows, session_ids, original, anonymized = make_case()
    for tag, noise in (("u07_09", 1.5), ("a050", 3.0), ("a070", 1.0)):
        f = ga.config_files(tag, tmp_path)
        write_tsv(f["manifest"], rows)
        anon = original + noise * np.random.default_rng(len(tag) + int(noise)).normal(size=original.shape)
        np.savez(f["embeddings"], session_ids=np.array(session_ids, dtype=str), original=original,
                 anonymized=anon.astype(np.float32))
        write_tsv(f["transcripts"], case["transcripts"])
    genders = ["male_masculine", "female_feminine", "male_masculine", ""]
    validated = [{"client_id": f"spk{s:02d}", "path": "x.mp3", "gender": genders[s % 4]} for s in range(12)]
    write_tsv(tmp_path / CLIPS_SUBDIR.parent / "validated.tsv", validated)
    return tmp_path


def run_main(monkeypatch, data, *extra):
    monkeypatch.setattr(sys, "argv", ["group_analysis.py", "--data-dir", str(data), "--n-boot", "5",
                                      "--n-perm", "30", "--max-far", "0.05", *extra])
    with matplotlib.rc_context():
        ga.main()


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_main_por_estrato_tablas_pruebas_y_figuras(monkeypatch, toy_data):
    run_main(monkeypatch, toy_data, "--group-by", "stratum")
    summary = read(toy_data / "por_grupo" / "resumen_estrato.tsv")
    assert [(r["config"], r["group"]) for r in summary][:5] == [
        ("a050", "caribe"), ("a050", "mexico"), ("a050", "sin_dato"), ("a050", "total"), ("a070", "caribe")]
    assert len(summary) == 3 * 4 and list(summary[0]) == ga.SUMMARY_COLUMNS
    assert {r["wer_note"] for r in summary if r["config"] == "a050"} == {"dominado por alucinaciones"}
    tests = read(toy_data / "por_grupo" / "pruebas_permutacion.tsv")
    assert [(t["metric"], t["config"]) for t in tests] == [
        ("impersonation_far", "a050"), ("impersonation_far", "a070"), ("wer_difference", "a070"),
        ("impersonation_far", "u07_09"), ("wer_difference", "u07_09")]
    assert {t["groups"] for t in tests} == {"caribe,mexico"} and {t["grouping"] for t in tests} == {"estrato"}
    assert all(float(t["p_holm"]) >= float(t["p_value"]) for t in tests)
    for name in ("suplantacion_estrato.pdf", "wer_estrato.pdf", "colombia.pdf"):
        assert (toy_data / "figuras" / "por_grupo" / name).read_bytes()[:4] == b"%PDF"


def test_main_genero_y_solo_masculino_se_suman_a_las_pruebas(monkeypatch, toy_data):
    run_main(monkeypatch, toy_data, "--group-by", "stratum")
    run_main(monkeypatch, toy_data, "--group-by", "gender")
    run_main(monkeypatch, toy_data, "--group-by", "stratum", "--only-gender", "masculino")
    tests = read(toy_data / "por_grupo" / "pruebas_permutacion.tsv")
    assert {t["grouping"] for t in tests} == {"estrato", "genero", "estrato_solo_masculino"}
    assert {t["groups"] for t in tests if t["grouping"] == "genero"} == {"masculino,femenino"}
    genero = read(toy_data / "por_grupo" / "resumen_genero.tsv")
    assert {r["group"] for r in genero} == {"masculino", "femenino", "sin_dato", "total"}
    solo = read(toy_data / "por_grupo" / "resumen_estrato_solo_masculino.tsv")
    assert {r["n_speakers"] for r in solo if r["group"] == "total"} == {"6"}
    assert (toy_data / "figuras" / "por_grupo" / "genero.pdf").exists()


def test_main_repetido_no_duplica_las_pruebas(monkeypatch, toy_data):
    run_main(monkeypatch, toy_data, "--group-by", "gender")
    run_main(monkeypatch, toy_data, "--group-by", "gender")
    assert len(read(toy_data / "por_grupo" / "pruebas_permutacion.tsv")) == 5


def test_main_holm_por_metrica_entre_configuraciones(monkeypatch, toy_data):
    run_main(monkeypatch, toy_data, "--group-by", "stratum")
    tests = read(toy_data / "por_grupo" / "pruebas_permutacion.tsv")
    for metric in ("impersonation_far", "wer_difference"):
        rows = [t for t in tests if t["metric"] == metric]
        np.testing.assert_allclose([float(t["p_holm"]) for t in rows], ga.holm([float(t["p_value"]) for t in rows]))


def test_main_total_igual_a_los_valores_globales(monkeypatch, toy_data, case):
    run_main(monkeypatch, toy_data, "--group-by", "all", "--configs", "u07_09")
    (row,) = read(toy_data / "por_grupo" / "resumen_total.tsv")[:1]
    f = ga.config_files("u07_09", toy_data)
    from run_eer import load_and_score
    _, _, results = load_and_score(f["embeddings"], f["manifest"], 42, 3, 2)
    expected = point_rates(results, 0.05)
    assert float(row["impersonation_far"]) == expected["far"]["impersonation_rate"]
    assert float(row["eer_oa"]) == results["o-a"]["eer"]


def test_main_umbral_global_recalculado_en_cada_replica(monkeypatch, toy_data):
    from eer_bootstrap import percentile_ci
    from run_eer import load_and_score
    run_main(monkeypatch, toy_data, "--group-by", "all", "--configs", "u07_09")
    (row,) = read(toy_data / "por_grupo" / "resumen_total.tsv")
    f = ga.config_files("u07_09", toy_data)
    splits, _, results = load_and_score(f["embeddings"], f["manifest"], 42, 3, 2)
    strata = [s["stratum"] for s in splits.values()]
    expected = bootstrap_rates(results, strata, 5, 2026, 0.05, progress_every=0)
    for point in ("eer", "far"):
        low, high = percentile_ci(expected[point]["impersonation_rate"])
        assert (float(row[f"impersonation_{point}_ci_low"]), float(row[f"impersonation_{point}_ci_high"])) == \
            pytest.approx((low, high), abs=1e-12)


def test_main_wer_estrato_solo_con_alpha_desde_0_6(monkeypatch, toy_data):
    calls = []
    real = ga.plot_heatmap
    monkeypatch.setattr(ga, "plot_heatmap", lambda s, g, c, m, *a, **k: calls.append((m, [x["tag"] for x in c]))
                        or real(s, g, c, m, *a, **k))
    run_main(monkeypatch, toy_data, "--group-by", "stratum")
    assert dict(calls) == {"impersonation_far": ["a050", "a070", "u07_09"], "wer_difference": ["a070", "u07_09"]}


def test_main_total_toma_el_ic_del_eer_del_json(monkeypatch, toy_data):
    f = ga.config_files("u07_09", toy_data)
    scenarios = {"o-o": (0.01, 0.02), "o-a": (0.11, 0.12), "a-a": (0.21, 0.22)}
    f["eer_json"].write_text(json.dumps({"level": 0.95, "scenarios": {
        s: {"eer": 0.0, "ci_low": lo, "ci_high": hi} for s, (lo, hi) in scenarios.items()}}), encoding="utf-8")
    run_main(monkeypatch, toy_data, "--group-by", "stratum", "--configs", "u07_09")
    (row,) = [r for r in read(toy_data / "por_grupo" / "resumen_estrato.tsv") if r["group"] == "total"]
    for s, name in zip(ga.SCENARIOS, ("eer_oo", "eer_oa", "eer_aa")):
        assert (float(row[f"{name}_ci_low"]), float(row[f"{name}_ci_high"])) == scenarios[s]


def test_adjust_by_metric_holm_por_separado_en_cada_metrica():
    rows = [{"metric": "impersonation_far", "config": c, "p_value": p} for c, p in (("a", 0.01), ("b", 0.04))]
    rows += [{"metric": "wer_difference", "config": c, "p_value": p} for c, p in (("a", 0.02), ("b", 0.03), ("c", 0.5))]
    ga.adjust_by_metric(rows)
    np.testing.assert_allclose([r["p_holm"] for r in rows[:2]], ga.holm([0.01, 0.04]))
    np.testing.assert_allclose([r["p_holm"] for r in rows[2:]], ga.holm([0.02, 0.03, 0.5]))
    assert rows[0]["p_holm"] == pytest.approx(0.02)  # mezclando las 5 sería 0,05


def test_main_figures_only_redibuja_sin_recalcular(monkeypatch, toy_data):
    run_main(monkeypatch, toy_data, "--group-by", "stratum")
    figures = toy_data / "figuras" / "por_grupo"
    for pdf in figures.glob("*.pdf"):
        pdf.unlink()
    monkeypatch.setattr(ga, "load_and_score", lambda *a, **k: pytest.fail("no debía recalcular"))
    run_main(monkeypatch, toy_data, "--group-by", "stratum", "--figures-only")
    assert sorted(p.name for p in figures.glob("*.pdf")) == ["colombia.pdf", "suplantacion_estrato.pdf",
                                                              "wer_estrato.pdf"]


def test_load_summary_lee_lo_que_escribe_write_tsv(tmp_path):
    rows = [{"config": "a070", "group": "caribe", "impersonation_far": 0.25, "wer_note": "", "n_words": None}]
    ga.write_tsv(rows, ["config", "group", "impersonation_far", "wer_note", "n_words"], tmp_path / "r.tsv")
    assert ga.load_summary(tmp_path / "r.tsv") == [{"config": "a070", "group": "caribe", "impersonation_far": 0.25,
                                                    "wer_note": "", "n_words": None}]
