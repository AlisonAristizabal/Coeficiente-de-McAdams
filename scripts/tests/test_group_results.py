"""Pruebas de los resultados por acento y género de un intervalo de α, con datos de juguete."""
import csv
import json

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")

import group_analysis as ga  # noqa: E402
import group_results as gr  # noqa: E402
from eer_bootstrap import resample_pairs, stratified_resample  # noqa: E402
from eer_protocol import split_enrollment_trial  # noqa: E402
from eer_scoring import compute_eer  # noqa: E402
from range_comparison import hallucination_diagnostic  # noqa: E402
from run_alpha_sweep import CLIPS_SUBDIR  # noqa: E402
from run_eer import build_row_index, run_scenarios  # noqa: E402
from wer_bootstrap import wer_values  # noqa: E402

DIM = 8
STRATA = ("caribe", "chileno", "mexico", "sin_dato")
GENDERS = ("male_masculine", "female_feminine", "male_masculine", "")


# ---------------------------------------------------------------------------
# Filtrado de pares dentro del grupo
# ---------------------------------------------------------------------------

def encoded_result(n_speakers=6, n_trials=2):
    """Escenario cuyas puntuaciones dicen de qué prueba y registro salen: 100·dueño + registro."""
    owner = np.repeat(np.arange(n_speakers), n_trials)
    scores = 100.0 * owner[:, None] + np.arange(n_speakers)[None, :]
    return {"scores": scores, "owner": owner}


def decode(scores):
    """(hablante de la prueba, hablante del registro) de cada puntuación de `encoded_result`."""
    scores = np.asarray(scores).astype(int)
    return scores // 100, scores % 100


def test_within_group_pairs_solo_pares_del_grupo():
    codes = np.array([0, 1, 0, 1, -1, 0])
    target, impostor = gr.within_group_pairs(encoded_result(), codes, 0)
    trial, enroll = decode(target)
    assert set(trial) == {0, 2, 5} and np.array_equal(trial, enroll)
    trial, enroll = decode(impostor)
    assert set(trial) <= {0, 2, 5} and set(enroll) <= {0, 2, 5} and np.all(trial != enroll)
    # 3 hablantes × 2 pruebas: 6 objetivo y 6 × 2 impostores.
    assert (target.size, impostor.size) == (6, 12)
    # El hablante sin grupo (-1) no aparece en ningún grupo.
    for g in (0, 1):
        assert 4 not in np.concatenate(decode(np.concatenate(gr.within_group_pairs(encoded_result(), codes, g))))


def test_within_group_pairs_excluye_copias_del_mismo_hablante():
    codes = np.array([0, 1, 0, 1, -1, 0])
    sample = np.array([0, 0, 2, 3, 4, 5])  # el hablante 0 sale dos veces y el 1 ninguna
    target, impostor = gr.within_group_pairs(encoded_result(), codes, 0, sample)
    # 4 copias × 2 pruebas objetivo; impostores: filas del 0 (4) contra 2 y 5, filas del 2 y del 5
    # (2 + 2) contra las dos copias del 0 y el otro → 4·2 + 2·3 + 2·3 = 20.
    assert (target.size, impostor.size) == (8, 20)
    trial, enroll = decode(impostor)
    assert np.all(trial != enroll)  # ningún par entre copias del hablante 0
    assert np.array_equal(*decode(target))
    np.testing.assert_array_equal(np.sort(np.concatenate(gr.within_group_pairs(encoded_result(), codes, 0, sample))),
                                  np.sort(np.concatenate(resample_pairs(encoded_result()["scores"],
                                                                        encoded_result()["owner"], [0, 0, 2, 5]))))


def test_pair_counts_por_grupo():
    codes = np.array([0, 1, 0, 1, -1, 0])
    assert gr.pair_counts(encoded_result(), codes, 2) == [(6, 12), (4, 4)]


def test_within_group_pairs_da_el_eer_de_eer_by_group(case):
    groups, codes = ga.group_codes(case["strata"])
    sample = stratified_resample(case["strata"], np.random.default_rng(4))
    for scenario in ("o-o", "o-a", "a-a"):
        result = case["results"][scenario]
        for s in (None, sample):
            expected = ga.eer_by_group(result, codes, len(groups), s)
            for g in range(len(groups)):
                assert compute_eer(*gr.within_group_pairs(result, codes, g, s))[0] == expected[g]


def test_coding_respeta_el_orden_y_deja_fuera_lo_demas():
    assert gr.coding(["femenino", "sin_dato", "masculino", "femenino"], ["masculino", "femenino"]).tolist() == [
        1, -1, 0, 1]


# ---------------------------------------------------------------------------
# Permutación con semilla fija
# ---------------------------------------------------------------------------

def rate_by_group(values, n_groups):
    """Función de grupo → promedio de `values` en cada grupo, para la prueba de permutación."""
    values = np.asarray(values, float)

    def group_values(codes):
        valid = codes >= 0
        sums = np.bincount(codes[valid], weights=values[valid], minlength=n_groups)
        return {"m": sums / np.bincount(codes[valid], minlength=n_groups)}
    return group_values


def test_range_statistic_maximo_menos_minimo_sin_nan():
    assert gr.range_statistic([0.2, np.nan, 0.05, 0.3]) == pytest.approx(0.25)


def test_range_permutation_test_estadistico_y_formula_a_mano():
    values = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0.0])
    codes = np.array([0, 0, 0, 1, 1, 1, -1, 2, 2])
    out = gr.range_permutation_test(rate_by_group(values, 3), codes, 99, seed=2026)["m"]
    np.testing.assert_allclose(out["values"], [2 / 3, 1 / 3, 1 / 2])
    assert out["observed"] == pytest.approx(1 / 3)
    rng = np.random.default_rng(2026)
    included = np.flatnonzero(codes >= 0)
    exceed = 0
    for _ in range(99):
        permuted = codes.copy()
        permuted[included] = rng.permutation(codes[included])
        exceed += gr.range_statistic(rate_by_group(values, 3)(permuted)["m"]) >= out["observed"] - 1e-12
    assert out["p_value"] == (1 + exceed) / 100


def test_range_permutation_test_reproducible_con_la_misma_semilla():
    rng = np.random.default_rng(0)
    values, codes = rng.random(60), np.repeat(np.arange(3), 20)
    first = gr.range_permutation_test(rate_by_group(values, 3), codes, 200, seed=2026)
    again = gr.range_permutation_test(rate_by_group(values, 3), codes, 200, seed=2026)
    assert first["m"]["p_value"] == again["m"]["p_value"]

    def permutations(seed):
        seen = []

        def spy(c):
            seen.append(c.copy())
            return {"m": np.zeros(3)}
        gr.range_permutation_test(spy, codes, 5, seed)
        return np.array(seen[1:])  # la primera llamada es la observada
    assert np.array_equal(permutations(2026), permutations(2026))
    assert not np.array_equal(permutations(2026), permutations(7))


def test_range_permutation_test_no_mueve_excluidos_y_conserva_tamanos():
    codes = np.array([0, 0, 0, -1, 1, 1, -1, 2, 2, 2])
    seen = []

    def spy(c):
        seen.append(c.copy())
        return {"m": np.zeros(3)}
    gr.range_permutation_test(spy, codes, 30, seed=1)
    assert len(seen) == 31
    for c in seen:
        assert np.array_equal(np.flatnonzero(c == -1), [3, 6])
        assert np.bincount(c[c >= 0], minlength=3).tolist() == [3, 2, 3]


def test_range_permutation_test_todas_las_metricas_con_las_mismas_permutaciones():
    codes = np.repeat(np.arange(3), 4)
    seen = []

    def two_metrics(c):
        seen.append(c.copy())
        return {"a": np.bincount(c, minlength=3).astype(float), "b": np.zeros(3)}
    out = gr.range_permutation_test(two_metrics, codes, 10, seed=3)
    assert set(out) == {"a", "b"} and len(seen) == 11
    assert out["b"]["p_value"] == 1.0  # estadístico 0 en todas las permutaciones


def test_range_permutation_test_diferencia_fuerte_p_minima_y_sin_diferencia_p_alta():
    rng = np.random.default_rng(0)
    codes = np.repeat(np.arange(3), 40)
    strong = rng.normal(np.repeat([0.0, 1.0, 2.0], 40), 0.1)
    assert gr.range_permutation_test(rate_by_group(strong, 3), codes, 500, 2026)["m"]["p_value"] == pytest.approx(
        1 / 501)
    flat = rng.normal(0.0, 1.0, size=120)
    assert gr.range_permutation_test(rate_by_group(flat, 3), codes, 500, 2026)["m"]["p_value"] > 0.05


# ---------------------------------------------------------------------------
# Diferencia de género
# ---------------------------------------------------------------------------

def test_group_difference_puntual_y_percentiles_a_mano():
    point = np.array([0.30, 0.10])
    reps = np.array([[0.31, 0.09], [0.28, 0.12], [0.35, 0.10], [0.29, 0.11], [0.30, 0.08]])
    estimate, low, high = gr.group_difference(point, reps, 0, 1, level=0.9)
    assert estimate == pytest.approx(0.20)
    expected = np.percentile(reps[:, 0] - reps[:, 1], [5, 95])
    assert (low, high) == pytest.approx(tuple(expected))


def test_group_difference_es_pareado():
    rng = np.random.default_rng(0)
    female = rng.uniform(0.05, 0.25, size=200)
    reps = np.column_stack([female + 0.02, female])  # los grupos varían juntos réplica por réplica
    estimate, low, high = gr.group_difference(np.array([0.17, 0.15]), reps, 0, 1)
    assert estimate == pytest.approx(0.02) and (low, high) == pytest.approx((0.02, 0.02))
    # Con réplicas no pareadas el IC sería mucho más ancho.
    unpaired = np.column_stack([female + 0.02, rng.permutation(female)])
    _, low, high = gr.group_difference(np.array([0.17, 0.15]), unpaired, 0, 1)
    assert high - low > 0.1


def test_group_difference_orden_y_nan():
    reps = np.array([[0.1, 0.3], [np.nan, 0.2], [0.2, 0.4]])
    estimate, low, high = gr.group_difference(np.array([0.15, 0.3]), reps, 0, 1)
    assert estimate == pytest.approx(-0.15) and (low, high) == pytest.approx(
        tuple(np.percentile([-0.2, -0.2], [2.5, 97.5])))
    assert gr.group_difference(np.array([0.15, 0.3]), reps, 1, 0)[0] == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# Datos de juguete
# ---------------------------------------------------------------------------

def make_case(n_speakers=16, seed=0):
    rng = np.random.default_rng(seed)
    rows, session_ids, original, anonymized = [], [], [], []
    for s in range(n_speakers):
        center = rng.normal(size=DIM)
        for c in range(5):
            sid = f"cv_{s:02d}_{c}"
            session_ids.append(sid)
            original.append(center + 0.6 * rng.normal(size=DIM))
            anonymized.append(center + 1.5 * rng.normal(size=DIM))
            rows.append({"client_id": f"spk{s:02d}", "stratum": STRATA[s % 4], "path": f"{sid}.mp3",
                         "session_id": sid, "coefficient": str(0.685 + 0.18 * rng.random()), "status": "ok",
                         "error": ""})
    return rows, session_ids, np.array(original, np.float32), np.array(anonymized, np.float32)


def make_transcripts(client_ids, strata):
    rng = np.random.default_rng(1)
    words = "el gato come pescado en la casa de mi abuela".split()
    rows = []
    for i, cid in enumerate(client_ids):
        ref = rng.choice(words, size=6).tolist()
        for version, p in (("original", 0.1), ("anonymized", 0.4)):
            hyp = [w if rng.random() > p else "x" for w in ref]
            if version == "anonymized" and i in (2, 9):
                hyp = hyp * 3  # alucinación: el triple de palabras que la referencia
            rows.append({"session_id": f"t{i}", "client_id": cid, "stratum": strata[i], "version": version,
                         "reference": " ".join(ref), "hypothesis": " ".join(hyp), "seconds": "1"})
    return rows


@pytest.fixture(scope="module")
def case():
    rows, session_ids, original, anonymized = make_case()
    splits, _ = split_enrollment_trial(rows)
    results = run_scenarios({"original": original, "anonymized": anonymized}, build_row_index(session_ids), splits)
    strata = [s["stratum"] for s in splits.values()]
    transcripts = make_transcripts(list(splits), strata)
    return {"results": results, "strata": strata, "client_ids": list(splits), "transcripts": transcripts,
            "manifest": rows, "counts": ga.aligned_wer_counts_from_rows(transcripts, list(splits))}


def test_group_metrics_un_solo_grupo_da_los_valores_globales(case):
    total = np.zeros(len(case["client_ids"]), dtype=int)
    metrics = gr.group_metrics(case["results"], case["counts"], total, 1)
    for name, scenario in (("eer_oo", "o-o"), ("eer_oa", "o-a"), ("eer_aa", "a-a")):
        assert metrics[name][0] == case["results"][scenario]["eer"]
    expected = wer_values(case["counts"])
    assert (metrics["wer_original"][0], metrics["wer_anonymized"][0]) == (expected["original"],
                                                                          expected["anonymized"])


def test_bootstrap_groups_misma_muestra_para_todas_las_agrupaciones(case):
    labels = [["masculino", "femenino", "masculino", "sin_dato"][i // 4] for i in range(len(case["client_ids"]))]
    gender = gr.coding(labels, ["masculino", "femenino"])
    _, accent = ga.group_codes(case["strata"])
    samples = ga.shared_samples(case["strata"], 6, 2026)
    reps = gr.bootstrap_groups(case["results"], case["counts"],
                               {"acento": (accent, 4, gr.ALL_METRICS), "genero": (gender, 2, gr.TEST_METRICS)},
                               samples, progress_every=0)
    assert reps["acento"]["eer_oa"].shape == (6, 4) and set(reps["genero"]) == set(gr.TEST_METRICS)
    for b, s in enumerate(samples):
        np.testing.assert_array_equal(reps["genero"]["eer_oa"][b], ga.eer_by_group(case["results"]["o-a"], gender, 2, s))
        np.testing.assert_array_equal(reps["acento"]["wer_anonymized"][b],
                                      ga.wer_by_group(case["counts"], accent, 4, s)["anonymized"])


def test_over_double_by_speaker_coincide_con_range_comparison(case):
    flags = gr.over_double_by_speaker(case["transcripts"], case["client_ids"])
    summary, _ = hallucination_diagnostic(case["transcripts"], {})
    assert flags.sum() == summary["n_hallucinated"] == 2 and np.flatnonzero(flags).tolist() == [2, 9]
    with pytest.raises(ValueError, match="no está en la partición"):
        gr.over_double_by_speaker(case["transcripts"], case["client_ids"][:-1])


def test_alpha_stats_sobre_todos_los_clips_del_grupo():
    manifest = [{"client_id": c, "coefficient": a, "status": st} for c, a, st in [
        ("a", "0.70", "ok"), ("a", "0.80", "ok"), ("b", "0.75", "ok"), ("b", "0.90", "error"), ("c", "0.86", "ok")]]
    alphas = gr.speaker_alphas(manifest, ["a", "b", "c"])
    stats = gr.alpha_stats(alphas, [0, 1])
    assert stats["alpha_mean"] == pytest.approx(0.75) and stats["alpha_sd"] == pytest.approx(0.05)
    assert (stats["alpha_min"], stats["alpha_max"]) == (0.70, 0.80)  # el clip con error no cuenta
    assert gr.alpha_stats(alphas, [])["alpha_mean"] is None


def test_eer_ci_from_json_solo_con_los_mismos_parametros(tmp_path, case):
    expected = {"seed": 42, "n_enroll": 3, "n_trial": 2, "n_boot": 1000, "boot_seed": 2026, "level": 0.95}
    scenarios = {s: {"eer": case["results"][s]["eer"], "ci_low": 0.01, "ci_high": 0.2} for s in ("o-o", "o-a", "a-a")}
    path = tmp_path / "eer.json"
    path.write_text(json.dumps({**expected, "scenarios": scenarios}), encoding="utf-8")
    assert gr.eer_ci_from_json(path, expected, case["results"]) == {s: (0.01, 0.2) for s in scenarios}
    assert gr.eer_ci_from_json(path, {**expected, "n_boot": 10}, case["results"]) is None
    assert gr.eer_ci_from_json(tmp_path / "no.json", expected, case["results"]) is None
    scenarios["o-a"]["eer"] += 0.01
    path.write_text(json.dumps({**expected, "scenarios": scenarios}), encoding="utf-8")
    assert gr.eer_ci_from_json(path, expected, case["results"]) is None


# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------

def summary_toy():
    rows = []
    for k, group in enumerate(("caribe", "chileno", "mexico", "sin_dato")):
        rows.append({"grouping": "acento", "group": group, "label": ga.GROUP_LABELS[group],
                     "eer_oo": 0.02, "eer_oa": 0.15 + 0.01 * ((k * 3) % 4), "eer_aa": 0.09,
                     "wer_original": 0.04, "wer_anonymized": 0.12})
    for group in ("masculino", "femenino", "sin_dato"):
        rows.append({"grouping": "genero", "group": group, "label": ga.GROUP_LABELS[group],
                     "eer_oo": 0.02, "eer_oa": 0.18, "eer_aa": 0.1, "wer_original": 0.04, "wer_anonymized": 0.11})
    for row in rows:
        for m in gr.ALL_METRICS:
            row[f"{m}_ci_low"], row[f"{m}_ci_high"] = row[m] - 0.01, row[m] + 0.02
    rows[-1].update({m: None for m in gr.ALL_METRICS})  # género vacío: solo conteos
    return rows


def draw(rows, sensitivity=None):
    with matplotlib.rc_context(gr.style()):
        fig = gr.plot_groups(rows, sensitivity)
        fig.canvas.draw()
    return fig


def vertical(ax):
    """Líneas verticales de todo el alto del panel (las metas)."""
    return [l.get_xdata()[0] for l in ax.get_lines()
            if len(l.get_xdata()) == 2 and l.get_xdata()[0] == l.get_xdata()[1] and list(l.get_ydata()) == [0, 1]]


def markers(ax, marker, hollow):
    """Marcadores de un tipo (sin línea), vacíos (cara blanca) o rellenos."""
    return [l for l in ax.get_lines() if l.get_marker() == marker and (l.get_markerfacecolor() == "white") == hollow]


def sensitivity_toy(group="caribe", value=0.08):
    return {"groups": {group: {"analysis": "sin_mas_del_doble", "row_type": "grupo", "group": group,
                               "wer_anonymized": str(value), "wer_anonymized_ci_low": str(value - 0.02),
                               "wer_anonymized_ci_high": str(value + 0.03)}}, "n_excluded": 2}


def test_plot_groups_dos_paneles_orden_separacion_metas_y_punto_decimal():
    fig = draw(summary_toy())
    a, b = fig.axes[:2]
    assert len(fig.axes) == 2
    assert [ax.get_title(loc="left") for ax in (a, b)] == ["$\\mathbf{(a)}$ Privacidad: EER o-a",
                                                           "$\\mathbf{(b)}$ Inteligibilidad: WER anonimizada"]
    # Acentos de mayor a menor EER o-a (chileno 0,18; mexico 0,17; caribe 0,15... sin_dato 0,16), después la
    # separación y masculino y femenino; el género vacío no se dibuja.
    assert [t.get_text() for t in a.get_yticklabels()] == ["Chileno", "México", "Sin dato", "Caribe",
                                                            "Masculino", "Femenino"]
    assert [round(y) for y in a.get_yticks()] == [6, 5, 4, 3, 1, 0]
    labels = [t.get_text() for ax in (a, b) for t in ax.get_xticklabels() if t.get_text()]
    assert labels and not any("," in t for t in labels)
    assert vertical(a) == [30.0] and vertical(b) == [15.0]
    assert [t.get_text() for t in a.texts] == ["Meta ≥ 30 %"] and [t.get_text() for t in b.texts] == ["Meta < 15 %"]
    assert len(markers(a, "o", hollow=True)) == 6 and len(markers(b, "o", hollow=True)) == 6
    assert markers(b, "D", hollow=True) == []  # sin sensibilidad no hay rombo
    assert [t.get_text() for t in fig.legends[0].get_texts()] == [
        "EER o-a (IC 95 %)", "WER anonimizada (IC 95 %)", "Voz original (EER o-o, WER original)", "Meta del proyecto"]
    assert fig.get_figwidth() == pytest.approx(gr.TEXT_WIDTH)


def test_plot_groups_rombo_solo_en_la_wer_del_grupo_y_desplazado():
    fig = draw(summary_toy(), sensitivity_toy("caribe", 0.08))
    a, b = fig.axes
    diamonds = markers(b, "D", hollow=True)
    assert len(diamonds) == 1 and markers(a, "D", hollow=True) == []
    caribe_y = 3  # renglón de Caribe (ver la prueba anterior)
    x, y = diamonds[0].get_xdata()[0], diamonds[0].get_ydata()[0]
    assert x == pytest.approx(8.0) and y == pytest.approx(caribe_y - gr.SENSITIVITY_OFFSET)
    assert diamonds[0].get_markeredgecolor() == gr.ANON
    bars = [l for l in b.get_lines() if l.get_marker() in ("None", "", None) and len(l.get_ydata()) == 2
            and l.get_ydata()[0] == l.get_ydata()[1] == pytest.approx(caribe_y - gr.SENSITIVITY_OFFSET)]
    assert len(bars) == 1 and list(bars[0].get_xdata()) == pytest.approx([6.0, 11.0])
    assert "WER sin las 2 transcripciones atípicas (IC 95 %)" in [t.get_text() for t in fig.legends[0].get_texts()]


def test_plot_groups_rombo_de_un_grupo_que_no_esta_no_se_dibuja():
    fig = draw(summary_toy(), sensitivity_toy("america_central"))
    assert markers(fig.axes[1], "D", hollow=True) == []
    assert len(fig.legends[0].get_texts()) == 4


def test_plot_groups_eje_de_la_wer_alcanza_el_ic_mas_alto():
    rows = summary_toy()
    rows[0].update({"wer_anonymized": 0.30, "wer_anonymized_ci_low": 0.084, "wer_anonymized_ci_high": 0.634})
    fig = draw(rows, sensitivity_toy("caribe"))
    assert fig.axes[1].get_xlim()[1] >= 63.4


def test_load_sensitivity_toma_la_fila_sin_las_atipicas(tmp_path):
    path = tmp_path / "sensibilidad_wer.csv"
    rows = [{"analysis": a, "row_type": t, "group": g, "wer_anonymized": v}
            for a, t, g, v in [("con_todas", "grupo", "america_central", "0.30"),
                               ("sin_mas_del_doble", "grupo", "america_central", "0.098"),
                               ("sin_mas_del_doble", "grupo", "caribe", "0.10"),
                               ("sin_mas_del_doble", "permutacion", "", ""),
                               ("sin_mas_del_doble", "transcripcion_excluida", "america_central", ""),
                               ("sin_mas_del_doble", "transcripcion_excluida", "america_central", "")]]
    gr.write_csv(rows, ["analysis", "row_type", "group", "wer_anonymized"], path)
    out = gr.load_sensitivity(path)
    assert list(out["groups"]) == ["america_central"] and out["groups"]["america_central"]["wer_anonymized"] == "0.098"
    assert out["n_excluded"] == 2
    assert gr.load_sensitivity(tmp_path / "no_existe.csv") == {}


def test_update_latex_agrega_y_luego_reemplaza_solo_su_bloque(tmp_path):
    path = tmp_path / "figuras_resultados.tex"
    other = ("\\begin{figure}[!htbp]\n    \\centering\n    \\caption{Otra.}\n    \\label{fig:res_otra}\n"
             "\\end{figure}\n")
    path.write_text(other, encoding="utf-8")
    gr.update_latex(path, gr.latex_block("u0685_0865"))
    gr.update_latex(path, gr.latex_block("u0685_0865"))  # la segunda vez reemplaza, no duplica
    text = path.read_text(encoding="utf-8")
    assert text.startswith(other) and text.count("\\label{fig:res_grupos}") == 1
    assert text.count("\\begin{figure*}[!t]") == 1 and "\\includegraphics[width=\\textwidth]" in text
    assert ("$\\mathcal{U}(0.685, 0.865)$: (a) EER o--a y (b) WER del audio anonimizado." in text
            and text.rstrip().endswith("transcripciones atípicas.}\n    \\label{fig:res_grupos}\n\\end{figure*}"))
    gr.update_latex(path, gr.latex_block("u0685_0865", sensitivity=False))
    text = path.read_text(encoding="utf-8")
    assert "rombo" not in text and text.startswith(other) and text.count("\\label{fig:res_grupos}") == 1


def test_point_number_punto_decimal():
    assert (gr.point_number(0.70), gr.point_number(12.5), gr.point_number(-2.5), gr.point_number(-0.0)) == (
        "0.7", "12.5", "−2.5", "0")


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
    f = ga.config_files("u0685_0865", tmp_path)
    write_tsv(f["manifest"], rows)
    np.savez(f["embeddings"], session_ids=np.array(session_ids, dtype=str), original=original, anonymized=anonymized)
    write_tsv(f["transcripts"], case["transcripts"])
    validated = [{"client_id": f"spk{s:02d}", "path": "x.mp3", "gender": GENDERS[(s // 4) % 4]} for s in range(16)]
    write_tsv(tmp_path / CLIPS_SUBDIR.parent / "validated.tsv", validated)
    return tmp_path


def run_main(data, *extra):
    gr.main(["--data-dir", str(data), "--n-boot", "5", "--n-perm", "20", "--formats", "pdf", "png", *extra],
            log=lambda *_: None)


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_main_tablas_pruebas_y_figura(toy_data, case):
    run_main(toy_data)
    out = toy_data / "por_grupo_u0685_0865"
    summary = read(out / "resumen_grupos.csv")
    assert list(summary[0]) == gr.SUMMARY_COLUMNS
    assert [(r["grouping"], r["group"]) for r in summary] == (
        [("acento", s) for s in STRATA] + [("genero", g) for g in ("masculino", "femenino", "sin_dato")]
        + [("total", "total")])
    by = {(r["grouping"], r["group"]): r for r in summary}
    assert by[("acento", "sin_dato")]["in_comparison"] == "False" and by[("acento", "caribe")]["in_comparison"] == "True"
    # 4 hablantes por estrato (5 clips y 1 transcripción cada uno): 8 objetivo y 8 · 3 impostores.
    caribe = by[("acento", "caribe")]
    assert [int(caribe[k]) for k in ("n_speakers", "n_clips", "n_wer_clips", "n_target_pairs", "n_impostor_pairs")] == [
        4, 20, 4, 8, 24]
    assert [int(by[("genero", g)]["n_speakers"]) for g in ("masculino", "femenino", "sin_dato")] == [8, 4, 4]
    empty = by[("genero", "sin_dato")]
    assert all(empty[m] == "" for m in gr.ALL_METRICS) and empty["n_target_pairs"] == ""
    total = by[("total", "total")]
    assert float(total["eer_oa"]) == case["results"]["o-a"]["eer"]
    assert float(total["wer_anonymized"]) == pytest.approx(wer_values(case["counts"])["anonymized"])
    assert total["eer_oa_ci_low"] == ""  # sin JSON de eer_bootstrap con los mismos parámetros
    assert int(total["n_over_double"]) == 2 and sum(int(by[("acento", s)]["n_over_double"]) for s in STRATA) == 2
    assert 0.685 <= float(total["alpha_min"]) <= float(total["alpha_max"]) <= 0.865
    for row in summary[:6]:
        for m in gr.ALL_METRICS:
            assert float(row[f"{m}_ci_low"]) <= float(row[f"{m}_ci_high"])

    tests = read(out / "pruebas_grupos.csv")
    assert [(t["test"], t["metric"]) for t in tests] == (
        [("diferencia_genero", m) for m in gr.TEST_METRICS] + [("permutacion_acento", m) for m in gr.TEST_METRICS])
    for t in tests[:3]:
        assert (t["group_a"], t["group_b"], t["p_value"]) == ("masculino", "femenino", "")
        assert float(t["estimate"]) == pytest.approx(float(by[("genero", "masculino")][t["metric"]])
                                                     - float(by[("genero", "femenino")][t["metric"]]))
    for t in tests[3:]:
        assert t["groups"] == "caribe,chileno,mexico" and int(t["n_speakers"]) == 12
        assert 1 / 21 <= float(t["p_value"]) <= 1 and float(t["estimate"]) == pytest.approx(
            float(t["value_a"]) - float(t["value_b"]))
    figures = toy_data / "figuras" / "resultados"
    assert (figures / "res_grupos.pdf").read_bytes()[:4] == b"%PDF"
    assert (figures / "res_grupos.png").read_bytes()[:4] == b"\x89PNG"
    latex = (figures / "figuras_resultados.tex").read_text(encoding="utf-8")
    assert latex.count("\\label{fig:res_grupos}") == 1 and "rombo" not in latex  # sin sensibilidad_wer.csv
    replicas = np.load(out / "replicas_grupos.npz")
    assert replicas["genero__eer_oa"].shape == (5, 2) and replicas["acento__eer_aa"].shape == (5, 4)


def test_main_mismos_resultados_con_la_misma_semilla(toy_data):
    run_main(toy_data)
    out = toy_data / "por_grupo_u0685_0865"
    first = (out / "resumen_grupos.csv").read_text(encoding="utf-8"), (out / "pruebas_grupos.csv").read_text(
        encoding="utf-8")
    run_main(toy_data)
    assert first == ((out / "resumen_grupos.csv").read_text(encoding="utf-8"),
                     (out / "pruebas_grupos.csv").read_text(encoding="utf-8"))


def test_main_figures_only_redibuja_sin_recalcular(toy_data):
    run_main(toy_data)
    figure = toy_data / "figuras" / "resultados" / "res_grupos.pdf"
    figure.unlink()
    (toy_data / "anon_k134" / "anon_manifest_u0685_0865.tsv").unlink()  # sin datos: no puede recalcular
    run_main(toy_data, "--figures-only")
    assert figure.read_bytes()[:4] == b"%PDF"
