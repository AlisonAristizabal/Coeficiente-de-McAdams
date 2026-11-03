"""Pruebas de range_comparison con un ..\\Datos de juguete (los JSON salen de los main reales, con pocas réplicas)."""
import csv
import json
import sys
from collections import Counter

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")

import eer_bootstrap  # noqa: E402
import group_analysis as ga  # noqa: E402
import impersonation  # noqa: E402
import range_comparison as rc  # noqa: E402
import wer_bootstrap  # noqa: E402

DIM = 8
N_BOOT = 20


# ---------------------------------------------------------------------------
# Diseño
# ---------------------------------------------------------------------------

EXPECTED_DESIGN = {
    "u0635_0815": (0.725, 0.18), "u065_08": (0.725, 0.15), "u0665_0785": (0.725, 0.12),
    "u066_084": (0.75, 0.18), "u0675_0825": (0.75, 0.15), "u069_081": (0.75, 0.12),
    "u0685_0865": (0.775, 0.18), "u07_085": (0.775, 0.15), "u0715_0835": (0.775, 0.12),
}


@pytest.mark.parametrize("tag, design", EXPECTED_DESIGN.items())
def test_nominal_design_desde_la_etiqueta(tag, design):
    assert rc.nominal_design(tag) == design


def test_nominal_design_referencia():
    assert rc.nominal_design("u07_09") == (0.8, 0.2)


def test_range_tags_son_los_nueve_del_diseno():
    assert set(rc.RANGE_TAGS) == set(EXPECTED_DESIGN)


def tags_configs(tags):
    return [{"tag": t} for t in tags]


def test_check_grid_completa():
    grid = rc.check_grid(tags_configs(rc.RANGE_TAGS))
    assert {cell: c["tag"] for cell, c in grid.items()} == {d: t for t, d in EXPECTED_DESIGN.items()}


def test_check_grid_falta_una_celda():
    with pytest.raises(ValueError, match=r"faltan \[\(0.75, 0.12\)\]"):
        rc.check_grid(tags_configs([t for t in rc.RANGE_TAGS if t != "u069_081"]))


def test_check_grid_celda_repetida():
    with pytest.raises(ValueError, match="repetidas"):
        rc.check_grid(tags_configs([*rc.RANGE_TAGS, "u069_081"]))


def test_check_grid_rango_fuera_del_diseno():
    with pytest.raises(ValueError, match="u07_09"):
        rc.check_grid(tags_configs([*rc.RANGE_TAGS, "u07_09"]))


def test_comparison_pairs_27_con_su_direccion():
    grid = rc.check_grid(tags_configs(rc.RANGE_TAGS))
    pairs = rc.comparison_pairs(grid, {"tag": "u07_09"})
    assert len(pairs) == 27
    assert Counter(kind for kind, _, _ in pairs) == {"mismo_punto_medio": 9, "mismo_ancho": 9,
                                                     "frente_a_referencia": 9}
    design = lambda c: rc.nominal_design(c["tag"])
    same_mid = [(design(a), design(b)) for k, a, b in pairs if k == "mismo_punto_medio"]
    assert all(a[0] == b[0] for a, b in same_mid)
    assert Counter((a[1], b[1]) for a, b in same_mid) == {(0.15, 0.18): 3, (0.12, 0.15): 3, (0.12, 0.18): 3}
    same_width = [(design(a), design(b)) for k, a, b in pairs if k == "mismo_ancho"]
    assert all(a[1] == b[1] for a, b in same_width)
    assert Counter((a[0], b[0]) for a, b in same_width) == {(0.725, 0.75): 3, (0.75, 0.775): 3, (0.725, 0.775): 3}
    ref = [(a["tag"], b["tag"]) for k, a, b in pairs if k == "frente_a_referencia"]
    assert {a for a, _ in ref} == {"u07_09"} and {b for _, b in ref} == set(rc.RANGE_TAGS)


# ---------------------------------------------------------------------------
# Diagnóstico de alucinaciones
# ---------------------------------------------------------------------------

def trow(sid, ref, hyp, version="anonymized"):
    return {"session_id": sid, "client_id": "c" + sid, "stratum": "caribe", "version": version,
            "reference": ref, "hypothesis": hyp, "seconds": "1"}


def test_hallucination_diagnostic_a_mano():
    rows = [
        trow("a", "uno dos tres", "uno dos tres"),                           # 0 errores
        trow("b", "uno dos tres", "uno dos tres cuatro cinco seis siete"),   # 7 > 2 × 3: alucinación, 4 errores
        trow("c", "uno dos tres", "uno dos tres cuatro cinco seis"),         # 6 = 2 × 3: no, 3 errores
        trow("d", "hola mundo", "chao"),                                     # 2 errores
        trow("e", "uno dos tres", "x", version="original"),                  # se ignora (original)
    ]
    summary, top = rc.hallucination_diagnostic(rows, {"a": 0.7, "b": 0.71, "c": 0.72, "d": 0.73})
    assert summary["n_clips"] == 4 and summary["n_hallucinated"] == 1
    assert (summary["words"], summary["errors"], summary["errors_hallucinated"]) == (11, 9, 4)
    assert summary["error_fraction_hallucinated"] == pytest.approx(4 / 9)
    assert summary["wer_official"] == pytest.approx(9 / 11)
    assert summary["wer_without_hallucinated"] == pytest.approx(5 / 8)
    assert [c["session_id"] for c in top] == ["b", "c", "d", "a"]
    assert [c["rank"] for c in top] == [1, 2, 3, 4]
    assert (top[0]["clip_alpha"], top[0]["hallucinated"], top[0]["hypothesis_words"]) == (0.71, True, 7)


def test_hallucination_diagnostic_solo_los_diez_con_mas_errores():
    rows = [trow(f"s{i:02d}", "uno dos tres", " ".join(["x"] * (i % 7))) for i in range(25)]
    _, top = rc.hallucination_diagnostic(rows, {})
    assert len(top) == 10
    errors = [c["clip_errors"] for c in top]
    assert errors == sorted(errors, reverse=True)


def test_hallucination_diagnostic_normaliza_numeros():
    # "4" se normaliza a "cuatro": no hay error ni alucinación.
    summary, _ = rc.hallucination_diagnostic([trow("a", "capítulo cuatro", "Capítulo 4.")], {})
    assert (summary["errors"], summary["n_hallucinated"]) == (0, 0)


# ---------------------------------------------------------------------------
# ..\Datos de juguete
# ---------------------------------------------------------------------------

def make_case(n_speakers=12, seed=0):
    rng = np.random.default_rng(seed)
    rows, session_ids, original = [], [], []
    for s in range(n_speakers):
        center = rng.normal(size=DIM)
        for c in range(5):
            sid = f"cv_{s:02d}_{c}"
            session_ids.append(sid)
            original.append(center + 0.6 * rng.normal(size=DIM))
            rows.append({"client_id": f"spk{s:02d}", "stratum": ["caribe", "mexico", "sin_dato"][s % 3],
                         "path": f"{sid}.mp3", "session_id": sid, "status": "ok", "error": ""})
    return rows, session_ids, np.array(original, np.float32)


def write_tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(monkeypatch, module, *argv):
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv])
    module.main()


def build_config(monkeypatch, data, tag, coefficients, noise, rows, session_ids, original):
    """Manifiesto, embeddings, transcripciones y los JSON de una configuración."""
    f = ga.config_files(tag, data)
    write_tsv(f["manifest"], [{**r, "coefficient": str(c)} for r, c in zip(rows, coefficients)])
    rng = np.random.default_rng(int(noise * 1000) + len(tag))
    anon = (original + noise * rng.normal(size=original.shape)).astype(np.float32)
    np.savez(f["embeddings"], session_ids=np.array(session_ids, dtype=str), original=original, anonymized=anon)
    words = "el gato come pescado en la casa de mi abuela".split()
    transcripts = []
    for i, cid in enumerate(sorted({r["client_id"] for r in rows})):
        stratum = next(r["stratum"] for r in rows if r["client_id"] == cid)
        ref = np.random.default_rng(i).choice(words, size=6).tolist()
        hyp_rng = np.random.default_rng(i + int(noise * 100))
        anon_hyp = [w if hyp_rng.random() > noise / 5 else "x" for w in ref]
        if i == 0 and noise > 1:  # una alucinación
            anon_hyp = anon_hyp + ["gracias"] * 10
        for version, hyp in (("original", ref), ("anonymized", anon_hyp)):
            transcripts.append({"session_id": f"cv_{cid[3:]}_0", "client_id": cid, "stratum": stratum,
                                "version": version, "reference": " ".join(ref), "hypothesis": " ".join(hyp),
                                "seconds": "1"})
    write_tsv(f["transcripts"], transcripts)
    common = ["--n-boot", str(N_BOOT)]
    run(monkeypatch, eer_bootstrap, "--embeddings", str(f["embeddings"]), "--manifest", str(f["manifest"]),
        "--output", str(f["eer_json"]), "--replicates-output", str(f["eer_replicates"]), *common)
    run(monkeypatch, impersonation, "--embeddings", str(f["embeddings"]), "--manifest", str(f["manifest"]),
        "--output", str(f["impersonation_json"]), "--max-far", "0.05", *common)
    run(monkeypatch, wer_bootstrap, "--transcripts", str(f["transcripts"]), "--output", str(f["wer_json"]), *common)
    return f


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    data = tmp_path_factory.mktemp("datos")
    mp = pytest.MonkeyPatch()
    rows, session_ids, original = make_case()
    n = len(rows)
    for k, (tag, (mid, width)) in enumerate(EXPECTED_DESIGN.items()):
        low, high = mid - width / 2, mid + width / 2
        coefficients = np.random.default_rng(k).uniform(low, high, n)
        build_config(mp, data, tag, coefficients, 1.0 + 0.2 * k, rows, session_ids, original)
    build_config(mp, data, "u07_09", np.random.default_rng(99).uniform(0.7, 0.9, n), 1.2, rows, session_ids, original)
    for alpha, noise in ((0.7, 1.6), (0.8, 0.8), (0.5, 3.0)):
        build_config(mp, data, f"a{round(alpha * 100):03d}", [alpha] * n, noise, rows, session_ids, original)
    mp.undo()
    return data


@pytest.fixture(scope="module")
def outputs(toy):
    mp = pytest.MonkeyPatch()
    mp.setattr(sys, "argv", ["range_comparison.py", "--data-dir", str(toy), "--n-boot", str(N_BOOT)])
    with matplotlib.rc_context():
        rc.main()
    mp.undo()

    def read(name):
        with open(toy / "rangos" / name, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f, delimiter="\t"))
    return {"summary": read("rangos_resumen.tsv"), "differences": read("rangos_diferencias.tsv"),
            "hallucinations": read("rangos_alucinaciones.tsv"), "data": toy}


def configs_by_tag(data):
    return {c["tag"]: c for c in ga.discover_configs(data)}


# ---------------------------------------------------------------------------
# Tabla resumen
# ---------------------------------------------------------------------------

def test_resumen_una_fila_por_rango_y_la_referencia(outputs):
    summary = outputs["summary"]
    assert [r["config"] for r in summary] == [*rc.RANGE_TAGS, "u07_09"]
    assert list(summary[0]) == rc.SUMMARY_COLUMNS
    for r in summary[:-1]:
        assert (float(r["midpoint"]), float(r["width"])) == EXPECTED_DESIGN[r["config"]]


def test_resumen_alpha_observado_del_manifiesto(outputs):
    configs = configs_by_tag(outputs["data"])
    for r in outputs["summary"]:
        values = [float(x["coefficient"]) for x in rc.read_tsv_rows(configs[r["config"]]["files"]["manifest"])]
        assert (float(r["alpha_obs_min"]), float(r["alpha_obs_max"])) == (min(values), max(values))
        assert float(r["alpha_obs_mean"]) == pytest.approx(np.mean(values))


def test_resumen_valores_e_ic_de_los_json(outputs):
    configs = configs_by_tag(outputs["data"])
    for r in outputs["summary"]:
        eer = json.loads(configs[r["config"]]["files"]["eer_json"].read_text(encoding="utf-8"))["scenarios"]
        wer = json.loads(configs[r["config"]]["files"]["wer_json"].read_text(encoding="utf-8"))["wer"]
        for name, s in (("eer_oo", "o-o"), ("eer_oa", "o-a"), ("eer_aa", "a-a")):
            assert (float(r[name]), float(r[f"{name}_ci_low"]), float(r[f"{name}_ci_high"])) == (
                eer[s]["eer"], eer[s]["ci_low"], eer[s]["ci_high"])
        for name, key in (("wer_original", "original"), ("wer_anonymized", "anonymized"),
                          ("wer_difference", "difference")):
            assert (float(r[name]), float(r[f"{name}_ci_low"]), float(r[f"{name}_ci_high"])) == (
                wer[key]["value"], wer[key]["ci_low"], wer[key]["ci_high"])


def test_compare_with_sweep_summary(outputs, tmp_path):
    rows = [{k: (float(v) if k != "config" else v) for k, v in r.items()} for r in outputs["summary"]]
    assert rc.compare_with_sweep_summary(rows, tmp_path / "no.tsv") is None
    columns = ["tag"] + [f"{m}{s}" for m in ("eer_oo", "eer_oa", "eer_aa", "wer_original", "wer_anonymized")
                         for s in ("", "_ci_low", "_ci_high")]
    sweep = [{"tag": r["config"], **{c: r[c] for c in columns[1:]}} for r in rows]
    write_tsv(tmp_path / "b.tsv", sweep)
    assert rc.compare_with_sweep_summary(rows, tmp_path / "b.tsv") == []
    sweep[0]["eer_oa"] = sweep[0]["eer_oa"] + 1e-9
    sweep[1]["wer_anonymized_ci_high"] = sweep[1]["wer_anonymized_ci_high"] + 1e-9
    write_tsv(tmp_path / "c.tsv", sweep)
    assert [(c, col) for c, col, _, _ in rc.compare_with_sweep_summary(rows, tmp_path / "c.tsv")] == [
        (rows[0]["config"], "eer_oa"), (rows[1]["config"], "wer_anonymized_ci_high")]


# ---------------------------------------------------------------------------
# Diferencias pareadas
# ---------------------------------------------------------------------------

def test_diferencias_81_filas_27_por_metrica(outputs):
    diffs = outputs["differences"]
    assert len(diffs) == 81 and list(diffs[0]) == rc.DIFFERENCE_COLUMNS
    assert Counter(r["metric"] for r in diffs) == {m: 27 for m in rc.METRICS}


def test_diferencia_puntual_es_la_resta_de_los_json(outputs):
    configs = configs_by_tag(outputs["data"])
    points = {t: rc.json_metrics(c) for t, c in configs.items() if t.startswith("u")}
    for r in outputs["differences"]:
        a, b = points[r["config_from"]][r["metric"]][0], points[r["config_to"]][r["metric"]][0]
        assert (float(r["value_from"]), float(r["value_to"])) == (a, b)
        assert float(r["difference"]) == b - a


def test_ic_pareado_del_eer_con_las_replicas_guardadas(outputs):
    configs = configs_by_tag(outputs["data"])
    for r in outputs["differences"]:
        if r["metric"] not in ("eer_oa", "eer_aa"):
            continue
        scenario = "o-a" if r["metric"] == "eer_oa" else "a-a"
        with np.load(configs[r["config_from"]]["files"]["eer_replicates"]) as a, \
                np.load(configs[r["config_to"]]["files"]["eer_replicates"]) as b:
            expected = eer_bootstrap.percentile_ci(b[scenario] - a[scenario])
        assert (float(r["ci_low"]), float(r["ci_high"])) == pytest.approx(expected, abs=1e-15)


def test_ic_pareado_del_wer_con_las_mismas_muestras(outputs):
    configs = configs_by_tag(outputs["data"])
    (row,) = [r for r in outputs["differences"] if r["metric"] == "wer_anonymized"
              and (r["config_from"], r["config_to"]) == ("u07_09", "u0715_0835")]
    splits, _, _ = rc.load_and_score(configs["u07_09"]["files"]["embeddings"], configs["u07_09"]["files"]["manifest"],
                                     42, 3, 2)
    samples = ga.shared_samples([s["stratum"] for s in splits.values()], N_BOOT, 2026)
    total = np.zeros(len(splits), dtype=int)
    reps = {}
    for tag in ("u07_09", "u0715_0835"):
        counts = ga.aligned_wer_counts(configs[tag]["files"]["transcripts"], list(splits))
        reps[tag] = np.array([ga.wer_by_group(counts, total, 1, s)["anonymized"][0] for s in samples])
    assert np.std(reps["u0715_0835"] - reps["u07_09"]) > 0  # el par sí varía entre réplicas
    expected = eer_bootstrap.percentile_ci(reps["u0715_0835"] - reps["u07_09"])
    assert (float(row["ci_low"]), float(row["ci_high"])) == pytest.approx(expected, abs=1e-15)


def test_configuracion_contra_si_misma_es_cero(outputs):
    configs = configs_by_tag(outputs["data"])
    c = configs["u069_081"]
    splits, _, results = rc.load_and_score(c["files"]["embeddings"], c["files"]["manifest"], 42, 3, 2)
    samples = ga.shared_samples([s["stratum"] for s in splits.values()], N_BOOT, 2026)
    params = {"seed": 42, "n_enroll": 3, "n_trial": 2, "n_boot": N_BOOT, "boot_seed": 2026}
    m = rc.config_replicates(c, results, samples, list(splits), params)
    rc.check_points(c, m)
    rows = rc.difference_rows([("prueba", c, c)], {c["tag"]: m}, 0.95)
    assert [(r["difference"], r["ci_low"], r["ci_high"]) for r in rows] == [(0.0, 0.0, 0.0)] * 3


def test_check_points_detecta_un_valor_distinto(outputs):
    c = configs_by_tag(outputs["data"])["u065_08"]
    good = {m: {"point": v[0], "reps": None} for m, v in rc.json_metrics(c).items() if m in rc.METRICS}
    rc.check_points(c, good)
    with pytest.raises(ValueError, match="eer_aa"):
        rc.check_points(c, {**good, "eer_aa": {"point": good["eer_aa"]["point"] + 1e-12, "reps": None}})


# ---------------------------------------------------------------------------
# Alucinaciones en main
# ---------------------------------------------------------------------------

def test_alucinaciones_configuraciones_y_estructura(outputs):
    rows = outputs["hallucinations"]
    summaries = [r for r in rows if r["row_type"] == "resumen"]
    # Los 9 rangos, la referencia y los α fijos entre 0,65 y 0,9 (no el 0,5).
    assert [r["config"] for r in summaries] == [*rc.RANGE_TAGS, "u07_09", "a070", "a080"]
    assert list(rows[0]) == rc.HALLUCINATION_COLUMNS
    for s in summaries:
        top = [r for r in rows if r["row_type"] == "top_clips" and r["config"] == s["config"]]
        assert len(top) == min(10, int(s["n_clips"])) and [int(r["rank"]) for r in top] == list(range(1, len(top) + 1))


def test_alucinaciones_wer_oficial_es_el_del_json(outputs):
    configs = configs_by_tag(outputs["data"])
    for r in (r for r in outputs["hallucinations"] if r["row_type"] == "resumen"):
        wer = json.loads(configs[r["config"]]["files"]["wer_json"].read_text(encoding="utf-8"))["wer"]["anonymized"]
        assert float(r["wer_official"]) == pytest.approx(wer["value"], abs=1e-15)


def test_alucinaciones_alpha_de_cada_clip(outputs):
    configs = configs_by_tag(outputs["data"])
    for r in (r for r in outputs["hallucinations"] if r["row_type"] == "top_clips"):
        coefficients = rc.manifest_coefficients(configs[r["config"]]["files"]["manifest"])
        assert float(r["clip_alpha"]) == coefficients[r["session_id"]]


def test_alucinaciones_detecta_la_del_juguete(outputs):
    hallucinated = [r for r in outputs["hallucinations"] if r["row_type"] == "top_clips" and r["hallucinated"] == "True"]
    assert hallucinated and all(int(r["hypothesis_words"]) > 2 * int(r["reference_words"]) for r in hallucinated)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def test_figuras_guardadas(outputs):
    for name in ("rangos_compromiso.pdf", "rangos_ancho.pdf"):
        assert (outputs["data"] / "figuras" / "rangos" / name).read_bytes()[:4] == b"%PDF"


def figure_rows(outputs):
    rows = [{k: (float(v) if k != "config" else v) for k, v in r.items()} for r in outputs["summary"]]
    return rows[:-1], rows[-1]


def test_compromiso_color_por_punto_medio_y_marcador_por_ancho(outputs):
    ranges, reference = figure_rows(outputs)
    fig = rc.plot_tradeoff(ranges, reference, [])
    ax = fig.axes[0]
    for r in ranges:
        (dot,) = [l for l in ax.get_lines() if l.get_marker() == rc.WIDTH_MARKERS[r["width"]]
                  and l.get_xdata()[0] == pytest.approx(100 * r["wer_anonymized"])
                  and l.get_ydata()[0] == pytest.approx(100 * r["eer_oa"])]
        assert matplotlib.colors.same_color(dot.get_color(), rc.MIDPOINT_COLORS[r["midpoint"]])
    (star,) = [l for l in ax.get_lines() if l.get_marker() == "*"]
    assert star.get_xdata()[0] == pytest.approx(100 * reference["wer_anonymized"])
    vlines = [l for l in ax.get_lines() if l.get_linestyle() == ":"]
    assert vlines[0].get_xdata()[0] == pytest.approx(15)
    assert fig.axes[1].get_ylabel() == "EER a-a (%)" and fig.axes[1].get_xlabel() == "WER anonimizado (%)"


def test_compromiso_barras_de_ic_en_los_dos_ejes(outputs):
    ranges, reference = figure_rows(outputs)
    fig = rc.plot_tradeoff(ranges[:1], reference, [])
    r = ranges[0]
    lines = fig.axes[1].get_lines()
    horizontal = [l for l in lines if np.allclose(sorted(l.get_xdata()), [100 * r["wer_anonymized_ci_low"],
                                                                          100 * r["wer_anonymized_ci_high"]])]
    vertical = [l for l in lines if np.allclose(sorted(l.get_ydata()), [100 * r["eer_aa_ci_low"],
                                                                        100 * r["eer_aa_ci_high"]])]
    assert horizontal and vertical


def test_compromiso_curva_de_alpha_fijo_con_sus_alpha(outputs):
    ranges, reference = figure_rows(outputs)
    fixed = [{"alpha": a, **{f"{m}{s}": v for m in ("wer_anonymized", "eer_oa", "eer_aa")
                              for s, v in (("", x), ("_ci_low", x * 0.9), ("_ci_high", x * 1.1))}}
             for a, x in ((0.8, 0.1), (0.7, 0.2))]
    fig = rc.plot_tradeoff(ranges, reference, fixed)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "0,7" in texts and "0,8" in texts
    (curve,) = [l for l in fig.axes[0].get_lines() if l.get_gid() == "alpha_fijo"]
    assert matplotlib.colors.same_color(curve.get_color(), rc.FIXED_COLOR)
    np.testing.assert_allclose(curve.get_xdata(), [20, 10])  # ordenada por α: 0,7 y luego 0,8


def test_ancho_tres_paneles_una_linea_por_punto_medio(outputs):
    ranges, _ = figure_rows(outputs)
    fig = rc.plot_by_width(ranges)
    assert len(fig.axes) == 3
    for ax, metric in zip(fig.axes, ("eer_oa", "eer_aa", "wer_anonymized")):
        lines = [l for l in ax.get_lines() if l.get_marker() == "o"]
        assert len(lines) == 3
        for line in lines:
            m = next(m for m, c in rc.MIDPOINT_COLORS.items() if matplotlib.colors.same_color(line.get_color(), c))
            expected = sorted((r["width"], 100 * r[metric]) for r in ranges if r["midpoint"] == m)
            np.testing.assert_allclose(line.get_ydata(), [y for _, y in expected])
            np.testing.assert_allclose(np.round(line.get_xdata(), 2), [w for w, _ in expected])
    assert [t.get_text() for t in fig.legends[0].get_texts()] == ["0,725", "0,75", "0,775"]
    assert fig.axes[2].xaxis.get_major_formatter()(0.15, 0) == "0,15"


def test_main_falla_si_no_esta_la_cuadricula(toy, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["range_comparison.py", "--data-dir", str(toy), "--output-dir", str(tmp_path),
                                      "--tags", *rc.RANGE_TAGS[:-1]])
    with pytest.raises(ValueError, match="cuadrícula"):
        rc.main()


def test_observed_alphas_ignora_los_clips_con_error(tmp_path):
    path = tmp_path / "m.tsv"
    header = ["client_id", "stratum", "path", "session_id", "coefficient", "status", "error"]
    rows = [["h", "s", "a.mp3", "a", "0.70", "ok", ""], ["h", "s", "b.mp3", "b", "0.80", "ok", ""],
            ["h", "s", "c.mp3", "c", "0.99", "error", "x"]]
    path.write_text("".join("\t".join(r) + "\n" for r in [header, *rows]), encoding="utf-8")
    assert rc.observed_alphas(path) == (0.7, 0.8, pytest.approx(0.75))


def test_main_falla_si_un_json_no_coincide_con_lo_calculado(toy, tmp_path, monkeypatch):
    import shutil
    data = tmp_path / "datos"
    shutil.copytree(toy, data, ignore=shutil.ignore_patterns("rangos", "figuras"))
    path = ga.config_files("u069_081", data)["eer_json"]
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["scenarios"]["a-a"]["eer"] += 0.01
    path.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["range_comparison.py", "--data-dir", str(data), "--n-boot", str(N_BOOT)])
    with pytest.raises(ValueError, match="u069_081: el valor puntual de eer_aa"):
        rc.main()
