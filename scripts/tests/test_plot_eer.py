import csv
import sys

import numpy as np
import pytest

# Las figuras necesitan matplotlib, que solo está en el entorno de evaluación
# (venv-eval). En otro entorno este archivo se salta en vez de romper la
# recolección de `pytest scripts`.
matplotlib = pytest.importorskip("matplotlib")

from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.stats import norm  # noqa: E402

import plot_eer  # noqa: E402
from eer_protocol import split_enrollment_trial
from eer_scoring import compute_eer, split_target_impostor
from plot_eer import (
    DET_TICKS,
    SCENARIO_STYLE,
    configure_fonts,
    decimal_comma,
    det_curve,
    plot_det,
    plot_score_distributions,
    probit,
)
from run_eer import build_row_index, run_scenarios


# ---------------------------------------------------------------------------
# Datos sintéticos
# ---------------------------------------------------------------------------

def overlapping_scores(seed=0, n_target=200, n_impostor=5000, shift=2.0):
    rng = np.random.default_rng(seed)
    return rng.normal(shift, 1, n_target), rng.normal(0, 1, n_impostor)


def brute_force(target, impostor, thresholds):
    """FAR y FRR contados uno por uno, con la convención de compute_eer."""
    frr = np.array([np.sum(target < t) for t in thresholds]) / len(target)
    far = np.array([np.sum(impostor >= t) for t in thresholds]) / len(impostor)
    return far, frr


def three_scenarios():
    s = {name: overlapping_scores(seed=i, shift=shift)
         for i, (name, shift) in enumerate([("o-o", 3.5), ("o-a", 2.0), ("a-a", 2.6)])}
    eers = {name: compute_eer(*pair)[0] for name, pair in s.items()}
    thresholds = {name: compute_eer(*pair)[1] for name, pair in s.items()}
    return s, eers, thresholds


# ---------------------------------------------------------------------------
# decimal_comma y probit
# ---------------------------------------------------------------------------

def test_decimal_comma_un_decimal_por_defecto():
    assert decimal_comma(0.5) == "0,5"


def test_decimal_comma_redondea_a_los_decimales_pedidos():
    assert decimal_comma(9.1791, 2) == "9,18"


def test_decimal_comma_negativos_y_enteros():
    assert decimal_comma(-0.25, 2) == "-0,25"
    assert decimal_comma(3, 0) == "3"


def test_probit_de_un_medio_es_cero():
    assert probit(0.5) == pytest.approx(0.0)


def test_probit_es_la_inversa_de_la_normal_acumulada():
    p = np.array([0.001, 0.05, 0.3, 0.9])
    np.testing.assert_allclose(norm.cdf(probit(p)), p)


def test_probit_es_creciente():
    p = np.linspace(0.01, 0.99, 50)
    assert np.all(np.diff(probit(p)) > 0)


# ---------------------------------------------------------------------------
# det_curve
# ---------------------------------------------------------------------------

def test_det_curve_misma_longitud_y_rango():
    far, frr = det_curve(*overlapping_scores())
    assert far.shape == frr.shape
    assert np.all((far >= 0) & (far <= 1)) and np.all((frr >= 0) & (frr <= 1))


def test_det_curve_far_decrece_y_frr_crece_con_el_umbral():
    far, frr = det_curve(*overlapping_scores())
    assert np.all(np.diff(far) <= 0) and np.all(np.diff(frr) >= 0)


def test_det_curve_empieza_aceptando_todo():
    far, frr = det_curve(*overlapping_scores())
    assert far[0] == 1.0 and frr[0] == 0.0


def test_det_curve_empate_en_el_umbral_se_acepta():
    # Umbrales 0,5 / 0,6 / 0,7 / 0,8. Con umbral 0,5 se aceptan el objetivo
    # 0,5 y los tres impostores (0,5 incluido), como en compute_eer.
    far, frr = det_curve([0.5, 0.7], [0.5, 0.6, 0.8], n_quantiles=2)
    np.testing.assert_allclose(far, [1, 2 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose(frr, [0, 0.5, 0.5, 1])


def test_det_curve_coincide_con_el_conteo_directo():
    target, impostor = overlapping_scores(n_target=60, n_impostor=400)
    far, frr = det_curve(target, impostor, n_quantiles=50)
    quantiles = np.quantile(impostor, np.linspace(0, 1, 51))
    thresholds = np.unique(np.concatenate([target, quantiles]))
    exp_far, exp_frr = brute_force(target, impostor, thresholds)
    np.testing.assert_allclose(far, exp_far)
    np.testing.assert_allclose(frr, exp_frr)


def test_det_curve_incluye_el_punto_de_cada_puntuacion_objetivo():
    target, impostor = overlapping_scores(n_target=40, n_impostor=300)
    far, frr = det_curve(target, impostor, n_quantiles=10)
    exp_far, exp_frr = brute_force(target, impostor, np.sort(target))
    points = set(zip(np.round(far, 9), np.round(frr, 9)))
    assert all((round(a, 9), round(b, 9)) in points for a, b in zip(exp_far, exp_frr))


def test_det_curve_pocos_puntos_con_muchos_impostores():
    rng = np.random.default_rng(1)
    far, _ = det_curve(rng.normal(2, 1, 1000), rng.normal(0, 1, 500_000), n_quantiles=2000)
    assert len(far) <= 1000 + 2001


def test_det_curve_pasa_cerca_del_eer():
    target, impostor = overlapping_scores(n_target=500, n_impostor=20000)
    eer, _ = compute_eer(target, impostor)
    far, frr = det_curve(target, impostor)
    i = np.argmin(np.abs(far - frr))
    assert (far[i] + frr[i]) / 2 == pytest.approx(eer, abs=0.005)


def test_det_curve_no_modifica_entradas():
    target, impostor = overlapping_scores(n_target=30, n_impostor=100)
    t, i = target.copy(), impostor.copy()
    det_curve(target, impostor)
    np.testing.assert_array_equal(target, t)
    np.testing.assert_array_equal(impostor, i)


def test_det_curve_acepta_listas():
    far, frr = det_curve([0.9, 0.8], [0.1, 0.2, 0.3])
    assert far[0] == 1.0 and frr[-1] < 1.0


# ---------------------------------------------------------------------------
# plot_det
# ---------------------------------------------------------------------------

def det_figure():
    s, eers, _ = three_scenarios()
    curves = {name: det_curve(*pair) for name, pair in s.items()}
    return plot_det(curves, eers), eers


def scenario_lines(ax, name):
    style = SCENARIO_STYLE[name]
    return [l for l in ax.get_lines() if matplotlib.colors.same_color(l.get_color(), style["color"])]


def test_plot_det_devuelve_figura_con_un_eje():
    fig, _ = det_figure()
    assert isinstance(fig, Figure) and len(fig.axes) == 1


def test_plot_det_una_curva_y_un_punto_por_escenario_mas_la_diagonal():
    fig, _ = det_figure()
    assert len(fig.axes[0].get_lines()) == 2 * 3 + 1


def test_plot_det_punto_de_eer_sobre_la_diagonal_en_el_valor_del_eer():
    fig, eers = det_figure()
    ax = fig.axes[0]
    for name, eer in eers.items():
        markers = [l for l in scenario_lines(ax, name) if l.get_marker() == "o"]
        assert len(markers) == 1
        x, y = markers[0].get_xdata()[0], markers[0].get_ydata()[0]
        assert x == pytest.approx(probit(eer)) and y == pytest.approx(probit(eer))


def test_plot_det_curvas_sin_infinitos():
    fig, _ = det_figure()
    for line in fig.axes[0].get_lines():
        assert np.all(np.isfinite(line.get_xdata())) and np.all(np.isfinite(line.get_ydata()))


def test_plot_det_curva_en_escala_probit():
    s, eers, _ = three_scenarios()
    far, frr = det_curve(*s["o-a"])
    fig = plot_det({"o-a": (far, frr)}, {"o-a": eers["o-a"]})
    curve = [l for l in scenario_lines(fig.axes[0], "o-a") if l.get_marker() != "o"][0]
    keep = (far > 0) & (far < 1) & (frr > 0) & (frr < 1)
    np.testing.assert_allclose(curve.get_xdata(), probit(far[keep]))
    np.testing.assert_allclose(curve.get_ydata(), probit(frr[keep]))


def test_plot_det_estilo_por_escenario():
    fig, _ = det_figure()
    for name, style in SCENARIO_STYLE.items():
        curve = [l for l in scenario_lines(fig.axes[0], name) if l.get_marker() != "o"][0]
        assert curve.get_linestyle() == matplotlib.lines.Line2D([], [], linestyle=style["linestyle"]).get_linestyle()


def test_scenario_style_colores_y_lineas_distintos_para_blanco_y_negro():
    assert set(SCENARIO_STYLE) == {"o-o", "o-a", "a-a"}
    assert len({v["color"] for v in SCENARIO_STYLE.values()}) == 3
    assert len({v["linestyle"] for v in SCENARIO_STYLE.values()}) == 3


def test_plot_det_leyenda_con_eer_y_coma_decimal():
    fig, eers = det_figure()
    texts = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    for name, eer in eers.items():
        assert f"{name} (EER {decimal_comma(100 * eer, 2)} %)" in texts


def test_plot_det_etiquetas_y_marcas_en_espanol():
    fig, _ = det_figure()
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Tasa de falsa aceptación (%)"
    assert ax.get_ylabel() == "Tasa de falso rechazo (%)"
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels[0] == "0,1" and labels[-1] == "40" and len(labels) == len(DET_TICKS)


def test_plot_det_escenario_perfectamente_separado_no_falla():
    target, impostor = np.array([0.9, 0.95]), np.array([0.1, 0.2, 0.3])
    eer, _ = compute_eer(target, impostor)
    fig = plot_det({"o-o": det_curve(target, impostor)}, {"o-o": eer})
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# plot_score_distributions
# ---------------------------------------------------------------------------

def dist_figure(bins=40):
    s, _, thresholds = three_scenarios()
    return plot_score_distributions(s, thresholds, bins=bins), s, thresholds


def bars(ax):
    """Barras del histograma relleno (impostor); el de objetivo es un contorno."""
    return [p for p in ax.patches if isinstance(p, Rectangle)]


def bar_lefts(ax):
    return sorted({round(p.get_x(), 9) for p in bars(ax)})


def test_plot_score_distributions_un_panel_por_escenario():
    fig, s, _ = dist_figure()
    assert len(fig.axes) == len(s)
    assert [ax.get_ylabel() for ax in fig.axes] == list(s)


def test_plot_score_distributions_comparten_eje_x():
    fig, _, _ = dist_figure()
    shared = fig.axes[0].get_shared_x_axes()
    assert all(shared.joined(fig.axes[0], ax) for ax in fig.axes[1:])


def test_plot_score_distributions_mismos_intervalos_en_todos_los_paneles():
    fig, s, _ = dist_figure(bins=40)
    lefts = [bar_lefts(ax) for ax in fig.axes]
    assert all(l == lefts[0] for l in lefts)
    assert len(lefts[0]) == 40


def test_plot_score_distributions_intervalos_cubren_todas_las_puntuaciones():
    fig, s, _ = dist_figure(bins=40)
    all_scores = np.concatenate([np.concatenate(p) for p in s.values()])
    rects = bars(fig.axes[0])
    assert min(p.get_x() for p in rects) == pytest.approx(all_scores.min())
    assert max(p.get_x() + p.get_width() for p in rects) == pytest.approx(all_scores.max())


def test_plot_score_distributions_impostor_normalizado():
    fig, _, _ = dist_figure()
    assert sum(p.get_width() * p.get_height() for p in bars(fig.axes[1])) == pytest.approx(1.0)


def test_plot_score_distributions_linea_en_el_umbral_de_cada_escenario():
    fig, s, thresholds = dist_figure()
    for ax, name in zip(fig.axes, s):
        vlines = [l for l in ax.get_lines() if len(set(l.get_xdata())) == 1]
        assert len(vlines) == 1
        assert vlines[0].get_xdata()[0] == pytest.approx(thresholds[name])


def test_plot_score_distributions_leyenda_y_etiqueta_x():
    fig, _, _ = dist_figure()
    texts = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert texts == ["Impostor", "Objetivo"]
    assert fig.axes[-1].get_xlabel() == "Similitud coseno"


def test_plot_score_distributions_coma_decimal_en_el_eje_x():
    fig, _, _ = dist_figure()
    assert fig.axes[-1].xaxis.get_major_formatter()(0.5, 0) == "0,5"


# ---------------------------------------------------------------------------
# configure_fonts
# ---------------------------------------------------------------------------

def test_configure_fonts():
    with matplotlib.rc_context():
        configure_fonts()
        assert matplotlib.rcParams["font.size"] == 8
        assert matplotlib.rcParams["pdf.fonttype"] == 42


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def write_inputs(tmp_path, n_speakers=8, seed=0):
    rng = np.random.default_rng(seed)
    rows, session_ids, original, anonymized = [], [], [], []
    for s in range(n_speakers):
        center = rng.normal(size=8)
        for c in range(5):
            sid = f"spk{s:02d}_c{c}"
            session_ids.append(sid)
            original.append(center + 0.3 * rng.normal(size=8))
            anonymized.append(center + 1.5 * rng.normal(size=8))
            rows.append({"client_id": f"spk{s:02d}", "stratum": f"st{s % 2}", "path": sid + ".mp3",
                         "session_id": sid, "coefficient": "0.8", "status": "ok", "error": ""})
    npz = tmp_path / "emb.npz"
    np.savez(npz, session_ids=np.array(session_ids, dtype=str),
             original=np.array(original, np.float32), anonymized=np.array(anonymized, np.float32))
    manifest = tmp_path / "manifest.tsv"
    with open(manifest, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    expected = run_scenarios(
        {"original": np.array(original, np.float32), "anonymized": np.array(anonymized, np.float32)},
        build_row_index(session_ids), split_enrollment_trial(rows)[0])
    return npz, manifest, expected


def run_main(monkeypatch, npz, manifest, out_dir, *extra):
    monkeypatch.setattr(sys, "argv", ["plot_eer.py", "--embeddings", str(npz), "--manifest", str(manifest),
                                      "--output-dir", str(out_dir), *extra])
    with matplotlib.rc_context():
        plot_eer.main()


def test_main_guarda_los_dos_pdf_en_una_carpeta_nueva(monkeypatch, tmp_path, capsys):
    npz, manifest, _ = write_inputs(tmp_path)
    out_dir = tmp_path / "figuras" / "eer"
    run_main(monkeypatch, npz, manifest, out_dir)
    for name in ("det.pdf", "distribuciones.pdf"):
        assert (out_dir / name).read_bytes()[:4] == b"%PDF"
    assert str(out_dir) in capsys.readouterr().out


@pytest.fixture
def captured(monkeypatch):
    calls = {}

    class DummyFigure:
        def savefig(self, path):
            calls.setdefault("paths", []).append(path.name)

    def fake_det(curves, eers):
        calls["curves"], calls["eers"] = curves, eers
        return DummyFigure()

    def fake_dist(scores_by_scenario, thresholds):
        calls["scores"], calls["thresholds"] = scores_by_scenario, thresholds
        return DummyFigure()

    monkeypatch.setattr(plot_eer, "plot_det", fake_det)
    monkeypatch.setattr(plot_eer, "plot_score_distributions", fake_dist)
    return calls


def test_main_pasa_las_puntuaciones_de_cada_escenario(monkeypatch, tmp_path, captured):
    npz, manifest, expected = write_inputs(tmp_path)
    run_main(monkeypatch, npz, manifest, tmp_path / "f")
    assert set(captured["scores"]) == {"o-o", "o-a", "a-a"}
    for name, r in expected.items():
        target, impostor = split_target_impostor(r["scores"], r["owner"])
        np.testing.assert_allclose(captured["scores"][name][0], target)
        np.testing.assert_allclose(captured["scores"][name][1], impostor)


def test_main_pasa_eer_y_umbral_de_cada_escenario(monkeypatch, tmp_path, captured):
    npz, manifest, expected = write_inputs(tmp_path)
    run_main(monkeypatch, npz, manifest, tmp_path / "f")
    for name, r in expected.items():
        assert captured["eers"][name] == pytest.approx(r["eer"])
        assert captured["thresholds"][name] == pytest.approx(r["threshold"])


def test_main_curvas_salen_de_det_curve(monkeypatch, tmp_path, captured):
    npz, manifest, expected = write_inputs(tmp_path)
    run_main(monkeypatch, npz, manifest, tmp_path / "f")
    for name, r in expected.items():
        far, frr = det_curve(*split_target_impostor(r["scores"], r["owner"]))
        np.testing.assert_allclose(captured["curves"][name][0], far)
        np.testing.assert_allclose(captured["curves"][name][1], frr)


def test_main_nombres_de_archivo(monkeypatch, tmp_path, captured):
    npz, manifest, _ = write_inputs(tmp_path)
    run_main(monkeypatch, npz, manifest, tmp_path / "f")
    assert captured["paths"] == ["det.pdf", "distribuciones.pdf"]


def test_main_respeta_n_trial(monkeypatch, tmp_path, captured):
    npz, manifest, _ = write_inputs(tmp_path, n_speakers=6)
    run_main(monkeypatch, npz, manifest, tmp_path / "f", "--n-enroll", "2", "--n-trial", "3")
    assert len(captured["scores"]["o-o"][0]) == 18


def test_main_respeta_la_semilla(monkeypatch, tmp_path, captured):
    npz, manifest, _ = write_inputs(tmp_path)
    run_main(monkeypatch, npz, manifest, tmp_path / "f", "--seed", "7")
    seed7 = dict(captured["thresholds"])
    run_main(monkeypatch, npz, manifest, tmp_path / "f")
    assert seed7 != captured["thresholds"]

def test_main_usa_load_and_score(monkeypatch, tmp_path, captured):
    from run_eer import load_and_score
    npz, manifest, _ = write_inputs(tmp_path)
    calls = []

    def spy(*args):
        calls.append(args)
        return load_and_score(*args)

    monkeypatch.setattr(plot_eer, "load_and_score", spy)
    run_main(monkeypatch, npz, manifest, tmp_path / "f", "--seed", "7", "--n-enroll", "2", "--n-trial", "3")
    assert calls == [(str(npz), str(manifest), 7, 2, 3)]


# ---------------------------------------------------------------------------
# plot_eer_ci
# ---------------------------------------------------------------------------

import json  # noqa: E402

from plot_eer import COLUMN_WIDTH, plot_eer_ci  # noqa: E402

# Intervalos asimétricos a propósito: o-a está 5 puntos sobre ci_low y 7 bajo ci_high.
CI = {
    "o-o": {"eer": 0.02, "ci_low": 0.015, "ci_high": 0.03},
    "o-a": {"eer": 0.40, "ci_low": 0.35, "ci_high": 0.47},
    "a-a": {"eer": 0.0918, "ci_low": 0.08, "ci_high": 0.11},
}


def ci_parts(fig, name):
    """Punto (marcador "o") y barra del escenario, buscados por su color."""
    lines = [l for l in fig.axes[0].get_lines()
             if matplotlib.colors.same_color(l.get_color(), SCENARIO_STYLE[name]["color"])]
    points = [l for l in lines if l.get_marker() == "o"]
    bars = [l for l in lines if l.get_marker() != "o"]
    assert len(points) == 1 and len(bars) == 1
    return points[0], bars[0]


def vertical_lines(fig):
    return [l for l in fig.axes[0].get_lines() if len(set(l.get_xdata())) == 1 and len(l.get_xdata()) == 2
            and l.get_ydata()[0] != l.get_ydata()[1]]


def test_plot_eer_ci_devuelve_figura_de_una_columna_sin_titulo():
    fig = plot_eer_ci(CI)
    assert isinstance(fig, Figure) and len(fig.axes) == 1
    assert fig.get_figwidth() == pytest.approx(COLUMN_WIDTH)
    assert fig.axes[0].get_title() == "" and fig._suptitle is None


def test_plot_eer_ci_punto_en_el_eer_en_porcentaje():
    fig = plot_eer_ci(CI)
    for name, s in CI.items():
        point, _ = ci_parts(fig, name)
        assert point.get_xdata()[0] == pytest.approx(100 * s["eer"])


def test_plot_eer_ci_barra_va_de_ci_low_a_ci_high():
    fig = plot_eer_ci(CI)
    for name, s in CI.items():
        point, bar = ci_parts(fig, name)
        np.testing.assert_allclose(sorted(bar.get_xdata()), [100 * s["ci_low"], 100 * s["ci_high"]])
        assert set(bar.get_ydata()) == {point.get_ydata()[0]}


def test_plot_eer_ci_barra_asimetrica_no_se_vuelve_simetrica():
    fig = plot_eer_ci(CI)
    _, bar = ci_parts(fig, "o-a")
    low, high = sorted(bar.get_xdata())
    assert 40 - low == pytest.approx(5) and high - 40 == pytest.approx(7)


def test_plot_eer_ci_intervalo_que_no_contiene_el_eer():
    scenarios = {"a-a": {"eer": 0.10, "ci_low": 0.11, "ci_high": 0.14}}
    _, bar = ci_parts(plot_eer_ci(scenarios), "a-a")
    np.testing.assert_allclose(sorted(bar.get_xdata()), [11, 14])


def test_plot_eer_ci_orden_o_o_arriba_y_a_a_abajo():
    fig = plot_eer_ci({name: CI[name] for name in ("a-a", "o-o", "o-a")})
    y = {name: ci_parts(fig, name)[0].get_ydata()[0] for name in CI}
    assert y["o-o"] > y["o-a"] > y["a-a"]
    ax = fig.axes[0]
    labels = sorted(zip(ax.get_yticks(), [t.get_text() for t in ax.get_yticklabels()]), reverse=True)
    assert [text for _, text in labels] == ["o-o", "o-a", "a-a"]


def test_plot_eer_ci_colores_por_escenario():
    fig = plot_eer_ci(CI)
    for name in CI:
        point, bar = ci_parts(fig, name)
        assert matplotlib.colors.same_color(point.get_color(), SCENARIO_STYLE[name]["color"])
        assert matplotlib.colors.same_color(bar.get_color(), SCENARIO_STYLE[name]["color"])


def test_plot_eer_ci_meta_en_30_por_ciento_por_defecto():
    fig = plot_eer_ci(CI)
    lines = vertical_lines(fig)
    assert len(lines) == 1
    assert lines[0].get_xdata()[0] == pytest.approx(30)
    assert lines[0].get_linestyle() == ":"
    assert "Meta (30 %)" in [t.get_text() for t in fig.axes[0].texts]


def test_plot_eer_ci_meta_en_target():
    fig = plot_eer_ci(CI, target=0.125)
    assert vertical_lines(fig)[0].get_xdata()[0] == pytest.approx(12.5)
    assert "Meta (12,5 %)" in [t.get_text() for t in fig.axes[0].texts]


def test_plot_eer_ci_rotulo_de_la_meta_sobre_su_linea():
    fig = plot_eer_ci(CI, target=0.25)
    text = [t for t in fig.axes[0].texts if t.get_text().startswith("Meta")][0]
    assert text.get_position()[0] == pytest.approx(25)


def test_plot_eer_ci_eje_x_en_porcentaje_con_coma_decimal():
    fig = plot_eer_ci(CI)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "EER (%)"
    formatter = ax.xaxis.get_major_formatter()
    assert formatter(12.5, 0) == "12,5"
    assert formatter(10, 0) == "10"


def test_plot_eer_ci_eje_x_muestra_la_meta_y_todos_los_intervalos():
    for target in (0.30, 0.60):
        ax = plot_eer_ci(CI, target=target).axes[0]
        low, high = ax.get_xlim()
        # Puede empezar antes de 0 para dar espacio al número del límite
        # inferior, pero sin marcas negativas.
        assert low <= 0 and 0 in ax.get_xticks() and min(ax.get_xticks()) == 0
        assert high > max(100 * target, 47)


def test_plot_eer_ci_no_modifica_la_entrada():
    scenarios = json.loads(json.dumps(CI))
    plot_eer_ci(scenarios)
    assert scenarios == CI


# ---------------------------------------------------------------------------
# main con --bootstrap-json
# ---------------------------------------------------------------------------

def write_bootstrap_json(tmp_path, scenarios=CI):
    path = tmp_path / "boot.json"
    path.write_text(json.dumps({"n_boot": 1000, "scenarios": scenarios}), encoding="utf-8")
    return path


def test_main_sin_bootstrap_json_no_crea_eer_ic(monkeypatch, tmp_path):
    npz, manifest, _ = write_inputs(tmp_path)
    out_dir = tmp_path / "figuras"
    run_main(monkeypatch, npz, manifest, out_dir)
    assert sorted(p.name for p in out_dir.iterdir()) == ["det.pdf", "distribuciones.pdf"]


def test_main_con_bootstrap_json_crea_eer_ic(monkeypatch, tmp_path, capsys):
    npz, manifest, _ = write_inputs(tmp_path)
    out_dir = tmp_path / "figuras"
    run_main(monkeypatch, npz, manifest, out_dir, "--bootstrap-json", str(write_bootstrap_json(tmp_path)))
    assert sorted(p.name for p in out_dir.iterdir()) == ["det.pdf", "distribuciones.pdf", "eer_ic.pdf"]
    assert (out_dir / "eer_ic.pdf").read_bytes()[:4] == b"%PDF"


def test_main_pasa_los_escenarios_del_json_a_plot_eer_ci(monkeypatch, tmp_path, captured):
    npz, manifest, _ = write_inputs(tmp_path)

    def fake_ci(scenarios):
        captured["ci"] = scenarios
        return captured_figure

    class DummyFigure:
        def savefig(self, path):
            captured.setdefault("paths", []).append(path.name)

    captured_figure = DummyFigure()
    monkeypatch.setattr(plot_eer, "plot_eer_ci", fake_ci)
    run_main(monkeypatch, npz, manifest, tmp_path / "f", "--bootstrap-json", str(write_bootstrap_json(tmp_path)))
    assert captured["ci"] == CI
    assert captured["paths"] == ["det.pdf", "distribuciones.pdf", "eer_ic.pdf"]


def test_main_con_el_json_que_genera_eer_bootstrap(monkeypatch, tmp_path):
    import eer_bootstrap
    npz, manifest, _ = write_inputs(tmp_path)
    boot_json = tmp_path / "boot.json"
    monkeypatch.setattr(sys, "argv", ["eer_bootstrap.py", "--embeddings", str(npz), "--manifest", str(manifest),
                                      "--output", str(boot_json), "--replicates-output", str(tmp_path / "r.npz"),
                                      "--n-boot", "20"])
    eer_bootstrap.main()
    out_dir = tmp_path / "figuras"
    run_main(monkeypatch, npz, manifest, out_dir, "--bootstrap-json", str(boot_json))
    assert (out_dir / "eer_ic.pdf").read_bytes()[:4] == b"%PDF"


# ---------------------------------------------------------------------------
# Elementos explicativos: descripciones, meta y composición del EER
# ---------------------------------------------------------------------------

from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402

from plot_eer import (  # noqa: E402
    DET_LIMITS,
    FALSE_ACCEPT_COLOR,
    GOAL_EER,
    SCENARIO_LABEL,
    TARGET_COLOR,
    shade_tail,
)
from run_eer import SCENARIOS  # noqa: E402


def texts_of(ax):
    return [t.get_text() for t in ax.texts]


def fills(ax):
    return [c for c in ax.collections if isinstance(c, PolyCollection)]


def polygon_area(collection):
    x, y = collection.get_paths()[0].vertices.T
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def test_scenario_label_describe_cada_escenario_segun_scenarios():
    assert set(SCENARIO_LABEL) == set(SCENARIOS)
    words = {"original": ("original", "original"), "anonymized": ("anonimizado", "anonimizada")}
    for name, (enroll, trial) in SCENARIOS.items():
        assert SCENARIO_LABEL[name] == f"Registro {words[enroll][0]}, prueba {words[trial][1]}"


def test_goal_eer_es_30_por_ciento():
    assert GOAL_EER == 0.30


# --- shade_tail -------------------------------------------------------------

def shaded(values, threshold, above, edges=np.linspace(0, 1, 11)):
    fig = Figure()
    ax = fig.subplots()
    shade_tail(ax, np.asarray(values, float), edges, threshold, above=above, color="k")
    return fills(ax)[0].get_paths()[0].vertices


def test_shade_tail_arriba_empieza_en_el_umbral_y_llega_al_final():
    v = shaded(np.linspace(0.01, 0.99, 200), 0.43, above=True)
    assert v[:, 0].min() == pytest.approx(0.43) and v[:, 0].max() == pytest.approx(1.0)


def test_shade_tail_abajo_empieza_al_inicio_y_termina_en_el_umbral():
    v = shaded(np.linspace(0.01, 0.99, 200), 0.43, above=False)
    assert v[:, 0].min() == pytest.approx(0.0) and v[:, 0].max() == pytest.approx(0.43)


def test_shade_tail_area_es_la_fraccion_del_lado_sombreado():
    # Con el umbral en un borde de intervalo, el área es exactamente la
    # fracción de valores de ese lado (histograma con área 1).
    values = np.random.default_rng(0).uniform(0, 1, 5000)
    fig = Figure()
    ax = fig.subplots()
    shade_tail(ax, values, np.linspace(0, 1, 11), 0.7, above=True, color="k")
    shade_tail(ax, values, np.linspace(0, 1, 11), 0.7, above=False, color="k")
    above, below = (polygon_area(c) for c in fills(ax))
    assert above == pytest.approx(np.mean(values >= 0.7))
    assert below == pytest.approx(np.mean(values < 0.7))


def test_shade_tail_sigue_la_altura_del_histograma():
    values = [0.05] * 3 + [0.55] * 1  # densidad 7,5 en [0; 0,1) y 2,5 en [0,5; 0,6)
    v = shaded(values, 0.0, above=True)
    assert v[:, 1].max() == pytest.approx(7.5)


# --- plot_det -----------------------------------------------------------------

def test_plot_det_zona_de_la_meta_desde_la_meta_hasta_el_limite():
    fig, _ = det_figure()
    ax = fig.axes[0]
    zone = fills(ax)
    assert len(zone) == 1
    v = zone[0].get_paths()[0].vertices
    high = probit(DET_LIMITS[1] / 100)
    assert v[:, 0].min() == pytest.approx(probit(0.30)) and v[:, 1].min() == pytest.approx(probit(0.30))
    assert v[:, 0].max() == pytest.approx(high) and v[:, 1].max() == pytest.approx(high)
    assert "Meta\nEER ≥ 30 %" in texts_of(ax)


def test_plot_det_zona_de_la_meta_respeta_goal():
    s, eers, _ = three_scenarios()
    fig = plot_det({n: det_curve(*p) for n, p in s.items()}, eers, goal=0.2)
    v = fills(fig.axes[0])[0].get_paths()[0].vertices
    assert v[:, 0].min() == pytest.approx(probit(0.2))
    assert "Meta\nEER ≥ 20 %" in texts_of(fig.axes[0])


def test_plot_det_rotulo_de_reidentificacion_y_leyenda_con_titulo_bajo_el_grafico():
    fig, _ = det_figure()
    ax = fig.axes[0]
    assert "↙ más fácil reidentificar" in texts_of(ax)
    legend = ax.get_legend()
    assert legend.get_title().get_text() == "Escenario (registro-prueba)"
    renderer = FigureCanvasAgg(fig).get_renderer()
    fig.draw(renderer)
    assert legend.get_window_extent(renderer).y1 < ax.get_window_extent(renderer).y0


# --- plot_score_distributions -----------------------------------------------

def test_plot_score_distributions_titulos_con_descripcion_y_eer():
    fig, s, thresholds = dist_figure()
    for ax, name in zip(fig.axes, s):
        target, impostor = s[name]
        eer, _ = compute_eer(target, impostor)
        assert ax.get_title(loc="left") == SCENARIO_LABEL[name]
        assert ax.get_title(loc="right") == f"EER {decimal_comma(100 * eer, 2)} %"


def test_plot_score_distributions_sombrea_falsa_aceptacion_y_falso_rechazo_en_el_umbral():
    fig, s, thresholds = dist_figure()
    for ax, name in zip(fig.axes, s):
        fa, fr = fills(ax)
        assert matplotlib.colors.same_color(fa.get_facecolor()[0][:3], FALSE_ACCEPT_COLOR)
        assert matplotlib.colors.same_color(fr.get_facecolor()[0][:3], TARGET_COLOR)
        assert fa.get_paths()[0].vertices[:, 0].min() == pytest.approx(thresholds[name])
        assert fr.get_paths()[0].vertices[:, 0].max() == pytest.approx(thresholds[name])


def test_plot_score_distributions_areas_sombreadas_son_las_tasas_de_error():
    fig, s, thresholds = dist_figure(bins=40)
    for ax, name in zip(fig.axes, s):
        target, impostor = s[name]
        fa, fr = fills(ax)
        # Tolerancia: la parte del intervalo cortado por el umbral.
        assert polygon_area(fa) == pytest.approx(np.mean(impostor >= thresholds[name]), abs=0.02)
        assert polygon_area(fr) == pytest.approx(np.mean(target < thresholds[name]), abs=0.02)


def test_plot_score_distributions_leyenda_de_errores_en_la_figura():
    fig, _, _ = dist_figure()
    texts = [t.get_text() for t in fig.legends[0].get_texts()]
    assert texts == ["Falsa aceptación", "Falso rechazo", "Umbral del EER"]


def test_plot_score_distributions_leyenda_de_errores_no_tapa_los_paneles():
    fig, _, _ = dist_figure()
    renderer = FigureCanvasAgg(fig).get_renderer()
    fig.draw(renderer)
    lowest = min(ax.get_tightbbox(renderer).y0 for ax in fig.axes)
    assert fig.legends[0].get_window_extent(renderer).y1 <= lowest + 1


# --- plot_eer_ci ----------------------------------------------------------------

def test_plot_eer_ci_sin_descripciones_ni_columna_de_valores():
    texts = texts_of(plot_eer_ci(CI).axes[0])
    assert not any(label.split(",")[0] in t for label in SCENARIO_LABEL.values() for t in texts)
    assert not any("IC" in t for t in texts)


def ci_label(ax, text):
    matches = [t for t in ax.texts if t.get_text() == text]
    assert len(matches) == 1, text
    return matches[0]


def test_plot_eer_ci_numeros_en_los_extremos_y_en_el_punto():
    fig = plot_eer_ci(CI)
    ax = fig.axes[0]
    for name, eer, low, high in [("o-o", "2,00 %", "1,50", "3,00"), ("o-a", "40,00 %", "35,00", "47,00"),
                                 ("a-a", "9,18 %", "8,00", "11,00")]:
        point, bar = ci_parts(fig, name)
        y = point.get_ydata()[0]
        bar_low, bar_high = sorted(bar.get_xdata())
        low_label, high_label, eer_label = ci_label(ax, low), ci_label(ax, high), ci_label(ax, eer)
        # Inferior: anclado al extremo inferior y escrito hacia la izquierda.
        assert low_label.xy == pytest.approx((bar_low, y))
        assert low_label.get_ha() == "right" and low_label.xyann[0] < 0
        # Superior: anclado al extremo superior y escrito hacia la derecha.
        assert high_label.xy == pytest.approx((bar_high, y))
        assert high_label.get_ha() == "left" and high_label.xyann[0] > 0
        # EER: sobre el punto.
        assert eer_label.xy == pytest.approx((point.get_xdata()[0], y))
        assert eer_label.xyann[1] > 0 and eer_label.get_va() == "bottom"


def rendered(fig):
    renderer = FigureCanvasAgg(fig).get_renderer()
    fig.draw(renderer)
    return renderer


def test_plot_eer_ci_numeros_no_se_salen_por_la_izquierda_con_eer_cercano_a_cero():
    scenarios = {"o-o": {"eer": 0.0172, "ci_low": 0.0138, "ci_high": 0.0220},
                 "o-a": {"eer": 0.1485, "ci_low": 0.1424, "ci_high": 0.1561},
                 "a-a": {"eer": 0.0918, "ci_low": 0.0848, "ci_high": 0.0977}}
    fig = plot_eer_ci(scenarios)
    ax = fig.axes[0]
    renderer = rendered(fig)
    assert ax.get_xlim()[0] < 0
    label = ci_label(ax, "1,38")
    assert label.get_window_extent(renderer).x0 >= ax.get_window_extent(renderer).x0


def test_plot_eer_ci_eje_desde_cero_si_los_numeros_caben():
    scenarios = {name: {"eer": 0.12, "ci_low": 0.10, "ci_high": 0.14} for name in CI}
    assert plot_eer_ci(scenarios).axes[0].get_xlim()[0] == 0


def test_plot_eer_ci_numeros_de_un_renglon_no_se_enciman():
    fig = plot_eer_ci(CI)
    ax = fig.axes[0]
    renderer = rendered(fig)
    for eer, low, high in [("2,00 %", "1,50", "3,00"), ("40,00 %", "35,00", "47,00"), ("9,18 %", "8,00", "11,00")]:
        boxes = [ci_label(ax, t).get_window_extent(renderer) for t in (low, high, eer)]
        for i in range(3):
            for j in range(i + 1, 3):
                assert not boxes[i].overlaps(boxes[j])


def test_plot_eer_ci_zona_de_la_meta_sombreada():
    fig = plot_eer_ci(CI, target=0.25)
    ax = fig.axes[0]
    zones = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert len(zones) == 1
    assert zones[0].get_x() == pytest.approx(25)
    assert zones[0].get_x() + zones[0].get_width() == pytest.approx(ax.get_xlim()[1])


def test_plot_eer_ci_la_zona_de_la_meta_se_ve_aunque_los_eer_sean_bajos():
    low = {name: {"eer": 0.01, "ci_low": 0.005, "ci_high": 0.02} for name in CI}
    assert plot_eer_ci(low).axes[0].get_xlim()[1] >= 1.3 * 30


def test_plot_eer_ci_indica_hacia_donde_hay_mas_privacidad():
    assert "más privacidad →" in texts_of(plot_eer_ci(CI).axes[0])


def test_plot_eer_ci_textos_dentro_de_la_figura():
    fig = plot_eer_ci(CI)
    renderer = FigureCanvasAgg(fig).get_renderer()
    fig.draw(renderer)
    box = fig.bbox
    for text in fig.axes[0].texts:
        extent = text.get_window_extent(renderer)
        assert extent.x0 >= box.x0 - 1 and extent.x1 <= box.x1 + 1
        assert extent.y0 >= box.y0 - 1 and extent.y1 <= box.y1 + 1


def test_ningun_texto_lleva_el_color_de_un_escenario():
    # El color identifica la marca; el texto va en tinta neutra para leerse impreso.
    s, eers, thresholds = three_scenarios()
    figures = [
        plot_det({n: det_curve(*p) for n, p in s.items()}, eers),
        plot_score_distributions(s, thresholds),
        plot_eer_ci(CI),
    ]
    colors = [style["color"] for style in SCENARIO_STYLE.values()]
    for fig in figures:
        for ax in fig.axes:
            for text in ax.texts + [ax.title, ax._left_title, ax._right_title]:
                assert not any(matplotlib.colors.same_color(text.get_color(), c) for c in colors), text.get_text()


def test_plot_eer_ci_marcas_del_eje_redondas():
    for scenarios in (CI, {n: {"eer": 0.0172, "ci_low": 0.0138, "ci_high": 0.022} for n in CI}):
        ticks = plot_eer_ci(scenarios).axes[0].get_xticks()
        steps = set(np.round(np.diff(ticks), 9))
        assert len(steps) == 1 and steps <= {1, 2, 5, 10, 20, 25, 50}


def test_plot_eer_ci_punto_mas_chico_que_las_tapas():
    fig = plot_eer_ci(CI)
    for name in CI:
        point, bar = ci_parts(fig, name)
        assert point.get_markersize() < bar.get_markersize()
