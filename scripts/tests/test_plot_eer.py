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