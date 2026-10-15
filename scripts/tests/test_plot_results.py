import json
import sys

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")

from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import plot_results  # noqa: E402
from plot_eer import COLUMN_WIDTH, SCENARIO_STYLE  # noqa: E402
from plot_results import (  # noqa: E402
    ANONYMIZED_COLOR,
    CASE_LABELS,
    CASE_MARKERS,
    ORIGINAL_COLOR,
    eer_wer_config,
    plot_eer_wer,
    plot_impersonation,
    plot_wer,
    tradeoff_cases,
)


def iv(value, low, high):
    return {"value": value, "ci_low": low, "ci_high": high}


# Intervalos asimétricos a propósito.
OPS = {
    "eer": {"threshold": 0.31, "far_oo": 0.017, "genuine_rate": iv(0.983, 0.977, 0.988),
            "impersonation_rate": iv(0.45, 0.40, 0.48)},
    "far": {"threshold": 0.52, "far_oo": 0.0098, "max_far": 0.01, "genuine_rate": iv(0.92, 0.905, 0.934),
            "impersonation_rate": iv(0.03, 0.021, 0.04)},
}
WER = {"original": iv(0.061, 0.052, 0.070), "anonymized": iv(0.114, 0.10, 0.13), "difference": iv(0.053, 0.04, 0.066)}


def lines_of(ax, color):
    return [l for l in ax.get_lines() if matplotlib.colors.same_color(l.get_color(), color)]


def series(ax, color):
    """Puntos (marcador "o") y barras horizontales (marcador "|") de un color, ordenados de arriba abajo."""
    lines = lines_of(ax, color)
    points = sorted((l for l in lines if l.get_marker() == "o"), key=lambda l: -l.get_ydata()[0])
    bars = sorted((l for l in lines if l.get_marker() == "|"), key=lambda l: -l.get_ydata()[0])
    return points, bars


def vertical_lines(ax):
    return [l for l in ax.get_lines() if len(l.get_xdata()) == 2 and l.get_xdata()[0] == l.get_xdata()[1]
            and l.get_ydata()[0] != l.get_ydata()[1] and l.get_linestyle() == ":"]


def horizontal_lines(ax):
    return [l for l in ax.get_lines() if len(l.get_ydata()) == 2 and l.get_ydata()[0] == l.get_ydata()[1]
            and l.get_linestyle() == ":"]


def texts_of(ax):
    return [t.get_text() for t in ax.texts]


def rendered(fig):
    renderer = FigureCanvasAgg(fig).get_renderer()
    fig.draw(renderer)
    return renderer


def assert_texts_inside(fig):
    renderer = rendered(fig)
    box = fig.bbox
    for ax in fig.axes:
        for text in ax.texts:
            e = text.get_window_extent(renderer)
            assert e.x0 >= box.x0 - 1 and e.x1 <= box.x1 + 1 and e.y0 >= box.y0 - 1 and e.y1 <= box.y1 + 1, \
                text.get_text()


def assert_no_text_in_series_color(fig):
    for ax in fig.axes:
        for text in ax.texts:
            for color in (ORIGINAL_COLOR, ANONYMIZED_COLOR):
                assert not matplotlib.colors.same_color(text.get_color(), color), text.get_text()


def assert_interval(point, bar, interval):
    assert point.get_xdata()[0] == pytest.approx(100 * interval["value"])
    np.testing.assert_allclose(sorted(bar.get_xdata()), [100 * interval["ci_low"], 100 * interval["ci_high"]])
    assert set(bar.get_ydata()) == {point.get_ydata()[0]}


# ---------------------------------------------------------------------------
# plot_wer
# ---------------------------------------------------------------------------

def test_plot_wer_figura_de_una_columna_sin_titulo():
    fig = plot_wer(WER)
    assert isinstance(fig, Figure) and len(fig.axes) == 1
    assert fig.get_figwidth() == pytest.approx(COLUMN_WIDTH) and fig.axes[0].get_title() == ""


def test_plot_wer_puntos_y_extremos_de_cada_intervalo():
    ax = plot_wer(WER).axes[0]
    for key, color in (("original", ORIGINAL_COLOR), ("anonymized", ANONYMIZED_COLOR)):
        (point,), (bar,) = series(ax, color)
        assert_interval(point, bar, WER[key])


def test_plot_wer_intervalo_asimetrico_no_se_vuelve_simetrico():
    ax = plot_wer(WER).axes[0]
    (point,), (bar,) = series(ax, ANONYMIZED_COLOR)
    low, high = sorted(bar.get_xdata())
    assert 11.4 - low == pytest.approx(1.4) and high - 11.4 == pytest.approx(1.6)


def test_plot_wer_original_arriba_y_rotulos():
    ax = plot_wer(WER).axes[0]
    (orig,), _ = series(ax, ORIGINAL_COLOR)
    (anon,), _ = series(ax, ANONYMIZED_COLOR)
    assert orig.get_ydata()[0] > anon.get_ydata()[0]
    ticks = dict(zip(ax.get_yticks(), [t.get_text() for t in ax.get_yticklabels()]))
    assert ticks[orig.get_ydata()[0]] == "Original" and ticks[anon.get_ydata()[0]] == "Anonimizado"


def test_plot_wer_meta_punteada_en_15_por_ciento():
    ax = plot_wer(WER).axes[0]
    (line,) = vertical_lines(ax)
    assert line.get_xdata()[0] == pytest.approx(15)
    assert "Meta (15 %)" in texts_of(ax)


def test_plot_wer_meta_en_target():
    ax = plot_wer(WER, target=0.125).axes[0]
    assert vertical_lines(ax)[0].get_xdata()[0] == pytest.approx(12.5)
    assert "Meta (12,5 %)" in texts_of(ax)
    label = [t for t in ax.texts if t.get_text().startswith("Meta")][0]
    assert label.get_position()[0] == pytest.approx(12.5)


def test_plot_wer_numeros_y_coma_decimal():
    ax = plot_wer(WER).axes[0]
    for text in ("6,10 %", "5,20", "7,00", "11,40 %", "10,00", "13,00"):
        assert text in texts_of(ax)
    assert ax.get_xlabel() == "WER (%)"
    assert ax.xaxis.get_major_formatter()(12.5, 0) == "12,5"


def test_plot_wer_eje_incluye_meta_e_intervalos():
    low, high = plot_wer(WER).axes[0].get_xlim()
    assert low <= 0 and high > 15
    low, high = plot_wer({**WER, "anonymized": iv(0.4, 0.35, 0.5)}).axes[0].get_xlim()
    assert high > 50


def test_plot_wer_textos_dentro_y_en_tinta_neutra():
    fig = plot_wer(WER)
    assert_texts_inside(fig)
    assert_no_text_in_series_color(fig)


# ---------------------------------------------------------------------------
# plot_impersonation
# ---------------------------------------------------------------------------

def test_plot_impersonation_dos_series_por_punto_de_operacion():
    ax = plot_impersonation(OPS).axes[0]
    for key, color in (("genuine_rate", ORIGINAL_COLOR), ("impersonation_rate", ANONYMIZED_COLOR)):
        points, bars = series(ax, color)
        assert len(points) == len(bars) == 2
        for point, bar, name in zip(points, bars, ("eer", "far")):  # de arriba abajo
            assert_interval(point, bar, OPS[name][key])


def test_plot_impersonation_intervalo_asimetrico():
    ax = plot_impersonation(OPS).axes[0]
    points, bars = series(ax, ANONYMIZED_COLOR)
    low, high = sorted(bars[0].get_xdata())
    assert 45 - low == pytest.approx(5) and high - 45 == pytest.approx(3)


def test_plot_impersonation_grupos_y_orden():
    ax = plot_impersonation(OPS).axes[0]
    genuine, _ = series(ax, ORIGINAL_COLOR)
    anon, _ = series(ax, ANONYMIZED_COLOR)
    ticks = sorted(zip(ax.get_yticks(), [t.get_text() for t in ax.get_yticklabels()]), reverse=True)
    assert [t for _, t in ticks] == ["Umbral EER (o-o)", "Umbral FAR ≤ 1 %"]
    for i, (center, _) in enumerate(ticks):
        # En cada grupo, la persona real arriba y la grabación anonimizada abajo.
        assert genuine[i].get_ydata()[0] > center > anon[i].get_ydata()[0]


def test_plot_impersonation_rotulo_far_sale_del_json():
    ops = {**OPS, "far": {**OPS["far"], "max_far": 0.05}}
    labels = [t.get_text() for t in plot_impersonation(ops).axes[0].get_yticklabels()]
    assert "Umbral FAR ≤ 5 %" in labels


def test_plot_impersonation_leyenda_y_colores_distintos():
    fig = plot_impersonation(OPS)
    (legend,) = fig.legends
    assert [t.get_text() for t in legend.get_texts()] == ["Persona real", "Grabación anonimizada"]
    colors = [h.get_color() for h in legend.legend_handles]
    assert matplotlib.colors.same_color(colors[0], ORIGINAL_COLOR)
    assert matplotlib.colors.same_color(colors[1], ANONYMIZED_COLOR)
    assert not matplotlib.colors.same_color(ORIGINAL_COLOR, ANONYMIZED_COLOR)


def test_plot_impersonation_eje_de_0_a_100():
    ax = plot_impersonation(OPS).axes[0]
    low, high = ax.get_xlim()
    assert high == 100 and low <= 0
    assert min(ax.get_xticks()) == 0 and max(ax.get_xticks()) == 100
    assert ax.get_xlabel() == "Aceptación (%)"
    assert ax.xaxis.get_major_formatter()(2.5, 0) == "2,5"


def test_plot_impersonation_numeros_con_coma():
    texts = texts_of(plot_impersonation(OPS).axes[0])
    for text in ("98,30 %", "97,70", "98,80", "45,00 %", "40,00", "48,00", "3,00 %", "2,10", "4,00"):
        assert text in texts


def test_plot_impersonation_textos_dentro_y_en_tinta_neutra():
    fig = plot_impersonation(OPS)
    assert_texts_inside(fig)
    assert_no_text_in_series_color(fig)


# ---------------------------------------------------------------------------
# plot_eer_wer, eer_wer_config y tradeoff_cases
# ---------------------------------------------------------------------------

def eer_json(oo=(0.0172, 0.0138, 0.0220), oa=(0.1485, 0.1424, 0.1561), aa=(0.0918, 0.0848, 0.0977)):
    return {"scenarios": {name: {"eer": v, "ci_low": lo, "ci_high": hi, "n_target": 10, "n_impostor": 90}
                          for name, (v, lo, hi) in (("o-o", oo), ("o-a", oa), ("a-a", aa))}}


def wer_json(original=(0.0384, 0.034, 0.043), anonymized=(0.0986, 0.088, 0.112)):
    return {"wer": {"original": iv(*original), "anonymized": iv(*anonymized), "difference": iv(0.06, 0.05, 0.07)}}


ONE_CONFIG = [eer_wer_config("U(0,7; 0,9)", eer_json(), wer_json())]
THREE_CONFIGS = ONE_CONFIG + [
    eer_wer_config("U(0,6; 0,8)", eer_json(oa=(0.24, 0.20, 0.25), aa=(0.16, 0.151, 0.168)),
                   wer_json(anonymized=(0.135, 0.125, 0.146))),
    eer_wer_config("U(0,5; 0,7)", eer_json(oa=(0.33, 0.32, 0.34), aa=(0.22, 0.21, 0.23)),
                   wer_json(anonymized=(0.19, 0.178, 0.203))),
]


def case_parts(ax, scenario):
    """Puntos, barras horizontales (IC del WER) y verticales (IC del EER) de un escenario."""
    lines = lines_of(ax, SCENARIO_STYLE[scenario]["color"])
    return ([l for l in lines if l.get_marker() == CASE_MARKERS[scenario]],
            [l for l in lines if l.get_marker() == "|"], [l for l in lines if l.get_marker() == "_"])


def test_eer_wer_config_toma_eer_por_escenario_y_los_dos_wer():
    config = eer_wer_config("a", eer_json(), wer_json())
    assert config["label"] == "a"
    assert config["eer"]["o-a"] == {"value": 0.1485, "ci_low": 0.1424, "ci_high": 0.1561}
    assert set(config["eer"]) == {"o-o", "o-a", "a-a"}
    assert config["wer"] == {"original": iv(0.0384, 0.034, 0.043), "anonymized": iv(0.0986, 0.088, 0.112)}


def test_tradeoff_cases_una_configuracion():
    cases = tradeoff_cases(ONE_CONFIG)
    config = ONE_CONFIG[0]
    assert [(c["scenario"], c["label"]) for c in cases] == [("o-o", ""), ("o-a", "U(0,7; 0,9)"),
                                                            ("a-a", "U(0,7; 0,9)")]
    assert cases[0]["x"] == config["wer"]["original"] and cases[0]["y"] == config["eer"]["o-o"]
    for case in cases[1:]:
        assert case["x"] == config["wer"]["anonymized"] and case["y"] == config["eer"][case["scenario"]]


def test_tradeoff_cases_sin_anonimizar_una_sola_vez():
    cases = tradeoff_cases(THREE_CONFIGS)
    assert [c["scenario"] for c in cases].count("o-o") == 1 and len(cases) == 1 + 2 * 3
    assert [c["label"] for c in cases if c["scenario"] == "o-a"] == ["U(0,7; 0,9)", "U(0,6; 0,8)", "U(0,5; 0,7)"]


@pytest.mark.parametrize("configs", [ONE_CONFIG, THREE_CONFIGS])
def test_plot_eer_wer_puntos_e_intervalos_de_cada_caso(configs):
    ax = plot_eer_wer(configs).axes[0]
    cases = tradeoff_cases(configs)
    for scenario in ("o-o", "o-a", "a-a"):
        expected = [c for c in cases if c["scenario"] == scenario]
        dots, h_bars, v_bars = case_parts(ax, scenario)
        assert len(dots) == len(h_bars) == len(v_bars) == len(expected)
        for c, dot, h, v in zip(expected, dots, h_bars, v_bars):
            x, y = 100 * c["x"]["value"], 100 * c["y"]["value"]
            assert (dot.get_xdata()[0], dot.get_ydata()[0]) == pytest.approx((x, y))
            np.testing.assert_allclose(sorted(h.get_xdata()), [100 * c["x"]["ci_low"], 100 * c["x"]["ci_high"]])
            np.testing.assert_allclose(h.get_ydata(), [y, y])
            np.testing.assert_allclose(sorted(v.get_ydata()), [100 * c["y"]["ci_low"], 100 * c["y"]["ci_high"]])
            np.testing.assert_allclose(v.get_xdata(), [x, x])


def test_plot_eer_wer_intervalos_asimetricos():
    # Segunda configuración: EER o-a 24 [20; 25] y WER 13,5 [12,5; 14,6].
    ax = plot_eer_wer(THREE_CONFIGS).axes[0]
    dots, h_bars, v_bars = case_parts(ax, "o-a")
    np.testing.assert_allclose(sorted(v_bars[1].get_ydata()), [20, 25])
    np.testing.assert_allclose(sorted(h_bars[1].get_xdata()), [12.5, 14.6])


@pytest.mark.parametrize("configs", [ONE_CONFIG, THREE_CONFIGS])
def test_plot_eer_wer_rotulo_de_cada_configuracion_junto_a_su_punto_o_a(configs):
    ax = plot_eer_wer(configs).axes[0]
    for config in configs:
        (label,) = [t for t in ax.texts if t.get_text() == config["label"]]
        x, y = 100 * config["wer"]["anonymized"]["value"], 100 * config["eer"]["o-a"]["value"]
        assert label.xy == pytest.approx((x, y))
    assert len([t for t in ax.texts if t.get_text().startswith("U(")]) == len(configs)


def test_plot_eer_wer_metas_en_su_valor():
    ax = plot_eer_wer(ONE_CONFIG).axes[0]
    (v_line,), (h_line,) = vertical_lines(ax), horizontal_lines(ax)
    assert v_line.get_xdata()[0] == pytest.approx(15) and h_line.get_ydata()[0] == pytest.approx(30)
    assert "Meta WER (15 %)" in texts_of(ax) and "Meta EER (30 %)" in texts_of(ax)


def test_plot_eer_wer_metas_configurables():
    ax = plot_eer_wer(ONE_CONFIG, wer_target=0.125, eer_target=0.4).axes[0]
    assert vertical_lines(ax)[0].get_xdata()[0] == pytest.approx(12.5)
    assert horizontal_lines(ax)[0].get_ydata()[0] == pytest.approx(40)
    assert "Meta WER (12,5 %)" in texts_of(ax) and "Meta EER (40 %)" in texts_of(ax)


def test_plot_eer_wer_zona_donde_se_cumplen_las_dos_metas():
    ax = plot_eer_wer(ONE_CONFIG).axes[0]
    (zone,) = [c for c in ax.collections if isinstance(c, PolyCollection)]
    x, y = zone.get_paths()[0].vertices.T
    assert (x.min(), x.max()) == pytest.approx((0, 15))
    assert y.min() == pytest.approx(30) and y.max() == pytest.approx(ax.get_ylim()[1])
    assert "↖ mejor" in texts_of(ax)


def test_plot_eer_wer_ejes():
    ax = plot_eer_wer(THREE_CONFIGS).axes[0]
    assert ax.get_xlabel() == "WER (%)" and ax.get_ylabel() == "EER (%)"
    assert ax.get_xlim()[0] == 0 and ax.get_xlim()[1] > 20.3
    assert ax.get_ylim()[0] == 0 and ax.get_ylim()[1] > 34
    assert ax.xaxis.get_major_formatter()(2.5, 0) == "2,5" and ax.yaxis.get_major_formatter()(0.5, 0) == "0,5"
    low_y = plot_eer_wer(ONE_CONFIG).axes[0].get_ylim()[1]
    assert low_y > 30  # la meta se ve aunque los EER sean bajos


def test_plot_eer_wer_leyenda_con_colores_y_marcadores_de_cada_escenario():
    fig = plot_eer_wer(ONE_CONFIG)
    (legend,) = fig.legends
    assert [t.get_text() for t in legend.get_texts()] == [CASE_LABELS[n] for n in ("o-o", "o-a", "a-a")]
    for handle, name in zip(legend.legend_handles, ("o-o", "o-a", "a-a")):
        assert matplotlib.colors.same_color(handle.get_color(), SCENARIO_STYLE[name]["color"])
        assert handle.get_marker() == CASE_MARKERS[name]
    assert len(set(CASE_MARKERS.values())) == 3


def test_plot_eer_wer_figura_de_una_columna_textos_dentro_y_en_tinta_neutra():
    fig = plot_eer_wer(THREE_CONFIGS)
    assert fig.get_figwidth() == pytest.approx(COLUMN_WIDTH) and fig.axes[0].get_title() == ""
    assert_texts_inside(fig)
    for text in fig.axes[0].texts:
        for style in SCENARIO_STYLE.values():
            assert not matplotlib.colors.same_color(text.get_color(), style["color"]), text.get_text()


def test_plot_eer_wer_leyenda_no_tapa_el_grafico():
    fig = plot_eer_wer(ONE_CONFIG)
    renderer = rendered(fig)
    assert fig.legends[0].get_window_extent(renderer).y1 <= fig.axes[0].get_tightbbox(renderer).y0 + 1


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

@pytest.fixture
def jsons(tmp_path):
    imp, wer, eer = tmp_path / "imp.json", tmp_path / "wer.json", tmp_path / "eer.json"
    imp.write_text(json.dumps({"operating_points": OPS}), encoding="utf-8")
    wer.write_text(json.dumps({"wer": WER}), encoding="utf-8")
    eer.write_text(json.dumps(eer_json()), encoding="utf-8")
    return imp, wer, eer


def run_main(monkeypatch, out_dir, *args):
    monkeypatch.setattr(sys, "argv", ["plot_results.py", "--output-dir", str(out_dir), *args])
    with matplotlib.rc_context():
        plot_results.main()


def created(out_dir):
    return sorted(p.name for p in out_dir.iterdir()) if out_dir.exists() else []


def test_main_solo_suplantacion(monkeypatch, tmp_path, jsons):
    run_main(monkeypatch, tmp_path / "f", "--impersonation-json", str(jsons[0]))
    assert created(tmp_path / "f") == ["suplantacion.pdf"]
    assert (tmp_path / "f" / "suplantacion.pdf").read_bytes()[:4] == b"%PDF"


def test_main_solo_wer(monkeypatch, tmp_path, jsons):
    run_main(monkeypatch, tmp_path / "f", "--wer-json", str(jsons[1]))
    assert created(tmp_path / "f") == ["wer.pdf"]


def test_main_solo_compromiso(monkeypatch, tmp_path, jsons):
    run_main(monkeypatch, tmp_path / "f", "--config", "U(0,7; 0,9)", str(jsons[2]), str(jsons[1]))
    assert created(tmp_path / "f") == ["compromiso.pdf"]


def test_main_todas(monkeypatch, tmp_path, jsons, capsys):
    imp, wer, eer = (str(p) for p in jsons)
    run_main(monkeypatch, tmp_path / "f", "--impersonation-json", imp, "--wer-json", wer,
             "--config", "a", eer, wer)
    assert created(tmp_path / "f") == ["compromiso.pdf", "suplantacion.pdf", "wer.pdf"]
    assert "compromiso.pdf" in capsys.readouterr().out


def test_main_sin_json_no_crea_nada(monkeypatch, tmp_path, capsys):
    run_main(monkeypatch, tmp_path / "f")
    assert created(tmp_path / "f") == []
    assert "ningún JSON" in capsys.readouterr().out


def test_main_pasa_los_datos_de_cada_json(monkeypatch, tmp_path, jsons):
    calls = {}

    class Dummy:
        def savefig(self, path):
            calls.setdefault("saved", []).append(path.name)

    monkeypatch.setattr(plot_results, "plot_impersonation", lambda ops: calls.update(ops=ops) or Dummy())
    monkeypatch.setattr(plot_results, "plot_wer", lambda w: calls.update(wer=w) or Dummy())
    monkeypatch.setattr(plot_results, "plot_eer_wer", lambda configs: calls.update(configs=configs) or Dummy())
    imp, wer, eer = (str(p) for p in jsons)
    run_main(monkeypatch, tmp_path / "f", "--impersonation-json", imp, "--wer-json", wer,
             "--config", "a", eer, wer, "--config", "b", eer, wer, "--config", "c", eer, wer)
    assert calls["ops"] == OPS and calls["wer"] == WER
    assert [c["label"] for c in calls["configs"]] == ["a", "b", "c"]
    assert calls["configs"][0] == eer_wer_config("a", eer_json(), {"wer": WER})


def label_named(ax, text):
    (label,) = [t for t in ax.texts if t.get_text() == text]
    return label


def test_draw_interval_numeros_anclados_a_su_extremo():
    ax = plot_wer(WER).axes[0]
    (point,), _ = series(ax, ANONYMIZED_COLOR)
    y = point.get_ydata()[0]
    low, high, value = label_named(ax, "10,00"), label_named(ax, "13,00"), label_named(ax, "11,40 %")
    assert low.xy == pytest.approx((10, y)) and low.get_ha() == "right" and low.xyann[0] < 0
    assert high.xy == pytest.approx((13, y)) and high.get_ha() == "left" and high.xyann[0] > 0
    assert value.xy == pytest.approx((11.4, y)) and value.xyann[1] > 0


def test_plot_impersonation_valor_de_la_grabacion_debajo_del_punto():
    ax = plot_impersonation(OPS).axes[0]
    assert label_named(ax, "45,00 %").xyann[1] < 0 and label_named(ax, "98,30 %").xyann[1] > 0


def test_plot_wer_cerca_de_cero_el_numero_inferior_no_se_sale():
    wer = {**WER, "original": iv(0.004, 0.002, 0.007)}
    fig = plot_wer(wer)
    ax = fig.axes[0]
    renderer = rendered(fig)
    assert ax.get_xlim()[0] < 0
    assert label_named(ax, "0,20").get_window_extent(renderer).x0 >= ax.get_window_extent(renderer).x0


def test_plot_impersonation_marcas_cada_20():
    assert list(plot_impersonation(OPS).axes[0].get_xticks()) == [0, 20, 40, 60, 80, 100]


def test_eer_wer_config_json_de_suplantacion_error_claro():
    with pytest.raises(ValueError, match="eer_bootstrap.py.*impersonation.py"):
        eer_wer_config("U(0,7; 0,9)", {"operating_points": OPS}, wer_json())


def test_eer_wer_config_json_de_run_eer_sin_intervalos_error_claro():
    run_eer_json = {"scenarios": {"o-o": {"eer": 0.02, "threshold": 0.4, "n_target": 1, "n_impostor": 9}}}
    with pytest.raises(ValueError, match="eer_bootstrap.py"):
        eer_wer_config("a", run_eer_json, wer_json())


def test_eer_wer_config_json_del_wer_equivocado():
    with pytest.raises(ValueError, match="wer_bootstrap.py"):
        eer_wer_config("a", eer_json(), {"operating_points": OPS})


def test_main_config_con_el_json_equivocado_avisa_sin_traceback_y_no_crea_nada(monkeypatch, tmp_path, jsons):
    imp, wer, _ = (str(p) for p in jsons)
    with pytest.raises(SystemExit) as exit_info:
        run_main(monkeypatch, tmp_path / "f", "--impersonation-json", imp, "--wer-json", wer,
                 "--config", "U(0,7; 0,9)", imp, wer)
    message = str(exit_info.value)
    assert "impersonation.py" in message and f"EER_JSON = {imp}" in message
    assert "eer_bootstrap_k134.json" in message  # muestra la forma correcta del comando
    assert created(tmp_path / "f") == []


def test_read_json_acepta_bom(tmp_path):
    path = tmp_path / "con_bom.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"wer": WER}).encode("utf-8"))
    assert plot_results.read_json(path) == {"wer": WER}
