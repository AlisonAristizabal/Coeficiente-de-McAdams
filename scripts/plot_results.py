"""
Figuras de resultados a partir de los JSON ya generados (no recalcula nada).

- suplantacion.pdf: tasa de aceptación de la persona real y de la grabación
  anonimizada en cada punto de operación (JSON de `impersonation.py`).
- wer.pdf: WER original y anonimizado con sus intervalos (JSON de
  `wer_bootstrap.py`).
- compromiso.pdf: EER frente a WER (privacidad contra utilidad): el caso sin
  anonimizar y los casos o-a y a-a de cada configuración de α (JSON de
  `eer_bootstrap.py` y `wer_bootstrap.py`).

Siguen el estilo de `plot_eer.plot_eer_ci`: punto con barra de intervalo
(de `ci_low` a `ci_high` tal cual, no simétrica), los números en los
extremos y sobre el punto, coma decimal, ancho de una columna y sin título.
"""
import argparse
import json
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

from plot_eer import (
    CI_LABEL_GAP,
    COLUMN_WIDTH,
    GOAL_EER,
    GOAL_FILL,
    INK,
    MUTED_INK,
    SCENARIO_STYLE,
    SMALL_FONT,
    configure_fonts,
    decimal_comma,
    left_limit,
    percent_label,
)

# Azul para la voz original y bermellón para la anonimizada (la misma pareja
# de la paleta validada que usan o-o y o-a en plot_eer).
ORIGINAL_COLOR = "#0072B2"
ANONYMIZED_COLOR = "#D55E00"

WER_TARGET = 0.15

# Figura de compromiso: el color de cada caso es el de su escenario en
# plot_eer y además lleva un marcador propio, para leerse en blanco y negro.
CASE_MARKERS = {"o-o": "o", "o-a": "s", "a-a": "^"}
CASE_LABELS = {
    "o-o": "o-o: sin anonimizar (WER original)",
    "o-a": "o-a: registro original, prueba anonimizada",
    "a-a": "a-a: registro y prueba anonimizados",
}

# Puntos de operación de impersonation.py, de arriba abajo.
OPERATING_POINTS = ("eer", "far")
# Series de la figura de suplantación: (llave del JSON, rótulo, color).
ACCEPTANCE_SERIES = (
    ("genuine_rate", "Persona real", ORIGINAL_COLOR),
    ("impersonation_rate", "Grabación anonimizada", ANONYMIZED_COLOR),
)


def draw_interval(ax, interval, y, color, value_side="above"):
    """
    Dibuja un punto con su barra de intervalo horizontal y sus números.

    La barra va de `ci_low` a `ci_high` y el punto en `value`, todo pasado a
    porcentaje. El límite inferior se escribe a la izquierda de su extremo,
    el superior a la derecha del suyo y el valor en negrita encima (o
    debajo) del punto.

    Parámetros:
        ax: eje donde se dibuja.
        interval: diccionario {"value", "ci_low", "ci_high"} como fracciones.
        y: renglón (coordenada y).
        color: color de la barra y el punto.
        value_side: "above" o "below", dónde va el número del valor.

    Retorna:
        La anotación del límite inferior (para `left_limit`).
    """
    value, low, high = (100 * interval[k] for k in ("value", "ci_low", "ci_high"))
    ax.plot([low, high], [y, y], color=color, linewidth=1.6, marker="|", markersize=10)
    ax.plot(value, y, color=color, marker="o", markersize=4, markeredgewidth=0, linestyle="none", zorder=3)
    number = {"fontsize": SMALL_FONT, "textcoords": "offset points"}
    low_label = ax.annotate(decimal_comma(low, 2), (low, y), xytext=(-CI_LABEL_GAP, 0), ha="right", va="center",
                            color=MUTED_INK, **number)
    ax.annotate(decimal_comma(high, 2), (high, y), xytext=(CI_LABEL_GAP, 0), ha="left", va="center",
                color=MUTED_INK, **number)
    above = value_side == "above"
    ax.annotate(f"{decimal_comma(value, 2)} %", (value, y), xytext=(0, 5 if above else -5), ha="center",
                va="bottom" if above else "top", color=INK, fontweight="bold", **number)
    return low_label


def percent_axis(ax, x_max, label):
    """Eje x en porcentaje desde 0, con marcas redondas y coma decimal."""
    ax.set_xticks([t for t in MaxNLocator(nbins=5, steps=[1, 2, 5, 10]).tick_values(0, x_max) if 0 <= t <= x_max])
    ax.set_xlim(0, x_max)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: percent_label(x)))
    ax.set_xlabel(label)
    ax.grid(axis="x", color="0.9", linewidth=0.5)


def goal_line(ax, target, label):
    """Línea vertical punteada en `target` (fracción) con su rótulo encima del gráfico."""
    ax.axvline(100 * target, color="0.3", linestyle=":", linewidth=0.8)
    ax.text(100 * target, 1.01, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom")


def plot_impersonation(operating_points):
    """
    Dibuja la aceptación de la persona real y de la grabación anonimizada.

    Un grupo de renglones por punto de operación ("eer" arriba, "far"
    abajo). En cada grupo, arriba la persona real con su voz original
    (genuine_rate) y abajo la grabación anonimizada (impersonation_rate),
    cada una con su intervalo. El eje x va de 0 a 100 %; si el número del
    límite inferior no cabe a la izquierda, el eje empieza un poco antes de 0.

    Parámetros:
        operating_points: llave "operating_points" del JSON de
            `impersonation.py`.

    Retorna:
        Una `Figure` de matplotlib.
    """
    names = [name for name in OPERATING_POINTS if name in operating_points]
    max_far = operating_points.get("far", {}).get("max_far", 0.01)
    titles = {"eer": "Umbral EER (o-o)", "far": f"Umbral FAR ≤ {percent_label(100 * max_far)} %"}
    height = 0.8 * len(names) + 1.1
    fig = Figure(figsize=(COLUMN_WIDTH, height))
    ax = fig.subplots()
    centers = [len(names) - 1 - i for i in range(len(names))]
    low_labels = []
    for center, name in zip(centers, names):
        for (key, _, color), offset, side in zip(ACCEPTANCE_SERIES, (0.2, -0.2), ("above", "below")):
            low_labels.append(draw_interval(ax, operating_points[name][key], center + offset, color, side))
    ax.set_yticks(centers, [titles[name] for name in names])
    ax.set_ylim(-0.65, len(names) - 0.35)
    percent_axis(ax, 100, "Aceptación (%)")
    handles = [Line2D([], [], color=color, linewidth=1.6, marker="o", markersize=4, label=label)
               for _, label, color in ACCEPTANCE_SERIES]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=SMALL_FONT)
    rect = (0, 0.25 / height, 1, 1)
    fig.tight_layout(rect=rect)
    ax.set_xlim(left_limit(ax, low_labels, 100), 100)
    fig.tight_layout(rect=rect)
    return fig


def plot_wer(wer_summary, target=WER_TARGET):
    """
    Dibuja el WER original y el anonimizado con sus intervalos y la meta.

    Parámetros:
        wer_summary: llave "wer" del JSON de `wer_bootstrap.py`, con
            "original" y "anonymized" ({"value", "ci_low", "ci_high"}).
        target: meta de WER como fracción.

    Retorna:
        Una `Figure` de matplotlib.
    """
    rows = (("original", "Original", ORIGINAL_COLOR, 1), ("anonymized", "Anonimizado", ANONYMIZED_COLOR, 0))
    fig = Figure(figsize=(COLUMN_WIDTH, 0.45 * len(rows) + 0.8))
    ax = fig.subplots()
    x_max = 100 * max([1.3 * target] + [1.12 * wer_summary[key]["ci_high"] for key, *_ in rows])
    goal_line(ax, target, f"Meta ({percent_label(100 * target)} %)")
    low_labels = [draw_interval(ax, wer_summary[key], y, color) for key, _, color, y in rows]
    ax.set_yticks([y for *_, y in rows], [label for _, label, _, _ in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    percent_axis(ax, x_max, "WER (%)")
    fig.tight_layout()
    ax.set_xlim(left_limit(ax, low_labels, x_max), x_max)
    fig.tight_layout()
    return fig


def eer_wer_config(label, eer_summary, wer_summary):
    """
    Reúne los datos de una configuración de α para la figura de compromiso.

    Parámetros:
        label: rótulo de la configuración, por ejemplo "U(0,7; 0,9)".
        eer_summary: JSON de `eer_bootstrap.py` ya leído (llave "scenarios"
            con "eer", "ci_low" y "ci_high" por escenario).
        wer_summary: JSON de `wer_bootstrap.py` ya leído (llave "wer").

    Lanza `ValueError` con un mensaje claro si algún JSON no es del tipo
    esperado (por ejemplo, el de `impersonation.py` en lugar del de
    `eer_bootstrap.py`, o el de `run_eer.py`, que no trae intervalos).

    Retorna:
        Diccionario {"label", "eer": {escenario: {"value", "ci_low",
        "ci_high"}}, "wer": {"original", "anonymized"}}.
    """
    scenarios = eer_summary.get("scenarios")
    if not scenarios or any("ci_low" not in s for s in scenarios.values()):
        raise ValueError(f"{label}: el JSON del EER debe ser el de eer_bootstrap.py (llave 'scenarios' con "
                         "intervalos); ¿se pasó el de impersonation.py o el de run_eer.py?")
    if "wer" not in wer_summary:
        raise ValueError(f"{label}: el JSON del WER debe ser el de wer_bootstrap.py (llave 'wer')")
    eer = {name: {"value": s["eer"], "ci_low": s["ci_low"], "ci_high": s["ci_high"]}
           for name, s in eer_summary["scenarios"].items()}
    return {"label": label, "eer": eer,
            "wer": {key: wer_summary["wer"][key] for key in ("original", "anonymized")}}


def tradeoff_cases(configs):
    """
    Arma los casos de la figura de compromiso: un punto (WER, EER) por caso.

    El caso sin anonimizar (o-o con el WER original) es el mismo para todas
    las configuraciones, porque el audio original no depende de α, así que se
    toma de la primera y aparece una sola vez. Cada configuración aporta dos
    casos con su WER anonimizado: o-a (registro original, prueba anonimizada)
    y a-a (ambos anonimizados).

    Parámetros:
        configs: lista de salidas de `eer_wer_config`.

    Retorna:
        Lista de diccionarios {"scenario", "label", "x": WER, "y": EER}, con
        x e y como {"value", "ci_low", "ci_high"}. `label` es el rótulo de la
        configuración (vacío para el caso sin anonimizar).
    """
    first = configs[0]
    cases = [{"scenario": "o-o", "label": "", "x": first["wer"]["original"], "y": first["eer"]["o-o"]}]
    for config in configs:
        for scenario in ("o-a", "a-a"):
            cases.append({"scenario": scenario, "label": config["label"],
                          "x": config["wer"]["anonymized"], "y": config["eer"][scenario]})
    return cases


def plot_eer_wer(configs, wer_target=WER_TARGET, eer_target=GOAL_EER):
    """
    Dibuja EER frente a WER: privacidad contra utilidad en cada caso.

    Cada caso de `tradeoff_cases` es un punto con su intervalo en x (WER) y
    en y (EER), con el color y el marcador de su escenario. Lo deseable es
    arriba a la izquierda: mucho EER (el sistema confunde a los hablantes) y
    poco WER (se entiende lo que se dice). Las metas son líneas punteadas y
    la zona donde se cumplen las dos va sombreada. El rótulo de cada
    configuración se escribe a la izquierda de su punto o-a, que suele
    quedar libre porque al bajar α los puntos avanzan hacia arriba y a la
    derecha. Sirve con una
    configuración o con varias.

    Parámetros:
        configs: lista de salidas de `eer_wer_config`.
        wer_target: meta de WER como fracción (se cumple por debajo).
        eer_target: meta de EER como fracción (se cumple por encima).

    Retorna:
        Una `Figure` de matplotlib.
    """
    cases = tradeoff_cases(configs)
    fig = Figure(figsize=(COLUMN_WIDTH, COLUMN_WIDTH + 0.35))
    ax = fig.subplots()
    x_max = 100 * max([1.3 * wer_target] + [1.3 * c["x"]["ci_high"] for c in cases])
    y_max = 100 * max([1.3 * eer_target] + [1.15 * c["y"]["ci_high"] for c in cases])
    ax.fill_between([0, 100 * wer_target], 100 * eer_target, y_max, color=GOAL_FILL, linewidth=0, zorder=0)
    ax.text(0.015, 0.985, "↖ mejor", transform=ax.transAxes, ha="left", va="top", fontsize=SMALL_FONT,
            color=MUTED_INK)
    goal_line(ax, wer_target, f"Meta WER ({percent_label(100 * wer_target)} %)")
    ax.axhline(100 * eer_target, color="0.3", linestyle=":", linewidth=0.8)
    ax.text(0.99, 100 * eer_target, f"Meta EER ({percent_label(100 * eer_target)} %)",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=SMALL_FONT, color=MUTED_INK)
    for c in cases:
        style = SCENARIO_STYLE[c["scenario"]]
        x, y = 100 * c["x"]["value"], 100 * c["y"]["value"]
        ax.plot([100 * c["x"]["ci_low"], 100 * c["x"]["ci_high"]], [y, y], color=style["color"],
                linewidth=1.4, marker="|", markersize=8)
        ax.plot([x, x], [100 * c["y"]["ci_low"], 100 * c["y"]["ci_high"]], color=style["color"],
                linewidth=1.4, marker="_", markersize=8)
        ax.plot(x, y, color=style["color"], marker=CASE_MARKERS[c["scenario"]], markersize=5, markeredgewidth=0,
                linestyle="none", zorder=3)
        if c["scenario"] == "o-a":
            ax.annotate(c["label"], (x, y), xytext=(-7, 3), textcoords="offset points", ha="right", va="bottom",
                        fontsize=SMALL_FONT, color=INK)
    percent_axis(ax, x_max, "WER (%)")
    ax.set_yticks([t for t in MaxNLocator(nbins=5, steps=[1, 2, 5, 10]).tick_values(0, y_max) if 0 <= t <= y_max])
    ax.set_ylim(0, y_max)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: percent_label(v)))
    ax.set_ylabel("EER (%)")
    ax.grid(axis="y", color="0.9", linewidth=0.5)
    handles = [Line2D([], [], color=SCENARIO_STYLE[name]["color"], marker=CASE_MARKERS[name], markersize=5,
                      linewidth=1.4, label=CASE_LABELS[name]) for name in ("o-o", "o-a", "a-a")]
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False, fontsize=SMALL_FONT)
    height = fig.get_figheight()
    fig.tight_layout(rect=(0, 0.6 / height, 1, 1))
    return fig


def read_json(path):
    """Lee un JSON en UTF-8, con o sin BOM (PowerShell 5.1 y el Bloc de notas lo agregan)."""
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def parse_args():
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description="Figuras de suplantación, WER y EER frente a WER a partir de los JSON.")
    parser.add_argument("--output-dir", required=True, help="Carpeta donde se guardan las figuras.")
    parser.add_argument("--impersonation-json", help="JSON de impersonation.py; si se da, guarda suplantacion.pdf.")
    parser.add_argument("--wer-json", help="JSON de wer_bootstrap.py; si se da, guarda wer.pdf.")
    parser.add_argument("--config", nargs=3, action="append", default=[],
                        metavar=("LABEL", "EER_JSON", "WER_JSON"),
                        help="Una configuración de α para compromiso.pdf (se puede repetir).")
    return parser.parse_args()


def load_configs(config_args):
    """
    Lee y valida los JSON de cada --config antes de dibujar nada.

    Si un JSON no es del tipo esperado, termina con un mensaje corto (sin
    traceback) que dice qué archivo se recibió y cómo debe ir el comando.

    Parámetros:
        config_args: lista de [LABEL, EER_JSON, WER_JSON] de argparse.

    Retorna:
        Lista de salidas de `eer_wer_config`.
    """
    configs = []
    for label, eer_path, wer_path in config_args:
        try:
            configs.append(eer_wer_config(label, read_json(eer_path), read_json(wer_path)))
        except ValueError as error:
            raise SystemExit(
                f"Error en --config: {error}\n"
                f"  Se recibió: EER_JSON = {eer_path}, WER_JSON = {wer_path}\n"
                "  Forma correcta: --config ETIQUETA EER_JSON WER_JSON, por ejemplo:\n"
                '  --config "U(0,7; 0,9)" ..\\Datos\\eer_bootstrap_k134.json ..\\Datos\\wer_turbo_k134.json'
            ) from None
    return configs


def main():
    """Guarda solo las figuras cuyos JSON se dieron."""
    args = parse_args()
    configs = load_configs(args.config)
    configure_fonts()
    output_dir = Path(args.output_dir)
    saved = []
    figures = []
    if args.impersonation_json:
        figures.append(("suplantacion.pdf", lambda: plot_impersonation(
            read_json(args.impersonation_json)["operating_points"])))
    if args.wer_json:
        figures.append(("wer.pdf", lambda: plot_wer(read_json(args.wer_json)["wer"])))
    if configs:
        figures.append(("compromiso.pdf", lambda: plot_eer_wer(configs)))
    if not figures:
        print("No se dio ningún JSON: no se generó ninguna figura.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, build in figures:
        build().savefig(output_dir / name)
        saved.append(name)
    print(f"Figuras guardadas en {output_dir}: {', '.join(saved)}")


if __name__ == "__main__":
    main()
