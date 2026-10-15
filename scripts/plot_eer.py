"""
Figuras del EER para el artículo: curvas DET, distribuciones de puntuaciones
y, opcionalmente, el EER con su intervalo de confianza.

Vuelve a calcular las puntuaciones con `run_scenarios` (no lee archivos
intermedios), de modo que las figuras siempre corresponden a los embeddings
y al manifiesto que se le pasan. La única excepción es la figura del
intervalo de confianza, que lee el JSON de `eer_bootstrap` para no repetir
las réplicas. Las figuras se guardan en PDF vectorial, al ancho de una
columna y sin título (el título va en el \\caption de LaTeX).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.stats import norm

from eer_scoring import split_target_impostor
from run_eer import load_and_score

# Ancho de una columna del artículo (pulgadas) y tamaño de letra (pt).
COLUMN_WIDTH = 3.35
FONT_SIZE = 8

# Cada escenario se distingue por color (paleta validada para daltonismo) y
# también por tipo de línea, para que se lea impreso en blanco y negro.
SCENARIO_STYLE = {
    "o-o": {"color": "#0072B2", "linestyle": "-"},
    "o-a": {"color": "#D55E00", "linestyle": "--"},
    "a-a": {"color": "#009E73", "linestyle": ":"},
}

TARGET_COLOR = "#0072B2"
IMPOSTOR_COLOR = "#BBBBBB"
# Impostores aceptados en el umbral del EER: un gris más oscuro que el del
# histograma de impostores, para que se lea como "la parte que se cuela".
FALSE_ACCEPT_COLOR = "#6E6E6E"

# Qué compara cada escenario, para que la figura se entienda sin conocer los
# códigos (registro primero, prueba después, como en `run_eer.SCENARIOS`).
SCENARIO_LABEL = {
    "o-o": "Registro original, prueba original",
    "o-a": "Registro original, prueba anonimizada",
    "a-a": "Registro anonimizado, prueba anonimizada",
}

# Meta de EER de la anonimización, como fracción.
GOAL_EER = 0.30
GOAL_FILL = "#EBEBEB"

# El texto va en tinta neutra, nunca con el color del escenario; el color lo
# lleva la marca de al lado.
INK = "#222222"
MUTED_INK = "#666666"
SMALL_FONT = 6.5
# Separación (pt) entre el extremo de una barra de IC y su número.
CI_LABEL_GAP = 3

# Marcas y límites de los ejes DET, en porcentaje.
DET_TICKS = [0.1, 0.5, 1, 2, 5, 10, 20, 40]
DET_LIMITS = (0.05, 60)


def decimal_comma(value, decimals=1):
    """Formatea un número con `decimals` decimales y coma decimal."""
    return f"{value:.{decimals}f}".replace(".", ",")


def det_curve(target, impostor, n_quantiles=2000):
    """
    Calcula los puntos (FAR, FRR) de la curva DET.

    Usa la misma convención que `compute_eer`: se acepta un par si su
    puntuación es mayor o igual que el umbral. Como umbrales se toman todas
    las puntuaciones objetivo (donde cambia el FRR) y `n_quantiles + 1`
    cuantiles de las de impostor, en vez de los millones de puntuaciones de
    impostor: la curva queda igual a la vista y el PDF pesa poco.

    Retorna:
        Una tupla (far, frr) de arreglos 1D del mismo largo, como
        fracciones entre 0 y 1, ordenados por umbral creciente.
    """
    target = np.sort(np.asarray(target, dtype=np.float64))
    impostor = np.sort(np.asarray(impostor, dtype=np.float64))
    quantiles = np.quantile(impostor, np.linspace(0, 1, n_quantiles + 1))
    thresholds = np.unique(np.concatenate([target, quantiles]))
    # frr: fracción de objetivos rechazados (puntuación < umbral).
    # far: fracción de impostores aceptados (puntuación >= umbral).
    frr = np.searchsorted(target, thresholds, side="left") / target.size
    far = (impostor.size - np.searchsorted(impostor, thresholds, side="left")) / impostor.size
    return far, frr


def probit(p):
    """Transforma probabilidades a la escala de desviación normal de la DET."""
    return norm.ppf(p)


def plot_det(curves, eers, goal=GOAL_EER):
    """
    Dibuja las curvas DET de varios escenarios en una figura.

    Cada curva lleva su punto de EER, que cae sobre la diagonal FAR = FRR.
    El cuadrado sombreado de la esquina superior derecha es la zona de la
    meta: una curva lo atraviesa si y solo si su EER es mayor o igual que
    `goal`. Un rótulo en la esquina opuesta recuerda que hacia allá el
    hablante es más fácil de reidentificar, porque en anonimización un EER
    alto es lo deseable (al revés que en verificación). La leyenda va debajo
    del gráfico para no tapar las curvas. Los puntos con FAR o FRR igual a 0
    o 1 no se dibujan, porque en escala probit quedan en el infinito.

    Parámetros:
        curves: diccionario {escenario: (far, frr)}, de `det_curve`.
        eers: diccionario {escenario: EER como fracción}.
        goal: meta de EER como fracción.

    Retorna:
        Una `Figure` de matplotlib.
    """
    fig = Figure(figsize=(COLUMN_WIDTH, COLUMN_WIDTH + 0.6))
    ax = fig.subplots()
    low, high = probit(np.array(DET_LIMITS) / 100)
    goal_x = probit(goal)
    ax.fill_between([goal_x, high], goal_x, high, color=GOAL_FILL, linewidth=0, zorder=0)
    ax.text(high - 0.08, high - 0.08, f"Meta\nEER ≥ {percent_label(100 * goal)} %", ha="right", va="top",
            color=MUTED_INK, fontsize=SMALL_FONT)
    ax.text(low + 0.08, low + 0.08, "↙ más fácil reidentificar", ha="left", va="bottom",
            color=MUTED_INK, fontsize=SMALL_FONT)
    for name, (far, frr) in curves.items():
        keep = (far > 0) & (far < 1) & (frr > 0) & (frr < 1)
        label = f"{name} (EER {decimal_comma(100 * eers[name], 2)} %)"
        ax.plot(probit(far[keep]), probit(frr[keep]), label=label, linewidth=1.4, **SCENARIO_STYLE[name])
        ax.plot(probit(eers[name]), probit(eers[name]), marker="o", markersize=5, markeredgecolor="white",
                markeredgewidth=0.8, color=SCENARIO_STYLE[name]["color"], linestyle="none", zorder=3)
    ax.plot([low, high], [low, high], color="0.6", linewidth=0.6)
    ticks = probit(np.array(DET_TICKS) / 100)
    labels = [decimal_comma(t, 1) if t < 1 else f"{t:g}" for t in DET_TICKS]
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Tasa de falsa aceptación (%)")
    ax.set_ylabel("Tasa de falso rechazo (%)")
    ax.grid(color="0.9", linewidth=0.5)
    ax.legend(title="Escenario (registro-prueba)", frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.17), fontsize=SMALL_FONT, title_fontsize=SMALL_FONT)
    fig.tight_layout()
    return fig


def plot_score_distributions(scores_by_scenario, thresholds, bins=60):
    """
    Dibuja, por escenario, el histograma de puntuaciones objetivo e impostor.

    Un panel por escenario, con el mismo eje x y los mismos intervalos en
    todos para que se puedan comparar. Los histogramas están normalizados
    (área 1) porque hay más de mil impostores por cada par objetivo. La
    línea vertical marca el umbral del EER de ese escenario y las áreas
    sombreadas muestran de qué está hecho el EER en ese umbral: los
    impostores que lo superan (falsa aceptación) y los objetivos que quedan
    por debajo (falso rechazo). Cada panel lleva arriba lo que compara y su
    EER, que es el promedio de esas dos tasas.

    Parámetros:
        scores_by_scenario: diccionario {escenario: (target, impostor)}.
        thresholds: diccionario {escenario: umbral del EER}.
        bins: número de intervalos del histograma.

    Retorna:
        Una `Figure` de matplotlib.
    """
    names = list(scores_by_scenario)
    height = 1.25 * len(names) + 0.8
    fig = Figure(figsize=(COLUMN_WIDTH, height))
    axes = fig.subplots(len(names), 1, sharex=True, squeeze=False)[:, 0]
    all_scores = np.concatenate([np.concatenate(pair) for pair in scores_by_scenario.values()])
    edges = np.linspace(all_scores.min(), all_scores.max(), bins + 1)
    for ax, name in zip(axes, names):
        target, impostor = (np.asarray(s, dtype=np.float64) for s in scores_by_scenario[name])
        threshold = thresholds[name]
        ax.hist(impostor, bins=edges, density=True, color=IMPOSTOR_COLOR, label="Impostor")
        shade_tail(ax, impostor, edges, threshold, above=True, color=FALSE_ACCEPT_COLOR)
        shade_tail(ax, target, edges, threshold, above=False, color=TARGET_COLOR, alpha=0.3)
        ax.hist(target, bins=edges, density=True, histtype="step", color=TARGET_COLOR, linewidth=1.2, label="Objetivo")
        ax.axvline(threshold, color=INK, linestyle="--", linewidth=0.8)
        # Misma convención que compute_eer: se acepta si la puntuación >= umbral.
        eer = (np.mean(impostor >= threshold) + np.mean(target < threshold)) / 2
        ax.set_title(SCENARIO_LABEL.get(name, ""), loc="left", fontsize=SMALL_FONT, color=MUTED_INK)
        ax.set_title(f"EER {decimal_comma(100 * eer, 2)} %", loc="right", fontsize=SMALL_FONT, color=INK)
        ax.set_ylabel(name)
        ax.set_yticks([])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: decimal_comma(x, 1)))
    axes[0].legend(frameon=False, loc="upper left", fontsize=SMALL_FONT)
    axes[-1].set_xlabel("Similitud coseno")
    error_handles = [
        Patch(color=FALSE_ACCEPT_COLOR, label="Falsa aceptación"),
        Patch(color=TARGET_COLOR, alpha=0.3, label="Falso rechazo"),
        Line2D([], [], color=INK, linestyle="--", linewidth=0.8, label="Umbral del EER"),
    ]
    fig.legend(handles=error_handles, loc="lower center", ncol=3, frameon=False, fontsize=SMALL_FONT,
               handlelength=1.5, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.3 / height, 1, 1))
    return fig


def shade_tail(ax, values, edges, threshold, above, **style):
    """
    Sombrea el histograma de `values` de un lado del umbral.

    Usa los mismos intervalos (`edges`) y la misma normalización que el
    histograma, de modo que el área sombreada encaja dentro de sus barras,
    y corta exactamente en `threshold`.

    Parámetros:
        ax: eje donde se dibuja.
        values: puntuaciones (arreglo 1D).
        edges: bordes de los intervalos del histograma.
        threshold: umbral donde empieza o termina el sombreado.
        above: True sombrea desde el umbral hacia la derecha; False, desde
            la izquierda hasta el umbral.
        style: argumentos de estilo para `fill_between`.
    """
    heights, _ = np.histogram(values, bins=edges, density=True)
    if above:
        xs = np.concatenate([[threshold], edges[edges > threshold]])
    else:
        xs = np.concatenate([edges[edges < threshold], [threshold]])
    bins_of = np.clip(np.searchsorted(edges, xs, side="right") - 1, 0, heights.size - 1)
    ax.fill_between(xs, heights[bins_of], step="post", linewidth=0, **style)


def plot_eer_ci(scenarios, target=GOAL_EER):
    """
    Dibuja el EER de cada escenario con su intervalo de confianza.

    Un renglón por escenario, en el orden de `SCENARIO_STYLE` (o-o arriba,
    a-a abajo): un punto en el EER y una barra horizontal de `ci_low` a
    `ci_high`. La barra se dibuja con sus extremos tal cual y no como
    error simétrico alrededor del punto, porque el intervalo por percentiles
    puede ser asimétrico e incluso no contener el EER de todos los datos.
    Una línea vertical punteada marca la meta y la zona sombreada a su
    derecha es donde el EER la alcanza. Cada barra lleva sus números: el
    límite inferior a la izquierda de su extremo, el superior a la derecha
    del suyo y el EER encima del punto. Si el límite inferior no cabe a la
    izquierda (EER cerca de 0), el eje empieza un poco antes de 0 para darle
    espacio, sin marcas negativas.

    Parámetros:
        scenarios: diccionario {escenario: {"eer", "ci_low", "ci_high"}} como
            fracciones, por ejemplo la llave "scenarios" del JSON de
            `eer_bootstrap`.
        target: meta de EER como fracción.

    Retorna:
        Una `Figure` de matplotlib.
    """
    names = [name for name in SCENARIO_STYLE if name in scenarios]
    fig = Figure(figsize=(COLUMN_WIDTH, 0.45 * len(names) + 0.8))
    ax = fig.subplots()
    rows = range(len(names) - 1, -1, -1)
    # El eje llega al menos a 1,3 veces la meta para que su zona se vea.
    x_max = 100 * max([1.3 * target] + [1.12 * scenarios[name]["ci_high"] for name in names])
    ax.axvspan(100 * target, x_max, color=GOAL_FILL, linewidth=0, zorder=0)
    ax.axvline(100 * target, color="0.3", linestyle=":", linewidth=0.8, gid="target")
    ax.text(100 * target, 1.01, f"Meta ({percent_label(100 * target)} %)", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom")
    number = {"fontsize": SMALL_FONT, "textcoords": "offset points"}
    low_labels = []
    for y, name in zip(rows, names):
        s, color = scenarios[name], SCENARIO_STYLE[name]["color"]
        low, high, eer = 100 * s["ci_low"], 100 * s["ci_high"], 100 * s["eer"]
        ax.plot([low, high], [y, y], color=color, linewidth=1.6, marker="|", markersize=10, gid=f"ci {name}")
        # El punto es más chico que las tapas para no tapar un intervalo angosto.
        ax.plot(eer, y, color=color, marker="o", markersize=4, markeredgewidth=0,
                linestyle="none", zorder=3, gid=f"eer {name}")
        low_labels.append(ax.annotate(decimal_comma(low, 2), (low, y), xytext=(-CI_LABEL_GAP, 0), ha="right",
                                      va="center", color=MUTED_INK, **number))
        ax.annotate(decimal_comma(high, 2), (high, y), xytext=(CI_LABEL_GAP, 0), ha="left", va="center",
                    color=MUTED_INK, **number)
        ax.annotate(f"{decimal_comma(eer, 2)} %", (eer, y), xytext=(0, 5), ha="center", va="bottom",
                    color=INK, fontweight="bold", **number)
    ax.text(0.012, 0.03, "más privacidad →", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=SMALL_FONT, color=MUTED_INK)
    ax.set_yticks(list(rows), names)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xticks([t for t in MaxNLocator(nbins=5, steps=[1, 2, 5, 10]).tick_values(0, x_max) if 0 <= t <= x_max])
    ax.set_xlim(0, x_max)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: percent_label(x)))
    ax.set_xlabel("EER (%)")
    ax.grid(axis="x", color="0.9", linewidth=0.5)
    fig.tight_layout()
    ax.set_xlim(left_limit(ax, low_labels, x_max), x_max)
    fig.tight_layout()
    return fig


def left_limit(ax, labels, x_max, pad=3):
    """
    Calcula el inicio del eje x para que las etiquetas no se salgan por la izquierda.

    Cada etiqueta termina `CI_LABEL_GAP` puntos a la izquierda de su ancla
    (el límite inferior de una barra). Si con el eje desde 0 alguna no cabe
    entre el eje y su ancla, se baja el inicio lo justo para que quepa.

    Parámetros:
        ax: eje ya acomodado con `tight_layout`.
        labels: anotaciones alineadas a la derecha de su ancla.
        x_max: final del eje x, que no cambia.
        pad: margen extra en píxeles.

    Retorna:
        El inicio del eje x: 0 si todas caben, un valor negativo si no.
    """
    width = ax.get_window_extent().width
    gap = CI_LABEL_GAP * ax.figure.dpi / 72
    start = 0.0
    for label in labels:
        needed = label.get_window_extent().width + gap + pad
        anchor = label.xy[0]
        # (anchor - start) / (x_max - start) * width >= needed, despejado para start.
        start = min(start, (anchor * width - needed * x_max) / (width - needed))
    return start


def percent_label(value):
    """Formatea un porcentaje sin ceros sobrantes y con coma decimal (12.5 → "12,5")."""
    return f"{value:g}".replace(".", ",")


def configure_fonts():
    """Fija el tamaño de letra y guarda las fuentes como TrueType en el PDF."""
    import matplotlib
    matplotlib.rcParams.update({"font.size": FONT_SIZE, "pdf.fonttype": 42})


def parse_args():
    """Lee los argumentos de la línea de comandos (valores por defecto de D2)."""
    parser = argparse.ArgumentParser(description="Genera las figuras DET y de distribución de puntuaciones.")
    parser.add_argument("--embeddings", required=True, help="Archivo .npz generado por extract_embeddings.")
    parser.add_argument("--manifest", required=True, help="Manifiesto TSV generado por anonymize_subset.")
    parser.add_argument("--output-dir", required=True, help="Carpeta donde se guardan las figuras.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de la partición registro/prueba.")
    parser.add_argument("--n-enroll", type=int, default=3, help="Clips de registro por hablante.")
    parser.add_argument("--n-trial", type=int, default=2, help="Clips de prueba por hablante.")
    parser.add_argument("--bootstrap-json", help="JSON de eer_bootstrap; si se da, también se guarda eer_ic.pdf.")
    return parser.parse_args()


def main():
    """
    Calcula los escenarios y guarda det.pdf y distribuciones.pdf.

    Si se pasa --bootstrap-json, guarda además eer_ic.pdf con el EER y el
    intervalo de confianza de la llave "scenarios" de ese JSON.
    """
    args = parse_args()
    configure_fonts()
    _, _, results = load_and_score(args.embeddings, args.manifest, args.seed, args.n_enroll, args.n_trial)

    scores_by_scenario = {name: split_target_impostor(r["scores"], r["owner"]) for name, r in results.items()}
    curves = {name: det_curve(*pair) for name, pair in scores_by_scenario.items()}
    eers = {name: r["eer"] for name, r in results.items()}
    thresholds = {name: r["threshold"] for name, r in results.items()}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_det(curves, eers).savefig(output_dir / "det.pdf")
    plot_score_distributions(scores_by_scenario, thresholds).savefig(output_dir / "distribuciones.pdf")
    if args.bootstrap_json:
        with open(args.bootstrap_json, encoding="utf-8") as f:
            ci_scenarios = json.load(f)["scenarios"]
        plot_eer_ci(ci_scenarios).savefig(output_dir / "eer_ic.pdf")
    print(f"Figuras guardadas en: {output_dir}")


if __name__ == "__main__":
    main()