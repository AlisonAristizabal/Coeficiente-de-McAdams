"""
Figuras del EER para el artículo: curvas DET y distribuciones de puntuaciones.

Vuelve a calcular las puntuaciones con `run_scenarios` (no lee archivos
intermedios), de modo que las figuras siempre corresponden a los embeddings
y al manifiesto que se le pasan. Las figuras se guardan en PDF vectorial, al
ancho de una columna y sin título (el título va en el \\caption de LaTeX).
"""
import argparse
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
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


def plot_det(curves, eers):
    """
    Dibuja las curvas DET de varios escenarios en una figura.

    Cada curva lleva su punto de EER, que cae sobre la diagonal FAR = FRR.
    Los puntos con FAR o FRR igual a 0 o 1 no se dibujan, porque en escala
    probit quedan en el infinito.

    Parámetros:
        curves: diccionario {escenario: (far, frr)}, de `det_curve`.
        eers: diccionario {escenario: EER como fracción}.

    Retorna:
        Una `Figure` de matplotlib.
    """
    fig = Figure(figsize=(COLUMN_WIDTH, COLUMN_WIDTH))
    ax = fig.subplots()
    low, high = probit(np.array(DET_LIMITS) / 100)
    for name, (far, frr) in curves.items():
        keep = (far > 0) & (far < 1) & (frr > 0) & (frr < 1)
        label = f"{name} (EER {decimal_comma(100 * eers[name], 2)} %)"
        ax.plot(probit(far[keep]), probit(frr[keep]), label=label, linewidth=1.2, **SCENARIO_STYLE[name])
        ax.plot(probit(eers[name]), probit(eers[name]), marker="o", markersize=4,
                color=SCENARIO_STYLE[name]["color"], linestyle="none")
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
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return fig


def plot_score_distributions(scores_by_scenario, thresholds, bins=60):
    """
    Dibuja, por escenario, el histograma de puntuaciones objetivo e impostor.

    Un panel por escenario, con el mismo eje x y los mismos intervalos en
    todos para que se puedan comparar. Los histogramas están normalizados
    (área 1) porque hay más de mil impostores por cada par objetivo. La
    línea vertical marca el umbral del EER de ese escenario.

    Parámetros:
        scores_by_scenario: diccionario {escenario: (target, impostor)}.
        thresholds: diccionario {escenario: umbral del EER}.
        bins: número de intervalos del histograma.

    Retorna:
        Una `Figure` de matplotlib.
    """
    names = list(scores_by_scenario)
    fig = Figure(figsize=(COLUMN_WIDTH, 1.1 * len(names) + 0.4))
    axes = fig.subplots(len(names), 1, sharex=True, squeeze=False)[:, 0]
    all_scores = np.concatenate([np.concatenate(pair) for pair in scores_by_scenario.values()])
    edges = np.linspace(all_scores.min(), all_scores.max(), bins + 1)
    for ax, name in zip(axes, names):
        target, impostor = scores_by_scenario[name]
        ax.hist(impostor, bins=edges, density=True, color=IMPOSTOR_COLOR, label="Impostor")
        ax.hist(target, bins=edges, density=True, histtype="step", color=TARGET_COLOR, linewidth=1.2, label="Objetivo")
        ax.axvline(thresholds[name], color="black", linestyle="--", linewidth=0.8)
        ax.set_ylabel(name)
        ax.set_yticks([])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: decimal_comma(x, 1)))
    axes[0].legend(frameon=False, loc="upper left")
    axes[-1].set_xlabel("Similitud coseno")
    fig.tight_layout()
    return fig


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
    return parser.parse_args()


def main():
    """Calcula los escenarios y guarda det.pdf y distribuciones.pdf."""
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
    print(f"Figuras guardadas en: {output_dir}")


if __name__ == "__main__":
    main()