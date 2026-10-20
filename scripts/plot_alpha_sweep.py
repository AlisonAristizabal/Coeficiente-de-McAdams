"""
Resumen y figura del barrido de α a partir de los JSON de cada configuración.

No recalcula nada: lee, para cada α, el JSON del bootstrap del EER, el de
suplantación y el del WER (los nombres salen de `run_alpha_sweep.config_paths`)
y escribe:

- una tabla TSV con una fila por configuración: EER o-o, o-a y a-a,
  suplantación en los dos umbrales y WER original y anonimizado, cada uno
  con su intervalo (como fracciones, con punto decimal, para leerla con
  cualquier programa; vacío si falta ese JSON);
- una figura de tres paneles con el eje α compartido: EER o-a, suplantación
  con FAR ≤ 1 % y WER anonimizado, con sus intervalos, la referencia sin
  anonimizar y las metas. El a-a va en la tabla, no en la figura.

Si falta algún JSON (por ejemplo, el WER todavía no se ha corrido), esa
columna queda vacía y ese punto no se dibuja.
"""
import argparse
import csv
import json
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

from plot_eer import COLUMN_WIDTH, GOAL_EER, INK, MUTED_INK, SMALL_FONT, configure_fonts, percent_label
from plot_results import ANONYMIZED_COLOR, ORIGINAL_COLOR, WER_TARGET
from run_alpha_sweep import DEFAULT_ALPHAS, REPO, build_configs, config_paths

# (prefijo de columna, de dónde sale cada valor en los JSON).
METRICS = (
    ("eer_oo", "eer", ("scenarios", "o-o")),
    ("eer_oa", "eer", ("scenarios", "o-a")),
    ("eer_aa", "eer", ("scenarios", "a-a")),
    ("impersonation_eer", "impersonation", ("operating_points", "eer", "impersonation_rate")),
    ("impersonation_far", "impersonation", ("operating_points", "far", "impersonation_rate")),
    ("wer_original", "wer", ("wer", "original")),
    ("wer_anonymized", "wer", ("wer", "anonymized")),
)
SUMMARY_COLUMNS = ["tag", "alpha_min", "alpha_max"] + [
    f"{prefix}{suffix}" for prefix, _, _ in METRICS for suffix in ("", "_ci_low", "_ci_high")]


def read_interval(summary, keys):
    """
    Saca (valor, ci_low, ci_high) de un JSON ya leído.

    Acepta los dos formatos que hay: {"eer", "ci_low", "ci_high"} (EER) y
    {"value", "ci_low", "ci_high"} (suplantación y WER).
    """
    node = summary
    for key in keys:
        node = node[key]
    return (node["eer"] if "eer" in node else node["value"]), node["ci_low"], node["ci_high"]


def collect_rows(configs, settings):
    """
    Arma una fila por configuración con todas las métricas y sus intervalos.

    Parámetros:
        configs: salida de `run_alpha_sweep.build_configs`.
        settings: objeto con `anon_root`, `output_dir`, `dataset_tag` y
            `whisper_model` (para las rutas de `config_paths`).

    Retorna:
        Lista de diccionarios con las columnas de `SUMMARY_COLUMNS`; las
        métricas cuyo JSON no existe quedan en None.
    """
    rows = []
    for config in configs:
        paths = config_paths(config["tag"], settings)
        loaded = {}
        for source in ("eer", "impersonation", "wer"):
            if Path(paths[source]).exists():
                with open(paths[source], encoding="utf-8-sig") as f:
                    loaded[source] = json.load(f)
        row = {"tag": config["tag"], "alpha_min": config["alpha_min"], "alpha_max": config["alpha_max"]}
        for prefix, source, keys in METRICS:
            values = read_interval(loaded[source], keys) if source in loaded else (None, None, None)
            row.update(zip((prefix, f"{prefix}_ci_low", f"{prefix}_ci_high"), values))
        rows.append(row)
    return rows


def write_summary(rows, output_path):
    """Escribe la tabla en TSV (UTF-8, punto decimal, celdas vacías para lo que falta)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(SUMMARY_COLUMNS)
        for row in rows:
            # csv.writer escribe None como celda vacía.
            writer.writerow([row[c] for c in SUMMARY_COLUMNS])
    return output_path


def _series(rows, prefix):
    """Puntos (α, valor, ci_low, ci_high) en porcentaje de las filas con α fijo y valor disponible."""
    return [(r["alpha_min"], 100 * r[prefix], 100 * r[f"{prefix}_ci_low"], 100 * r[f"{prefix}_ci_high"])
            for r in rows if r["alpha_min"] == r["alpha_max"] and r[prefix] is not None]


def _reference(rows, prefix):
    """Primer valor disponible de una métrica que no depende de α (o-o, WER original), en porcentaje."""
    for r in rows:
        if r[prefix] is not None:
            return 100 * r[prefix], 100 * r[f"{prefix}_ci_low"], 100 * r[f"{prefix}_ci_high"]
    return None


def _draw_panel(ax, points, reference=None, target=None, target_label=None):
    """Dibuja una serie con intervalos verticales, su referencia horizontal y la meta."""
    if reference is not None:
        value, low, high = reference
        ax.axhspan(low, high, color=ORIGINAL_COLOR, alpha=0.12, linewidth=0)
        ax.axhline(value, color=ORIGINAL_COLOR, linestyle="--", linewidth=1.0)
    if target is not None:
        ax.axhline(100 * target, color="0.3", linestyle=":", linewidth=0.8)
        ax.text(1.0, 100 * target, target_label, transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=SMALL_FONT, color=MUTED_INK)
    if points:
        xs = [p[0] for p in points]
        ax.plot(xs, [p[1] for p in points], color=ANONYMIZED_COLOR, linewidth=0.8, alpha=0.6)
        for x, value, low, high in points:
            ax.plot([x, x], [low, high], color=ANONYMIZED_COLOR, linewidth=1.4, marker="_", markersize=7)
        ax.plot(xs, [p[1] for p in points], color=ANONYMIZED_COLOR, marker="o", markersize=4, markeredgewidth=0,
                linestyle="none", zorder=3)
    tops = [p[3] for p in points] + ([reference[2]] if reference else []) + ([130 * target] if target else [])
    y_max = max(tops + [1.0]) * 1.1
    ax.set_ylim(0, y_max)
    ax.set_yticks([t for t in MaxNLocator(nbins=4, steps=[1, 2, 5, 10]).tick_values(0, y_max) if 0 <= t <= y_max])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: percent_label(v)))
    ax.grid(axis="y", color="0.9", linewidth=0.5)


def plot_alpha_sweep(rows, eer_target=GOAL_EER, wer_target=WER_TARGET):
    """
    Figura del barrido: tres paneles con el eje α compartido.

    (1) EER o-a con su intervalo, el EER o-o como referencia (línea
    discontinua y franja de su intervalo) y la meta; (2) tasa de
    suplantación con FAR ≤ 1 % con su intervalo; (3) WER anonimizado con
    su intervalo, el WER original como referencia y la meta. Solo entran las
    configuraciones de α fijo. El a-a no se dibuja.

    Parámetros:
        rows: salida de `collect_rows`.
        eer_target: meta de EER (fracción).
        wer_target: meta de WER (fracción).

    Retorna:
        Una `Figure` de matplotlib.
    """
    fig = Figure(figsize=(COLUMN_WIDTH, 4.6))
    axes = fig.subplots(3, 1, sharex=True)
    _draw_panel(axes[0], _series(rows, "eer_oa"), _reference(rows, "eer_oo"), eer_target,
                f"Meta ({percent_label(100 * eer_target)} %)")
    axes[0].set_ylabel("EER o-a (%)")
    _draw_panel(axes[1], _series(rows, "impersonation_far"))
    axes[1].set_ylabel("Suplantación\nFAR ≤ 1 % (%)")
    _draw_panel(axes[2], _series(rows, "wer_anonymized"), _reference(rows, "wer_original"), wer_target,
                f"Meta ({percent_label(100 * wer_target)} %)")
    axes[2].set_ylabel("WER (%)")

    alphas = sorted({r["alpha_min"] for r in rows if r["alpha_min"] == r["alpha_max"]})
    if alphas:
        pad = 0.05 * max(alphas[-1] - alphas[0], 0.2)
        axes[2].set_xlim(alphas[0] - pad, alphas[-1] + pad)
        axes[2].set_xticks(alphas)
    axes[2].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    axes[2].set_xlabel("α (coeficiente de McAdams)")
    handles = [
        Line2D([], [], color=ANONYMIZED_COLOR, marker="o", markersize=4, linewidth=1.4, label="Anonimizado (o-a)"),
        Line2D([], [], color=ORIGINAL_COLOR, linestyle="--", linewidth=1.0, label="Sin anonimizar (o-o)"),
        Line2D([], [], color="0.3", linestyle=":", linewidth=0.8, label="Meta"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=SMALL_FONT,
               handlelength=1.6, columnspacing=1.0, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.3 / fig.get_figheight(), 1, 1))
    return fig


def parse_args(argv=None):
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description="Tabla resumen y figura del barrido de α.")
    parser.add_argument("--alphas", type=float, nargs="+", default=None, help="α fijos (por defecto los 9 del barrido).")
    parser.add_argument("--range", type=float, nargs=2, action="append", default=[], metavar=("MIN", "MAX"),
                        dest="ranges", help="Rangos U(MIN, MAX) para la tabla (no van en la figura).")
    parser.add_argument("--data-dir", default=str(REPO.parent / "Datos"))
    parser.add_argument("--anon-root", default=None, help="Por defecto <data-dir>\\anon_k134.")
    parser.add_argument("--output-dir", default=None, help="Carpeta de los JSON (por defecto <data-dir>).")
    parser.add_argument("--dataset-tag", default="k134")
    parser.add_argument("--whisper-model", default="turbo")
    parser.add_argument("--summary", default=None, help="Por defecto <data-dir>\\barrido_alfa_resumen.tsv.")
    parser.add_argument("--figure", default=None, help="Por defecto <data-dir>\\figuras\\barrido_alfa.pdf.")
    args = parser.parse_args(argv)
    data = Path(args.data_dir)
    args.anon_root = args.anon_root or str(data / "anon_k134")
    args.output_dir = args.output_dir or str(data)
    args.summary = args.summary or str(data / "barrido_alfa_resumen.tsv")
    args.figure = args.figure or str(data / "figuras" / "barrido_alfa.pdf")
    if args.alphas is None and not args.ranges:
        args.alphas = list(DEFAULT_ALPHAS)
    return args


def main():
    """Escribe la tabla resumen y la figura del barrido."""
    args = parse_args()
    configs = build_configs(args.alphas, args.ranges)
    rows = collect_rows(configs, args)
    write_summary(rows, args.summary)
    missing = [r["tag"] for r in rows if any(r[p] is None for p, _, _ in METRICS)]
    configure_fonts()
    Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    plot_alpha_sweep(rows).savefig(args.figure)
    print(f"Tabla: {args.summary}")
    print(f"Figura: {args.figure}")
    if missing:
        print(f"Configuraciones con algún JSON faltante (quedan vacías): {', '.join(missing)}")


if __name__ == "__main__":
    main()
