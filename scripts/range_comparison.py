"""
Comparación de las 9 configuraciones de rango uniforme (3 puntos medios × 3 anchos).

Usa solo resultados existentes (manifiestos, embeddings, transcripciones,
JSON y réplicas del EER): no anonimiza, no extrae embeddings ni transcribe.
U(0,7; 0,9) es la referencia.

Salidas:
- rangos_resumen.tsv: una fila por configuración con su punto medio y ancho
  nominales (de la etiqueta), los α observados en el manifiesto, el EER o-o,
  o-a y a-a, y el WER original, anonimizado y su diferencia, con los IC de
  los JSON.
- rangos_diferencias.tsv: diferencias pareadas (destino − origen) del EER
  o-a, el EER a-a y el WER anonimizado, con IC por bootstrap. Las réplicas
  del EER son las de eer_bootstrap.py (*_replicas.npz), después de comprobar
  sus parámetros y recalcular la primera y la última; las del WER se
  calculan con las mismas muestras de hablantes. Sin corrección por
  comparaciones múltiples.
- rangos_alucinaciones.tsv: diagnóstico de clips cuya hipótesis anonimizada
  tiene más del doble de palabras que la referencia (no cambia el WER).
- Figuras rangos_compromiso.pdf y rangos_ancho.pdf.
"""
import argparse
import csv
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

from alpha_differences import eer_replicates, missing_outputs, paired_difference
from common_voice_corpus import read_tsv_rows
from group_analysis import aligned_wer_counts, discover_configs, is_fixed, parse_tag, shared_samples, wer_by_group
from plot_eer import COLUMN_WIDTH, INK, MUTED_INK, SMALL_FONT, configure_fonts, percent_label
from plot_results import WER_TARGET
from run_alpha_sweep import REPO
from run_eer import load_and_score
from transcribe import normalize_final
from wer_bootstrap import row_counts

RANGE_TAGS = ("u0635_0815", "u065_08", "u0665_0785", "u066_084", "u0675_0825", "u069_081", "u0685_0865",
              "u07_085", "u0715_0835")
REFERENCE_TAG = "u07_09"
MIDPOINTS = (0.725, 0.75, 0.775)
WIDTHS = (0.12, 0.15, 0.18)
METRICS = ("eer_oa", "eer_aa", "wer_anonymized")
FIXED_WINDOW = (0.65, 0.9)
HALLUCINATION_RATIO = 2
TOP_CLIPS = 10

# Color por punto medio (Okabe-Ito) y marcador por ancho.
MIDPOINT_COLORS = {0.725: "#0072B2", 0.75: "#D55E00", 0.775: "#009E73"}
WIDTH_MARKERS = {0.12: "^", 0.15: "s", 0.18: "o"}
FIXED_COLOR = "#8C8C8C"
REFERENCE_COLOR = "#000000"


# ---------------------------------------------------------------------------
# Diseño
# ---------------------------------------------------------------------------

def nominal_design(tag):
    """Punto medio y ancho nominales de un rango, desde su etiqueta, redondeados a 3 decimales."""
    low, high = parse_tag(tag)
    return round((low + high) / 2, 3), round(high - low, 3)


def check_grid(configs, midpoints=MIDPOINTS, widths=WIDTHS):
    """
    Verifica que las configuraciones formen la cuadrícula completa punto medio × ancho.

    Retorna:
        Diccionario {(punto medio, ancho): configuración}. Lanza `ValueError`
        con las celdas que faltan, sobran o se repiten.
    """
    grid, repeated, outside = {}, [], []
    for config in configs:
        cell = nominal_design(config["tag"])
        if cell not in set(product(midpoints, widths)):
            outside.append(f"{config['tag']} {cell}")
        elif cell in grid:
            repeated.append(f"{grid[cell]['tag']} y {config['tag']} {cell}")
        else:
            grid[cell] = config
    missing = [cell for cell in product(midpoints, widths) if cell not in grid]
    if missing or repeated or outside:
        raise ValueError(f"los rangos no forman la cuadrícula {len(midpoints)} × {len(widths)}: "
                         f"faltan {missing}, repetidas {repeated}, fuera del diseño {outside}")
    return grid


def observed_alphas(manifest_path):
    """α mínimo, máximo y promedio de los clips con estado ok del manifiesto."""
    values = np.array([float(r["coefficient"]) for r in read_tsv_rows(manifest_path)
                       if r["status"] == "ok" and r.get("coefficient") not in (None, "")])
    return float(values.min()), float(values.max()), float(values.mean())


# ---------------------------------------------------------------------------
# Tabla resumen (de los JSON)
# ---------------------------------------------------------------------------

def read_json(path):
    """Lee un JSON en UTF-8 (con o sin BOM)."""
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


SUMMARY_METRICS = ("eer_oo", "eer_oa", "eer_aa", "wer_original", "wer_anonymized", "wer_difference")
SUMMARY_COLUMNS = (["config", "midpoint", "width", "alpha_obs_min", "alpha_obs_max", "alpha_obs_mean"]
                   + [f"{m}{s}" for m in SUMMARY_METRICS for s in ("", "_ci_low", "_ci_high")])


def json_metrics(config):
    """
    EER o-o, o-a, a-a y WER original, anonimizado y diferencia, con sus IC, desde los JSON.

    Retorna:
        Diccionario {métrica: (valor, ci_low, ci_high)} con las métricas de
        `SUMMARY_METRICS`.
    """
    eer = read_json(config["files"]["eer_json"])["scenarios"]
    wer = read_json(config["files"]["wer_json"])["wer"]
    out = {f"eer_{s.replace('-', '')}": (eer[s]["eer"], eer[s]["ci_low"], eer[s]["ci_high"])
           for s in ("o-o", "o-a", "a-a")}
    for key, name in (("original", "wer_original"), ("anonymized", "wer_anonymized"), ("difference", "wer_difference")):
        out[name] = (wer[key]["value"], wer[key]["ci_low"], wer[key]["ci_high"])
    return out


def summary_row(config):
    """Fila de la tabla resumen de una configuración de rango."""
    midpoint, width = nominal_design(config["tag"])
    low, high, mean = observed_alphas(config["files"]["manifest"])
    row = {"config": config["tag"], "midpoint": midpoint, "width": width,
           "alpha_obs_min": low, "alpha_obs_max": high, "alpha_obs_mean": mean}
    for metric, (value, ci_low, ci_high) in json_metrics(config).items():
        row.update({metric: value, f"{metric}_ci_low": ci_low, f"{metric}_ci_high": ci_high})
    return row


# ---------------------------------------------------------------------------
# Diferencias pareadas
# ---------------------------------------------------------------------------

def config_replicates(config, results, samples, client_ids, params):
    """
    Valor puntual y réplicas del EER o-a, el EER a-a y el WER anonimizado.

    El EER usa `alpha_differences.eer_replicates` (réplicas guardadas si son
    válidas); el WER se calcula con las mismas muestras de hablantes.

    Retorna:
        Diccionario {métrica: {"point", "reps"}} con las métricas de `METRICS`.
    """
    total = np.zeros(len(client_ids), dtype=int)
    counts = aligned_wer_counts(config["files"]["transcripts"], client_ids)
    return {
        "eer_oa": {"point": results["o-a"]["eer"], "reps": eer_replicates(config, results, samples, params, "o-a")},
        "eer_aa": {"point": results["a-a"]["eer"], "reps": eer_replicates(config, results, samples, params, "a-a")},
        "wer_anonymized": {"point": wer_by_group(counts, total, 1)["anonymized"][0],
                           "reps": np.array([wer_by_group(counts, total, 1, s)["anonymized"][0] for s in samples])},
    }


def check_points(config, metrics):
    """Lanza `ValueError` si algún valor puntual no es exactamente el de los JSON."""
    expected = json_metrics(config)
    wrong = [m for m in METRICS if metrics[m]["point"] != expected[m][0]]
    if wrong:
        raise ValueError(f"{config['tag']}: el valor puntual de {', '.join(wrong)} no coincide con su JSON")


def comparison_pairs(grid, reference):
    """
    Pares (tipo, origen, destino) de las comparaciones del diseño.

    (a) mismo punto medio: ancho 0,15→0,18, 0,12→0,15 y 0,12→0,18;
    (b) mismo ancho: punto medio 0,725→0,75, 0,75→0,775 y 0,725→0,775;
    (c) cada rango frente a la referencia (origen = referencia).
    """
    w1, w2, w3 = sorted(WIDTHS)
    m1, m2, m3 = sorted(MIDPOINTS)
    pairs = []
    for m in MIDPOINTS:
        for a, b in ((w2, w3), (w1, w2), (w1, w3)):
            pairs.append(("mismo_punto_medio", grid[(m, a)], grid[(m, b)]))
    for w in WIDTHS:
        for a, b in ((m1, m2), (m2, m3), (m1, m3)):
            pairs.append(("mismo_ancho", grid[(a, w)], grid[(b, w)]))
    for m, w in product(MIDPOINTS, WIDTHS):
        pairs.append(("frente_a_referencia", reference, grid[(m, w)]))
    return pairs


DIFFERENCE_COLUMNS = ["comparison", "config_from", "config_to", "midpoint_from", "width_from", "midpoint_to",
                      "width_to", "metric", "value_from", "value_to", "difference", "ci_low", "ci_high"]


def difference_rows(pairs, metrics, level):
    """Una fila por par y métrica con los dos valores, la diferencia (destino − origen) y su IC."""
    rows = []
    for kind, a, b in pairs:
        (ma, wa), (mb, wb) = nominal_design(a["tag"]), nominal_design(b["tag"])
        for metric, (diff, low, high) in paired_difference(metrics[a["tag"]], metrics[b["tag"]], level,
                                                           METRICS).items():
            rows.append({"comparison": kind, "config_from": a["tag"], "config_to": b["tag"],
                         "midpoint_from": ma, "width_from": wa, "midpoint_to": mb, "width_to": wb, "metric": metric,
                         "value_from": metrics[a["tag"]][metric]["point"],
                         "value_to": metrics[b["tag"]][metric]["point"],
                         "difference": diff, "ci_low": low, "ci_high": high})
    return rows


# ---------------------------------------------------------------------------
# Diagnóstico de alucinaciones
# ---------------------------------------------------------------------------

HALLUCINATION_COLUMNS = [
    "row_type", "config", "alpha", "midpoint", "width", "n_clips", "n_hallucinated", "words", "errors",
    "errors_hallucinated", "error_fraction_hallucinated", "wer_official", "wer_without_hallucinated",
    "rank", "session_id", "client_id", "stratum", "clip_alpha", "reference_words", "hypothesis_words",
    "clip_errors", "hallucinated", "reference", "hypothesis"]


def hallucination_diagnostic(transcript_rows, coefficients, normalize=normalize_final,
                             ratio=HALLUCINATION_RATIO, top=TOP_CLIPS):
    """
    Diagnóstico de alucinaciones de la versión anonimizada de una configuración.

    Un clip se marca si su hipótesis normalizada tiene más de `ratio` veces
    las palabras de la referencia. El WER oficial no cambia: el WER sin esos
    clips es solo un análisis de sensibilidad.

    Parámetros:
        transcript_rows: filas del TSV de transcribe.py.
        coefficients: {session_id: coeficiente de McAdams del clip}.
        normalize: normalización del texto (la final).
        ratio: umbral de palabras de la hipótesis frente a la referencia.
        top: cuántos clips con más errores se devuelven.

    Retorna:
        Una tupla (resumen, clips): el resumen con los conteos y los dos WER,
        y la lista de los `top` clips con más errores.
    """
    clips = []
    for row in transcript_rows:
        if row["version"] != "anonymized":
            continue
        errors, ref_words = row_counts(row["reference"], row["hypothesis"], normalize)
        hyp_words = len(normalize(row["hypothesis"]).split())
        clips.append({"session_id": row["session_id"], "client_id": row["client_id"], "stratum": row["stratum"],
                      "clip_alpha": coefficients.get(row["session_id"]), "reference_words": ref_words,
                      "hypothesis_words": hyp_words, "clip_errors": errors,
                      "hallucinated": hyp_words > ratio * ref_words,
                      "reference": row["reference"], "hypothesis": row["hypothesis"]})
    words = sum(c["reference_words"] for c in clips)
    errors = sum(c["clip_errors"] for c in clips)
    flagged = [c for c in clips if c["hallucinated"]]
    errors_h = sum(c["clip_errors"] for c in flagged)
    words_h = sum(c["reference_words"] for c in flagged)
    summary = {"n_clips": len(clips), "n_hallucinated": len(flagged), "words": words, "errors": errors,
               "errors_hallucinated": errors_h,
               "error_fraction_hallucinated": errors_h / errors if errors else float("nan"),
               "wer_official": errors / words,
               "wer_without_hallucinated": (errors - errors_h) / (words - words_h) if words > words_h else float("nan")}
    ranked = sorted(clips, key=lambda c: (-c["clip_errors"], c["session_id"]))[:top]
    return summary, [{**c, "rank": i + 1} for i, c in enumerate(ranked)]


def manifest_coefficients(manifest_path):
    """Coeficiente de cada clip del manifiesto ({session_id: α})."""
    return {r["session_id"]: float(r["coefficient"]) for r in read_tsv_rows(manifest_path)
            if r["status"] == "ok" and r.get("coefficient") not in (None, "")}


def hallucination_rows(config):
    """Filas del diagnóstico de una configuración: un resumen y los clips con más errores."""
    summary, top = hallucination_diagnostic(read_tsv_rows(config["files"]["transcripts"]),
                                            manifest_coefficients(config["files"]["manifest"]))
    design = (None, None) if is_fixed(config) else nominal_design(config["tag"])
    base = {"config": config["tag"], "alpha": config["alpha"] if is_fixed(config) else None,
            "midpoint": design[0], "width": design[1]}
    return [{"row_type": "resumen", **base, **summary}] + [{"row_type": "top_clips", **base, **c} for c in top]


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def _interval_point(ax, x, y, color, marker, size=5, zorder=3):
    """Punto con barras de IC en x y en y (cada una de (valor, ci_low, ci_high), en %)."""
    ax.plot([x[1], x[2]], [y[0], y[0]], color=color, linewidth=0.9, zorder=zorder - 1)
    ax.plot([x[0], x[0]], [y[1], y[2]], color=color, linewidth=0.9, zorder=zorder - 1)
    ax.plot(x[0], y[0], color=color, marker=marker, markersize=size, markeredgecolor="white",
            markeredgewidth=0.4, linestyle="none", zorder=zorder)


def _percent(row, metric):
    return tuple(100 * row[f"{metric}{s}"] for s in ("", "_ci_low", "_ci_high"))


def plot_tradeoff(range_rows, reference_row, fixed_rows, wer_target=WER_TARGET):
    """
    Compromiso EER frente a WER anonimizado: panel 1 EER o-a, panel 2 EER a-a.

    Cada rango es un punto con IC en los dos ejes, con color según el punto
    medio y marcador según el ancho. Los α fijos forman una curva gris con su
    α escrito junto a cada punto; U(0,7; 0,9) es una estrella negra. La línea
    vertical punteada marca WER = `wer_target`.

    Parámetros:
        range_rows: filas de `summary_row` de los rangos.
        reference_row: fila de la referencia.
        fixed_rows: filas con "alpha" y las métricas de los α fijos.
        wer_target: meta de WER (fracción).

    Retorna:
        Una `Figure` de matplotlib.
    """
    fig = Figure(figsize=(COLUMN_WIDTH, 5.4))
    axes = fig.subplots(2, 1, sharex=True)
    fixed_rows = sorted(fixed_rows, key=lambda r: r["alpha"])
    # Los rótulos de α fijo van del lado de la curva donde no hay rangos: a la izquierda en el
    # panel del EER o-a y abajo a la derecha en el del EER a-a.
    label_offsets = {"eer_oa": ((-4, 0), "right", "center"), "eer_aa": ((4, -3), "left", "top")}
    for ax, metric, label in ((axes[0], "eer_oa", "EER o-a (%)"), (axes[1], "eer_aa", "EER a-a (%)")):
        offset, ha, va = label_offsets[metric]
        if fixed_rows:
            xs = [100 * r["wer_anonymized"] for r in fixed_rows]
            ys = [100 * r[metric] for r in fixed_rows]
            ax.plot(xs, ys, color=FIXED_COLOR, linewidth=0.9, zorder=1, gid="alpha_fijo")
            for r in fixed_rows:
                _interval_point(ax, _percent(r, "wer_anonymized"), _percent(r, metric), FIXED_COLOR, "o", size=3,
                                zorder=2)
                ax.annotate(f"{r['alpha']:g}".replace(".", ","), (100 * r["wer_anonymized"], 100 * r[metric]),
                            xytext=offset, textcoords="offset points", ha=ha, va=va, fontsize=SMALL_FONT - 1.5,
                            color=MUTED_INK)
        for r in range_rows:
            _interval_point(ax, _percent(r, "wer_anonymized"), _percent(r, metric), MIDPOINT_COLORS[r["midpoint"]],
                            WIDTH_MARKERS[r["width"]])
        _interval_point(ax, _percent(reference_row, "wer_anonymized"), _percent(reference_row, metric),
                        REFERENCE_COLOR, "*", size=8, zorder=4)
        ax.axvline(100 * wer_target, color="0.3", linestyle=":", linewidth=0.8)
        ax.set_ylabel(label)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: percent_label(round(v, 6))))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
        ax.grid(color="0.92", linewidth=0.5)
    axes[0].text(100 * wer_target, 1.01, f"WER {percent_label(100 * wer_target)} %",
                 transform=axes[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=SMALL_FONT,
                 color=MUTED_INK)
    axes[1].set_xlabel("WER anonimizado (%)")
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: percent_label(round(v, 6))))
    handles = ([Line2D([], [], color=MIDPOINT_COLORS[m], marker="o", linestyle="none", markersize=5,
                       label=f"Punto medio {m:g}".replace(".", ",")) for m in MIDPOINTS]
               + [Line2D([], [], color=INK, marker=WIDTH_MARKERS[w], linestyle="none", markersize=5,
                         label=f"Ancho {w:g}".replace(".", ",")) for w in WIDTHS]
               + [Line2D([], [], color=FIXED_COLOR, marker="o", markersize=3, linewidth=0.9, label="α fijo"),
                  Line2D([], [], color=REFERENCE_COLOR, marker="*", linestyle="none", markersize=8,
                         label="U(0,7; 0,9)")])
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=SMALL_FONT - 0.5,
               handlelength=1.4, columnspacing=0.8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.75 / fig.get_figheight(), 1, 1))
    return fig


def plot_by_width(range_rows):
    """
    Tres paneles frente al ancho del rango (EER o-a, EER a-a, WER anonimizado),
    una línea por punto medio, con IC y un pequeño desplazamiento horizontal.
    """
    fig = Figure(figsize=(COLUMN_WIDTH, 5.0))
    axes = fig.subplots(3, 1, sharex=True)
    offsets = dict(zip(MIDPOINTS, (-0.0015, 0.0, 0.0015)))
    panels = (("eer_oa", "EER o-a (%)"), ("eer_aa", "EER a-a (%)"), ("wer_anonymized", "WER anonimizado (%)"))
    for ax, (metric, label) in zip(axes, panels):
        for m in MIDPOINTS:
            rows = sorted((r for r in range_rows if r["midpoint"] == m), key=lambda r: r["width"])
            xs = [r["width"] + offsets[m] for r in rows]
            for x, r in zip(xs, rows):
                value, low, high = _percent(r, metric)
                ax.plot([x, x], [low, high], color=MIDPOINT_COLORS[m], linewidth=1.0, marker="_", markersize=5)
            ax.plot(xs, [100 * r[metric] for r in rows], color=MIDPOINT_COLORS[m], linewidth=1.2, marker="o",
                    markersize=3.5, markeredgewidth=0)
        ax.set_ylabel(label, fontsize=SMALL_FONT)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: percent_label(round(v, 6))))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        ax.grid(axis="y", color="0.9", linewidth=0.5)
    axes[2].set_xticks(list(WIDTHS))
    axes[2].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    axes[2].set_xlabel("Ancho del rango de α")
    handles = [Line2D([], [], color=MIDPOINT_COLORS[m], marker="o", markersize=3.5, linewidth=1.2,
                      label=f"{m:g}".replace(".", ",")) for m in MIDPOINTS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=SMALL_FONT, labelcolor=INK,
               title="Punto medio", title_fontsize=SMALL_FONT)
    fig.tight_layout(rect=(0, 0.45 / fig.get_figheight(), 1, 1))
    return fig


# ---------------------------------------------------------------------------
# Programa
# ---------------------------------------------------------------------------

def write_tsv(rows, columns, path):
    """Escribe una tabla TSV (UTF-8, punto decimal; None y NaN como celda vacía)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def cell(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        return float(value) if isinstance(value, np.floating) else value

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([cell(row.get(c)) for c in columns])
    return path


def compare_with_sweep_summary(rows, path):
    """
    Compara los valores de la tabla con barrido_alfa_resumen.tsv.

    Retorna:
        Lista de diferencias (configuración, columna, valor aquí, valor allá);
        vacía si todo coincide. None si el archivo no existe.
    """
    if not Path(path).exists():
        return None
    sweep = {r["tag"]: r for r in read_tsv_rows(path)}
    mismatches = []
    for row in rows:
        other = sweep.get(row["config"])
        for metric in ("eer_oo", "eer_oa", "eer_aa", "wer_original", "wer_anonymized"):
            for suffix in ("", "_ci_low", "_ci_high"):
                column = f"{metric}{suffix}"
                if other is None or other.get(column) in (None, "") or float(other[column]) != row[column]:
                    mismatches.append((row["config"], column, row[column], None if other is None else other.get(column)))
    return mismatches


def parse_args(argv=None):
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description="Comparación de los rangos uniformes de α (3 × 3) con U(0,7; 0,9).")
    parser.add_argument("--data-dir", default=str(REPO.parent / "Datos"))
    parser.add_argument("--output-dir", default=None, help="Por defecto <data-dir>\\rangos.")
    parser.add_argument("--figures-dir", default=None, help="Por defecto <data-dir>\\figuras\\rangos.")
    parser.add_argument("--tags", nargs="+", default=list(RANGE_TAGS), help="Los 9 rangos del diseño.")
    parser.add_argument("--reference", default=REFERENCE_TAG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-enroll", type=int, default=3)
    parser.add_argument("--n-trial", type=int, default=2)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--boot-seed", type=int, default=2026)
    parser.add_argument("--level", type=float, default=0.95)
    args = parser.parse_args(argv)
    data = Path(args.data_dir)
    args.output_dir = args.output_dir or str(data / "rangos")
    args.figures_dir = args.figures_dir or str(data / "figuras" / "rangos")
    return args


def main():
    """Tabla, diferencias pareadas, diagnóstico de alucinaciones y figuras de los rangos."""
    args = parse_args()
    start = time.perf_counter()
    configs = discover_configs(args.data_dir)
    by_tag = {c["tag"]: c for c in configs}
    missing = [t for t in [*args.tags, args.reference] if t not in by_tag]
    if missing:
        raise SystemExit(f"Faltan configuraciones (sin manifiesto o embeddings): {', '.join(missing)}")
    ranges = [by_tag[t] for t in args.tags]
    reference = by_tag[args.reference]
    grid = check_grid(ranges)
    incomplete = {c["tag"]: missing_outputs(c) for c in [*ranges, reference] if missing_outputs(c)}
    if incomplete:
        raise SystemExit(f"Configuraciones sin todas sus salidas: {incomplete}")
    output_dir, figures_dir = Path(args.output_dir), Path(args.figures_dir)

    # 1-2. Tabla resumen.
    summary = [summary_row(c) for c in [*ranges, reference]]
    write_tsv(summary, SUMMARY_COLUMNS, output_dir / "rangos_resumen.tsv")
    check = compare_with_sweep_summary(summary, Path(args.data_dir) / "barrido_alfa_resumen.tsv")
    print("Tabla frente a barrido_alfa_resumen.tsv: "
          + ("no existe" if check is None else ("coincide" if not check else f"{len(check)} diferencias: {check[:5]}")),
          flush=True)

    # 3. Diferencias pareadas.
    used = [*ranges, reference]
    first_splits, _, _ = load_and_score(used[0]["files"]["embeddings"], used[0]["files"]["manifest"],
                                        args.seed, args.n_enroll, args.n_trial)
    client_ids = list(first_splits)
    samples = shared_samples([s["stratum"] for s in first_splits.values()], args.n_boot, args.boot_seed)
    params = {"seed": args.seed, "n_enroll": args.n_enroll, "n_trial": args.n_trial, "n_boot": args.n_boot,
              "boot_seed": args.boot_seed}
    metrics, oo_reps = {}, {}
    for c in used:
        splits, _, results = load_and_score(c["files"]["embeddings"], c["files"]["manifest"], args.seed,
                                            args.n_enroll, args.n_trial)
        if splits != first_splits:
            raise ValueError(f"{c['tag']}: la partición registro/prueba no coincide con la de {used[0]['tag']}")
        metrics[c["tag"]] = config_replicates(c, results, samples, client_ids, params)
        check_points(c, metrics[c["tag"]])
        with np.load(c["files"]["eer_replicates"]) as data:
            oo_reps[c["tag"]] = np.array(data["o-o"])
    same_oo = all(np.array_equal(v, oo_reps[used[0]["tag"]]) for v in oo_reps.values())
    print(f"EER o-o por réplica idéntico en las {len(used)} configuraciones (mismas muestras): "
          f"{'sí' if same_oo else 'NO'}", flush=True)
    self_check = paired_difference(metrics[reference["tag"]], metrics[reference["tag"]], args.level, METRICS)
    print(f"Referencia contra sí misma: {self_check}", flush=True)
    pairs = comparison_pairs(grid, reference)
    differences = difference_rows(pairs, metrics, args.level)
    write_tsv(differences, DIFFERENCE_COLUMNS, output_dir / "rangos_diferencias.tsv")
    counts = {m: sum(r["metric"] == m for r in differences) for m in METRICS}
    print(f"Comparaciones por métrica (sin corrección por comparaciones múltiples): {counts}", flush=True)

    # 4. Diagnóstico de alucinaciones.
    low, high = FIXED_WINDOW
    fixed = [c for c in configs if is_fixed(c) and low - 1e-9 <= c["alpha"] <= high + 1e-9
             and Path(c["files"]["transcripts"]).exists()]
    diagnostic = [row for c in [*ranges, reference, *fixed] for row in hallucination_rows(c)]
    write_tsv(diagnostic, HALLUCINATION_COLUMNS, output_dir / "rangos_alucinaciones.tsv")

    # 5. Figuras.
    fixed_rows = [{"alpha": c["alpha"], **{f"{m}{s}": v for m, values in json_metrics(c).items()
                                          for s, v in zip(("", "_ci_low", "_ci_high"), values)}}
                  for c in fixed if not missing_outputs(c)]
    configure_fonts()
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_tradeoff(summary[:-1], summary[-1], fixed_rows).savefig(figures_dir / "rangos_compromiso.pdf")
    plot_by_width(summary[:-1]).savefig(figures_dir / "rangos_ancho.pdf")
    print(f"Salidas en {output_dir} y {figures_dir} ({time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main()
