"""
Resultados por acento y por género de un intervalo de α (por defecto el del
prototipo, U(0.685, 0.865), etiqueta u0685_0865).

Usa solo resultados ya generados (manifiesto, embeddings ECAPA y
transcripciones de Whisper): no anonimiza, no extrae embeddings ni
transcribe. Reutiliza las funciones del pipeline: la partición y las
puntuaciones de `run_eer.load_and_score`, el EER dentro del grupo de
`group_analysis.eer_by_group` (con `compute_eer` y `resample_pairs`), el WER
de corpus por grupo de `group_analysis.wer_by_group` (normalización final),
el género de `group_analysis.load_genders` y el criterio de alucinación de
`range_comparison` (más del doble de palabras que la referencia).

Grupos:
- Acento: los 10 estratos del subconjunto (9 acentos y sin_dato).
- Género: la etiqueta declarada en validated.tsv. Solo masculino y femenino
  entran en la comparación; los demás valores y los vacíos solo se cuentan.

Intervalos de confianza: bootstrap de hablantes estratificado por acento
(`group_analysis.shared_samples`, las mismas réplicas de eer_bootstrap.py
con la misma semilla). En cada réplica se calcula la métrica de cada grupo
de acento, la de masculino, la de femenino y su diferencia; los IC son
percentiles de esas réplicas. Las copias de un mismo hablante no cuentan
como impostores.

Prueba por acento: permutación de las etiquetas de acento entre los
hablantes de los 9 acentos reales (sin_dato queda fuera). El estadístico es
el valor máximo menos el mínimo entre los 9 grupos; p = (1 + n.º de
permutaciones con estadístico >= observado) / (1 + n.º de permutaciones).

Salidas (punto decimal; las métricas van en fracción, no en porcentaje):
- <output-dir>\\resumen_grupos.csv: una fila por grupo y una del total.
- <output-dir>\\pruebas_grupos.csv: diferencias masculino − femenino con IC
  y valores p de la permutación por acento.
- <output-dir>\\replicas_grupos.npz: las réplicas del bootstrap.
- <figures-dir>\\res_grupos.pdf y .png: (a) EER o-a y (b) WER anonimizada,
  para figure*. Si existe <output-dir>\\sensibilidad_wer.csv (de
  wer_sensitivity.py), (b) agrega un rombo con la WER de América Central sin
  las transcripciones atípicas.
- El bloque figure* de res_grupos en <figures-dir>\\figuras_resultados.tex
  (se reemplaza si ya está; los demás bloques no cambian). Ojo:
  figuras_resultados.py reescribe ese archivo sin este bloque; después de
  correrlo, vuelva a correr este script con --figures-only.

Uso, desde la raíz del repositorio y con venv-eval:

    & $py scripts\\group_results.py
    & $py scripts\\group_results.py --n-boot 10 --n-perm 10 --output-dir <carpeta de prueba>
    & $py scripts\\group_results.py --figures-only
"""
import argparse
import csv
import re
import time
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

from common_voice_corpus import read_tsv_rows
from eer_bootstrap import resample_pairs
from figuras_resultados import (ANON, BASE, EER_GOAL, GRID, INK, SELECTED_TAG, TEXT_WIDTH, TINY, WER_GOAL, Interval,
                                interval_point, new_figure, nice_limit, panel_title, read_json, save_figure, style,
                                value_grid)
from group_analysis import (GROUP_LABELS, TOTAL, aligned_wer_counts_from_rows, ci, config_files, eer_by_group,
                            group_codes, load_genders, parse_tag, shared_samples, test_codes, wer_by_group)
from range_comparison import HALLUCINATION_RATIO
from run_alpha_sweep import CLIPS_SUBDIR, REPO
from run_eer import load_and_score
from transcribe import normalize_final

# Métrica → escenario del EER o versión del WER.
METRIC_SOURCES = {"eer_oo": "o-o", "eer_oa": "o-a", "eer_aa": "a-a",
                  "wer_original": "original", "wer_anonymized": "anonymized"}
ALL_METRICS = tuple(METRIC_SOURCES)
WER_METRICS = ("wer_original", "wer_anonymized")
TEST_METRICS = ("eer_oa", "eer_aa", "wer_anonymized")
GENDER_COMPARISON = ("masculino", "femenino")
ACCENT, GENDER = "acento", "genero"

SUMMARY_COLUMNS = (["grouping", "group", "label", "in_comparison", "n_speakers", "n_clips", "n_wer_clips",
                    "n_target_pairs", "n_impostor_pairs"]
                   + [f"{m}{s}" for m in ALL_METRICS for s in ("", "_ci_low", "_ci_high")]
                   + ["n_words", "n_errors_original", "n_errors_anonymized", "n_over_double",
                      "alpha_mean", "alpha_sd", "alpha_min", "alpha_max"])
TEST_COLUMNS = ["test", "metric", "groups", "n_speakers", "group_a", "value_a", "group_b", "value_b", "estimate",
                "ci_low", "ci_high", "n_resamples", "seed", "p_value"]


# ---------------------------------------------------------------------------
# Grupos y pares
# ---------------------------------------------------------------------------

def coding(labels, groups):
    """Código de cada hablante según su posición en `groups` (-1 si su etiqueta no está)."""
    index = {g: i for i, g in enumerate(groups)}
    return np.array([index.get(label, -1) for label in labels], dtype=int)


def within_group_pairs(result, codes, group, sample=None):
    """
    Pares objetivo e impostor de un grupo: prueba y registro de hablantes del grupo.

    Toma los hablantes del grupo (o sus copias en la muestra bootstrap) y
    arma los pares con `eer_bootstrap.resample_pairs`, así que un par entre
    copias de un mismo hablante no cuenta como impostor. Es el mismo
    filtrado que usa `group_analysis.eer_by_group`.

    Parámetros:
        result: escenario de `run_scenarios` (con "scores" y "owner").
        codes: grupo de cada hablante (-1 si no está en ningún grupo).
        group: código del grupo.
        sample: índices de hablantes de una réplica; None usa todos una vez.

    Retorna:
        Una tupla (target, impostor) de arreglos 1D.
    """
    codes = np.asarray(codes)
    sample = np.arange(codes.size) if sample is None else np.asarray(sample)
    members = sample[codes[sample] == group]
    return resample_pairs(result["scores"], result["owner"], members)


def pair_counts(result, codes, n_groups):
    """Número de pares objetivo e impostor dentro de cada grupo, con todos los hablantes."""
    counts = []
    for g in range(n_groups):
        target, impostor = within_group_pairs(result, codes, g)
        counts.append((int(target.size), int(impostor.size)))
    return counts


def group_sums(values, codes, n_groups):
    """Suma de un valor por hablante dentro de cada grupo (los hablantes con código -1 no cuentan)."""
    codes = np.asarray(codes)
    valid = codes >= 0
    return np.bincount(codes[valid], weights=np.asarray(values, float)[valid], minlength=n_groups)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def group_metrics(results, wer_counts, codes, n_groups, sample=None, metrics=ALL_METRICS):
    """
    EER y WER de cada grupo para todos los hablantes o para una muestra bootstrap.

    El EER es el de `eer_by_group` (solo pares dentro del grupo) y el WER el
    de corpus de `wer_by_group`. NaN donde un grupo no tiene datos.

    Retorna:
        Diccionario {métrica: arreglo (n_groups,)} con las métricas pedidas.
    """
    out, wer = {}, None
    for metric in metrics:
        source = METRIC_SOURCES[metric]
        if metric.startswith("eer_"):
            out[metric] = eer_by_group(results[source], codes, n_groups, sample)
        else:
            if wer is None:
                wer = wer_by_group(wer_counts, codes, n_groups, sample)
            out[metric] = wer[source]
    return out


def bootstrap_groups(results, wer_counts, codings, samples, progress_every=100, log=print):
    """
    Réplicas bootstrap de las métricas de varias agrupaciones con las mismas muestras.

    En cada réplica la misma muestra de hablantes sirve para todas las
    agrupaciones (acento, género y total), de modo que la diferencia entre
    dos grupos se puede calcular réplica por réplica.

    Parámetros:
        codings: {nombre: (códigos, número de grupos, métricas)}.
        samples: muestras de `group_analysis.shared_samples`.
        progress_every: cada cuántas réplicas se imprime el avance (0 o None: nunca).

    Retorna:
        {nombre: {métrica: arreglo (réplicas, número de grupos)}}.
    """
    reps = {name: {m: np.empty((len(samples), n)) for m in metrics} for name, (_, n, metrics) in codings.items()}
    start = time.perf_counter()
    for b, sample in enumerate(samples):
        for name, (codes, n, metrics) in codings.items():
            for metric, values in group_metrics(results, wer_counts, codes, n, sample, metrics).items():
                reps[name][metric][b] = values
        if progress_every and (b + 1) % progress_every == 0:
            log(f"  Réplica {b + 1}/{len(samples)} ({time.perf_counter() - start:.0f} s)")
    return reps


def group_difference(point, reps, a, b, level=0.95):
    """
    Diferencia entre dos grupos (a − b) con IC de las mismas réplicas.

    El valor puntual sale de todos los datos; el IC, de los percentiles de
    la diferencia réplica por réplica (bootstrap pareado). Las réplicas con
    NaN se ignoran.

    Parámetros:
        point: valor de cada grupo con todos los datos.
        reps: arreglo (réplicas, grupos).
        a, b: códigos de los dos grupos.

    Retorna:
        Una tupla (diferencia, ci_low, ci_high).
    """
    reps = np.asarray(reps, float)
    low, high = ci(reps[:, a] - reps[:, b], level)
    return float(point[a] - point[b]), low, high


def range_statistic(values):
    """Valor máximo menos valor mínimo entre grupos (sin contar los NaN)."""
    return float(np.nanmax(values) - np.nanmin(values))


def range_permutation_test(group_values, codes, n_perm, seed, progress_every=0, log=print):
    """
    Prueba de permutación global del rango (máximo − mínimo) entre grupos.

    Baraja las etiquetas de grupo entre los hablantes que tienen código >= 0
    (cada grupo conserva su tamaño); los hablantes con código -1 no se
    mueven y quedan fuera. Todas las métricas usan las mismas permutaciones.
    p = (1 + n.º de permutaciones con estadístico >= observado) / (1 + n_perm).

    Parámetros:
        group_values: función que recibe los códigos de los hablantes y
            devuelve {métrica: valor de cada grupo}.
        codes: grupo de cada hablante (-1 si no entra en la prueba).
        n_perm: número de permutaciones.
        seed: semilla de `np.random.default_rng`.

    Retorna:
        {métrica: {"values": valores observados por grupo, "observed":
        estadístico observado, "p_value": p}}.
    """
    codes = np.asarray(codes)
    included = np.flatnonzero(codes >= 0)
    observed_values = group_values(codes)
    observed = {m: range_statistic(v) for m, v in observed_values.items()}
    exceed = dict.fromkeys(observed, 0)
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    for k in range(n_perm):
        permuted = codes.copy()
        permuted[included] = rng.permutation(codes[included])
        for metric, values in group_values(permuted).items():
            exceed[metric] += range_statistic(values) >= observed[metric] - 1e-12
        if progress_every and (k + 1) % progress_every == 0:
            log(f"  Permutación {k + 1}/{n_perm} ({time.perf_counter() - start:.0f} s)")
    return {m: {"values": np.asarray(observed_values[m], float), "observed": observed[m],
                "p_value": (1 + exceed[m]) / (1 + n_perm)} for m in observed}


def over_double_by_speaker(transcript_rows, client_ids, normalize=normalize_final, ratio=HALLUCINATION_RATIO):
    """
    Transcripciones anonimizadas de cada hablante con más de `ratio` veces las palabras de su referencia.

    Es el criterio de `range_comparison.hallucination_diagnostic`: palabras
    después de la normalización final. Lanza `ValueError` si una fila es de
    un hablante que no está en `client_ids`.
    """
    index = {c: i for i, c in enumerate(client_ids)}
    out = np.zeros(len(client_ids), dtype=int)
    for row in transcript_rows:
        if row["version"] != "anonymized":
            continue
        if row["client_id"] not in index:
            raise ValueError(f"el hablante {row['client_id']} de las transcripciones no está en la partición")
        ref_words = len(normalize(row["reference"]).split())
        hyp_words = len(normalize(row["hypothesis"]).split())
        out[index[row["client_id"]]] += hyp_words > ratio * ref_words
    return out


def speaker_alphas(manifest_rows, client_ids):
    """α de todos los clips "ok" del manifiesto de cada hablante, en el orden de `client_ids`."""
    index = {c: i for i, c in enumerate(client_ids)}
    out = [[] for _ in client_ids]
    for row in manifest_rows:
        if row["status"] == "ok" and row["client_id"] in index and row.get("coefficient") not in (None, ""):
            out[index[row["client_id"]]].append(float(row["coefficient"]))
    return [np.array(values) for values in out]


def alpha_stats(alphas, members):
    """Media, desviación estándar muestral (ddof = 1), mínimo y máximo del α de los clips de `members`."""
    values = np.concatenate([alphas[i] for i in members]) if len(members) else np.array([])
    if values.size == 0:
        return {"alpha_mean": None, "alpha_sd": None, "alpha_min": None, "alpha_max": None}
    return {"alpha_mean": float(values.mean()), "alpha_sd": float(values.std(ddof=1)) if values.size > 1 else None,
            "alpha_min": float(values.min()), "alpha_max": float(values.max())}


# ---------------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------------

def count_rows(grouping, groups, codes, speaker, alphas, pairs, compared):
    """
    Filas de conteos de una agrupación: hablantes, clips, pares, palabras, alucinaciones y α.

    Parámetros:
        speaker: {columna: valor por hablante} que se suma dentro del grupo
            (n_clips, n_wer_clips, n_words, n_errors_original,
            n_errors_anonymized, n_over_double).
        alphas: salida de `speaker_alphas`.
        pairs: salida de `pair_counts` (o None para dejar los pares vacíos).
        compared: grupos que entran en la comparación.
    """
    rows = []
    for g, group in enumerate(groups):
        members = np.flatnonzero(np.asarray(codes) == g)
        row = {"grouping": grouping, "group": group, "label": GROUP_LABELS.get(group, group),
               "in_comparison": group in compared, "n_speakers": int(members.size),
               "n_target_pairs": pairs[g][0] if pairs else None, "n_impostor_pairs": pairs[g][1] if pairs else None}
        row.update({key: int(np.sum(np.asarray(values)[members])) for key, values in speaker.items()})
        row.update(alpha_stats(alphas, members))
        rows.append(row)
    return rows


def metric_cells(point, reps, g, level):
    """Valor puntual e IC de cada métrica para el grupo de código `g`."""
    cells = {}
    for metric, values in point.items():
        low, high = ci(reps[metric][:, g], level) if reps is not None and metric in reps else (None, None)
        cells.update({metric: float(values[g]), f"{metric}_ci_low": low, f"{metric}_ci_high": high})
    return cells


def eer_ci_from_json(path, expected, results):
    """
    IC del EER total del JSON de eer_bootstrap.py, si se calculó con los mismos parámetros.

    Comprueba la semilla, la partición, el número de réplicas, la semilla
    del bootstrap y el nivel, y que el EER puntual del JSON sea el de
    `results`. Si algo no coincide o no existe el archivo, retorna None.
    """
    if not Path(path).exists():
        return None
    summary = read_json(path)
    if any(summary.get(k) != v for k, v in expected.items()):
        return None
    scenarios = summary.get("scenarios", {})
    if any(abs(scenarios.get(s, {}).get("eer", np.nan) - results[s]["eer"]) > 1e-12 for s in ("o-o", "o-a", "a-a")):
        return None
    return {s: (scenarios[s]["ci_low"], scenarios[s]["ci_high"]) for s in ("o-o", "o-a", "a-a")}


def write_csv(rows, columns, path):
    """Escribe una tabla CSV en UTF-8 con punto decimal (None y NaN como celda vacía)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def cell(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([cell(row.get(c)) for c in columns])
    return path


def read_csv(path):
    """Lee una tabla escrita por `write_csv` como lista de diccionarios de texto."""
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------

def point_number(value):
    """Número sin ceros sobrantes, con punto decimal y signo menos tipográfico: 0.70 → "0.7"."""
    value = round(float(value), 6)
    if value == 0:
        value = 0.0
    return f"{value:g}".replace("-", "−")


POINT_FORMATTER = FuncFormatter(lambda v, _: point_number(v))

# (letra, título, rótulo del eje, métrica, referencia original, color, marcador, meta, signo de la meta)
PANELS = (
    ("a", "Privacidad: EER o-a", "EER (%)", "eer_oa", "eer_oo", ANON, "s", EER_GOAL, "≥"),
    ("b", "Inteligibilidad: WER anonimizada", "WER (%)", "wer_anonymized", "wer_original", ANON, "o", WER_GOAL, "<"),
)

# Sensibilidad de la WER (wer_sensitivity.py): WER sin las transcripciones con más del doble de
# palabras que su referencia, como rombo vacío debajo del punto del grupo en el panel de la WER.
SENSITIVITY_FILE = "sensibilidad_wer.csv"
SENSITIVITY_ANALYSIS = "sin_mas_del_doble"
SENSITIVITY_GROUPS = ("america_central",)
SENSITIVITY_OFFSET = 0.32  # desplazamiento vertical del rombo, en renglones
FIGURE_NAME = "res_grupos"
LATEX_FILE = "figuras_resultados.tex"


def load_sensitivity(path, groups=SENSITIVITY_GROUPS):
    """
    WER sin las transcripciones atípicas de `groups`, de sensibilidad_wer.csv.

    Retorna:
        Diccionario {"groups": {grupo: fila (analysis = sin_mas_del_doble,
        row_type = grupo)}, "n_excluded": transcripciones excluidas de esos
        grupos}; vacío si el archivo no existe.
    """
    if not Path(path).exists():
        return {}
    table = read_csv(path)
    rows = {r["group"]: r for r in table
            if r["analysis"] == SENSITIVITY_ANALYSIS and r["row_type"] == "grupo" and r["group"] in groups}
    excluded = sum(r["row_type"] == "transcripcion_excluida" and r["group"] in groups for r in table)
    return {"groups": rows, "n_excluded": excluded} if rows else {}


def _value(row, key):
    """Valor numérico de una celda (texto o número); None si está vacía."""
    value = row.get(key)
    return None if value in (None, "") else float(value)


def figure_order(rows):
    """
    Renglones de la figura, de arriba abajo: los acentos de mayor a menor
    EER o-a, None (la separación) y después masculino y femenino.
    """
    accents = sorted((r for r in rows if r["grouping"] == ACCENT), key=lambda r: -_value(r, "eer_oa"))
    genders = [r for g in GENDER_COMPARISON for r in rows if r["grouping"] == GENDER and r["group"] == g]
    return accents + ([None] + genders if genders else [])


def plot_groups(rows, sensitivity=None):
    """
    EER o-a y WER anonimizada por acento y por género, con el eje vertical compartido.

    Dos paneles lado a lado, para el ancho de toda la página (figure*). Cada
    grupo lleva su punto y su IC del 95 % (el eje x llega hasta el extremo
    superior del IC más alto); el marcador hueco gris es el valor sin
    anonimizar (EER o-o en (a) y WER original en (b)) y las líneas punteadas
    son las metas del proyecto (EER o-a ≥ 30 % y WER < 15 %). Con
    `sensitivity`, en (b) se agrega para esos grupos un rombo vacío con su IC,
    un poco más abajo, con la WER sin las transcripciones atípicas. Punto
    decimal en todos los números.

    Parámetros:
        rows: filas de resumen_grupos.csv (números o texto).
        sensitivity: salida de `load_sensitivity`, o None.

    Retorna:
        Una `Figure` de matplotlib.
    """
    order = figure_order(rows)
    n = len(order)
    accents = {r["group"] for r in order if r is not None and r["grouping"] == ACCENT}
    alternatives = {group: Interval.from_row(row, "wer_anonymized")
                    for group, row in (sensitivity or {}).get("groups", {}).items() if group in accents}
    alternatives = {group: interval for group, interval in alternatives.items() if interval is not None}
    drawn_sensitivity = bool(alternatives)
    # La leyenda con el rombo ocupa dos filas: se agrega ese alto para no apretar los renglones.
    fig = new_figure(TEXT_WIDTH, 0.165 * n + 1.05 + (0.15 if drawn_sensitivity else 0.0))
    axes = fig.subplots(1, len(PANELS), sharey=True)
    positions = {i: n - 1 - i for i in range(n)}
    for ax, (letter, title, xlabel, metric, base, color, marker, goal, sign) in zip(axes, PANELS):
        extent = []
        for i, row in enumerate(order):
            if row is None:
                ax.axhline(positions[i], color=GRID, linewidth=0.8, zorder=0.5)
                continue
            interval = Interval.from_row(row, metric)
            if interval is not None:
                interval_point(ax, None, positions[i], interval, color, marker, size=4.0, orientation="h",
                               linewidth=1.0)
                extent += [interval.low, interval.high]
            alternative = alternatives.get(row["group"]) if metric == "wer_anonymized" and row["grouping"] == ACCENT \
                else None
            if alternative is not None:
                interval_point(ax, None, positions[i] - SENSITIVITY_OFFSET, alternative, color, "D", size=3.6,
                               orientation="h", filled=False, linewidth=1.0)
                extent += [alternative.low, alternative.high]
            original = _value(row, base) if base else None
            if original is not None:
                ax.plot(100 * original, positions[i], linestyle="none", marker="o", markersize=4.0,
                        markerfacecolor="white", markeredgecolor=BASE, markeredgewidth=0.9, zorder=4)
                extent.append(100 * original)
        if goal is not None:
            # Misma línea que goal_line de figuras_resultados; el rótulo va en el renglón de la
            # separación (vacío), para no tapar el IC de ningún grupo.
            ax.axvline(goal, color=INK, linestyle=(0, (1.2, 1.6)), linewidth=0.9, zorder=2)
            label_y = next((positions[i] for i, r in enumerate(order) if r is None), n - 1)
            ax.annotate(f"Meta {sign} {point_number(goal)} %", (goal, label_y), xytext=(3, 0),
                        textcoords="offset points", ha="left", va="center", fontsize=TINY + 0.5, color=INK, zorder=5,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.85})
            extent.append(goal * 1.22)
        ax.set_xlim(0, nice_limit(max(extent + [1.0]) * 1.04))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.xaxis.set_major_formatter(POINT_FORMATTER)
        ax.set_xlabel(xlabel)
        value_grid(ax, "x")
        panel_title(ax, letter, title)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks([positions[i] for i, r in enumerate(order) if r is not None],
                       [r["label"] or GROUP_LABELS.get(r["group"], r["group"]) for r in order if r is not None])
    axes[0].set_ylim(-0.6, n - 0.4)

    def handle(color, marker, filled=True):
        return Line2D([], [], color=color, linewidth=1.0, marker=marker, markersize=4.0,
                      markerfacecolor=color if filled else "white", markeredgecolor="white" if filled else color,
                      markeredgewidth=0.6 if filled else 0.9)

    series = [(handle(ANON, "s"), "EER o-a (IC 95 %)"), (handle(ANON, "o"), "WER anonimizada (IC 95 %)")]
    if drawn_sensitivity:
        series.append((Line2D([], [], color=ANON, linewidth=1.0, marker="D", markersize=3.6,
                              markerfacecolor="white", markeredgecolor=ANON, markeredgewidth=0.9),
                       f"WER sin las {sensitivity['n_excluded']} transcripciones atípicas (IC 95 %)"))
    references = [(Line2D([], [], linestyle="none", marker="o", markersize=4.0, markerfacecolor="white",
                          markeredgecolor=BASE, markeredgewidth=0.9), "Voz original (EER o-o, WER original)"),
                  (Line2D([], [], color=INK, linestyle=(0, (1.2, 1.6)), linewidth=0.9), "Meta del proyecto")]
    if drawn_sensitivity:
        # Dos filas para que quepa en \textwidth: arriba las series y abajo las referencias. La leyenda
        # se llena por columnas, así que las entradas se intercalan.
        entries = [series[0], references[0], series[1], references[1], series[2]]
        ncol = 3
    else:
        entries, ncol = series + references, len(series) + len(references)
    fig.legend([h for h, _ in entries], [label for _, label in entries], loc="outside lower center", ncol=ncol,
               columnspacing=1.2, handlelength=1.8)
    return fig


def latex_block(tag, sensitivity=True, name=FIGURE_NAME, folder="figuras/resultados"):
    """Bloque figure* (ancho de página) de la figura, con el formato de figuras_resultados.latex_snippet."""
    low, high = parse_tag(tag)
    caption = (f"Resultados por acento y género con el intervalo elegido, $\\mathcal{{U}}({low:g}, {high:g})$: "
               "(a) EER o--a y (b) WER del audio anonimizado. Las barras son intervalos de confianza del 95~\\% "
               "por \\textit{bootstrap}, los círculos grises son los valores con la voz original y la línea "
               "punteada es la meta del proyecto.")
    if sensitivity:
        caption += " En (b), el rombo muestra la WER de América Central sin las dos transcripciones atípicas."
    return (f"\\begin{{figure*}}[!t]\n    \\centering\n"
            f"    \\includegraphics[width=\\textwidth]{{{folder}/{name}.pdf}}\n"
            f"    \\caption{{{caption}}}\n    \\label{{fig:{name}}}\n\\end{{figure*}}\n")


def update_latex(path, block, name=FIGURE_NAME):
    """
    Pone `block` en el archivo LaTeX de las figuras: reemplaza el entorno
    figure o figure* que tenga \\label{fig:<name>} o, si no hay, lo agrega al
    final. Los demás bloques no cambian.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    for match in re.finditer(r"\\begin\{(figure\*?)\}.*?\\end\{\1\}\n?", text, flags=re.S):
        if f"\\label{{fig:{name}}}" in match.group(0):
            text = text[:match.start()] + block + text[match.end():]
            break
    else:
        text = (text.rstrip("\n") + "\n\n" if text.strip() else "") + block
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def save_group_figure(rows, figures_dir, formats=("pdf", "png"), sensitivity=None, tag=SELECTED_TAG,
                      name=FIGURE_NAME):
    """
    Dibuja `plot_groups` con el estilo de figuras_resultados.py, la guarda en
    cada formato y actualiza su bloque en figuras_resultados.tex.

    Retorna:
        Lista de rutas: las de la figura y la del archivo LaTeX.
    """
    with matplotlib.rc_context(style()):
        Path(figures_dir).mkdir(parents=True, exist_ok=True)
        paths = save_figure(plot_groups(rows, sensitivity), figures_dir, name, formats)
    block = latex_block(tag, sensitivity=bool(sensitivity), name=name)
    return paths + [update_latex(Path(figures_dir) / LATEX_FILE, block, name)]


# ---------------------------------------------------------------------------
# Programa
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description="EER y WER por acento y por género de un intervalo de α.")
    parser.add_argument("--data-dir", default=str(REPO.parent / "Datos"))
    parser.add_argument("--tag", default=SELECTED_TAG, help="Configuración (por defecto u0685_0865).")
    parser.add_argument("--validated", default=None, help="validated.tsv; por defecto el de Common Voice en --data-dir.")
    parser.add_argument("--output-dir", default=None, help="Por defecto <data-dir>\\por_grupo_<tag>.")
    parser.add_argument("--figures-dir", default=None, help="Por defecto <data-dir>\\figuras\\resultados.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-enroll", type=int, default=3)
    parser.add_argument("--n-trial", type=int, default=2)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--boot-seed", type=int, default=2026)
    parser.add_argument("--level", type=float, default=0.95)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--perm-seed", type=int, default=2026)
    parser.add_argument("--formats", nargs="+", choices=("pdf", "png", "svg"), default=["pdf", "png"])
    parser.add_argument("--figures-only", action="store_true",
                        help="Solo redibuja la figura desde resumen_grupos.csv y sensibilidad_wer.csv, sin recalcular.")
    args = parser.parse_args(argv)
    data = Path(args.data_dir)
    args.output_dir = args.output_dir or str(data / f"por_grupo_{args.tag}")
    args.figures_dir = args.figures_dir or str(data / "figuras" / "resultados")
    args.validated = args.validated or str(data / CLIPS_SUBDIR.parent / "validated.tsv")
    return args


def main(argv=None, log=print):
    """Calcula las tablas por grupo, las pruebas y la figura de una configuración."""
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    if args.figures_only:
        paths = save_group_figure(read_csv(output_dir / "resumen_grupos.csv"), args.figures_dir, args.formats,
                                  load_sensitivity(output_dir / SENSITIVITY_FILE), args.tag)
        log(f"Figura: {', '.join(str(p) for p in paths)}")
        return
    start_all = time.perf_counter()
    files = config_files(args.tag, args.data_dir)
    log(f"Configuración {args.tag}\n  manifiesto: {files['manifest']}\n  embeddings: {files['embeddings']}\n"
        f"  transcripciones: {files['transcripts']}\n  género: {args.validated}")

    splits, excluded, results = load_and_score(files["embeddings"], files["manifest"], args.seed, args.n_enroll,
                                               args.n_trial)
    client_ids = list(splits)
    strata = [s["stratum"] for s in splits.values()]
    transcript_rows = list(read_tsv_rows(files["transcripts"]))
    wer_counts = aligned_wer_counts_from_rows(transcript_rows, client_ids, files["transcripts"])
    manifest_rows = list(read_tsv_rows(files["manifest"]))
    genders = load_genders(args.validated, client_ids)
    gender_labels = [genders[c] for c in client_ids]
    log(f"Hablantes: {len(client_ids)} (excluidos: {len(excluded)}); "
        "EER global o-o/o-a/a-a: " + ", ".join(f"{100 * results[s]['eer']:.2f} %" for s in ("o-o", "o-a", "a-a")))

    # Agrupaciones: acento (10 estratos; la prueba solo con los 9 acentos reales) y género.
    accent_groups, accent_codes = group_codes(strata)
    tested_accents, perm_codes = test_codes(accent_groups, accent_codes, "stratum")
    all_genders, _ = group_codes(gender_labels)
    gender_groups = [g for g in GENDER_COMPARISON if g in all_genders] + \
                    [g for g in all_genders if g not in GENDER_COMPARISON]
    gender_codes = coding(gender_labels, gender_groups)
    compared_genders = [g for g in GENDER_COMPARISON if g in gender_groups]
    compared_codes = coding(gender_labels, compared_genders)
    total_codes = np.zeros(len(client_ids), dtype=int)
    log("Género por hablante: " + ", ".join(f"{g} {int(np.sum(gender_codes == i))}"
                                            for i, g in enumerate(gender_groups)))

    codings = {ACCENT: (accent_codes, len(accent_groups), ALL_METRICS),
               GENDER: (compared_codes, len(compared_genders), ALL_METRICS),
               TOTAL: (total_codes, 1, WER_METRICS)}
    point = {name: group_metrics(results, wer_counts, codes, n, None, metrics)
             for name, (codes, n, metrics) in codings.items()}

    start = time.perf_counter()
    samples = shared_samples(strata, args.n_boot, args.boot_seed)
    reps = bootstrap_groups(results, wer_counts, codings, samples, progress_every=100, log=log)
    boot_seconds = time.perf_counter() - start
    log(f"Bootstrap: {args.n_boot} réplicas en {boot_seconds:.1f} s ({boot_seconds / args.n_boot:.2f} s por réplica)")

    start = time.perf_counter()
    permutation = range_permutation_test(
        lambda codes: group_metrics(results, wer_counts, codes, len(tested_accents), None, TEST_METRICS),
        perm_codes, args.n_perm, args.perm_seed, progress_every=100, log=log)
    perm_seconds = time.perf_counter() - start
    log(f"Permutación: {args.n_perm} permutaciones en {perm_seconds:.1f} s "
        f"({perm_seconds / args.n_perm:.3f} s por permutación)")

    # Tabla por grupo.
    speaker = {"n_clips": np.array([len(s["enroll"]) + len(s["trial"]) for s in splits.values()]),
               "n_wer_clips": wer_counts["clips"]["anonymized"], "n_words": wer_counts["words"]["original"],
               "n_errors_original": wer_counts["errors"]["original"],
               "n_errors_anonymized": wer_counts["errors"]["anonymized"],
               "n_over_double": over_double_by_speaker(transcript_rows, client_ids)}
    alphas = speaker_alphas(manifest_rows, client_ids)
    oo = results["o-o"]
    summary = count_rows(ACCENT, accent_groups, accent_codes, speaker, alphas,
                         pair_counts(oo, accent_codes, len(accent_groups)), tested_accents)
    for g, row in enumerate(summary):
        row.update(metric_cells(point[ACCENT], reps[ACCENT], g, args.level))
    gender_rows = count_rows(GENDER, gender_groups, gender_codes, speaker, alphas,
                             pair_counts(oo, gender_codes, len(gender_groups)), compared_genders)
    for row in gender_rows:
        if row["group"] in compared_genders:
            row.update(metric_cells(point[GENDER], reps[GENDER], compared_genders.index(row["group"]), args.level))
        else:
            row["n_target_pairs"] = row["n_impostor_pairs"] = None  # no entra en la comparación
    total = count_rows(TOTAL, [TOTAL], total_codes, speaker, alphas, pair_counts(oo, total_codes, 1), [])[0]
    total.update(metric_cells(point[TOTAL], reps[TOTAL], 0, args.level))
    expected = {"seed": args.seed, "n_enroll": args.n_enroll, "n_trial": args.n_trial, "n_boot": args.n_boot,
                "boot_seed": args.boot_seed, "level": args.level}
    json_ci = eer_ci_from_json(files["eer_json"], expected, results)
    for metric in ("eer_oo", "eer_oa", "eer_aa"):
        total[metric] = float(results[METRIC_SOURCES[metric]]["eer"])
        low, high = json_ci[METRIC_SOURCES[metric]] if json_ci else (None, None)
        total[f"{metric}_ci_low"], total[f"{metric}_ci_high"] = low, high
    if json_ci is None:
        log(f"  Aviso: {Path(files['eer_json']).name} no existe o tiene otros parámetros; el total va sin IC del EER")
    summary += gender_rows + [total]

    # Pruebas.
    tests = []
    if len(compared_genders) == 2:
        for metric in TEST_METRICS:
            estimate, low, high = group_difference(point[GENDER][metric], reps[GENDER][metric], 0, 1, args.level)
            tests.append({"test": "diferencia_genero", "metric": metric, "groups": ",".join(compared_genders),
                          "n_speakers": int(np.sum(compared_codes >= 0)), "group_a": compared_genders[0],
                          "value_a": point[GENDER][metric][0], "group_b": compared_genders[1],
                          "value_b": point[GENDER][metric][1], "estimate": estimate, "ci_low": low, "ci_high": high,
                          "n_resamples": args.n_boot, "seed": args.boot_seed})
    for metric in TEST_METRICS:
        values = permutation[metric]["values"]
        high_g, low_g = int(np.nanargmax(values)), int(np.nanargmin(values))
        tests.append({"test": "permutacion_acento", "metric": metric, "groups": ",".join(tested_accents),
                      "n_speakers": int(np.sum(perm_codes >= 0)), "group_a": tested_accents[high_g],
                      "value_a": values[high_g], "group_b": tested_accents[low_g], "value_b": values[low_g],
                      "estimate": permutation[metric]["observed"], "n_resamples": args.n_perm,
                      "seed": args.perm_seed, "p_value": permutation[metric]["p_value"]})

    summary_path = write_csv(summary, SUMMARY_COLUMNS, output_dir / "resumen_grupos.csv")
    tests_path = write_csv(tests, TEST_COLUMNS, output_dir / "pruebas_grupos.csv")
    np.savez(output_dir / "replicas_grupos.npz", samples=np.array(samples),
             **{f"{name}__{metric}": values for name, by_metric in reps.items() for metric, values in by_metric.items()},
             accent_groups=np.array(accent_groups), gender_groups=np.array(compared_genders))
    figure_paths = save_group_figure(summary, args.figures_dir, args.formats,
                                     load_sensitivity(output_dir / SENSITIVITY_FILE), args.tag)
    log(f"Tablas: {summary_path}, {tests_path}\nFigura: {', '.join(str(p) for p in figure_paths)}")
    log(f"Tiempo total: {time.perf_counter() - start_all:.0f} s")


if __name__ == "__main__":
    main()
