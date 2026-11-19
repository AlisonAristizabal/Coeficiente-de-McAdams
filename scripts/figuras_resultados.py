"""
Figuras de la sección de Resultados: privacidad (EER), inteligibilidad (WER)
y tiempo real (RTF).

No recalcula ninguna métrica. Lee las salidas que ya dejaron los scripts del
pipeline y las organiza según los experimentos de la Sección 3.6.3 del
artículo. Cada figura se dibuja solo si existen sus archivos; si falta
alguno, se avisa y se sigue con las demás.

| Figura                     | Experimento                                   | Lee (dentro de --data-dir)                         |
|----------------------------|-----------------------------------------------|----------------------------------------------------|
| res_resumen_prototipo      | Intervalo del prototipo frente a las metas    | JSON del EER y del WER de --selected; en_vivo\\    |
| res_etapa1_barrido_alfa    | Etapa 1: α fijo de 0,2 a 1,0 (paso 0,1)       | JSON del EER y del WER de cada α fijo              |
| res_etapa2_barrido_fino    | Etapa 2: α fijo de 0,65 a 0,875 (paso 0,025)  | JSON del EER y del WER de cada α fijo              |
| res_etapa3_intervalos      | Etapa 3: nueve intervalos uniformes           | JSON de los rangos y de U(0,7; 0,9)                |
| res_diferencias_alfa       | Bootstrap pareado entre α vecinos             | barrido_alfa_diferencias.tsv (alpha_differences)   |
| res_tiempo_real            | Prototipo: RTF, tiempo por bloque y latencia  | en_vivo\\<corrida>\\resumen.json y tiempos.tsv      |

Ninguna figura muestra la tasa de falsa aceptación (FAR) ni la suplantación:
la privacidad se reporta solo con el EER (escenarios o-o, o-a y a-a). Por
ahora tampoco hay figuras por acento ni por género (siguen en las tablas de
group_analysis.py).

Estilo: tipografía con remates (Times, como newtxtext del artículo), coma
decimal, ancho de una columna (3,27 in) o de toda la página (6,85 in) del
artículo, sin título (va en el \\caption), letras de panel, metas punteadas,
intervalos de confianza del 95 % tal como salen del bootstrap (pueden ser
asimétricos) y paleta validada para daltonismo. Cada figura se guarda en PDF
vectorial y en PNG a 300 ppp, y se escribe además `figuras_resultados.tex`
con el código LaTeX para incluirlas.

Uso, desde la raíz del repositorio y con venv-eval:

    & $py scripts\\figuras_resultados.py
    & $py scripts\\figuras_resultados.py --only etapa1 etapa2 --formats pdf
    & $py scripts\\figuras_resultados.py --live-runs ..\\Datos\\en_vivo\\20261115_101500_anonimizado ...
"""
import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator, NullFormatter

from group_analysis import config_files, manifest_alpha, parse_tag
from run_alpha_sweep import REPO

# ---------------------------------------------------------------------------
# Estilo
# ---------------------------------------------------------------------------

# Anchos del artículo (letterpaper, márgenes de 2,1 cm, separación de 0,8 cm).
COLUMN_WIDTH = 3.27
TEXT_WIDTH = 6.85

FONT_SIZE = 8
SMALL = 7
TINY = 6

INK = "#1A1A1A"         # texto principal
MUTED = "#5E5E5E"       # texto secundario
AXIS = "#4D4D4D"        # ejes y marcas
GRID = "#E4E4E4"        # rejilla
GOAL_FILL = "#EFEFEF"   # zona donde se cumple una meta
BAND = "#F3F3F3"        # zonas de α (barrido fino, intervalo del prototipo)
HIGHLIGHT_ROW = "#E8F1FA"

# Paleta validada (scripts/validate_palette.js de la guía de visualización):
# anonimizado/o-a en azul, a-a en bermellón y la referencia sin anonimizar en
# gris claro, que además va con línea discontinua o marcador hueco.
ANON = "#0072B2"
AA = "#D55E00"
BASE = "#8A8A8A"

SCENARIO_STYLE = {
    "o-o": {"color": BASE, "marker": "o", "label": "Original (o-o)"},
    "o-a": {"color": ANON, "marker": "s", "label": "Ignorante (o-a)"},
    "a-a": {"color": AA, "marker": "^", "label": "Semi-informado (a-a)"},
}

# Diseño de la etapa 3. El ancho va en una rampa ordinal de un solo tono
# (claro → oscuro, más ancho) y U(0,7; 0,9) es la línea roja discontinua.
MIDPOINTS = (0.725, 0.75, 0.775)
WIDTHS = (0.12, 0.15, 0.18)
WIDTH_COLORS = {0.12: "#86B6EF", 0.15: "#2A78D6", 0.18: "#104281"}
REFERENCE_COLOR = "#C0392B"
LATENCY_COLORS = ("#104281", "#2A78D6", "#86B6EF")

# Metas del proyecto (Tabla 2 del artículo), en porcentaje salvo el RTF.
EER_GOAL = 30.0
WER_GOAL = 15.0
RTF_GOAL = 0.1
# Referencia de la Recomendación UIT-T G.114 (retardo en un sentido, ms).
LATENCY_REFERENCE_MS = 150.0

SELECTED_TAG = "u0685_0865"
REFERENCE_TAG = "u07_09"
FINE_WINDOW = (0.65, 0.875)
FINE_FIGURE_WINDOW = (0.65, 0.9)
DIFFERENCE_MIN_ALPHA = 0.6

FIGURE_NAMES = {
    "resumen": "res_resumen_prototipo",
    "etapa1": "res_etapa1_barrido_alfa",
    "etapa2": "res_etapa2_barrido_fino",
    "intervalos": "res_etapa3_intervalos",
    "diferencias": "res_diferencias_alfa",
    "tiempo_real": "res_tiempo_real",
}


def style():
    """Parámetros de matplotlib comunes a todas las figuras (para `matplotlib.rc_context`)."""
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": SMALL,
        "ytick.labelsize": SMALL,
        "legend.fontsize": SMALL,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlepad": 4,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.2,
        "legend.handletextpad": 0.5,
        "lines.solid_capstyle": "round",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.constrained_layout.h_pad": 0.03,
        "figure.constrained_layout.w_pad": 0.03,
    }


def new_figure(width, height):
    """Figura con diseño restringido, para que nada se salga del lienzo."""
    return Figure(figsize=(width, height), layout="constrained")


# ---------------------------------------------------------------------------
# Formato de números (coma decimal)
# ---------------------------------------------------------------------------

def num(value, decimals=1):
    """Número con `decimals` decimales y coma decimal: 12.345 → "12,3"."""
    return f"{value:.{decimals}f}".replace(".", ",")


def gnum(value):
    """Número sin ceros sobrantes, coma decimal y signo menos tipográfico: 0.70 → "0,7", -2.5 → "−2,5"."""
    value = round(value, 6)
    if value == 0:
        value = 0.0  # evita "-0"
    return f"{value:g}".replace(".", ",").replace("-", "\u2212")


def alpha_text(alpha):
    """α con coma decimal: 0.725 → "0,725"."""
    return gnum(alpha)


def range_text(low, high):
    """Intervalo uniforme de α: (0.685, 0.865) → "U(0,685; 0,865)"."""
    return f"U({gnum(low)}; {gnum(high)})"


def tag_label(tag):
    """Rótulo de una etiqueta de configuración ("a080" → "α = 0,8", "u07_09" → "U(0,7; 0,9)")."""
    low, high = parse_tag(tag)
    return f"α = {alpha_text(low)}" if low == high else range_text(low, high)


GNUM_FORMATTER = FuncFormatter(lambda v, _: gnum(v))


# ---------------------------------------------------------------------------
# Lectura de resultados
# ---------------------------------------------------------------------------

@dataclass
class Interval:
    """Valor puntual y su intervalo de confianza, en las unidades de la figura."""
    value: float
    low: float
    high: float

    @classmethod
    def from_json(cls, node, scale=100.0):
        """Desde {"eer"|"value", "ci_low", "ci_high"} en fracciones."""
        value = node["eer"] if "eer" in node else node["value"]
        return cls(scale * value, scale * node["ci_low"], scale * node["ci_high"])

    @classmethod
    def from_row(cls, row, prefix, scale=100.0):
        """Desde las columnas `prefix`, `prefix_ci_low` y `prefix_ci_high` de una tabla; None si falta alguna."""
        values = [row.get(f"{prefix}{s}") for s in ("", "_ci_low", "_ci_high")]
        if any(v in (None, "") for v in values) or any(_is_nan(v) for v in values):
            return None
        return cls(*(scale * float(v) for v in values))

    def excludes_zero(self):
        """Si el intervalo no contiene el 0 (diferencia distinta de cero)."""
        return self.low > 0 or self.high < 0


def _is_nan(value):
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


@dataclass
class Config:
    """Resultados de una configuración de α, en porcentaje."""
    tag: str
    fixed: bool
    alpha_min: float
    alpha_max: float
    eer: dict                       # {escenario: Interval}
    wer: dict = field(default_factory=dict)  # {"original", "anonymized", "difference": Interval}

    @property
    def alpha(self):
        return self.alpha_min if self.fixed else None

    @property
    def nominal(self):
        """Rango nominal de la etiqueta (el α observado en el manifiesto puede no tocar los extremos)."""
        try:
            return parse_tag(self.tag)
        except ValueError:
            return self.alpha_min, self.alpha_max

    @property
    def design(self):
        """Punto medio y ancho nominales, redondeados a 3 decimales."""
        low, high = self.nominal
        return round((low + high) / 2, 3), round(high - low, 3)

    @property
    def label(self):
        return f"α = {alpha_text(self.alpha)}" if self.fixed else range_text(*self.nominal)


def read_json(path):
    """Lee un JSON en UTF-8, con o sin BOM."""
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def read_tsv(path):
    """Lee una tabla TSV como lista de diccionarios de texto."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_configs(data_dir, dataset_tag="k134", whisper_model="turbo", log=print):
    """
    Lee el EER y el WER de todas las configuraciones que tienen manifiesto y JSON del EER.

    El α sale del manifiesto (columna coefficient), igual que en
    `group_analysis.discover_configs`, pero aquí no se exigen los embeddings:
    solo se leen los JSON. Si falta el JSON del WER, la configuración entra
    sin WER.

    Retorna:
        Diccionario {etiqueta: Config}.
    """
    anon_root = Path(data_dir) / f"anon_{dataset_tag}"
    configs = {}
    for manifest in sorted(anon_root.glob("anon_manifest_*.tsv")):
        tag = manifest.name[len("anon_manifest_"):-len(".tsv")]
        files = config_files(tag, data_dir, dataset_tag, whisper_model)
        if not Path(files["eer_json"]).exists():
            log(f"  [{tag}] sin {Path(files['eer_json']).name}: no entra en las figuras")
            continue
        fixed, alpha_min, alpha_max = manifest_alpha(files["manifest"])
        eer = {name: Interval.from_json(node) for name, node in read_json(files["eer_json"])["scenarios"].items()}
        wer = {}
        if Path(files["wer_json"]).exists():
            wer = {key: Interval.from_json(node) for key, node in read_json(files["wer_json"])["wer"].items()}
        else:
            log(f"  [{tag}] sin {Path(files['wer_json']).name}: entra sin WER")
        configs[tag] = Config(tag, fixed, alpha_min, alpha_max, eer, wer)
    return configs


def fixed_configs(configs, window=None, grid=None):
    """
    Configuraciones de α fijo ordenadas por α.

    Parámetros:
        window: (mínimo, máximo) inclusive, o None para todas.
        grid: paso de la rejilla de α (0.1 para la etapa 1); None no filtra.
    """
    out = []
    for c in configs.values():
        if not c.fixed:
            continue
        if window and not (window[0] - 1e-9 <= c.alpha <= window[1] + 1e-9):
            continue
        if grid and abs(c.alpha / grid - round(c.alpha / grid)) > 1e-6:
            continue
        out.append(c)
    return sorted(out, key=lambda c: c.alpha)


def range_configs(configs, reference=REFERENCE_TAG):
    """Los intervalos del diseño 3 × 3 (punto medio × ancho), sin la referencia."""
    out = []
    for c in configs.values():
        if c.fixed or c.tag == reference:
            continue
        midpoint, width = c.design
        if midpoint in MIDPOINTS and width in WIDTHS:
            out.append(c)
    return sorted(out, key=lambda c: c.design)


def baseline(configs, key):
    """
    Métrica que no depende de α (EER o-o o WER original), de la primera configuración que la tenga.

    `key` es ("eer", "o-o") o ("wer", "original").
    """
    group, name = key
    for c in configs.values():
        source = c.eer if group == "eer" else c.wer
        if name in source:
            return source[name]
    return None


@dataclass
class LiveRun:
    """Una corrida del prototipo en vivo."""
    name: str
    rtf: float
    block_ms: np.ndarray
    block_duration_ms: float
    input_latency_ms: float | None
    output_latency_ms: float | None
    algorithmic_ms: float
    cuts: int
    coefficient: float | None

    @property
    def latency_ms(self):
        if self.input_latency_ms is None or self.output_latency_ms is None:
            return None
        return self.input_latency_ms + self.output_latency_ms + self.algorithmic_ms


def load_live_runs(live_dir=None, run_dirs=None, mode="anonimizado"):
    """
    Lee las corridas del prototipo (resumen.json y tiempos.tsv de mcadams_app.live.run).

    Parámetros:
        live_dir: carpeta con una subcarpeta por corrida; se usan todas las
            del modo `mode`, en orden de fecha (el nombre empieza por
            AAAAMMDD_HHMMSS).
        run_dirs: carpetas de corrida explícitas (tienen prioridad).
        mode: modo de las corridas que se usan.

    Retorna:
        Lista de `LiveRun` (vacía si no hay corridas).
    """
    if run_dirs:
        candidates = [Path(p) for p in run_dirs]
    elif live_dir and Path(live_dir).is_dir():
        candidates = sorted(p for p in Path(live_dir).iterdir() if (p / "resumen.json").exists())
    else:
        return []
    runs = []
    for run_dir in candidates:
        summary = read_json(run_dir / "resumen.json")
        config, session = summary["config"], summary["session"]
        if config["mode"] != mode or session.get("rtf") is None:
            continue
        times = np.array([float(r["process_ms"]) for r in read_tsv(run_dir / "tiempos.tsv")])
        latency = summary.get("stream_latency_ms", {})
        block_ms = 1000 * config["blocksize"] / config["sample_rate"]
        delay_ms = float(summary.get("anonymizer", {}).get("delay_ms") or 0.0)
        cuts = sum(int(session.get(k, 0)) for k in ("input_overflows", "input_underflows", "output_overflows",
                                                      "output_underflows"))
        runs.append(LiveRun(run_dir.name, float(session["rtf"]), times, block_ms, latency.get("input"),
                            latency.get("output"), block_ms + delay_ms, cuts,
                            summary.get("anonymizer", {}).get("coefficient")))
    return runs


# ---------------------------------------------------------------------------
# Piezas comunes de las figuras
# ---------------------------------------------------------------------------

def panel_title(ax, letter, text):
    """Título de panel a la izquierda: letra en negrita y descripción."""
    ax.set_title(f"$\\mathbf{{({letter})}}$ {text}", loc="left", color=INK)


def value_grid(ax, axis="y"):
    """Rejilla recesiva de líneas finas continuas en el eje de los valores."""
    ax.grid(axis=axis, which="major", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def nice_limit(top, steps=(1, 2, 2.5, 5, 10)):
    """Límite superior redondo para un eje que empieza en 0."""
    if top <= 0:
        return 1.0
    ticks = MaxNLocator(nbins=5, steps=list(steps)).tick_values(0, top)
    return float(ticks[ticks >= top - 1e-9][0]) if np.any(ticks >= top - 1e-9) else float(top)


def goal_line(ax, value, label, orientation="h", where=0.995, side="above"):
    """Meta como línea punteada con su rótulo junto a un extremo del eje."""
    style = {"color": INK, "linestyle": (0, (1.2, 1.6)), "linewidth": 0.9, "zorder": 2}
    if orientation == "h":
        ax.axhline(value, **style)
        ax.text(where, value, label, transform=ax.get_yaxis_transform(), ha="right",
                va="bottom" if side == "above" else "top", fontsize=TINY + 0.5, color=INK, zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.85})
    else:
        ax.axvline(value, **style)
        ax.text(value, where, label, transform=ax.get_xaxis_transform(), ha="left" if side == "right" else "right",
                va="top", fontsize=TINY + 0.5, color=INK, zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.85})


def reference_band(ax, interval, orientation="h"):
    """Referencia sin anonimizar: línea discontinua gris y franja de su IC."""
    if orientation == "h":
        ax.axhspan(interval.low, interval.high, color=BASE, alpha=0.18, linewidth=0, zorder=0.5)
        ax.axhline(interval.value, color=BASE, linestyle=(0, (4, 2)), linewidth=1.0, zorder=1)
    else:
        ax.axvspan(interval.low, interval.high, color=BASE, alpha=0.18, linewidth=0, zorder=0.5)
        ax.axvline(interval.value, color=BASE, linestyle=(0, (4, 2)), linewidth=1.0, zorder=1)


def reference_handle(label):
    """Entrada de leyenda para `reference_band`."""
    return (Patch(facecolor=BASE, alpha=0.18, edgecolor="none"),
            Line2D([], [], color=BASE, linestyle=(0, (4, 2)), linewidth=1.0)), label


def goal_handle(label="Meta del proyecto"):
    return Line2D([], [], color=INK, linestyle=(0, (1.2, 1.6)), linewidth=0.9), label


def interval_point(ax, x, y, interval, color, marker="o", size=4.0, orientation="v", filled=True, zorder=3,
                   linewidth=0.9):
    """
    Punto con su IC dibujado de extremo a extremo (no como error simétrico,
    porque el IC por percentiles puede ser asimétrico).

    orientation "v": el IC va en y (x es la posición); "h": el IC va en x.
    """
    if orientation == "v":
        ax.plot([x, x], [interval.low, interval.high], color=color, linewidth=linewidth, zorder=zorder - 0.5,
                solid_capstyle="butt")
        px, py = x, interval.value
    else:
        ax.plot([interval.low, interval.high], [y, y], color=color, linewidth=linewidth, zorder=zorder - 0.5,
                solid_capstyle="butt")
        px, py = interval.value, y
    ax.plot(px, py, linestyle="none", marker=marker, markersize=size, color=color,
            markerfacecolor=color if filled else "white", markeredgecolor="white" if filled else color,
            markeredgewidth=0.6 if filled else 0.9, zorder=zorder)


def value_label(ax, interval, y, text):
    """Valor escrito a la derecha del extremo superior de su IC."""
    ax.annotate(text, (max(interval.high, interval.value), y), xytext=(4, 0), textcoords="offset points",
                ha="left", va="center", fontsize=SMALL, color=INK, zorder=6,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.85})


def percent_axis(axis, top=None):
    """Formato de un eje en porcentaje (coma decimal; el % va en el rótulo del eje)."""
    axis.set_major_formatter(GNUM_FORMATTER)
    if top is not None:
        axis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))


# ---------------------------------------------------------------------------
# Figura 0: intervalo del prototipo frente a las metas
# ---------------------------------------------------------------------------

def plot_summary(config, runs=()):
    """
    El intervalo del prototipo frente a las tres metas, una métrica por panel.

    (a) EER de los escenarios o-o, o-a y a-a con su IC; la meta (EER ≥ 30 %)
    se evalúa en o-a. (b) WER original y anonimizada con su IC; meta
    WER < 15 %. (c) RTF de cada ensayo del prototipo; meta RTF < 0,1. El
    panel (c) solo aparece si hay corridas. La zona gris de cada panel es
    donde se cumple la meta.

    Retorna:
        Una `Figure` de matplotlib.
    """
    panels = [("eer", 3), ("wer", 2)] + ([("rtf", len(runs))] if runs else [])
    heights = [n + 1.25 for _, n in panels]
    fig = new_figure(COLUMN_WIDTH, 0.27 * sum(heights) + 0.35)
    axes = fig.subplots(len(panels), 1, gridspec_kw={"height_ratios": heights})

    # (a) EER
    ax = axes[0]
    scenarios = [s for s in ("o-o", "o-a", "a-a") if s in config.eer]
    top = nice_limit(max([EER_GOAL * 1.35] + [config.eer[s].high * 1.12 for s in scenarios]))
    ax.axvspan(EER_GOAL, top, color=GOAL_FILL, linewidth=0, zorder=0)
    goal_line(ax, EER_GOAL, f"Meta o-a ≥ {gnum(EER_GOAL)} %", orientation="v", side="left", where=0.98)
    for i, s in enumerate(scenarios):
        y = len(scenarios) - 1 - i
        st = SCENARIO_STYLE[s]
        interval_point(ax, None, y, config.eer[s], st["color"], st["marker"], size=4.6, orientation="h",
                       filled=s != "o-o", linewidth=1.2)
        value_label(ax, config.eer[s], y, f"{num(config.eer[s].value)} %")
    ax.set_yticks(range(len(scenarios)), [SCENARIO_STYLE[s]["label"] for s in reversed(scenarios)])
    ax.set_ylim(-0.6, len(scenarios) - 0.4)
    ax.set_xlim(0, top)
    ax.xaxis.set_major_formatter(GNUM_FORMATTER)
    ax.set_xlabel("EER (%)")
    value_grid(ax, "x")
    panel_title(ax, "a", "Privacidad")

    # (b) WER
    ax = axes[1]
    rows = [(k, lab) for k, lab in (("original", "Original"), ("anonymized", "Anonimizada")) if k in config.wer]
    top = nice_limit(max([WER_GOAL * 1.35] + [config.wer[k].high * 1.12 for k, _ in rows]))
    ax.axvspan(0, WER_GOAL, color=GOAL_FILL, linewidth=0, zorder=0)
    goal_line(ax, WER_GOAL, f"Meta < {gnum(WER_GOAL)} %", orientation="v", side="right", where=0.98)
    for i, (k, _) in enumerate(rows):
        y = len(rows) - 1 - i
        color = BASE if k == "original" else ANON
        interval_point(ax, None, y, config.wer[k], color, "o", size=4.6, orientation="h",
                       filled=k != "original", linewidth=1.2)
        value_label(ax, config.wer[k], y, f"{num(config.wer[k].value)} %")
    ax.set_yticks(range(len(rows)), [lab for _, lab in reversed(rows)])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, top)
    ax.xaxis.set_major_formatter(GNUM_FORMATTER)
    ax.set_xlabel("WER (%)")
    value_grid(ax, "x")
    panel_title(ax, "b", "Inteligibilidad")

    # (c) RTF
    if runs:
        ax = axes[2]
        top = nice_limit(max([RTF_GOAL * 1.3] + [r.rtf * 1.15 for r in runs]), steps=(1, 2, 2.5, 5, 10))
        ax.axvspan(0, RTF_GOAL, color=GOAL_FILL, linewidth=0, zorder=0)
        goal_line(ax, RTF_GOAL, f"Meta < {gnum(RTF_GOAL)}", orientation="v", side="right", where=0.98)
        for i, run in enumerate(runs):
            y = len(runs) - 1 - i
            ax.plot([0, run.rtf], [y, y], color=ANON, linewidth=1.0, alpha=0.35, solid_capstyle="butt")
            ax.plot(run.rtf, y, "o", color=ANON, markersize=4.6, markeredgecolor="white", markeredgewidth=0.6)
            ax.annotate(num(run.rtf, 3), (run.rtf, y), xytext=(5, 0), textcoords="offset points", ha="left",
                        va="center", fontsize=SMALL, color=INK)
        ax.set_yticks(range(len(runs)), [f"Ensayo {len(runs) - i}" for i in range(len(runs))])
        ax.set_ylim(-0.6, len(runs) - 0.4)
        ax.set_xlim(0, top)
        ax.xaxis.set_major_formatter(GNUM_FORMATTER)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        ax.set_xlabel("RTF (tiempo de procesamiento / duración del audio)")
        value_grid(ax, "x")
        panel_title(ax, "c", "Tiempo real")
    for ax in axes:
        ax.tick_params(axis="y", length=0)
    fig.get_layout_engine().set(hspace=0.06)
    return fig


# ---------------------------------------------------------------------------
# Figuras 1 y 2: barridos con α fijo
# ---------------------------------------------------------------------------

def plot_alpha_sweep(points, eer_oo=None, wer_original=None, zone=None, zone_label=None, fine=False):
    """
    Barrido con α fijo: EER o-a (panel a) y WER anonimizada (panel b) frente a α.

    Cada α lleva su punto y su IC; la referencia sin anonimizar (EER o-o y
    WER original, que no dependen de α) es la línea discontinua gris con la
    franja de su IC, y las metas son las líneas punteadas. El eje x usa el α
    real con su espaciado real.

    Parámetros:
        points: configuraciones de α fijo (`fixed_configs`).
        eer_oo, wer_original: `Interval` de referencia (o None).
        zone: (α mínimo, α máximo) que se sombrea en los dos paneles (la
            zona del barrido fino o el intervalo del prototipo), o None.
        zone_label: rótulo de esa zona.
        fine: True para el barrido fino (marcas cada 0,05 con submarcas
            cada 0,025); False para el grueso (marcas cada 0,1).

    Retorna:
        Una `Figure` de matplotlib.
    """
    fig = new_figure(COLUMN_WIDTH, 4.15)
    ax_eer, ax_wer = fig.subplots(2, 1, sharex=True)
    alphas = [c.alpha for c in points]
    for ax in (ax_eer, ax_wer):
        value_grid(ax, "y")
        if zone:
            ax.axvspan(*zone, color=BAND, linewidth=0, zorder=0)

    # (a) EER o-a
    eer_points = [c for c in points if "o-a" in c.eer]
    if eer_oo:
        reference_band(ax_eer, eer_oo)
    goal_line(ax_eer, EER_GOAL, f"Meta ≥ {gnum(EER_GOAL)} %")
    if eer_points:
        ax_eer.plot([c.alpha for c in eer_points], [c.eer["o-a"].value for c in eer_points], color=ANON,
                    linewidth=1.1, zorder=2)
        for c in eer_points:
            interval_point(ax_eer, c.alpha, None, c.eer["o-a"], ANON, "o", size=4.0)
    eer_top = nice_limit(max([EER_GOAL * 1.25] + [c.eer["o-a"].high * 1.06 for c in eer_points]
                             + ([eer_oo.high] if eer_oo else [])))
    ax_eer.set_ylim(0, eer_top)
    ax_eer.yaxis.set_major_formatter(GNUM_FORMATTER)
    ax_eer.set_ylabel("EER (%)")
    panel_title(ax_eer, "a", "Privacidad: EER, escenario o-a")

    # (b) WER anonimizada
    wer_points = [c for c in points if "anonymized" in c.wer]
    if wer_original:
        reference_band(ax_wer, wer_original)
    goal_line(ax_wer, WER_GOAL, f"Meta < {gnum(WER_GOAL)} %")
    if wer_points:
        ax_wer.plot([c.alpha for c in wer_points], [c.wer["anonymized"].value for c in wer_points], color=ANON,
                    linewidth=1.1, zorder=2)
        for c in wer_points:
            interval_point(ax_wer, c.alpha, None, c.wer["anonymized"], ANON, "o", size=4.0)
    wer_top = nice_limit(max([WER_GOAL * 1.4] + [c.wer["anonymized"].high * 1.06 for c in wer_points]
                             + ([wer_original.high] if wer_original else [])))
    ax_wer.set_ylim(0, wer_top)
    ax_wer.yaxis.set_major_formatter(GNUM_FORMATTER)
    ax_wer.set_ylabel("WER (%)")
    panel_title(ax_wer, "b", "Inteligibilidad: WER")

    if zone and zone_label:
        # El rótulo va en el panel del WER, arriba: ahí la zona casi siempre está vacía.
        ax_wer.text((zone[0] + zone[1]) / 2, 0.97, zone_label, transform=ax_wer.get_xaxis_transform(),
                    ha="center", va="top", fontsize=TINY + 0.5, color=MUTED)

    if alphas:
        low, high = min(alphas), max(alphas)
        if zone:
            low, high = min(low, zone[0]), max(high, zone[1])
        pad = 0.04 * max(high - low, 0.2)
        ax_wer.set_xlim(low - pad, high + pad)
    if fine:
        ax_wer.xaxis.set_major_locator(MultipleLocator(0.05))
        ax_wer.xaxis.set_minor_locator(MultipleLocator(0.025))
    else:
        ax_wer.xaxis.set_major_locator(MultipleLocator(0.1))
    ax_wer.xaxis.set_major_formatter(GNUM_FORMATTER)
    ax_wer.set_xlabel("Coeficiente de McAdams, α")

    handles = [
        (Line2D([], [], color=ANON, linewidth=1.1, marker="o", markersize=4, markeredgecolor="white",
                markeredgewidth=0.6), "Voz anonimizada (IC 95 %)"),
        reference_handle("Voz original"),
        goal_handle(),
    ]
    fig.legend([h for h, _ in handles], [l for _, l in handles], loc="outside upper center", ncol=3,
               handlelength=1.8, columnspacing=1.0)
    fig.align_ylabels([ax_eer, ax_wer])
    return fig


# ---------------------------------------------------------------------------
# Figura 3: los nueve intervalos uniformes (etapa 3)
# ---------------------------------------------------------------------------

BAR_WIDTH = 0.25
BAR_GAP = 0.04


def bar_position(midpoint, width):
    """Posición x de una barra: grupo por punto medio y, dentro, el ancho de menor a mayor."""
    return MIDPOINTS.index(midpoint) + (WIDTHS.index(width) - 1) * (BAR_WIDTH + BAR_GAP)


def _side_labels(ax, items, y_max):
    """
    Rótulos a la derecha del panel, fuera del área de datos, a la altura de su línea.

    Si dos quedan muy cerca, se separan lo justo para que no se monten.

    Parámetros:
        items: lista de (valor y, texto, color, negrita).
        y_max: límite superior del eje (para medir la separación).
    """
    items = sorted(items, key=lambda item: item[0])
    min_gap = 0.13 * y_max
    placed = []
    for value, text, color, bold in items:
        y = value if not placed or value - placed[-1] >= min_gap else placed[-1] + min_gap
        placed.append(y)
        ax.annotate(text, (1.0, value), xycoords=ax.get_yaxis_transform(), xytext=(4, (y - value) / y_max * 120),
                    textcoords="offset points", ha="left", va="center", fontsize=TINY + 0.5, color=color,
                    fontweight="bold" if bold else "normal", annotation_clip=False, linespacing=1.0)


def plot_range_bars(ranges, reference=None, selected_tag=SELECTED_TAG):
    """
    Los nueve intervalos uniformes en barras agrupadas por punto medio.

    Tres paneles apilados que comparten el eje x, de modo que cada intervalo
    queda en la misma columna en todos: (a) EER o-a, (b) EER a-a y (c) WER
    anonimizada. Cada barra es un intervalo U(α mínimo; α máximo): su color
    indica el ancho (más oscuro, más ancho) y debajo se escriben sus dos
    extremos. El bigote es el IC del 95 % y el número encima, el valor
    puntual. La línea roja discontinua es U(0,7; 0,9), el punto de partida, y
    la punteada negra es la meta (EER o-a ≥ 30 % y WER < 15 %; la meta de EER
    no aplica al a-a). La barra del intervalo del prototipo lleva borde negro
    y su rótulo va en negrita.

    Parámetros:
        ranges: intervalos del diseño 3 × 3 (`range_configs`).
        reference: configuración de U(0,7; 0,9), o None.
        selected_tag: etiqueta del intervalo del prototipo.

    Retorna:
        Una `Figure` de matplotlib.
    """
    fig = new_figure(TEXT_WIDTH, 6.1)
    axes = fig.subplots(3, 1, sharex=True)
    panels = (
        ("a", "Privacidad, atacante ignorante: EER o-a (más alto es mejor)", "EER o-a (%)",
         lambda c: c.eer.get("o-a"), EER_GOAL),
        ("b", "Privacidad, atacante semi-informado: EER a-a (más alto es mejor)", "EER a-a (%)",
         lambda c: c.eer.get("a-a"), None),
        ("c", "Inteligibilidad: WER del audio anonimizado (más bajo es mejor)", "WER (%)",
         lambda c: c.wer.get("anonymized"), WER_GOAL),
    )
    bars = sorted(ranges, key=lambda c: bar_position(*c.design))
    cap = BAR_WIDTH * 0.16
    for ax, (letter, title, ylabel, getter, goal) in zip(axes, panels):
        tops = []
        for c in bars:
            interval = getter(c)
            if interval is None:
                continue
            x = bar_position(*c.design)
            selected = c.tag == selected_tag
            ax.bar(x, interval.value, width=BAR_WIDTH, color=WIDTH_COLORS[c.design[1]],
                   edgecolor=INK if selected else "none", linewidth=1.1 if selected else 0, zorder=2)
            ax.vlines(x, interval.low, interval.high, color=INK, linewidth=0.7, zorder=3)
            ax.hlines([interval.low, interval.high], x - cap, x + cap, color=INK, linewidth=0.7, zorder=3)
            top = max(interval.high, interval.value)
            ax.annotate(num(interval.value), (x, top), xytext=(0, 1.5), textcoords="offset points", ha="center",
                        va="bottom", fontsize=TINY + 0.5, color=INK, zorder=6,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.85})
            tops.append(top)
        side = []
        ref = getter(reference) if reference else None
        if ref:
            ax.axhline(ref.value, color=REFERENCE_COLOR, linestyle=(0, (4, 2)), linewidth=1.0, zorder=5)
            side.append((ref.value, f"{range_text(*reference.nominal)}\n{num(ref.value)}", REFERENCE_COLOR, False))
        if goal is not None:
            ax.axhline(goal, color=INK, linestyle=(0, (1.2, 1.6)), linewidth=0.9, zorder=5)
            side.append((goal, f"Meta\n{gnum(goal)}", INK, True))
        y_max = nice_limit(max(tops + [v for v, *_ in side] + [1.0]) * 1.15)
        ax.set_ylim(0, y_max)
        _side_labels(ax, side, y_max)
        ax.yaxis.set_major_formatter(GNUM_FORMATTER)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
        ax.set_ylabel(ylabel)
        value_grid(ax, "y")
        panel_title(ax, letter, title)
        ax.tick_params(axis="x", length=0)

    # Eje x (compartido): los extremos de α bajo cada barra y, debajo, el punto medio de cada grupo.
    bottom = axes[-1]
    bottom.set_xlim(-0.6, len(MIDPOINTS) - 0.4)
    bottom.set_xticks([bar_position(*c.design) for c in bars],
                      [f"{alpha_text(c.nominal[0])}–{alpha_text(c.nominal[1])}" for c in bars], fontsize=TINY + 0.5)
    for tick, c in zip(bottom.get_xticklabels(), bars):
        if c.tag == selected_tag:
            tick.set_fontweight("bold")
    bottom.tick_params(axis="x", pad=2)
    groups = bottom.secondary_xaxis("bottom")
    groups.set_xticks(range(len(MIDPOINTS)), [f"Punto medio {gnum(m)}" for m in MIDPOINTS])
    groups.tick_params(axis="x", length=0, pad=2 + 1.25 * (TINY + 0.5) + 3, labelsize=SMALL)
    groups.spines["bottom"].set_visible(False)
    groups.set_xlabel("Intervalo uniforme de α: α mínimo–α máximo")
    fig.align_ylabels(axes)

    handles = [Patch(facecolor=WIDTH_COLORS[w], edgecolor="none") for w in WIDTHS]
    labels = [f"Ancho {gnum(w)}" for w in WIDTHS]
    handles.append(Line2D([], [], color=INK, linestyle="none", marker="|", markersize=7, markeredgewidth=0.8))
    labels.append("IC 95 %")
    if reference:
        handles.append(Line2D([], [], color=REFERENCE_COLOR, linestyle=(0, (4, 2)), linewidth=1.0))
        labels.append(f"{range_text(*reference.nominal)}, punto de partida")
    handles.append(Line2D([], [], color=INK, linestyle=(0, (1.2, 1.6)), linewidth=0.9))
    labels.append("Meta del proyecto")
    if any(c.tag == selected_tag for c in bars):
        handles.append(Patch(facecolor="white", edgecolor=INK, linewidth=1.1))
        labels.append(f"Intervalo del prototipo, {tag_label(selected_tag)}")
    fig.legend(handles, labels, loc="outside lower center", ncol=4, columnspacing=1.4, handlelength=1.6)
    return fig


# ---------------------------------------------------------------------------
# Figura 4: diferencias pareadas entre α vecinos (bootstrap)
# ---------------------------------------------------------------------------

def _forest_panel(ax, labels, intervals, color=ANON, highlight=None):
    """Panel de diferencias: un renglón por comparación, el 0 como referencia."""
    n = len(labels)
    for i, interval in enumerate(intervals):
        y = n - 1 - i
        if highlight is not None and highlight == i:
            ax.axhspan(y - 0.5, y + 0.5, color=HIGHLIGHT_ROW, linewidth=0, zorder=0)
        if interval is None:
            continue
        interval_point(ax, None, y, interval, color, "o", size=4.2, orientation="h",
                       filled=interval.excludes_zero(), linewidth=1.0)
    ax.axvline(0, color=AXIS, linewidth=0.7, zorder=1)
    ax.set_ylim(-0.6, n - 0.4)
    ax.xaxis.set_major_formatter(GNUM_FORMATTER)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    value_grid(ax, "x")
    ax.tick_params(axis="y", length=0)


def _symmetric_pad(ax, intervals):
    """Margen horizontal para que el 0 y todos los IC quepan."""
    values = [v for i in intervals if i for v in (i.low, i.high)] + [0.0]
    low, high = min(values), max(values)
    pad = 0.08 * max(high - low, 1e-6)
    ax.set_xlim(low - pad, high + pad)


def forest_handles():
    """Leyenda de relleno: IC que excluye o incluye el 0."""
    return ([Line2D([], [], linestyle="none", marker="o", markersize=4.2, color=ANON, markeredgecolor="white"),
             Line2D([], [], linestyle="none", marker="o", markersize=4.2, markerfacecolor="white",
                    markeredgecolor=ANON, markeredgewidth=0.9)],
            ["IC 95 % excluye el 0", "IC 95 % incluye el 0"])


def plot_alpha_differences(rows, min_alpha=DIFFERENCE_MIN_ALPHA):
    """
    Diferencias pareadas entre α vecinos: (a) EER o-a y (b) WER anonimizada.

    La diferencia es la del α mayor menos la del α menor, en puntos
    porcentuales, con su IC por bootstrap pareado (`alpha_differences.py`).
    Un marcador relleno indica que el IC no contiene el 0. Solo entran los
    pares con α ≥ `min_alpha`: por debajo, la WER está dominada por
    alucinaciones del ASR y sus diferencias aplastan la escala.

    Retorna:
        Una `Figure` de matplotlib.
    """
    rows = sorted((r for r in rows if float(r["alpha_from"]) >= min_alpha - 1e-9),
                  key=lambda r: float(r["alpha_from"]))
    labels = [f"{alpha_text(float(r['alpha_from']))} → {alpha_text(float(r['alpha_to']))}" for r in rows]
    eer = [Interval.from_row(_difference_row(r, "eer_oa"), "d") for r in rows]
    wer = [Interval.from_row(_difference_row(r, "wer_anonymized"), "d") for r in rows]
    fig = new_figure(COLUMN_WIDTH, 0.17 * len(rows) + 1.25)
    axes = fig.subplots(1, 2, sharey=True)
    for ax, intervals, letter, title, xlabel in ((axes[0], eer, "a", "EER o-a", "Diferencia (pp)"),
                                                 (axes[1], wer, "b", "WER anonimizada", "Diferencia (pp)")):
        _forest_panel(ax, labels, intervals)
        _symmetric_pad(ax, intervals)
        ax.set_xlabel(xlabel)
        panel_title(ax, letter, title)
    axes[0].set_yticks(range(len(labels)), list(reversed(labels)))
    axes[0].set_ylabel("Par de α vecinos (menor → mayor)")
    handles, labels_ = forest_handles()
    fig.legend(handles, labels_, loc="outside lower center", ncol=2)
    return fig


def _difference_row(row, metric):
    """Columnas de alpha_differences.py renombradas a d, d_ci_low, d_ci_high."""
    return {"d": row.get(f"{metric}_difference"), "d_ci_low": row.get(f"{metric}_ci_low"),
            "d_ci_high": row.get(f"{metric}_ci_high")}


# ---------------------------------------------------------------------------
# Figura 5: prototipo en tiempo real
# ---------------------------------------------------------------------------

def plot_real_time(runs):
    """
    Desempeño del prototipo en vivo, un renglón por ensayo.

    (a) RTF de cada ensayo frente a la meta (RTF < 0,1). (b) Tiempo de
    procesamiento por bloque en escala logarítmica: la línea va del mínimo
    al máximo, la caja del percentil 25 al 75 y se marcan la mediana, el
    percentil 99 y el máximo, frente a la duración de un bloque, que es el
    tiempo disponible para procesarlo. (c) Latencia estimada: latencias de
    entrada y de salida que reporta el dispositivo y retardo algorítmico (un
    bloque más el retraso del adaptador), frente a los 150 ms de la
    Recomendación UIT-T G.114.

    Retorna:
        Una `Figure` de matplotlib.
    """
    n = len(runs)
    fig = new_figure(TEXT_WIDTH, 0.3 * n + 1.45)
    axes = fig.subplots(1, 3, sharey=True, gridspec_kw={"width_ratios": [1, 1.15, 1.15]})
    ys = [n - 1 - i for i in range(n)]
    for ax in axes:
        value_grid(ax, "x")
        ax.tick_params(axis="y", length=0)

    # (a) RTF
    ax = axes[0]
    top = nice_limit(max([RTF_GOAL * 1.3] + [r.rtf * 1.25 for r in runs]))
    ax.axvspan(0, RTF_GOAL, color=GOAL_FILL, linewidth=0, zorder=0)
    goal_line(ax, RTF_GOAL, f"Meta < {gnum(RTF_GOAL)}", orientation="v", side="left", where=0.99)
    for run, y in zip(runs, ys):
        ax.plot([0, run.rtf], [y, y], color=ANON, linewidth=1.0, alpha=0.35, solid_capstyle="butt")
        ax.plot(run.rtf, y, "o", color=ANON, markersize=4.6, markeredgecolor="white", markeredgewidth=0.6)
        ax.annotate(num(run.rtf, 3), (run.rtf, y), xytext=(5, 0), textcoords="offset points", ha="left",
                    va="center", fontsize=SMALL, color=INK)
    ax.set_xlim(0, top)
    ax.xaxis.set_major_formatter(GNUM_FORMATTER)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    ax.set_xlabel("RTF")
    panel_title(ax, "a", "Factor de tiempo real")
    total_cuts = sum(r.cuts for r in runs)
    cut_note = "Sin cortes de audio" if total_cuts == 0 else f"Cortes de audio: {', '.join(str(r.cuts) for r in runs)}"
    ax.text(0.02, 0.015, cut_note, transform=ax.transAxes, ha="left", va="bottom", fontsize=TINY, color=MUTED)

    # (b) Tiempo por bloque
    ax = axes[1]
    ax.set_xscale("log")
    budget = runs[0].block_duration_ms
    lows = []
    for run, y in zip(runs, ys):
        t = run.block_ms[run.block_ms > 0]
        if t.size == 0:
            continue
        p25, p50, p75, p99 = np.percentile(t, [25, 50, 75, 99])
        lo, hi = float(t.min()), float(t.max())
        lows.append(lo)
        ax.plot([lo, hi], [y, y], color=BASE, linewidth=0.8, zorder=1.5, solid_capstyle="butt")
        ax.add_patch(Rectangle((p25, y - 0.17), p75 - p25, 0.34, facecolor="#CFE0F5", edgecolor=ANON,
                               linewidth=0.6, zorder=2))
        ax.plot(p50, y, "o", color=ANON, markersize=4.2, markeredgecolor="white", markeredgewidth=0.6, zorder=3)
        ax.plot(p99, y, "D", markerfacecolor="white", markeredgecolor=ANON, markeredgewidth=0.9, markersize=3.6,
                zorder=3)
        ax.plot(hi, y, "x", color=INK, markersize=4, markeredgewidth=0.9, zorder=3)
    goal_line(ax, budget, f"Duración del bloque ({gnum(budget)} ms)", orientation="v", side="left", where=0.99)
    left = 10 ** math.floor(math.log10(min(lows))) if lows else 0.01
    ax.set_xlim(left, budget * 3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: gnum(v)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Tiempo de procesamiento por bloque (ms, escala log.)")
    panel_title(ax, "b", "Tiempo por bloque")

    # (c) Latencia estimada
    ax = axes[2]
    tops = [LATENCY_REFERENCE_MS]
    for run, y in zip(runs, ys):
        if run.latency_ms is None:
            ax.text(0.02, y, "sin dato del dispositivo", transform=ax.get_yaxis_transform(), va="center",
                    fontsize=TINY, color=MUTED)
            continue
        left = 0.0
        for value, color in zip((run.input_latency_ms, run.output_latency_ms, run.algorithmic_ms), LATENCY_COLORS):
            ax.barh(y, value, left=left, height=0.5, color=color, edgecolor="white", linewidth=0.8, zorder=2)
            left += value
        ax.annotate(f"{num(run.latency_ms, 0)} ms", (left, y), xytext=(4, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=SMALL, color=INK)
        tops.append(left)
    goal_line(ax, LATENCY_REFERENCE_MS, f"UIT-T G.114 ({gnum(LATENCY_REFERENCE_MS)} ms)", orientation="v",
              side="left", where=0.99)
    ax.set_xlim(0, nice_limit(max(tops) * 1.12))
    ax.xaxis.set_major_formatter(GNUM_FORMATTER)
    ax.set_xlabel("Latencia estimada (ms)")
    panel_title(ax, "c", "Latencia")

    axes[0].set_yticks(ys, [f"Ensayo {i + 1}" for i in range(n)])
    axes[0].set_ylim(-0.6, n - 0.4)
    handles = [Line2D([], [], linestyle="none", marker="o", color=ANON, markersize=4.2, markeredgecolor="white"),
               Patch(facecolor="#CFE0F5", edgecolor=ANON, linewidth=0.6),
               Line2D([], [], linestyle="none", marker="D", markerfacecolor="white", markeredgecolor=ANON,
                      markersize=3.6, markeredgewidth=0.9),
               Line2D([], [], linestyle="none", marker="x", color=INK, markersize=4, markeredgewidth=0.9)]
    labels = ["Mediana", "Percentiles 25–75", "Percentil 99", "Máximo"]
    handles += [Patch(facecolor=c, edgecolor="none") for c in LATENCY_COLORS]
    labels += ["Dispositivo: entrada", "Dispositivo: salida", "Algorítmica (bloque + adaptador)"]
    fig.legend(handles, labels, loc="outside lower center", ncol=7, columnspacing=1.0, handlelength=1.3)
    fig.get_layout_engine().set(wspace=0.06)
    return fig


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

CAPTIONS = {
    "resumen": ("figure", "Desempeño del intervalo del prototipo frente a las metas del proyecto: (a) EER en los "
                "escenarios o--o, o--a y a--a; (b) WER del audio original y anonimizado; (c) RTF de cada ensayo "
                "del prototipo. Las barras son intervalos de confianza del 95~\\% por \\textit{bootstrap} y la "
                "zona gris indica dónde se cumple cada meta."),
    "etapa1": ("figure", "Barrido con $\\alpha$ fijo (etapa 1): (a) EER en el escenario o--a y (b) WER del audio "
               "anonimizado. La línea discontinua y la franja gris son el valor sin anonimizar y su intervalo de "
               "confianza del 95~\\%; la zona sombreada es el rango del barrido fino."),
    "etapa2": ("figure", "Barrido fino con $\\alpha$ fijo (etapa 2): (a) EER en el escenario o--a y (b) WER del "
               "audio anonimizado. La zona sombreada corresponde al intervalo usado en el prototipo."),
    "intervalos": ("figure*", "Resultados de los nueve intervalos uniformes de $\\alpha$ (etapa 3), agrupados por "
                   "punto medio: (a) EER o--a, (b) EER a--a y (c) WER del audio anonimizado. Debajo de cada barra "
                   "están los extremos del intervalo, $\\alpha_{\\min}$--$\\alpha_{\\max}$, y el color indica su "
                   "ancho. Los bigotes son intervalos de confianza del 95~\\% por \\textit{bootstrap}; la línea roja "
                   "discontinua es $\\mathcal{U}(0.7,0.9)$ y la punteada negra, la meta del proyecto. El borde negro "
                   "marca el intervalo del prototipo."),
    "diferencias": ("figure", "Diferencias pareadas entre valores vecinos de $\\alpha$ fijo ($\\alpha$ mayor menos "
                    "$\\alpha$ menor), con su intervalo de confianza del 95~\\% por \\textit{bootstrap} pareado: "
                    "(a) EER o--a y (b) WER del audio anonimizado."),
    "tiempo_real": ("figure*", "Desempeño del prototipo en tiempo real en cada ensayo de 60~s: (a) RTF, (b) tiempo "
                    "de procesamiento por bloque frente a la duración del bloque y (c) latencia estimada frente a "
                    "la referencia de 150~ms de la Recomendación UIT-T G.114."),
}


def latex_snippet(names, folder="figuras/resultados"):
    """Código LaTeX para incluir las figuras generadas (una por entorno)."""
    blocks = []
    for key in names:
        env, caption = CAPTIONS[key]
        width = "\\columnwidth" if env == "figure" else "\\textwidth"
        placement = "[!htbp]" if env == "figure" else "[!t]"
        blocks.append(f"\\begin{{{env}}}{placement}\n    \\centering\n"
                      f"    \\includegraphics[width={width}]{{{folder}/{FIGURE_NAMES[key]}.pdf}}\n"
                      f"    \\caption{{{caption}}}\n    \\label{{fig:{FIGURE_NAMES[key]}}}\n\\end{{{env}}}\n")
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Programa
# ---------------------------------------------------------------------------

def save_figure(fig, output_dir, name, formats):
    """Guarda la figura en cada formato pedido y devuelve las rutas."""
    paths = []
    for fmt in formats:
        path = Path(output_dir) / f"{name}.{fmt}"
        fig.savefig(path, dpi=300 if fmt == "png" else None, facecolor="white")
        paths.append(path)
    return paths


def build_figures(args, log=print):
    """
    Arma todas las figuras posibles con los archivos que existan.

    Retorna:
        Lista de (clave, Figure); avisa con `log` las que se saltan.
    """
    data = Path(args.data_dir)
    wanted = set(args.only or FIGURE_NAMES)
    log("Leyendo configuraciones...")
    configs = load_configs(data, args.dataset_tag, args.whisper_model, log)
    runs = load_live_runs(args.live_dir, args.live_runs)
    eer_oo = baseline(configs, ("eer", "o-o"))
    wer_original = baseline(configs, ("wer", "original"))
    selected = configs.get(args.selected)
    reference = configs.get(args.reference)
    out = []

    def skip(key, reason):
        if key in wanted:
            log(f"  Se omite {FIGURE_NAMES[key]}: {reason}")

    if "resumen" in wanted:
        if selected:
            out.append(("resumen", plot_summary(selected, runs)))
        else:
            skip("resumen", f"no hay resultados de {args.selected}")

    coarse = fixed_configs(configs, grid=0.1)
    if "etapa1" in wanted:
        if coarse:
            out.append(("etapa1", plot_alpha_sweep(coarse, eer_oo, wer_original, zone=FINE_WINDOW,
                                                   zone_label="Barrido fino\n(etapa 2)")))
        else:
            skip("etapa1", "no hay configuraciones de α fijo en la rejilla de 0,1")

    fine = fixed_configs(configs, window=FINE_FIGURE_WINDOW)
    if "etapa2" in wanted:
        if fine:
            low, high = parse_tag(args.selected)
            out.append(("etapa2", plot_alpha_sweep(fine, eer_oo, wer_original, zone=(low, high),
                                                   zone_label=f"Intervalo del prototipo\n{range_text(low, high)}",
                                                   fine=True)))
        else:
            skip("etapa2", "no hay configuraciones de α fijo entre 0,65 y 0,9")

    ranges = range_configs(configs, args.reference)
    if "intervalos" in wanted:
        if ranges:
            if reference is None:
                log(f"  {args.reference} no tiene resultados: la figura de intervalos va sin la línea de referencia")
            out.append(("intervalos", plot_range_bars(ranges, reference, args.selected)))
        else:
            skip("intervalos", "no hay intervalos del diseño 3 × 3")

    differences = data / "barrido_alfa_diferencias.tsv"
    if "diferencias" in wanted:
        if differences.exists():
            out.append(("diferencias", plot_alpha_differences(read_tsv(differences))))
        else:
            skip("diferencias", f"no existe {differences.name} (correr alpha_differences.py)")

    if "tiempo_real" in wanted:
        if runs:
            out.append(("tiempo_real", plot_real_time(runs)))
        else:
            skip("tiempo_real", f"no hay corridas en modo anonimizado en {args.live_dir}")
    return out


def parse_args(argv=None):
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description="Figuras de resultados (EER, WER y RTF) a partir de las salidas "
                                                 "del pipeline.")
    parser.add_argument("--data-dir", default=str(REPO.parent / "Datos"))
    parser.add_argument("--output-dir", default=None, help="Por defecto <data-dir>\\figuras\\resultados.")
    parser.add_argument("--dataset-tag", default="k134")
    parser.add_argument("--whisper-model", default="turbo")
    parser.add_argument("--selected", default=SELECTED_TAG, help="Intervalo del prototipo (por defecto u0685_0865).")
    parser.add_argument("--reference", default=REFERENCE_TAG, help="Intervalo de referencia (por defecto u07_09).")
    parser.add_argument("--live-dir", default=None, help="Corridas del prototipo; por defecto <data-dir>\\en_vivo.")
    parser.add_argument("--live-runs", nargs="+", default=None,
                        help="Carpetas de corrida que se usan (por defecto todas las de modo anonimizado).")
    parser.add_argument("--only", nargs="+", choices=sorted(FIGURE_NAMES), default=None,
                        help="Solo estas figuras.")
    parser.add_argument("--formats", nargs="+", choices=("pdf", "png", "svg"), default=["pdf", "png"])
    args = parser.parse_args(argv)
    data = Path(args.data_dir)
    args.output_dir = args.output_dir or str(data / "figuras" / "resultados")
    args.live_dir = args.live_dir or str(data / "en_vivo")
    for tag in (args.selected, args.reference):
        parse_tag(tag)  # falla temprano si la etiqueta no es válida
    return args


def main(argv=None):
    """Genera las figuras, las guarda y escribe el código LaTeX para incluirlas."""
    args = parse_args(argv)
    with matplotlib.rc_context(style()):
        figures = build_figures(args)
        if not figures:
            print("No se generó ninguna figura: revisa --data-dir.")
            return
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for key, fig in figures:
            save_figure(fig, output_dir, FIGURE_NAMES[key], args.formats)
    keys = [key for key, _ in figures]
    (output_dir / "figuras_resultados.tex").write_text(latex_snippet(keys), encoding="utf-8")
    print(f"Figuras en {output_dir}: {', '.join(FIGURE_NAMES[k] for k in keys)}")
    print(f"Código LaTeX: {output_dir / 'figuras_resultados.tex'}")


if __name__ == "__main__":
    main()