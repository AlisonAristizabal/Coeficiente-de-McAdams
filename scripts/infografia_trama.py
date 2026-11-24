"""Infografía del procesamiento de una trama (Sección 4.3), con datos calculados por mcadams-core.

Versión 4 (comentarios del asesor):
  - el nombre de cada etapa empieza con un verbo;
  - cada flecha dice con palabras lo que sale de una etapa y entra en la siguiente,
    y su color indica si es voz original (azul), voz anonimizada (naranja) o excitación (gris);
  - la etapa 1 muestra un tramo de la voz con la trama de 20 ms resaltada y ampliada debajo;
  - letra más grande y fuente tipo Times, como la del artículo;
  - cuatro columnas: las etapas 5 a 7 van una columna a la izquierda, y las tarjetas son más anchas.

Uso:  python infografia_trama.py --wav clip.wav --t 0.6 --alpha 0.775 --out infografia_trama
En el artículo conviene incluir el PDF (vectorial), no el PNG.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, FancyBboxPatch
from scipy import signal

from mcadams_core.config.lpc_defaults import compute_lpc_order
from mcadams_core.io.audio_io import load_wav
from mcadams_core.signal_transform.framing import compute_frame_params, hann_window
from mcadams_core.signal_transform.lpc import coeffs_from_poles, compute_lpc, poles_from_coeffs
from mcadams_core.signal_transform.mcadams import transform_poles
from mcadams_core.signal_transform.resynth import get_excitation, resynthesize_signal

AZUL, NARANJA = "#2a78d6", "#eb6834"
XMAX = 5.0  # kHz mostrados en las tarjetas 3 y 4
TINTA, TINTA2, GRIS = "#0b0b0b", "#52514e", "#8a8984"
FONDO, BORDE = "#f7f7f5", "#d9d8d2"
FONDO_ARRIBA, FONDO_ABAJO = "#eef4fb", "#fbf1ed"

# Tamaños de letra (pt). La figura ocupa el ancho de la página (\textwidth).
F_TIT, F_SUB, F_FLE, F_ANOT, F_LEY = 7.8, 6.1, 6.6, 6.3, 7.0
CONTEXTO_S = 0.20  # segundos de voz que se muestran en la etapa 1


def datos(path, t, alpha):
    sr, x = load_wav(path)
    n, h = compute_frame_params(sr)
    orden = compute_lpc_order(sr)
    w = hann_window(n)
    k0 = int(round(t * sr / h))
    inicio = k0 * h

    def procesa(k):
        s = x[k * h:k * h + n] * w
        a = compute_lpc(s, orden)
        a_t = coeffs_from_poles(transform_poles(poles_from_coeffs(a), alpha))
        e = get_excitation(s, a)
        return s, a, a_t, e, resynthesize_signal(e, a_t)

    s, a, a_t, e, y = procesa(k0)
    polos = poles_from_coeffs(a)
    sup = polos[(polos.imag > 0) & (np.abs(polos) > 0.85)]
    sup = sup[np.argsort(-np.abs(sup))]
    f_o = np.angle(sup) * sr / (2 * np.pi)
    f_n = np.angle(sup) ** alpha * sr / (2 * np.pi)
    sel = f_o < 5500
    movs = sorted(zip(f_o[sel][:4], f_n[sel][:4]))
    # tres tramas vecinas para ilustrar el overlap-add
    vecinas = [procesa(k)[4] for k in (k0 - 1, k0, k0 + 1)]
    ola = np.zeros(n + 2 * h)
    for i, yk in enumerate(vecinas):
        ola[i * h:i * h + n] += yk
    f, ho = signal.freqz(1, a, worN=1024, fs=sr)
    _, ht = signal.freqz(1, a_t, worN=1024, fs=sr)
    # tramo de voz alrededor de la trama, para la etapa 1
    m = int(CONTEXTO_S * sr / 2)
    c0, c1 = max(0, inicio + n // 2 - m), min(len(x), inicio + n // 2 + m)
    return dict(movs=movs, sr=sr, n=n, h=h, w=w, crudo=x[inicio:inicio + n], s=s, e=e, y=y, ola=ola,
                vecinas=vecinas, f=f, env_o=20 * np.log10(np.abs(ho)), env_t=20 * np.log10(np.abs(ht)),
                contexto=x[c0:c1], t_contexto=(np.arange(c0, c1) - inicio) / sr * 1000)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True)
    p.add_argument("--t", type=float, default=0.6)
    p.add_argument("--alpha", type=float, default=0.775)
    p.add_argument("--out", default="infografia_trama")
    args = p.parse_args()
    d = datos(args.wav, args.t, args.alpha)

    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": F_SUB, "pdf.fonttype": 42})
    W, H = 6.85, 4.1
    fig = plt.figure(figsize=(W, H))
    iy = 1 / H                                  # una pulgada en fracción de la altura
    ml, mr, gx = 0.062, 0.09, 0.066             # márgenes para la entrada y la salida; espacio entre tarjetas
    cw = (1 - ml - mr - 3 * gx) / 4             # cuatro columnas: tarjetas más anchas
    col = [ml + i * (cw + gx) for i in range(4)]
    ch = 1.55 * iy                              # alto de cada tarjeta (1.55 in)
    fila = {"arriba": 1 - 0.03 * iy - ch, "abajo": 0.03 * iy}

    def tarjeta(c, f, num, titulo, sub, fondo=FONDO, ejes=True):
        x0, y0 = col[c], fila[f]
        top = y0 + ch
        fig.patches.append(FancyBboxPatch((x0, y0), cw, ch, boxstyle="round,pad=0,rounding_size=0.012",
                                          transform=fig.transFigure, facecolor=fondo, edgecolor=BORDE,
                                          linewidth=0.6, zorder=-2))
        fig.text(x0 + 0.014, top - 0.15 * iy, str(num), color="white", fontsize=F_TIT - 1.2, fontweight="bold",
                 ha="center", va="center", zorder=3,
                 bbox={"boxstyle": "circle,pad=0.25", "facecolor": TINTA2, "edgecolor": "none"})
        fig.text(x0 + 0.028, top - 0.07 * iy, titulo, fontsize=F_TIT, fontweight="bold", color=TINTA,
                 va="top", linespacing=1.05)
        fig.text(x0 + 0.008, top - 0.46 * iy, sub, fontsize=F_SUB, color=TINTA2, va="center")
        if not ejes:
            return x0, y0
        ax = fig.add_axes([x0 + 0.01, y0 + 0.07 * iy, cw - 0.02, ch - 0.64 * iy])
        ax.set_axis_off()
        return ax

    def flecha(xy0, xy1, color=TINTA2, estilo="arc3"):
        fig.patches.append(FancyArrowPatch(xy0, xy1, transform=fig.transFigure, arrowstyle="-|>",
                                           mutation_scale=9, linewidth=1.3, color=color,
                                           connectionstyle=estilo, zorder=5))

    def codo(puntos, color):
        """Flecha con quiebres: línea por los puntos y punta en el último tramo."""
        xs, ys = zip(*puntos)
        fig.add_artist(plt.Line2D(xs[:-1], ys[:-1], transform=fig.transFigure, color=color, linewidth=1.3,
                                  solid_joinstyle="miter", zorder=5))
        flecha(puntos[-2], puntos[-1], color)

    def rotulo(x, y, texto, color=TINTA2, ha="center", va="bottom", size=F_FLE):
        fig.text(x, y, texto, fontsize=size, color=color, ha=ha, va=va, zorder=6, linespacing=1.0,
                 fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.9})

    n, h, sr = d["n"], d["h"], d["sr"]
    tms = np.arange(n) / sr * 1000

    # 1. Tomar una trama: tramo de voz arriba, trama ampliada abajo
    x0, y0 = tarjeta(0, "arriba", 1, "Tomar\nuna trama", "20 ms de voz", ejes=False)
    alto = ch - 0.64 * iy
    ax_v = fig.add_axes([x0 + 0.012, y0 + 0.07 * iy + alto * 0.60, cw - 0.024, alto * 0.38])
    ax_t = fig.add_axes([x0 + 0.012, y0 + 0.07 * iy + alto * 0.12, cw - 0.024, alto * 0.34])
    for ax in (ax_v, ax_t):
        ax.set_axis_off()
    c, tc = d["contexto"], d["t_contexto"]
    # silueta de la voz: amplitud máxima en ventanas de 2 ms (se lee mejor que la onda completa)
    paso = max(1, int(0.002 * sr))
    k = len(c) // paso
    env = np.abs(c[:k * paso]).reshape(k, paso).max(axis=1)
    te = tc[:k * paso].reshape(k, paso).mean(axis=1)
    ax_v.fill_between(te, -env, env, color=GRIS, alpha=0.55, linewidth=0)
    ax_v.axvspan(0, tms[-1], color=AZUL, alpha=0.35, linewidth=0)
    lim_v = 1.05 * np.max(np.abs(c))
    ax_v.set_ylim(-lim_v, lim_v)
    ax_v.set_xlim(tc[0], tc[-1])
    ax_t.plot(tms, d["crudo"], color=AZUL, linewidth=1.1)
    lim_t = 1.1 * np.max(np.abs(d["crudo"]))
    ax_t.set_ylim(-lim_t, lim_t)
    ax_t.set_xlim(0, tms[-1])
    for xa, xb in ((0, 0), (tms[-1], tms[-1])):
        fig.add_artist(ConnectionPatch(xyA=(xa, -lim_v), coordsA=ax_v.transData, xyB=(xb, lim_t),
                                       coordsB=ax_t.transData, color=AZUL, linewidth=0.6, linestyle="--",
                                       alpha=0.8))
    ax_t.text(tms[-1] / 2, -lim_t, "trama ampliada", fontsize=F_ANOT, color=AZUL, va="top", ha="center")

    # 2. Suavizar los bordes
    ax = tarjeta(1, "arriba", 2, "Suavizar\nlos bordes", "ventana de Hann")
    a_max = np.max(np.abs(d["crudo"]))
    ax.plot(tms, d["crudo"], color=AZUL, linewidth=0.6, alpha=0.3)
    ax.plot(tms, d["w"] * a_max, color=TINTA2, linewidth=0.8, linestyle="--")
    ax.plot(tms, -d["w"] * a_max, color=TINTA2, linewidth=0.8, linestyle="--")
    ax.plot(tms, d["s"], color=AZUL, linewidth=1.0)
    ax.set_ylim(-1.15 * a_max, 1.15 * a_max)

    # 3. Identificar los formantes
    ax = tarjeta(2, "arriba", 3, "Identificar\nformantes", "LPC (Ec. 10–11)", fondo=FONDO_ARRIBA)
    fk = d["f"] / 1000
    ax.plot(fk, d["env_o"], color=AZUL, linewidth=1.1)
    fo_k = np.array([fo for fo, _ in d["movs"]]) / 1000
    fn_k = np.array([fn for _, fn in d["movs"]]) / 1000
    ax.plot(fo_k, np.interp(fo_k, fk, d["env_o"]), "o", color=AZUL, markersize=3.2)
    lo, hi = min(d["env_o"].min(), d["env_t"].min()), max(d["env_o"].max(), d["env_t"].max())
    ax.set_ylim(lo - 0.36 * (hi - lo), hi + 0.18 * (hi - lo))
    ax.set_xlim(0, XMAX)
    yb = lo - 0.27 * (hi - lo)

    def eje_frecuencia(ax):
        ax.annotate("", xy=(XMAX - 0.2, yb), xytext=(0.2, yb),
                    arrowprops={"arrowstyle": "-|>", "color": GRIS, "linewidth": 0.7, "mutation_scale": 6})
        ax.text(0.2, yb + 0.06 * (hi - lo), "grave", fontsize=F_ANOT, color=TINTA2, va="bottom")
        ax.text(XMAX - 0.2, yb + 0.06 * (hi - lo), "agudo", fontsize=F_ANOT, color=TINTA2, va="bottom",
                ha="right")

    eje_frecuencia(ax)
    ax.text(XMAX * 0.97, hi + 0.12 * (hi - lo), "picos = formantes", fontsize=F_ANOT, color=AZUL,
            ha="right", va="center")
    lim_env = ax.get_ylim()

    # 4. Mover los formantes: flechas de la posición original a la nueva
    ax = tarjeta(3, "arriba", 4, "Mover\nformantes", "según $\\alpha$ (Ec. 5)", fondo=FONDO_ARRIBA)
    ax.plot(fk, d["env_o"], color=AZUL, linewidth=0.9, alpha=0.45)
    ax.plot(fk, d["env_t"], color=NARANJA, linewidth=1.1)
    yo = np.interp(fo_k, fk, d["env_o"])
    yn = np.interp(fn_k, fk, d["env_t"])
    ax.plot(fo_k, yo, "o", color=AZUL, markersize=2.6, alpha=0.6)
    ax.plot(fn_k, yn, "o", color=NARANJA, markersize=3.2)
    for a_, b_, ya_, yb_ in zip(fo_k, fn_k, yo, yn):
        if abs(b_ - a_) > 0.08:
            ax.annotate("", xy=(b_, yb_ + 0.05 * (hi - lo)), xytext=(a_, ya_ + 0.05 * (hi - lo)),
                        arrowprops={"arrowstyle": "-|>", "color": NARANJA, "linewidth": 0.8,
                                    "mutation_scale": 6, "shrinkA": 1, "shrinkB": 1})
    ax.set_ylim(*lim_env)
    ax.set_xlim(0, XMAX)
    eje_frecuencia(ax)

    # 5, 6 y 7: fila de abajo, una columna más a la izquierda
    ax = tarjeta(1, "abajo", 5, "Extraer la\nexcitación", "filtro inverso (Ec. 8)", fondo=FONDO_ABAJO)
    ax.plot(tms, d["e"], color=TINTA2, linewidth=0.9)
    em = np.max(np.abs(d["e"]))
    ax.set_ylim(-1.15 * em, 1.25 * em)

    ax = tarjeta(2, "abajo", 6, "Reconstruir\nla voz", "síntesis (Ec. 12)", fondo=FONDO_ABAJO)
    ax.plot(tms, d["s"], color=AZUL, linewidth=0.7, alpha=0.4)
    ax.plot(tms, d["y"], color=NARANJA, linewidth=1.0)
    ym = max(np.max(np.abs(d["y"])), np.max(np.abs(d["s"])))
    ax.set_ylim(-1.15 * ym, 1.15 * ym)

    ax = tarjeta(3, "abajo", 7, "Unir\nlas tramas", "overlap-add")
    to = np.arange(len(d["ola"])) / sr * 1000
    om = np.max(np.abs(d["ola"]))
    for i in range(len(d["vecinas"])):
        ax.plot((np.arange(n) + i * h) / sr * 1000, d["w"] * om * 1.25 + om * 0.15, color=GRIS,
                linewidth=0.7, linestyle="--")
    ax.plot(to, d["ola"], color=NARANJA, linewidth=0.9)
    ax.set_ylim(-1.15 * om, 1.55 * om)

    # Leyenda de colores (abajo a la izquierda)
    x0 = ml - 0.045
    for i, (color, lab) in enumerate(((AZUL, "voz original"), (NARANJA, "voz anonimizada"),
                                      (TINTA2, "excitación"))):
        ly = fila["abajo"] + ch * 0.62 - i * 0.2 * iy
        fig.add_artist(plt.Line2D([x0 + 0.002, x0 + 0.03], [ly, ly], transform=fig.transFigure, color=color,
                                  linewidth=1.8))
        fig.text(x0 + 0.037, ly, lab, fontsize=F_LEY, color=TINTA2, va="center")

    # Flechas: con palabras y con el color de lo que llevan
    ya = fila["arriba"] + ch * 0.40
    yb2 = fila["abajo"] + ch * 0.40
    sep = 0.025 * iy
    base_arr, techo_ab = fila["arriba"], fila["abajo"] + ch
    y_codo = (base_arr + techo_ab) / 2

    # entrada y salida del sistema
    flecha((0.003, ya), (col[0], ya), AZUL)
    rotulo(col[0] / 2, ya + sep, "voz\noriginal", AZUL)
    flecha((col[3] + cw, yb2), (0.997, yb2), NARANJA)
    rotulo((col[3] + cw + 1) / 2, yb2 + sep, "voz\nanonimizada", NARANJA)

    # fila de arriba
    for c0, lab in ((0, "trama"), (1, "trama\nsuavizada"), (2, "formantes")):
        flecha((col[c0] + cw, ya), (col[c0 + 1], ya), AZUL)
        rotulo(col[c0] + cw + gx / 2, ya + sep, lab, AZUL)
    # fila de abajo
    flecha((col[1] + cw, yb2), (col[2], yb2), TINTA2)
    rotulo(col[1] + cw + gx / 2, yb2 + sep, "excitación", TINTA2, size=F_FLE - 0.4)
    flecha((col[2] + cw, yb2), (col[3], yb2), NARANJA)
    rotulo(col[2] + cw + gx / 2, yb2 + sep, "trama\nnueva", NARANJA)

    # de 2 a 5: baja derecho
    x25 = col[1] + cw * 0.28
    flecha((x25, base_arr), (x25, techo_ab), AZUL)
    rotulo(x25 + 0.006, y_codo, "trama\nsuavizada", AZUL, ha="left", va="center")
    # de 3 a 5 (forma original) y de 4 a 6 (forma nueva): bajan, giran a la izquierda y entran por arriba
    x3, x35 = col[2] + cw * 0.5, col[1] + cw * 0.78
    codo([(x3, base_arr), (x3, y_codo), (x35, y_codo), (x35, techo_ab)], AZUL)
    rotulo((x3 + x35) / 2 + 0.01, y_codo + sep, "forma original", AZUL)
    x4, x46 = col[3] + cw * 0.5, col[2] + cw * 0.78
    codo([(x4, base_arr), (x4, y_codo), (x46, y_codo), (x46, techo_ab)], NARANJA)
    rotulo((x4 + x46) / 2 + 0.01, y_codo + sep, "forma nueva", NARANJA)

    # α de la llamada: entre las dos filas, a la derecha, con flecha hacia la etapa 4
    xa0, xa1 = col[3] + cw * 0.58, 0.997
    ya0, ya1 = techo_ab + 0.10 * iy, base_arr - 0.24 * iy
    fig.patches.append(FancyBboxPatch((xa0, ya0), xa1 - xa0, ya1 - ya0,
                                      boxstyle="round,pad=0,rounding_size=0.01", transform=fig.transFigure,
                                      facecolor="white", edgecolor=BORDE, linewidth=0.6, zorder=-2))
    fig.text((xa0 + xa1) / 2, ya1 - 0.1 * iy, "$\\alpha$ de la llamada", fontsize=F_SUB + 0.6,
             fontweight="bold", color=TINTA, ha="center", va="center")
    fig.text((xa0 + xa1) / 2, ya1 - 0.2 * iy, "al azar entre 0.685\ny 0.865, igual en\ntoda la llamada",
             fontsize=F_SUB, color=TINTA2, ha="center", va="top", linespacing=1.1)
    xa = xa0 + 0.03
    flecha((xa, ya1), (xa, base_arr), TINTA2)

    fig.savefig(f"{args.out}.pdf")
    fig.savefig(f"{args.out}.png", dpi=300)


if __name__ == "__main__":
    main()