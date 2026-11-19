"""Infografía del procesamiento de una trama (Sección 4.3), con datos calculados por mcadams-core.

Uso:  python infografia_trama.py --wav vocal.wav --t 0.6 --alpha 0.775 --out infografia_trama
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
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
    contexto = x[inicio - n:inicio + 2 * n]
    return dict(movs=movs, sr=sr, n=n, h=h, w=w, crudo=x[inicio:inicio + n], s=s, e=e, y=y, ola=ola,
                vecinas=vecinas, f=f, env_o=20 * np.log10(np.abs(ho)), env_t=20 * np.log10(np.abs(ht)),
                contexto=contexto)


def picos(f, env, fmax=5000):
    idx, _ = signal.find_peaks(env, prominence=3)
    return [i for i in idx if f[i] < fmax]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True)
    p.add_argument("--t", type=float, default=0.6)
    p.add_argument("--alpha", type=float, default=0.775)
    p.add_argument("--out", default="infografia_trama")
    args = p.parse_args()
    d = datos(args.wav, args.t, args.alpha)

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "font.size": 7,
                         "pdf.fonttype": 42})
    W, H = 6.85, 3.75
    fig = plt.figure(figsize=(W, H))
    m, cw, ch = 0.004, 0.178, 0.415
    gx = (1 - 2 * m - 5 * cw) / 4
    col = [m + i * (cw + gx) for i in range(5)]
    fila = {"arriba": 0.57, "abajo": 0.035}

    def tarjeta(c, f, num, titulo, sub, fondo=FONDO, alto=ch):
        x0, y0 = col[c], fila[f]
        fig.patches.append(FancyBboxPatch((x0, y0), cw, alto, boxstyle="round,pad=0,rounding_size=0.012",
                                          transform=fig.transFigure, facecolor=fondo, edgecolor=BORDE,
                                          linewidth=0.6, zorder=-2))
        if num is not None:
            cx, cy = x0 + 0.015, y0 + alto - 0.042
            fig.text(cx, cy, str(num), color="white", fontsize=6.2, fontweight="bold",
                     ha="center", va="center", zorder=3,
                     bbox={"boxstyle": "circle,pad=0.22", "facecolor": TINTA2, "edgecolor": "none"})
        fig.text(x0 + (0.029 if num is not None else 0.01), y0 + alto - 0.042, titulo, fontsize=6.9,
                 fontweight="bold", color=TINTA, va="center")
        fig.text(x0 + 0.01, y0 + alto - 0.092, sub, fontsize=5.7, color=TINTA2, va="center")
        ax = fig.add_axes([x0 + 0.012, y0 + 0.03, cw - 0.024, alto - 0.155])
        ax.set_axis_off()
        return ax

    def flecha(xy0, xy1, texto=None, dx=0, dy=0, ha="center", estilo="arc3"):
        fig.patches.append(FancyArrowPatch(xy0, xy1, transform=fig.transFigure, arrowstyle="-|>",
                                           mutation_scale=7, linewidth=0.8, color=TINTA2,
                                           connectionstyle=estilo, zorder=5))
        if texto:
            fig.text((xy0[0] + xy1[0]) / 2 + dx, (xy0[1] + xy1[1]) / 2 + dy, texto, fontsize=5.9,
                     color=TINTA2, ha=ha, va="center", zorder=6,
                     bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})

    n, h, sr = d["n"], d["h"], d["sr"]
    tms = np.arange(n) / sr * 1000

    # 1. Tomar 20 ms
    ax = tarjeta(0, "arriba", 1, "Tomar 20 ms", "trama $s_n$ (320 muestras)")
    c = d["contexto"]
    tc = (np.arange(len(c)) - n) / sr * 1000
    ax.plot(tc, c, color=GRIS, linewidth=0.5)
    ax.plot(tms, d["crudo"], color=AZUL, linewidth=0.9)
    ax.axvspan(0, tms[-1], color=AZUL, alpha=0.08, linewidth=0)
    lim = 1.15 * np.max(np.abs(c))
    ax.set_ylim(-lim, lim)
    ax.set_xlim(tc[0], tc[-1])

    # 2. Suavizar bordes
    ax = tarjeta(1, "arriba", 2, "Suavizar bordes", "ventana de Hann → $s^{\\prime}_n$")
    a_max = np.max(np.abs(d["crudo"]))
    ax.plot(tms, d["crudo"], color=AZUL, linewidth=0.5, alpha=0.35)
    ax.plot(tms, d["w"] * a_max, color=TINTA2, linewidth=0.7, linestyle="--")
    ax.plot(tms, -d["w"] * a_max, color=TINTA2, linewidth=0.7, linestyle="--")
    ax.plot(tms, d["s"], color=AZUL, linewidth=0.9)
    ax.set_ylim(-1.15 * a_max, 1.15 * a_max)

    # 3. Medir la forma de la boca
    ax = tarjeta(2, "arriba", 3, "Forma de la boca", "LPC → $A(z)$  (Ec. 10–11)",
                 fondo=FONDO_ARRIBA)
    fk = d["f"] / 1000
    ax.plot(fk, d["env_o"], color=AZUL, linewidth=1.0)
    fo_k = np.array([fo for fo, _ in d["movs"]]) / 1000
    ax.plot(fo_k, np.interp(fo_k, fk, d["env_o"]), "o", color=AZUL, markersize=2.6)
    lo, hi = min(d["env_o"].min(), d["env_t"].min()), max(d["env_o"].max(), d["env_t"].max())
    ax.set_ylim(lo - 0.32 * (hi - lo), hi + 0.12 * (hi - lo))
    ax.set_xlim(0, XMAX)
    yb = lo - 0.24 * (hi - lo)
    ax.annotate("", xy=(XMAX - 0.25, yb), xytext=(0.25, yb),
                arrowprops={"arrowstyle": "-|>", "color": GRIS, "linewidth": 0.6, "mutation_scale": 5})
    ax.text(0.25, yb + 0.07 * (hi - lo), "grave", fontsize=5.5, color=TINTA2, va="bottom")
    ax.text(XMAX - 0.25, yb + 0.07 * (hi - lo), "agudo", fontsize=5.5, color=TINTA2, va="bottom", ha="right")
    if len(fo_k):
        # El primer polo suele quedar cerca de F0 (pico de la fuente glotal), no es un formante: se señala el siguiente.
        f_lab = fo_k[1] if len(fo_k) > 1 else fo_k[0]
        ax.annotate("formantes", xy=(f_lab, np.interp(f_lab, fk, d["env_o"])), xytext=(0.62 * XMAX, hi),
                    fontsize=5.8, color=TINTA2, va="center",
                    arrowprops={"arrowstyle": "-", "color": GRIS, "linewidth": 0.5})
    lim_env = ax.get_ylim()

    # 4. Mover los formantes
    ax = tarjeta(3, "arriba", 4, "Mover formantes", "$\\varphi \\rightarrow \\varphi^{\\alpha}$ → $A^{\\prime}(z)$  (Ec. 5)",
                 fondo=FONDO_ARRIBA)
    ax.plot(fk, d["env_o"], color=AZUL, linewidth=0.8, alpha=0.55)
    ax.plot(fk, d["env_t"], color=NARANJA, linewidth=1.0)
    fo_k = np.array([fo for fo, _ in d["movs"]]) / 1000
    fn_k = np.array([fn for _, fn in d["movs"]]) / 1000
    ax.plot(fo_k, np.interp(fo_k, fk, d["env_o"]), "o", color=AZUL, markersize=2.2, alpha=0.6)
    ax.plot(fn_k, np.interp(fn_k, fk, d["env_t"]), "o", color=NARANJA, markersize=2.6)
    ax.set_ylim(*lim_env)
    ax.set_xlim(0, XMAX)
    ax.annotate("", xy=(XMAX - 0.25, yb), xytext=(0.25, yb),
                arrowprops={"arrowstyle": "-|>", "color": GRIS, "linewidth": 0.6, "mutation_scale": 5})
    ax.text(0.25, yb + 0.07 * (hi - lo), "grave", fontsize=5.5, color=TINTA2, va="bottom")
    ax.text(XMAX - 0.25, yb + 0.07 * (hi - lo), "agudo", fontsize=5.5, color=TINTA2, va="bottom", ha="right")

    # α de la sesión
    x0, y0 = col[4], fila["arriba"] + 0.12
    fig.patches.append(FancyBboxPatch((x0 + 0.01, y0), cw - 0.02, 0.17, boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.transFigure, facecolor="white", edgecolor=BORDE,
                                      linewidth=0.6, zorder=-2))
    fig.text(x0 + cw / 2, y0 + 0.128, f"$\\alpha$ de la llamada", fontsize=6.9, fontweight="bold",
             color=TINTA, ha="center", va="center")
    fig.text(x0 + cw / 2, y0 + 0.060, "al azar entre\n0.685 y 0.865,\nigual en toda la llamada", fontsize=5.7,
             color=TINTA2, ha="center", va="center", linespacing=1.3)
    flecha((x0 + 0.01, y0 + 0.085), (col[3] + cw, y0 + 0.085))

    # 5. Quitar la forma: queda el zumbido
    ax = tarjeta(2, "abajo", 5, "Quitar la forma", "filtro $A(z)$ → $e_n$  (Ec. 8)",
                 fondo=FONDO_ABAJO)
    fig.text(col[2] + 0.032, fila["abajo"] + ch - 0.045 - 0.0, "", fontsize=7)
    ax.plot(tms, d["e"], color=TINTA2, linewidth=0.8)
    ax.text(0.98, 0.95, "queda el zumbido", transform=ax.transAxes, fontsize=5.8,
            color=TINTA2, ha="right", va="top")
    em = np.max(np.abs(d["e"]))
    ax.set_ylim(-1.15 * em, 1.5 * em)

    # 6. Poner la forma nueva
    ax = tarjeta(3, "abajo", 6, "Forma nueva","filtro $1/A^{\\prime}(z)$ → $y_n$  (Ec. 12)",
                 fondo=FONDO_ABAJO)
    ax.plot(tms, d["s"], color=AZUL, linewidth=0.6, alpha=0.45)
    ax.plot(tms, d["y"], color=NARANJA, linewidth=0.9)
    ym = max(np.max(np.abs(d["y"])), np.max(np.abs(d["s"])))
    ax.set_ylim(-1.15 * ym, 1.15 * ym)

    # 7. Unir las tramas
    ax = tarjeta(4, "abajo", 7, "Unir tramas", "overlap-add")
    to = np.arange(len(d["ola"])) / sr * 1000
    om = np.max(np.abs(d["ola"]))
    for i, yk in enumerate(d["vecinas"]):
        ax.plot((np.arange(n) + i * h) / sr * 1000, d["w"] * om * 1.25 + om * 0.15, color=GRIS,
                linewidth=0.6, linestyle="--")
    ax.plot(to, d["ola"], color=NARANJA, linewidth=0.8)
    ax.set_ylim(-1.15 * om, 1.55 * om)

    # Leyenda (abajo a la izquierda)
    x0, y0 = col[0], fila["abajo"]
    fig.text(x0 + 0.008, y0 + ch - 0.04, "Cómo leer la figura", fontsize=6.9, fontweight="bold", color=TINTA)
    texto = ("Arriba (azul claro): se mide la forma\n"
             "de la boca y se cambia. Los formantes\n"
             "graves suben y los agudos bajan.\n"
             "Abajo (rosado): se separa el zumbido\n"
             "de las cuerdas vocales y se le pone\n"
             "la forma nueva. El zumbido no cambia,\n"
             "por eso el tono de la voz se conserva.")
    fig.text(x0 + 0.008, y0 + ch - 0.075, texto, fontsize=5.9, color=TINTA, va="top", linespacing=1.3)
    for i, (color, lab) in enumerate(((AZUL, "voz original"), (NARANJA, "voz anonimizada"),
                                      (TINTA2, "zumbido (excitación)"))):
        ly = y0 + 0.095 - i * 0.033
        fig.add_artist(plt.Line2D([x0 + 0.01, x0 + 0.035], [ly, ly], transform=fig.transFigure, color=color,
                                  linewidth=1.4))
        fig.text(x0 + 0.04, ly, lab, fontsize=5.8, color=TINTA2, va="center")

    # Flechas entre tarjetas
    ya = fila["arriba"] + ch * 0.42
    yb2 = fila["abajo"] + ch * 0.42
    for c0 in (0, 1, 2):
        flecha((col[c0] + cw, ya), (col[c0 + 1], ya))
    # de 2 (s'_n) hacia 5 (filtro inverso): baja y entra por la izquierda
    xm = col[1] + cw * 0.82
    flecha((xm, fila["arriba"]), (col[2], yb2), estilo="angle,angleA=-90,angleB=180,rad=0")
    fig.text(xm + 0.006, (fila["arriba"] + yb2) / 2 + 0.03, "$s^{\\prime}_n$", fontsize=6.5, color=TINTA2, va="center")
    # de 3 a 5 (forma original) y de 4 a 6 (forma nueva)
    flecha((col[2] + cw * 0.5, fila["arriba"]), (col[2] + cw * 0.5, fila["abajo"] + ch), "forma original $A(z)$")
    flecha((col[3] + cw * 0.5, fila["arriba"]), (col[3] + cw * 0.5, fila["abajo"] + ch), "forma nueva $A^{\\prime}(z)$")
    flecha((col[2] + cw, yb2), (col[3], yb2))
    flecha((col[3] + cw, yb2), (col[4], yb2))

    fig.savefig(f"{args.out}.pdf")
    fig.savefig(f"{args.out}.png", dpi=220)


if __name__ == "__main__":
    main()