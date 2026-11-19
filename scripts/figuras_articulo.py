"""Genera las figuras del artículo a partir de un clip real y del core.

Uso (desde la raíz del repositorio, con mcadams-core instalado):

    python scripts/figuras_articulo.py --wav ruta/al/clip.wav --alpha 0.775 --out figuras

Salidas (PDF vectorial para LaTeX y PNG de vista previa):
    fig_onda_espectrograma  forma de onda y espectrograma, original vs. anonimizado
    fig_zoom_onda           unos pocos ms de una vocal (por defecto 10), original y anonimizado superpuestos
    fig_zoom_onda_paneles   el mismo tramo en dos paneles con los mismos ejes
                            (las líneas discontinuas marcan un periodo de la voz)
    fig_polos_envolvente    polos en el plano z y envolvente LPC de una trama
    fig_curva_mcadams       frecuencia de salida vs. frecuencia de entrada para tres α

El clip debe ser un WAV mono a 16 kHz (los WAV convertidos del pipeline sirven).
Si no se indica --t-trama, se usa la trama con más energía del clip.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from mcadams_core.batch.utterance import anonymize_utterance
from mcadams_core.config.lpc_defaults import compute_lpc_order
from mcadams_core.io.audio_io import load_wav
from mcadams_core.signal_transform.framing import compute_frame_params, hann_window
from mcadams_core.signal_transform.lpc import coeffs_from_poles, compute_lpc, poles_from_coeffs
from mcadams_core.signal_transform.mcadams import transform_poles

# Paleta categórica validada (azul = original, naranja = anonimizado).
C_ORIG = "#2a78d6"
C_ANON = "#eb6834"
C_TERC = "#1baf7a"
C_GRIS = "#8a8984"
C_TEXTO = "#0b0b0b"
C_TEXTO2 = "#52514e"

ANCHO_DOBLE = 6.9   # pulgadas, ancho de página en dos columnas (figure*)
ANCHO_SIMPLE = 3.35  # pulgadas, una columna (figure)


def estilo():
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": C_GRIS,
        "axes.labelcolor": C_TEXTO,
        "xtick.color": C_TEXTO2,
        "ytick.color": C_TEXTO2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def guardar(fig, out, nombre):
    fig.savefig(out / f"{nombre}.pdf")
    fig.savefig(out / f"{nombre}.png", dpi=200)
    plt.close(fig)


def trama_con_ventana(audio, inicio, n):
    trama = audio[inicio:inicio + n]
    if len(trama) < n:
        trama = np.pad(trama, (0, n - len(trama)))
    return trama * hann_window(n)


def elegir_trama(audio, sr, t_trama):
    n, h = compute_frame_params(sr)
    if t_trama is not None:
        return int(round(t_trama * sr / h)) * h, n
    inicios = np.arange(0, max(len(audio) - n, 1), h)
    energias = [np.sum(trama_con_ventana(audio, i, n) ** 2) for i in inicios]
    return int(inicios[int(np.argmax(energias))]), n


def letra(ax, texto):
    ax.text(-0.02, 1.04, texto, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, fontweight="bold", color=C_TEXTO)


def espectrograma(x, sr, ms):
    nper = int(round(sr * ms / 1000))
    f, t, sxx = signal.spectrogram(x, fs=sr, window="hann", nperseg=nper,
                                   noverlap=nper - max(nper // 8, 1), nfft=max(512, nper),
                                   mode="psd")
    return f, t, 10 * np.log10(sxx + 1e-12)


def fig_onda_espectrograma(orig, anon, sr, alpha, inicio, n, ms_espec, out):
    t = np.arange(len(orig)) / sr
    lim = 1.05 * max(np.max(np.abs(orig)), np.max(np.abs(anon)))
    f, te, so = espectrograma(orig, sr, ms_espec)
    _, _, sa = espectrograma(anon, sr, ms_espec)
    vmax = so.max()
    vmin = vmax - 70

    fig, ejes = plt.subplots(2, 2, figsize=(ANCHO_DOBLE, 3.6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.4], "hspace": 0.18, "wspace": 0.08})
    t_trama = (inicio + n / 2) / sr
    titulos = ("Original", f"Anonimizado ($\\alpha$ = {alpha:.3f})")
    for col, (x, s, tit, color) in enumerate(((orig, so, titulos[0], C_ORIG), (anon, sa, titulos[1], C_ANON))):
        ax = ejes[0, col]
        ax.plot(t, x, color=color, linewidth=0.4)
        ax.set_ylim(-lim, lim)
        ax.set_title(tit, color=C_TEXTO)
        ax.axvline(t_trama, color=C_TEXTO2, linewidth=0.6, linestyle="--")
        letra(ax, "(a)" if col == 0 else "(b)")
        ax2 = ejes[1, col]
        im = ax2.pcolormesh(te, f / 1000, s, cmap="Greys", vmin=vmin, vmax=vmax,
                            shading="auto", rasterized=True)
        ax2.grid(False)
        ax2.set_xlabel("Tiempo (s)")
        letra(ax2, "(c)" if col == 0 else "(d)")
        if col == 0:
            ax.set_ylabel("Amplitud")
            ax2.set_ylabel("Frecuencia (kHz)")
        else:
            ax.tick_params(labelleft=False)
            ax2.tick_params(labelleft=False)
    ejes[0, 1].sharey(ejes[0, 0])
    ejes[1, 1].sharey(ejes[1, 0])
    cb = fig.colorbar(im, ax=ejes[1, :], fraction=0.025, pad=0.01)
    cb.set_label("dB", color=C_TEXTO)
    cb.outline.set_visible(False)
    guardar(fig, out, "fig_onda_espectrograma")


def ventana_zoom(orig, sr, inicio, n, ms_zoom):
    """Ventana de ms_zoom que empieza 1 ms antes del pico más alto cerca del centro de la trama,
    para que el zoom arranque al comienzo de un periodo de la voz."""
    centro = inicio + n // 2
    r = int(round(sr * 0.005))
    a0, b0 = max(centro - r, 0), min(centro + r, len(orig))
    pico = a0 + int(np.argmax(np.abs(orig[a0:b0])))
    a = max(pico - int(round(sr * 0.001)), 0)
    b = min(a + int(round(sr * ms_zoom / 1000)), len(orig))
    return a, b


def estimar_periodo(orig, sr, inicio, n, f0_min=60.0, f0_max=400.0):
    """Periodo de la voz (en muestras) por autocorrelación en 40 ms alrededor de la trama."""
    centro = inicio + n // 2
    r = int(round(sr * 0.020))
    seg = orig[max(centro - r, 0):min(centro + r, len(orig))]
    seg = seg - seg.mean()
    ac = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
    lo, hi = int(sr / f0_max), min(int(sr / f0_min), len(ac) - 1)
    return lo + int(np.argmax(ac[lo:hi]))


def marcas_periodo(orig, a, b, periodo):
    """Instantes (en muestras, relativos a a) separados un periodo, empezando en el pico más alto
    del original dentro del primer periodo de la ventana."""
    primero = int(np.argmax(orig[a:min(a + periodo, b)]))
    return np.arange(primero, b - a, periodo)


def fig_zoom(orig, anon, sr, inicio, n, ms_zoom, periodo, out):
    a, b = ventana_zoom(orig, sr, inicio, n, ms_zoom)
    marcas = marcas_periodo(orig, a, b, periodo) / sr * 1000
    t = (np.arange(a, b) - a) / sr * 1000
    lim = 1.08 * max(np.max(np.abs(orig[a:b])), np.max(np.abs(anon[a:b])))

    # Versión superpuesta
    fig, ax = plt.subplots(figsize=(ANCHO_SIMPLE, 1.9))
    ax.grid(axis="x", visible=False)
    for m in marcas:
        ax.axvline(m, color=C_TEXTO2, linewidth=0.7, linestyle="--", alpha=0.7)
    ax.axhline(0, color=C_GRIS, linewidth=0.5)
    ax.plot(t, orig[a:b], color=C_ORIG, linewidth=1.4, label="Original")
    ax.plot(t, anon[a:b], color=C_ANON, linewidth=1.4, label="Anonimizado")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Tiempo (ms)")
    ax.set_ylabel("Amplitud")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.5)
    guardar(fig, out, "fig_zoom_onda")

    # Versión en dos paneles, mismo eje de tiempo y de amplitud
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(ANCHO_SIMPLE, 2.6), sharex=True, sharey=True,
                                   gridspec_kw={"hspace": 0.25})
    for ax, x, color, tit in ((ax1, orig, C_ORIG, "Original"), (ax2, anon, C_ANON, "Anonimizado")):
        ax.grid(axis="x", visible=False)
        for m in marcas:
            ax.axvline(m, color=C_TEXTO2, linewidth=0.7, linestyle="--", alpha=0.7)
        ax.axhline(0, color=C_GRIS, linewidth=0.5)
        ax.plot(t, x[a:b], color=color, linewidth=1.4)
        ax.set_title(tit, loc="left", color=C_TEXTO)
        ax.set_ylabel("Amplitud")
    ax2.set_xlim(0, t[-1])
    ax2.set_ylim(-lim, lim)
    ax2.set_xlabel("Tiempo (ms)")
    guardar(fig, out, "fig_zoom_onda_paneles")
    return a / sr


def fig_polos_envolvente(orig, sr, alpha, inicio, n, out):
    orden = compute_lpc_order(sr)
    trama = trama_con_ventana(orig, inicio, n)
    a = compute_lpc(trama, orden)
    polos = poles_from_coeffs(a)
    polos_t = transform_poles(polos, alpha)
    a_t = coeffs_from_poles(polos_t)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(ANCHO_DOBLE, 2.7),
                                   gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.28})
    th = np.linspace(0, 2 * np.pi, 400)
    ax1.plot(np.cos(th), np.sin(th), color=C_GRIS, linewidth=0.8)
    ax1.axhline(0, color=C_GRIS, linewidth=0.4)
    ax1.axvline(0, color=C_GRIS, linewidth=0.4)
    ax1.plot([0, np.cos(1.0)], [0, np.sin(1.0)], color=C_TEXTO2, linewidth=0.6, linestyle=":")
    ax1.text(1.02 * np.cos(1.0), 1.02 * np.sin(1.0), f"$\\varphi$ = 1 rad\n({sr / (2 * np.pi) / 1000:.2f} kHz)",
             fontsize=6.5, color=C_TEXTO2, ha="left", va="bottom")
    sup = polos.imag > 1e-12
    for p in polos[sup]:
        q = abs(p) * np.exp(1j * np.angle(p) ** alpha)
        ax1.annotate("", xy=(q.real, q.imag), xytext=(p.real, p.imag),
                     arrowprops={"arrowstyle": "->", "color": C_TEXTO2, "linewidth": 0.6,
                                 "shrinkA": 2.5, "shrinkB": 2.5})
    ax1.plot(polos.real, polos.imag, "o", markersize=4.5, markerfacecolor="none",
             markeredgecolor=C_ORIG, markeredgewidth=1.0, linestyle="none", label="Original")
    ax1.plot(polos_t.real, polos_t.imag, "x", markersize=4.5,
             color=C_ANON, markeredgewidth=1.0, linestyle="none", label="Transformado")
    ax1.set_aspect("equal")
    ax1.set_xlim(-1.15, 1.15)
    ax1.set_ylim(-1.15, 1.15)
    ax1.set_xlabel("Parte real")
    ax1.set_ylabel("Parte imaginaria")
    ax1.grid(False)
    letra(ax1, "(a)")

    w, h_o = signal.freqz(1, a, worN=2048, fs=sr)
    _, h_t = signal.freqz(1, a_t, worN=2048, fs=sr)
    ax2.plot(w / 1000, 20 * np.log10(np.abs(h_o) + 1e-12), color=C_ORIG, linewidth=1.2, label="Original")
    ax2.plot(w / 1000, 20 * np.log10(np.abs(h_t) + 1e-12), color=C_ANON, linewidth=1.2,
             label=f"Anonimizado ($\\alpha$ = {alpha:.3f})")
    ax2.axvline(sr / (2 * np.pi) / 1000, color=C_TEXTO2, linewidth=0.6, linestyle=":")
    ax2.set_xlim(0, sr / 2000)
    ax2.set_xlabel("Frecuencia (kHz)")
    ax2.set_ylabel("Magnitud de $1/A(z)$ (dB)")
    letra(ax2, "(b)")
    from matplotlib.lines import Line2D
    manejadores = [
        Line2D([], [], color=C_ORIG, marker="o", markerfacecolor="none", markersize=4.5, linewidth=1.2,
               label="Original"),
        Line2D([], [], color=C_ANON, marker="x", markersize=4.5, linewidth=1.2,
               label=f"Anonimizado ($\\alpha$ = {alpha:.3f})"),
    ]
    fig.legend(handles=manejadores, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    guardar(fig, out, "fig_polos_envolvente")


def fig_curva(sr, alphas, out):
    f = np.linspace(1, sr / 2 - 1, 500)
    phi = 2 * np.pi * f / sr
    fig, ax = plt.subplots(figsize=(ANCHO_SIMPLE, 2.4))
    ax.plot(f / 1000, f / 1000, color=C_GRIS, linewidth=0.8, linestyle="--")
    ax.text(sr / 2000 * 0.97, sr / 2000 * 0.93, "sin cambio", fontsize=6.5, color=C_TEXTO2,
            ha="right", va="top", rotation=0)
    estilos = ("-", "--", "-.")
    for alpha, color, ls in zip(alphas, (C_ORIG, C_ANON, C_TERC), estilos):
        f_t = phi ** alpha * sr / (2 * np.pi)
        ax.plot(f / 1000, f_t / 1000, color=color, linewidth=1.2, linestyle=ls,
                label=f"$\\alpha$ = {alpha:.3f}")
    ax.axvline(sr / (2 * np.pi) / 1000, color=C_TEXTO2, linewidth=0.6, linestyle=":")
    ax.set_xlim(0, sr / 2000)
    ax.set_ylim(0, sr / 2000)
    ax.set_xlabel("Frecuencia original (kHz)")
    ax.set_ylabel("Frecuencia transformada (kHz)")
    ax.legend(loc="lower right")
    guardar(fig, out, "fig_curva_mcadams")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wav", required=True, help="Clip WAV mono a 16 kHz.")
    p.add_argument("--alpha", type=float, default=0.775, help="Coeficiente fijo para la figura.")
    p.add_argument("--out", default="figuras", help="Carpeta de salida.")
    p.add_argument("--t-trama", type=float, default=None,
                   help="Instante (s) de la trama para el zoom y los polos. Por defecto, la de más energía.")
    p.add_argument("--zoom-ms", type=float, default=10.0, help="Duración de la ventana del zoom (ms).")
    p.add_argument("--periodo-ms", type=float, default=None,
                   help="Periodo de la voz (ms) para las líneas del zoom. Por defecto se estima del audio.")
    p.add_argument("--espec-ms", type=float, default=8.0,
                   help="Ventana del espectrograma (ms). Corta = se ven los formantes.")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sr, orig = load_wav(args.wav)
    if sr != 16000:
        print(f"Aviso: el clip está a {sr} Hz; el pipeline trabaja a 16 kHz.")
    anon, alpha = anonymize_utterance(orig, sr, "figura", alpha_min=args.alpha, alpha_max=args.alpha)
    inicio, n = elegir_trama(orig, sr, args.t_trama)
    print(f"alpha = {alpha:.3f}; trama en {inicio / sr:.3f} s ({n} muestras)")

    estilo()
    fig_onda_espectrograma(orig, anon, sr, alpha, inicio, n, args.espec_ms, out)
    if args.periodo_ms:
        periodo = int(round(args.periodo_ms * sr / 1000))
    else:
        periodo = estimar_periodo(orig, sr, inicio, n)
    print(f"periodo de la voz: {periodo / sr * 1000:.2f} ms (F0 ≈ {sr / periodo:.0f} Hz)")
    t_zoom = fig_zoom(orig, anon, sr, inicio, n, args.zoom_ms, periodo, out)
    print(f"zoom: {args.zoom_ms:.0f} ms desde {t_zoom:.4f} s")
    fig_polos_envolvente(orig, sr, alpha, inicio, n, out)
    fig_curva(sr, (0.685, 0.775, 0.865), out)
    print(f"Figuras guardadas en {out.resolve()}")


if __name__ == "__main__":
    main()