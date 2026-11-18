"""Extrae los 20 coeficientes LPC de una trama real para la figura del póster.

Toma un WAV a 16 kHz, busca la trama de 20 ms con más energía (normalmente
una vocal), le aplica la misma ventana de Hann periódica que usa el core y
llama a compute_lpc del propio core. Imprime los coeficientes y los guarda
en un CSV junto al WAV.

Uso:
    python extraer_coeficientes_lpc.py RUTA_AL_WAV
"""

import csv
import inspect
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal.windows import hann

from mcadams_core.signal_transform.lpc import compute_lpc

ORDER = 20
FRAME_MS = 20
HOP_MS = 10


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Uso: python extraer_coeficientes_lpc.py RUTA_AL_WAV")
    wav_path = Path(sys.argv[1])

    signal, sample_rate = sf.read(wav_path, dtype="float64")
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if sample_rate != 16000:
        sys.exit(f"El WAV está a {sample_rate} Hz; usa uno a 16 kHz.")

    frame_len = sample_rate * FRAME_MS // 1000
    hop = sample_rate * HOP_MS // 1000
    window = hann(frame_len, sym=False)

    # Trama con mayor energía: casi siempre cae en una vocal.
    starts = range(0, len(signal) - frame_len + 1, hop)
    energies = [np.sum(signal[s:s + frame_len] ** 2) for s in starts]
    best = list(starts)[int(np.argmax(energies))]
    frame = signal[best:best + frame_len] * window

    print("Firma de compute_lpc:", inspect.signature(compute_lpc))
    try:
        coeffs = compute_lpc(frame, ORDER)
    except TypeError:
        coeffs = compute_lpc(frame, order=ORDER)
    coeffs = np.asarray(coeffs)

    print(f"Archivo: {wav_path.name}")
    print(f"Trama: muestra {best} (t = {best / sample_rate:.3f} s)")
    print(f"Devuelto: {len(coeffs)} valores (se espera [1, a1, ..., a{ORDER}])")
    for k, value in enumerate(coeffs[1:], start=1):
        print(f"a{k:<2} = {value: .4f}")

    out_path = wav_path.with_name(wav_path.stem + "_lpc_poster.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["k", "a_k"])
        for k, value in enumerate(coeffs[1:], start=1):
            writer.writerow([k, f"{value:.6f}"])
    print(f"Guardado en: {out_path}")


if __name__ == "__main__":
    main()