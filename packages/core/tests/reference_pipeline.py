"""Implementación de referencia del enventanado y del overlap-add, solo para
pruebas.

Contiene frame_signal y overlap_add, que antes vivían en
mcadams_core.signal_transform. Es una implementación directa y no
incremental: segmenta toda la señal de una vez y la reconstruye sumando
todas las tramas al final. Se usa únicamente para verificar
AnonymizerStream, que hace lo mismo de forma incremental. No forma parte del
programa: ni los lotes ni el modo en vivo la usan.

El nombre del archivo no empieza por test_, para que pytest no lo recolecte
como archivo de pruebas. Las pruebas lo importan como
`from reference_pipeline import ...` porque packages/core/tests está en el
pythonpath de pytest (ver el pyproject.toml raíz)."""

import numpy as np


def frame_signal(audio_float, frame_length, hop_length):

    """Divide audio_float (1D) en fragmentos superpuestos usando
    np.lib.stride_tricks.sliding_window_view: se generan todas las
    ventanas de longitud frame_length (una por cada muestra inicial) y
    luego se toma una de cada hop_length. Devuelve un arreglo de forma
    (numero_fragmentos, longitud_frame), que es una vista de solo
    lectura sobre audio_float (no copia los datos)."""

    frames = np.lib.stride_tricks.sliding_window_view(audio_float, frame_length)[::hop_length]

    return frames


def overlap_add(frames, hop_length):
    """Reconstruye una señal 1D a partir de fragmentos superpuestos,
    sumándolos en sus posiciones correspondientes según el hop_length.
    frames tiene forma (n_frames, frame_length) y se asume que ya viene
    enventanado (por ejemplo, con apply_window), usando una ventana que
    satisface COLA para este hop_length — por eso no se aplica ninguna
    normalización adicional aquí.

    Args:
        frames: arreglo de forma (n_frames, frame_length), ya enventanado.
        hop_length: número de muestras entre el inicio de fragmentos
            consecutivos.

    Returns:
        Arreglo 1D de longitud (n_frames - 1) * hop_length + frame_length.
    """

    n_frames, frame_length = frames.shape
    output_length = (n_frames - 1) * hop_length + frame_length

    output = np.zeros(output_length)

    for i in range(n_frames):
        output[i * hop_length:i * hop_length + frame_length] += frames[i]

    return output
