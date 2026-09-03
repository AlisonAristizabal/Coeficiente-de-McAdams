import numpy as np

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