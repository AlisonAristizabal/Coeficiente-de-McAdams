import scipy.signal.windows as windows

def compute_frame_params(sample_rate, win_length_ms=20, hop_length_ms=10):
    """
    Calcula la longitud de ventana y el salto (hop) en muestras a partir
    de la frecuencia de muestreo y los valores en milisegundos.

    Args:
        sample_rate (int): Frecuencia de muestreo del audio.
        win_length_ms (float): Longitud de la ventana en milisegundos.
        hop_length_ms (float): Longitud del salto (hop) en milisegundos.

    Returns:
        tuple: Una tupla que contiene la longitud de la ventana y el salto
               en muestras (frame_length, hop_length).
    """
    frame_length = round(sample_rate * win_length_ms / 1000)
    hop_length = round(sample_rate * hop_length_ms / 1000)

    return frame_length, hop_length

def hann_window(frame_length):

    """Genera una ventana Hanning periódica (sym=False) de longitud
    frame_length, usando scipy.signal.windows.hann. Necesaria para que
    el overlap-add posterior satisfaga COLA a 50% de superposición."""

    hanning = windows.hann(frame_length, sym=False)

    return hanning

def apply_window(frames, window):

    """Multiplica cada fragmento por la ventana. fragmentos tiene forma
    (numero_fragmentos, longitud_frame); ventana tiene forma
    (longitud_frame,) se aplica a cada fila mediante broadcasting."""

    windowed_frames = frames * window

    return windowed_frames