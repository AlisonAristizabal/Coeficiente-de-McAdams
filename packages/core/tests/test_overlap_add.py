import numpy as np
from mcadams_core.signaltransform.framing import (
    frame_signal,
    hann_window,
    apply_window,
)
from mcadams_core.signaltransform.overlap_add import overlap_add

def test_overlap_add_small_manual():
    # Prueba manual con un ejemplo pequeño
    frames = np.array([
        [1, 1, 1, 1], 
        [2, 2, 2, 2],
        [3, 3, 3, 3]
        ])
    hop_length = 2

    resultado = overlap_add(frames, hop_length)

    # La señal reconstruida debería ser: [1, 1, 1+2, 1+2, 2+3, 2+3, 3, 3]=[1, 1, 3, 3, 5, 5, 3, 3]
    esperado = np.array([1, 1, 3, 3, 5, 5, 3, 3])
    assert np.array_equal(resultado, esperado)

