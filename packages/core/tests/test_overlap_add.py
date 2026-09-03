import numpy as np

from mcadams_core.signal_transform.framing import (
    compute_frame_params, 
    frame_signal, 
    hann_window, 
    apply_window,
)

from mcadams_core.signal_transform.overlap_add import overlap_add

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

def test_overlap_add_identity_reconstruction():
    frame_length = 320
    hop_length = 160
    n_frames = 10
    signal_length = (n_frames - 1) * hop_length + frame_length

    t = np.arange(signal_length)
    original = np.sin(2 * np.pi * t / 50)  # Señal de prueba

    frames = frame_signal(original, frame_length, hop_length)
    window = hann_window(frame_length)
    windowed_frames = apply_window(frames, window)
    reconstructed = overlap_add(windowed_frames, hop_length)

    edge = frame_length - hop_length
    original_interior = original[edge: -edge]
    reconstructed_interior = reconstructed[edge: -edge]

    assert np.allclose(original_interior, reconstructed_interior)