import numpy as np
from scipy.signal import check_COLA

from mcadams_core.signal_transform.framing import (
    compute_frame_params, 
    frame_signal, 
    hann_window, 
    apply_window,
)

def test_compute_frame_params():

    sample_rate = 16000

    frame_length, hop_length = compute_frame_params(sample_rate)

    assert frame_length == 320
    assert hop_length == 160

def test_frame_signal():

    audio = np.arange(10)
    frames = frame_signal(audio, frame_length=4, hop_length=2)

    assert frames.shape == (4, 4)
    assert np.array_equal(frames[0], np.array([0, 1, 2, 3]))
    assert np.array_equal(frames[1], np.array([2, 3, 4, 5]))
    assert np.array_equal(frames[2], np.array([4, 5, 6, 7]))
    assert np.array_equal(frames[3], np.array([6, 7, 8, 9]))

def test_hann_window():

    frame_length = 320
    hop_length = 160

    window = hann_window(frame_length)

    assert check_COLA(window, frame_length, frame_length-hop_length) == True

def test_apply_window():

    window = hann_window(320)
    frames = np.ones((5, 320))

    resultado = apply_window(frames, window)

    for fila in resultado:
        assert np.array_equal(fila, window)
