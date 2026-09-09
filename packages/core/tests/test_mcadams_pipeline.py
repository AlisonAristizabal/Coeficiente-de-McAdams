import numpy as np
from mcadams_core.config.lpc_defaults import compute_lpc_order
from mcadams_core.signal_transform.framing import compute_frame_params
from mcadams_core.signal_transform.framing import frame_signal, hann_window, apply_window
from mcadams_core.signal_transform.mcadams import process_frame
from mcadams_core.signal_transform.overlap_add import overlap_add



def test_full_pipeline_identity():
    sample_rate = 16000
    frame_length, hop_length = compute_frame_params(sample_rate)
    order = compute_lpc_order(sample_rate)
    n_frames = 10
    signal_length = (n_frames - 1) * hop_length + frame_length

    t = np.arange(signal_length)
    original = np.sin(2 * np.pi * t / 50)

    frames = frame_signal(original, frame_length, hop_length)
    window = hann_window(frame_length)
    windowed_frames = apply_window(frames, window)

    reconstructed_frames = np.array([
        process_frame(windowed_frames[i], order) for i in range(n_frames)
    ])

    reconstructed = overlap_add(reconstructed_frames, hop_length)

    edge = frame_length - hop_length
    assert np.allclose(original[edge:-edge], reconstructed[edge:-edge], atol=1e-6)