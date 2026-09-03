import numpy as np
from mcadams_core.signal_transform.lpc import (
    compute_lpc,
)
from mcadams_core.signal_transform.resynth import (
    get_excitation,
    resynthesize_signal
)

def test_get_excitation_and_resynthesize_signal():
    frame = np.sin(2 * np.pi * np.arange(320) / 50)  # Señal de prueba
    order = 20

    lpc_coeffs = compute_lpc(frame, order)
    excitation = get_excitation(frame, lpc_coeffs)
    resynthesized_frame = resynthesize_signal(excitation, lpc_coeffs)

    assert np.allclose(frame, resynthesized_frame)