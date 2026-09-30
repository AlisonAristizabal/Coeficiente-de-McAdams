import numpy as np
from mcadams_core.signal_transform.lpc import (
    compute_lpc,
    poles_from_coeffs,
    coeffs_from_poles,
)
from mcadams_core.config.lpc_defaults import compute_lpc_order
from mcadams_core.signal_transform.framing import hann_window

def test_compute_lpc_and_poles():

    frame = np.sin(2 * np.pi * np.arange(320) / 50)  # Señal de prueba
    order = 20

    coeficientes_originales = compute_lpc(frame, order)
    polos = poles_from_coeffs(coeficientes_originales)
    coeficientes_reconstruidos = coeffs_from_poles(polos)

    assert np.allclose(coeficientes_originales, coeficientes_reconstruidos, atol=1e-6)

def test_compute_lpc_silent_frame():

    order = 20
    frame = np.zeros(320)

    lpc_coeffs = compute_lpc(frame, order)

    expected = np.zeros(order + 1)
    expected[0] = 1.0
    assert lpc_coeffs.shape == (order + 1,)
    assert np.array_equal(lpc_coeffs, expected)

def test_compute_lpc_noise_frame_is_stable():

    # El método de autocorrelación garantiza un filtro estable:
    # todos los polos deben quedar dentro del círculo unitario
    rng = np.random.default_rng(1234)
    order = 20
    frame = rng.standard_normal(320) * hann_window(320)

    lpc_coeffs = compute_lpc(frame, order)
    poles = poles_from_coeffs(lpc_coeffs)

    assert lpc_coeffs.shape == (order + 1,)
    assert lpc_coeffs[0] == 1.0
    assert np.all(np.abs(poles) < 1)