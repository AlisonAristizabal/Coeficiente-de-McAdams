import numpy as np
from mcadams_core.signal_transform.lpc import (
    compute_lpc,
    poles_from_coeffs,
    coeffs_from_poles,
)
from mcadams_core.config.lpc_defaults import compute_lpc_order

def test_compute_lpc_and_poles():

    frame = np.sin(2 * np.pi * np.arange(320) / 50)  # Señal de prueba
    order = 20

    coeficientes_originales = compute_lpc(frame, order)
    polos = poles_from_coeffs(coeficientes_originales)
    coeficientes_reconstruidos = coeffs_from_poles(polos)

    assert np.allclose(coeficientes_originales, coeficientes_reconstruidos, atol=1e-6)