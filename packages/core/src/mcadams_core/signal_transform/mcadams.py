from mcadams_core.signal_transform.lpc import (
    compute_lpc,
    poles_from_coeffs,
    coeffs_from_poles,
)
from mcadams_core.signal_transform.resynth import (
    get_excitation,
    resynthesize_signal,
)


def process_frame(frame, order):
    """Procesa un único fragmento ya enventanado a través de todo el
    pipeline de análisis y resíntesis: calcula los coeficientes LPC,
    los convierte a polos, los reconstruye de vuelta a coeficientes, 
    extrae la excitación con los coeficientes originales, y resintetiza 
    con los coeficientes reconstruidos. Devuelve el fragmento reconstruido."""

    lpc_coefficients = compute_lpc(frame, order)
    poles = poles_from_coeffs(lpc_coefficients)
    new_lpc_coefficients = coeffs_from_poles(poles)
    excitation = get_excitation(frame, lpc_coefficients)
    # Resintetiza el fragmento usando la señal de excitación y los
    # coeficientes reconstruidos.
    reconstructed_frame = resynthesize_signal(excitation, new_lpc_coefficients)
    return reconstructed_frame
