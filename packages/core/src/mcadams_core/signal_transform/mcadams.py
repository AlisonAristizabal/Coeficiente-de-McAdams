import numpy as np
from mcadams_core.signal_transform.lpc import (
    compute_lpc,
    poles_from_coeffs,
    coeffs_from_poles,
)
from mcadams_core.signal_transform.resynth import (
    get_excitation,
    resynthesize_signal,
)


def process_frame(frame, order, mcadams_coefficient):

    """Procesa un único fragmento ya enventanado a través de todo el
    pipeline de análisis y resíntesis: calcula los coeficientes LPC,
    los convierte a polos, estos polos los transforma con el coeficiente
    de McAdams, los reconstruye de vuelta a coeficientes, extrae la excitación 
    con los coeficientes originales, y resintetiza con los coeficientes reconstruidos. 
    Devuelve el fragmento reconstruido y anonimizado en cierta medida según el 
    coeficiente utilizado."""

    lpc_coefficients = compute_lpc(frame, order)
    poles = poles_from_coeffs(lpc_coefficients)
    transformed_poles = transform_poles(poles, mcadams_coefficient)
    new_lpc_coefficients = coeffs_from_poles(transformed_poles)
    excitation = get_excitation(frame, lpc_coefficients)
    # Resintetiza el fragmento usando la señal de excitación y los
    # coeficientes reconstruidos.
    reconstructed_frame = resynthesize_signal(excitation, new_lpc_coefficients)
    return reconstructed_frame

def transform_angle(angle, mcadams_coefficient):

    """Aplica la fórmula central del método: ángulo ** coeficiente. Espera
    un ángulo positivo (radianes, en el rango (0, π)) — nunca se debe
    llamar con un ángulo negativo, porque una potencia fraccionaria de un
    número negativo no da un resultado real."""

    transformed_angle = angle ** mcadams_coefficient
    return transformed_angle

def transform_poles(poles, mcadams_coefficient):

    """Transforma un arreglo de polos según el método McAdams:
    - Polos reales: sin cambios.
    - Polos complejos con parte imaginaria positiva: se les aplica
      transform_angle sobre su ángulo, conservando el radio original.
    - Sus conjugados (parte imaginaria negativa): se reconstruyen como el
      conjugado del resultado ya calculado para su pareja — nunca se les
      aplica transform_angle directamente.
    Devuelve un arreglo del mismo tamaño que poles, preservando la
    estructura de pares conjugados necesaria para que coeffs_from_poles
    devuelva coeficientes reales."""

    real_poles = poles[np.isreal(poles)]
    complex_poles = poles[~np.isreal(poles)]
    positive_poles = complex_poles[np.imag(complex_poles) > 0]

    radii = np.abs(positive_poles)
    angles = np.angle(positive_poles)
    new_angles = transform_angle(angles, mcadams_coefficient)
    transformed_positive = radii * np.exp(1j * new_angles)
    transformed_negative = np.conj(transformed_positive)

    poles_transformed = np.concatenate([real_poles, transformed_positive, transformed_negative])
    return poles_transformed
