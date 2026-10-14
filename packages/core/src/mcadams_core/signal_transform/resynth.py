import scipy.signal as signal
import numpy as np
from mcadams_core.config.energy_defaults import MIN_FRAME_ENERGY

def get_excitation(frame, lpc_coeffs):

    """Calcula la señal de excitación de un fragmento de audio a partir de
    sus coeficientes LPC. Devuelve un arreglo de la misma longitud que
    frame."""

    # Filtra el fragmento con el filtro todo-polos definido por los
    # coeficientes LPC. La señal de excitación es la salida del filtro.
    excitation = signal.lfilter(lpc_coeffs, [1], frame)
    return excitation

def resynthesize_signal(excitation, lpc_coeffs):

    """Resintetiza un fragmento de audio a partir de su señal de excitación
    y sus coeficientes LPC. Devuelve un arreglo de la misma longitud que
    excitation."""

    # Filtra la señal de excitación con el filtro todo-polos definido por
    # los coeficientes LPC. La señal resintetizada es la salida del filtro.
    resynthesized = signal.lfilter([1], lpc_coeffs, excitation)
    return resynthesized

def match_energy(reconstructed, reference, min_energy=MIN_FRAME_ENERGY):
    """
    Escala un fragmento para que tenga la misma energía que uno de referencia.

    La energía es la suma de los cuadrados de las muestras. Resintetizar con
    polos modificados cambia la ganancia del filtro, por lo que el fragmento
    reconstruido sale con otra energía que el de entrada; esta función
    compensa esa diferencia multiplicándolo por un único factor. Si la energía
    de cualquiera de los dos fragmentos es menor o igual que `min_energy`
    (silencio), devuelve `reconstructed` sin cambios.

    Parámetros:
        reconstructed: fragmento resintetizado (arreglo 1D).
        reference: fragmento de entrada con la energía deseada (arreglo 1D).
        min_energy: umbral por debajo del cual no se reescala.

    Retorna:
        Un arreglo de la misma forma que `reconstructed`; no modifica las
        entradas.
    """
    reference_energy = np.sum(reference ** 2)
    reconstructed_energy = np.sum(reconstructed ** 2)
    if reference_energy <= min_energy or reconstructed_energy <= min_energy:
        return reconstructed
    gain = np.sqrt(reference_energy / reconstructed_energy)
    return reconstructed * gain