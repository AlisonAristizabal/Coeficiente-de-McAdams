import scipy.signal as signal
import numpy as np

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