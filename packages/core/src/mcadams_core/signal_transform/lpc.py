import librosa as lr
import numpy as np

def compute_lpc(frame, order):

    """Calcula los coeficientes LPC de un solo fragmento de audio usando
    el método de Burg (librosa.lpc). Devuelve un arreglo de longitud
    order+1, con a[0]=1 por convención — representa el polinomio
    denominador de un filtro todo-polos que modela ese fragmento."""

    lpc_coeffs = lr.lpc(frame, order = order)
    return lpc_coeffs

def poles_from_coeffs(lpc_coeffs):

    """Calcula los polos de un filtro todo-polos a partir de sus
    coeficientes LPC. Devuelve un arreglo de números complejos."""

    poles = np.roots(lpc_coeffs)
    return poles

def coeffs_from_poles(poles):

    """Calcula los coeficientes LPC de un filtro todo-polos a partir de
    sus polos. Devuelve un arreglo de longitud order+1, con a[0]=1."""

    lpc_coeffs = np.poly(poles)
    return lpc_coeffs.real

