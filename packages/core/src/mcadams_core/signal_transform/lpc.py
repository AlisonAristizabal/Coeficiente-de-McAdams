import numpy as np
from scipy.linalg import solve_toeplitz

def compute_lpc(frame, order):

    """Calcula los coeficientes LPC de un solo fragmento de audio usando
    el método de autocorrelación: se estima la autocorrelación r[0..order]
    del fragmento y se resuelven las ecuaciones normales de Yule-Walker
    (sistema de Toeplitz) con Levinson-Durbin, vía
    scipy.linalg.solve_toeplitz. Devuelve un arreglo de longitud
    order+1, con a[0]=1 por convención — representa el polinomio
    denominador de un filtro todo-polos que modela ese fragmento.

    Si el fragmento no tiene energía (silencio), devuelve [1, 0, ..., 0],
    es decir, un filtro sin polos que no altera la señal."""

    frame = np.asarray(frame, dtype=np.float64)
    n_samples = len(frame)

    # Autocorrelación para los retardos 0..order
    autocorr = np.array([
        np.dot(frame[:n_samples - lag], frame[lag:]) for lag in range(order + 1)
    ])

    # Fragmento de silencio: no hay nada que modelar
    if np.isclose(autocorr[0], 0.0):
        lpc_coeffs = np.zeros(order + 1)
        lpc_coeffs[0] = 1.0
        return lpc_coeffs

    # Corrección mínima de ruido blanco: mejora el condicionamiento de la
    # matriz de Toeplitz sin cambiar apreciablemente el resultado
    autocorr[0] *= 1 + 1e-9

    # Ecuaciones de Yule-Walker: R a = -r, con R de Toeplitz simétrica
    predictor = solve_toeplitz(autocorr[:order], -autocorr[1:order + 1])

    lpc_coeffs = np.concatenate(([1.0], predictor))
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

