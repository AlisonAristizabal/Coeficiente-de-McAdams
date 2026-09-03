def compute_lpc_order(sample_rate, offset = 4):
    """Calcula el orden del filtro LPC a partir de la frecuencia de
    muestreo, siguiendo la heurística de Markel & Gray (1976): orden =
    frecuencia de muestreo en kHz + offset (recomendado entre 4 y 5)."""

    order = round(sample_rate / 1000) + offset
    return order