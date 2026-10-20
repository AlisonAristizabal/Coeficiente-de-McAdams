import hashlib
import numpy as np
from mcadams_core.config.mcadams_defaults import (
    MIN_MCADAMS_COEFFICIENT,
    MAX_MCADAMS_COEFFICIENT,
)

def generate_coefficient(session_id, alpha_min=MIN_MCADAMS_COEFFICIENT, alpha_max=MAX_MCADAMS_COEFFICIENT):

    """Genera un coeficiente de McAdams determinístico a partir de un
    identificador de sesión (session_id). El mismo session_id siempre
    produce el mismo coeficiente, porque la aleatoriedad se deriva de un
    hash SHA-256 del identificador en lugar de una semilla del sistema —
    esto permite reproducir una sesión específica para depuración o
    evaluación. El coeficiente se muestrea de una distribución uniforme
    en el rango [alpha_min, alpha_max], que por defecto es
    [MIN_MCADAMS_COEFFICIENT, MAX_MCADAMS_COEFFICIENT]; con los valores
    por defecto el resultado es idéntico al de versiones anteriores.
    Si alpha_min == alpha_max, devuelve exactamente ese valor (alpha
    fijo, por ejemplo para un barrido). Lanza ValueError si no se cumple
    0 < alpha_min <= alpha_max <= 1.
    Devuelve una tupla (coefficient, seed), donde seed es el entero
    usado para sembrar el generador — útil para trazabilidad, no para
    volver a derivar el session_id original."""

    if not 0 < alpha_min <= alpha_max <= 1:
        raise ValueError(
            f"rango de alpha inválido: se requiere 0 < alpha_min <= alpha_max <= 1 "
            f"(alpha_min={alpha_min}, alpha_max={alpha_max})"
        )

    session_hash = hashlib.sha256(session_id.encode()).digest()
    seed = int.from_bytes(session_hash, 'big')
    if alpha_min == alpha_max:
        return float(alpha_min), seed

    generator = np.random.Generator(np.random.PCG64(seed))
    coefficient = generator.uniform(alpha_min, alpha_max)

    return coefficient, seed
