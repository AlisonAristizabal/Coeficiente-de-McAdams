import hashlib
import numpy as np
from mcadams_core.config.mcadams_defaults import (
    MIN_MCADAMS_COEFFICIENT,
    MAX_MCADAMS_COEFFICIENT,
)

def generate_coefficient(session_id):

    """Genera un coeficiente de McAdams determinístico a partir de un
    identificador de sesión (session_id). El mismo session_id siempre
    produce el mismo coeficiente, porque la aleatoriedad se deriva de un
    hash SHA-256 del identificador en lugar de una semilla del sistema —
    esto permite reproducir una sesión específica para depuración o
    evaluación. El coeficiente se muestrea de una distribución uniforme
    en el rango [MIN_MCADAMS_COEFFICIENT, MAX_MCADAMS_COEFFICIENT].
    Devuelve una tupla (coefficient, seed), donde seed es el entero
    usado para sembrar el generador — útil para trazabilidad, no para
    volver a derivar el session_id original."""

    session_hash = hashlib.sha256(session_id.encode()).digest()
    seed = int.from_bytes(session_hash, 'big')
    generator = np.random.Generator(np.random.PCG64(seed))
    coefficient = generator.uniform(MIN_MCADAMS_COEFFICIENT, MAX_MCADAMS_COEFFICIENT)

    return coefficient, seed