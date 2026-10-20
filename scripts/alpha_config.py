"""
Etiquetas de las configuraciones del coeficiente de McAdams.

Una configuración es un rango [alpha_min, alpha_max]; si los dos extremos
son iguales, el α es fijo. La etiqueta nombra las carpetas y archivos de
cada configuración (anon_<etiqueta>, anon_manifest_<etiqueta>.tsv, ...) y
sigue el patrón del rango por defecto, u07_09. Este módulo no importa
mcadams_core, para que lo puedan usar los scripts que corren sin él.
"""

DEFAULT_RANGE = (0.7, 0.9)


def _digits(value):
    """Escribe un α sin el punto decimal: 0.7 → "07", 0.65 → "065", 1.0 → "1"."""
    return f"{value:g}".replace(".", "")


def config_tag(alpha_min, alpha_max):
    """
    Etiqueta de una configuración de α.

    Un α fijo se escribe en centésimas con tres cifras (0.2 → "a020",
    1.0 → "a100"); un rango, como el de siempre (0.7–0.9 → "u07_09",
    0.6–0.8 → "u06_08").

    Parámetros:
        alpha_min: extremo inferior del rango.
        alpha_max: extremo superior (igual a alpha_min para un α fijo).

    Retorna:
        La etiqueta (texto).
    """
    if alpha_min == alpha_max:
        return f"a{round(alpha_min * 100):03d}"
    return f"u{_digits(alpha_min)}_{_digits(alpha_max)}"


def validate_range(alpha_min, alpha_max):
    """Lanza ValueError si no se cumple 0 < alpha_min <= alpha_max <= 1 (la misma regla del core)."""
    if not 0 < alpha_min <= alpha_max <= 1:
        raise ValueError(f"rango de alpha inválido: se requiere 0 < alpha_min <= alpha_max <= 1 "
                         f"(alpha_min={alpha_min}, alpha_max={alpha_max})")
