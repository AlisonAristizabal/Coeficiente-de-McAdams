SIN_DATO = "sin_dato"
OTRO = "otro"
ACCENT_TO_STRATUM = {
    "México": "mexico",
    "Andino-Pacífico: Colombia, Perú, Ecuador, oeste de Bolivia y Venezuela andina": "andino_pacifico",
    "España: Norte peninsular (Asturias, Castilla y León, Cantabria, País Vasco, Navarra, Aragón, La Rioja, Guadalajara, Cuenca)": "espana_norte",
    "Rioplatense: Argentina, Uruguay, este de Bolivia, Paraguay": "rioplatense",
    "Caribe: Cuba, Venezuela, Puerto Rico, República Dominicana, Panamá, Colombia caribeña, México caribeño, Costa del golfo de México": "caribe",
    "España: Centro-Sur peninsular (Madrid, Toledo, Castilla-La Mancha)": "espana_centro_sur",
    "América central": "america_central",
    "Chileno: Chile, Cuyo": "chileno",
    "España: Sur peninsular (Andalucia, Extremadura, Murcia)": "espana_sur",
}

def assign_stratum(speaker_accents):
    """
    Asigna un hablante de Common Voice a un estrato de acento.

    La asignación se hace por hablante, no por clip: recibe los valores
    crudos de la columna `accents` de todos sus clips. Los valores vacíos
    (o solo espacios) se ignoran, bajo el supuesto de que el acento es
    autorreportado en el perfil y un clip sin etiqueta no contradice a
    otro que sí la tiene.

    Reglas, en este orden:
    - Sin ninguna etiqueta no vacía: "sin_dato".
    - Más de una etiqueta distinta: "otro" (etiquetas en conflicto).
    - Una sola etiqueta: su estrato si está entre los nueve incluidos;
      en cualquier otro caso (valor combinado con "|", texto libre o
      acento fuera de los estratos) "otro".

    Parámetros:
        speaker_accents: iterable de strings con el `accents` crudo de
            cada clip del hablante.

    Retorna:
        El nombre del estrato como string.
    """
    labels = {a.strip() for a in speaker_accents if a and a.strip()}
    if not labels:
        return SIN_DATO
    if len(labels) > 1:
        return OTRO
    label = next(iter(labels))
    return ACCENT_TO_STRATUM.get(label, OTRO)
