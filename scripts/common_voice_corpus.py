import csv


def read_tsv_rows(file_path):
    """
    Lee un archivo TSV de Common Voice y entrega sus filas una por una.

    Es un generador: no carga el archivo completo en memoria. Cada fila
    se entrega como diccionario con las columnas del encabezado como
    llaves. Se desactiva el manejo de comillas (QUOTE_NONE) porque las
    transcripciones pueden contener comillas sueltas que, con el
    comportamiento por defecto de `csv`, unirían varias filas en una.

    Si una fila trae menos campos que el encabezado, `csv.DictReader`
    deja en `None` los campos faltantes; quien consuma esta función
    debe contemplarlo.

    Parámetros:
        file_path: ruta al archivo .tsv.

    Retorna:
        Un generador de diccionarios, uno por fila de datos.
    """
    with open(file_path, "r", encoding="utf-8", newline='') as f:
        yield from csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)


def load_durations(path):
    """
    Lee `clip_durations.tsv` de Common Voice y devuelve la duración de
    cada clip.

    Las llaves son los nombres de clip tal como aparecen en el archivo
    (con extensión `.mp3`, igual que la columna `path` de los TSV de
    datos) y los valores son enteros en milisegundos. Las filas cuya
    duración falta o no es un número entero se omiten, de modo que esos
    clips no aparecen en el resultado; quien consuma el diccionario debe
    tratar la ausencia de un clip como "duración desconocida". Si un
    clip se repite, prevalece su última aparición.

    Parámetros:
        path: ruta al archivo `clip_durations.tsv`.

    Retorna:
        Diccionario {nombre_de_clip: duración_en_ms}.
    """
    durations = {}
    for row in read_tsv_rows(path):
        raw = row['duration[ms]']
        if raw is None:
            continue
        try:
            duration_ms = int(raw)
        except ValueError:
            continue
        durations[row['clip']] = duration_ms
    return durations

def load_speakers(validated_path, durations, min_duration_ms):
    """
    Lee el TSV de datos de Common Voice y agrupa los clips por hablante.

    Para cada fila se consulta su duración en `durations` (llaves con
    extensión `.mp3`, igual que la columna `path`). Se descartan, en este
    orden y contando cada clip en una sola categoría, los clips sin
    duración conocida, los de duración menor a `min_duration_ms` y los
    sin `client_id`. El filtro de duración se aplica antes de agrupar
    para no construir estructuras de clips que luego se descartarían.

    Este paso no descarta hablantes por número de clips ni asigna
    estratos de acento: eso se hace después, sobre el resultado.

    Parámetros:
        validated_path: ruta al TSV de datos (por ejemplo `validated.tsv`).
        durations: diccionario {nombre_de_clip: duración_en_ms}, como el
            que devuelve `load_durations`.
        min_duration_ms: duración mínima en milisegundos; un clip con
            exactamente esa duración se conserva.

    Retorna:
        Una tupla (speakers, discarded):
        - speakers: diccionario {client_id: lista de clips}; cada clip es
          un diccionario con `path`, `accents`, `sentence` y
          `duration_ms`. `accents` se conserva crudo, puede ser "" o None.
        - discarded: diccionario con los contadores `sin_duracion`,
          `muy_corto` y `sin_client_id`.
    """
    speakers = {}
    discarded = {"sin_duracion": 0, "muy_corto": 0, "sin_client_id": 0}

    for row in read_tsv_rows(validated_path):
        client_id = row["client_id"]
        path = row["path"]
        accents = row["accents"]
        sentence = row["sentence"]
        duration_ms = durations.get(path)
        if duration_ms is None:
            discarded["sin_duracion"] += 1
            continue
        if duration_ms < min_duration_ms:
            discarded["muy_corto"] += 1
            continue
        if not client_id:
            discarded["sin_client_id"] += 1
            continue
        clip_info = {
            "path": path,
            "accents": accents,
            "sentence": sentence,
            "duration_ms": duration_ms}
        
        speakers.setdefault(client_id, []).append(clip_info)

    return speakers, discarded

