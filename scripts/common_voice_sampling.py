import random
from common_voice_strata import assign_stratum


def group_by_stratum(speakers):
    """
    Agrupa los hablantes por estrato de acento.

    Para cada hablante se reúnen los valores crudos de `accents` de todos
    sus clips y se asigna su estrato con `assign_stratum`. Se incluyen
    todos los estratos que aparezcan, también "otro" y "sin_dato"; decidir
    cuáles se muestrean es responsabilidad de quien llame a esta función.

    Las listas de `client_id` de cada estrato se devuelven ordenadas, para
    que un muestreo posterior con semilla fija dé el mismo resultado sin
    importar el orden de las filas del TSV original. No modifica el
    diccionario de entrada.

    Parámetros:
        speakers: diccionario {client_id: lista de clips}, como el que
            devuelve `filter_speakers`.

    Retorna:
        Diccionario {estrato: lista ordenada de client_id}.
    """
    strata = {}
    for client_id, clips in speakers.items():
        accents = [clip["accents"] for clip in clips]
        stratum = assign_stratum(accents)
        strata.setdefault(stratum, []).append(client_id)

    for stratum in strata:
        strata[stratum].sort()

    return strata

def sample_speakers(by_stratum, strata, k, seed):
    """
    Elige `k` hablantes de cada estrato, de forma reproducible.

    Para cada estrato de `strata` se mezcla la lista de sus hablantes con
    un generador sembrado por `seed` y el nombre del estrato, y se toman
    los primeros `k`. Como la mezcla no depende de `k`, el muestreo es
    anidado: con la misma semilla, los hablantes elegidos con un `k`
    menor son los primeros de los elegidos con uno mayor. Agregar o
    quitar estratos no cambia la muestra de los demás.

    Si un estrato tiene menos de `k` hablantes (o no aparece en
    `by_stratum`) se lanza `ValueError`, porque el diseño exige el mismo
    número de hablantes por estrato. No modifica `by_stratum`.

    Parámetros:
        by_stratum: diccionario {estrato: lista ordenada de client_id},
            como el que devuelve `group_by_stratum`.
        strata: lista de nombres de estratos a muestrear.
        k: número de hablantes por estrato.
        seed: entero que siembra el muestreo.

    Retorna:
        Diccionario {estrato: lista de `k` client_id}, en el orden de
        `strata`.
    """
    sampled = {}
    for stratum in strata:
        ids = by_stratum.get(stratum, [])
        if len(ids) < k:
            raise ValueError(f"estrato {stratum} tiene {len(ids)} "
                             f"hablantes, menos de {k}")
        rng = random.Random(f'{seed}-{stratum}')
        shuffled = list(ids)
        rng.shuffle(shuffled)
        sampled[stratum] = shuffled[:k]
    return sampled

def select_clips(clips, n, seed, client_id):
    """
    Elige `n` clips de un hablante, de forma reproducible.

    Los clips se ordenan por `path` antes de mezclarlos con un generador
    sembrado por `seed` y el `client_id`, de modo que el resultado no
    dependa del orden en que aparecían en el TSV. Se devuelven los
    primeros `n` de la mezcla. No modifica la lista de entrada.

    Si el hablante tiene menos de `n` clips se lanza `ValueError`.

    Parámetros:
        clips: lista de clips (diccionarios con al menos `path`) de un
            solo hablante.
        n: número de clips a elegir.
        seed: entero que siembra el muestreo.
        client_id: identificador del hablante, que diferencia la mezcla
            de cada uno.

    Retorna:
        Lista de `n` clips.
    """
    if len(clips) < n:
        raise ValueError(f"hablante {client_id} tiene {len(clips)} "
                         f"clips, menos de {n}")
    ordered = sorted(clips, key=lambda clip: clip["path"])
    rng = random.Random(f"{seed}-clips-{client_id}")
    rng.shuffle(ordered)
    return ordered[:n]