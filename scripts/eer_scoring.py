import numpy as np


def normalize_rows(matrix):
    """
    Normaliza cada fila de `matrix` a norma L2 igual a 1.

    No modifica la entrada. Lanza `ValueError` si alguna fila tiene norma
    cero, porque no tiene dirección y el coseno no está definido.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("hay embeddings con norma cero")
    return matrix / norms


def build_enrollment(embeddings, row_of, splits):
    """
    Calcula el embedding de registro de cada hablante (decisión D3).

    Para cada hablante de `splits`, en el orden del diccionario, normaliza
    los embeddings de sus clips de registro y los promedia. El promedio no
    se re-normaliza: `score_matrix` lo hace. Lanza `ValueError` si
    `splits` está vacío.

    Parámetros:
        embeddings: matriz (N, D), una fila por clip.
        row_of: diccionario {session_id: índice de fila en `embeddings`}.
        splits: salida de `split_enrollment_trial`.

    Retorna:
        Matriz (número de hablantes, D). La fila j es el hablante j-ésimo
        de `splits`.
    """
    if not splits:
        raise ValueError("splits está vacío")
    rows = []
    for client_id in splits:
        indices = [row_of[s] for s in splits[client_id]["enroll"]]
        normalized = normalize_rows(embeddings[indices])
        rows.append(normalized.mean(axis=0))
    return np.stack(rows)


def build_trials(embeddings, row_of, splits):
    """
    Reúne los embeddings de prueba de todos los hablantes.

    Recorre `splits` en el mismo orden que `build_enrollment`.

    Retorna:
        Una tupla (trials, owner). `trials` es una matriz (número de
        pruebas, D) sin normalizar. `owner[i]` es el índice del hablante
        dueño de la prueba i, según el orden de `splits`.
    """
    indices, owner = [], []
    for speaker_index, client_id in enumerate(splits):
        for session_id in splits[client_id]["trial"]:
            indices.append(row_of[session_id])
            owner.append(speaker_index)
    return embeddings[indices], np.array(owner)


def score_matrix(enrollment, trials):
    """
    Calcula la similitud coseno de cada prueba contra cada registro.

    Retorna:
        Matriz (número de pruebas, número de hablantes) donde
        `scores[i, j]` es el coseno entre la prueba i y el registro j.
    """
    return normalize_rows(trials) @ normalize_rows(enrollment).T


def split_target_impostor(scores, owner):
    """
    Separa las puntuaciones en pares objetivo y pares de impostor.

    El par (i, owner[i]) es objetivo; los demás pares de la fila i son
    impostores. Lanza `ValueError` si `owner` no tiene una entrada por fila.

    Retorna:
        Una tupla (target, impostor) de arreglos 1D.
    """
    scores = np.asarray(scores)
    owner = np.asarray(owner)
    if scores.shape[0] != owner.shape[0]:
        raise ValueError("owner no coincide con las filas de scores")
    is_target = np.zeros(scores.shape, dtype=bool)
    is_target[np.arange(scores.shape[0]), owner] = True
    return scores[is_target], scores[~is_target]


def compute_eer(target, impostor):
    """
    Calcula el EER con la convención del punto más cercano.

    Se acepta un par si su puntuación es mayor o igual que el umbral. Entre
    los umbrales observados se elige el de menor |FAR - FRR| y se reporta
    (FAR + FRR) / 2. En caso de empate gana el umbral más bajo. Lanza
    `ValueError` si algún conjunto está vacío o contiene NaN.

    Retorna:
        Una tupla (eer, umbral). El EER es una fracción entre 0 y 1.
    """
    target = np.sort(np.asarray(target, dtype=np.float64))
    impostor = np.sort(np.asarray(impostor, dtype=np.float64))
    if target.size == 0 or impostor.size == 0:
        raise ValueError("target e impostor no pueden estar vacíos")
    if np.isnan(target).any() or np.isnan(impostor).any():
        raise ValueError("las puntuaciones no pueden contener NaN")

    thresholds = np.unique(np.concatenate([target, impostor]))

    # frr_count: objetivos rechazados (puntuación < umbral).
    # far_count: impostores aceptados (puntuación >= umbral).
    frr_count = np.searchsorted(target, thresholds, side="left")
    far_count = impostor.size - np.searchsorted(impostor, thresholds, side="left")

    # |far_count/n_impostor - frr_count/n_target| con multiplicación
    # cruzada, para comparar enteros y no fracciones.
    gap = np.abs(far_count * target.size - frr_count * impostor.size)
    best = np.argmin(gap)
    eer = (far_count[best] / impostor.size + frr_count[best] / target.size) / 2
    return eer, thresholds[best]