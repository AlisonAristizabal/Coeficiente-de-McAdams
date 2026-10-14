from common_voice_sampling import select_clips


def split_enrollment_trial(rows, n_enroll=3, n_trial=2, seed=42):
    """
    Separa los clips de cada hablante en registro y prueba.

    Usa solo las filas del manifiesto con estado "ok". Para cada hablante con
    al menos `n_enroll + n_trial` clips, elige esa cantidad de forma
    reproducible con `select_clips`; los primeros `n_enroll` quedan como
    registro y los siguientes `n_trial` como prueba. Los hablantes con menos
    clips válidos se excluyen (no se lanza error) y se informan aparte para
    que quien llame los reporte. No modifica `rows`.

    Parámetros:
        rows: iterable de diccionarios del manifiesto, con las llaves
            `client_id`, `stratum`, `session_id` y `status`.
        n_enroll: clips de registro por hablante.
        n_trial: clips de prueba por hablante.
        seed: entero que siembra la elección.

    Retorna:
        Una tupla (splits, excluded). `splits` es un diccionario
        {client_id: {"stratum": str, "enroll": [session_id, ...],
        "trial": [session_id, ...]}}, con los hablantes ordenados por
        client_id. `excluded` es {client_id: número de clips válidos}.
    """
    by_speaker = {}
    for row in rows:
        clips = by_speaker.setdefault(row["client_id"], [])
        if row["status"] == "ok":
            clips.append(row)

    splits, excluded = {}, {}
    for client_id in sorted(by_speaker):
        clips = by_speaker[client_id]
        if len(clips) < n_enroll + n_trial:
            excluded[client_id] = len(clips)
            continue
        chosen = select_clips(clips, n_enroll + n_trial, seed, client_id)
        splits[client_id] = {
            "stratum": chosen[0]["stratum"],
            "enroll": [clip["session_id"] for clip in chosen[:n_enroll]],
            "trial": [clip["session_id"] for clip in chosen[n_enroll:n_enroll + n_trial]],
        }
    return splits, excluded