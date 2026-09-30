import numpy as np
from mcadams_core.streaming.anonymizer_stream import (
    AnonymizerStream
)

def anonymize_utterance(audio, sample_rate, session_id):

    """Anonimiza un único audio completo (utterance) usando el coeficiente
    de McAdams asociado a session_id. Internamente reutiliza AnonymizerStream:
    procesa todo el audio en un solo llamado a process_chunk y libera el
    remanente con flush, concatena ambos resultados y recorta la salida a
    la misma longitud que el audio original (flush siempre agrega una cola
    de relleno mayor o igual a hop_length). Devuelve una tupla con el audio
    anonimizado y el coeficiente de McAdams utilizado, para su registro en
    el manifiesto del batch."""

    stream = AnonymizerStream(sample_rate, session_id)
    processed_audio = stream.process_chunk(audio)
    flushed_audio = stream.flush()

    final_audio = np.concatenate([processed_audio,flushed_audio])

    sliced_audio = final_audio[:len(audio)]
    return sliced_audio, stream.mcadams_coefficient