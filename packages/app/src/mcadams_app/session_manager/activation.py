import uuid
from mcadams_core.streaming.anonymizer_stream import (
    AnonymizerStream
)

def start_anonymized_call(sample_rate):
    """Activa la anonimización para una llamada entrante, generando un
    identificador de sesión único (UUID) y creando el AnonymizerStream
    correspondiente para esa llamada.

    Devuelve una tupla (stream, session_id): el AnonymizerStream ya listo
    para recibir audio vía process_chunk, y el session_id (string) usado
    para generar su coeficiente de McAdams — útil para trazabilidad o
    para verificar en pruebas que el coeficiente del stream corresponde
    al session_id devuelto."""

    id_call = uuid.uuid4()
    id_string = str(id_call)

    stream = AnonymizerStream(sample_rate,id_string)
    return (stream, id_string)