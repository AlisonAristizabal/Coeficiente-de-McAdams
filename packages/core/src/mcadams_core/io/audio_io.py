import scipy.io.wavfile as wav
import numpy as np

def load_wav(path):

    """
    Lee un archivo WAV desde disco y lo devuelve normalizado y listo
    para trabajar con él como audio en punto flotante.

    Primero normaliza los valores crudos del archivo
    (que pueden venir en distintos formatos internos) a un rango
    consistente de [-1.0, 1.0], y luego colapsa a mono si el archivo
    viene en estéreo u otro número de canales.

    Parámetros:
        path: ruta al archivo .wav a leer.

    Retorna:
        (sample_rate, audio_float): la frecuencia de muestreo del
        archivo, y un arreglo 1D de NumPy en float64 normalizado.
    """
    sample_rate, data = wav.read(path)

    audio_float = _normalize_to_float(data)

    audio_float = _downmix_to_mono(audio_float)

    return sample_rate, audio_float


def _normalize_to_float(data):

    """
    Convierte un arreglo de audio crudo (en el dtype original del WAV)
    a punto flotante normalizado en el rango [-1.0, 1.0].

    scipy.io.wavfile no normaliza nada por sí solo: devuelve los valores
    exactamente en el formato en que estén guardados en el archivo, que
    puede variar. Por eso esta función despacha según el dtype detectado,
    en vez de asumir un solo formato:

    - int16: el más común en telefonía. Rango [-32768, 32767].
    - int32: PCM de mayor resolución. Rango [-2^31, 2^31 - 1].
    - uint8: sin signo, el silencio no es 0 sino 128 (el punto medio),
      por eso hay que restar 128 antes de escalar.
    - float32 / float64: ya suele venir normalizado, solo se homogeniza
      el tipo de dato.

    Si el dtype no es ninguno de los anteriores, se rechaza explícitamente
    en vez de intentar procesarlo con una fórmula que podría ser incorrecta.
    """

    if data.dtype == 'int16':
        return data.astype('float64') / 32768.0
    elif data.dtype == 'int32':
        return data.astype('float64') / (2**31)
    elif data.dtype == 'uint8':
        return (data.astype('float64') - 128) / 128.0
    elif data.dtype == 'float32' or data.dtype == 'float64':
        return data.astype('float64')
    else:
        raise ValueError(f"dtype de WAV no soportado: {data.dtype}")
    

def _downmix_to_mono(audio_float):

    """
    Colapsa audio multicanal a un solo canal (mono), promediando entre
    canales cuando sea necesario.

    En el contexto de telefonía móvil no existe la noción de voz en
    estéreo real: si un archivo llega con más de un canal, casi siempre
    es una configuración de grabación (ej. un micrófono que graba en
    2 canales por defecto), no contenido espacial genuino distinto por
    canal. Por eso se hace downmix automático, mostrando una advertencia
    para que quede visible que el archivo de entrada no vino en el
    formato mono esperado.
    """

    if audio_float.ndim == 1:
        return audio_float
    elif audio_float.ndim == 2:
        print(f"Advertencia: archivo con {audio_float.shape[1]} canales detectado, se combinó a mono")
        return audio_float.mean(axis=1)
    else:
        raise ValueError(f"Audio con más de 2 canales no soportado: {audio_float.ndim}D")

def save_wav(path, sample_rate, audio_float):

    """
    Escribe un arreglo de audio en punto flotante a un archivo WAV en
    disco, en formato int16 (el estándar para telefonía/voz).

    Antes de convertir, recorta (clip) cualquier valor que se haya salido
    del rango válido [-1.0, 1.0] durante el procesamiento.
    """

    audio_float_recortado = np.clip(audio_float, -1.0, 1.0)
    audio_int16 = float_to_int16(audio_float_recortado)
    wav.write(path, sample_rate, audio_int16)

def float_to_int16(audio_float):

    """
    Convierte audio en punto flotante [-1.0, 1.0] a enteros de 16 bits
    (int16), el formato estándar de WAV para voz.
    """
    
    return np.round(audio_float * 32767).astype('int16')