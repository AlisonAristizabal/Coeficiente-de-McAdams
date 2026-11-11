import scipy.io.wavfile as wav
import numpy as np
import soundfile

def load_wav(path):

    """
    Lee un archivo WAV desde disco y lo devuelve normalizado y listo
    para trabajar con él como audio en punto flotante.

    Usa soundfile.read (basada en libsndfile), la misma lectura que
    convert_mp3_to_wav. libsndfile identifica el formato interno del WAV
    según su cabecera (enteros de 8, 16, 24 o 32 bits, o punto flotante)
    y entrega las muestras normalizadas a [-1.0, 1.0] en float64: los
    enteros con signo de n bits se dividen entre 2^(n-1), los de 8 bits,
    que no tienen signo y cuyo silencio es 128, se calculan como
    (x - 128) / 128, y los de punto flotante se entregan sin escalar.
    Luego colapsa a mono si el archivo viene en estéreo.

    soundfile devuelve (audio, frecuencia), en el orden inverso al de
    scipy.io.wavfile; esta función devuelve (sample_rate, audio_float),
    que es el orden que esperan quienes la llaman.

    Parámetros:
        path: ruta al archivo .wav a leer.

    Retorna:
        (sample_rate, audio_float): la frecuencia de muestreo del
        archivo, y un arreglo 1D de NumPy en float64 normalizado.

    Lanza soundfile.LibsndfileError si libsndfile no reconoce el archivo
    como audio.
    """
    audio_float, sample_rate = soundfile.read(path, dtype="float64")

    audio_float = _downmix_to_mono(audio_float)

    return sample_rate, audio_float


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