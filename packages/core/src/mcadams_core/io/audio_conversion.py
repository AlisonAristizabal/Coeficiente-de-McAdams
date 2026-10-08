import math

import soundfile
from scipy.signal import resample_poly
from mcadams_core.io.audio_io import (
    save_wav,
)
from mcadams_core.config.audio_defaults import (
    TARGET_SAMPLE_RATE
)

def convert_mp3_to_wav(input_path, output_path):

    """Convierte un archivo MP3 a WAV, remuestreando el audio a
    TARGET_SAMPLE_RATE y guardándolo en output_path.

    Usa soundfile.read (libsndfile) para decodificar el MP3 directamente
    a float64 en [-1.0, 1.0]. Si el audio tiene más de un canal, se
    promedia a mono. Si la frecuencia de muestreo del archivo es distinta
    de TARGET_SAMPLE_RATE, se remuestrea con scipy.signal.resample_poly,
    usando el factor racional exacto up/down (reducido con el máximo
    común divisor); si ya coincide, no se remuestrea. El WAV final se
    escribe reutilizando save_wav, que se encarga del recorte a
    [-1.0, 1.0] y la conversión a int16.

    No devuelve ningún valor — el éxito de la conversión se señala por
    la ausencia de una excepción, no por un valor de retorno. Imprime
    un mensaje de confirmación con la ruta de salida."""

    audio, sample_rate = soundfile.read(input_path, dtype="float64")

    # soundfile devuelve (n_muestras, n_canales) cuando hay varios canales
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sample_rate != TARGET_SAMPLE_RATE:
        divisor = math.gcd(TARGET_SAMPLE_RATE, sample_rate)
        up = TARGET_SAMPLE_RATE // divisor
        down = sample_rate // divisor
        audio = resample_poly(audio, up, down)

    save_wav(output_path, TARGET_SAMPLE_RATE, audio)

    print(f"el audio se convirtió y guardó correctamente en el path {output_path}")
