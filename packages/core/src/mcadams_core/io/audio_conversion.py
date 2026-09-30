import subprocess

import numpy as np
from mcadams_core.io.audio_io import (
    save_wav,
)
from mcadams_core.config.audio_defaults import (
    TARGET_SAMPLE_RATE
)

def convert_mp3_to_wav(input_path, output_path):

    """Convierte un archivo MP3 a WAV, remuestreando el audio a
    TARGET_SAMPLE_RATE y guardándolo en output_path.

    Usa ffmpeg (proceso externo, debe estar en el PATH) para decodificar
    el MP3, mezclarlo a mono y remuestrearlo en un solo paso. ffmpeg
    entrega las muestras por stdout como float32 little-endian en
    [-1.0, 1.0], que se leen con np.frombuffer y se pasan a float64. El
    WAV final se escribe reutilizando save_wav, que se encarga del
    recorte a [-1.0, 1.0] y la conversión a int16.

    No devuelve ningún valor — el éxito de la conversión se señala por
    la ausencia de una excepción, no por un valor de retorno. Si ffmpeg
    falla se propaga subprocess.CalledProcessError (y FileNotFoundError
    si ffmpeg no está instalado). Imprime un mensaje de confirmación con
    la ruta de salida."""

    command = [
        "ffmpeg",
        "-v", "error",
        "-i", str(input_path),
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(TARGET_SAMPLE_RATE),
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=True)

    resampled_audio = np.frombuffer(result.stdout, dtype="<f4").astype("float64")
    save_wav(output_path, TARGET_SAMPLE_RATE, resampled_audio)

    print(f"el audio se convirtió y guardó correctamente en el path {output_path}")
