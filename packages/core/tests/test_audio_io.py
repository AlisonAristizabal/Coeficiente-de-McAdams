import numpy as np
import pytest
import scipy.io.wavfile as wav
import soundfile

from mcadams_core.io.audio_io import (
    load_wav,
    save_wav,
    float_to_int16,
    _downmix_to_mono,
    )

SAMPLE_RATE = 16000

def test_round_trip_mono_int16(tmp_path):
    sample_rate = 16000
    audio_original = np.array([0.5, -0.5, 0.0, 0.9, -0.9], dtype="float64")
    ruta = tmp_path / "prueba.wav"

    save_wav(ruta, sample_rate, audio_original)
    sr_leido, audio_leido = load_wav(ruta)

    assert sr_leido == sample_rate
    assert np.allclose(audio_original, audio_leido, atol=1 / 32767)


def test_downmix_stereo():
    audio_estereo = np.column_stack([
        np.ones(5),
        np.zeros(5),
    ])

    resultado = _downmix_to_mono(audio_estereo)

    esperado = np.full(5, 0.5)
    assert np.allclose(resultado, esperado)


def test_load_wav_rejects_non_audio_file(tmp_path):
    """Un archivo con extensión .wav que no es audio hace fallar a
    soundfile al abrirlo; load_wav no atrapa ese error."""
    path = tmp_path / "texto.wav"
    path.write_text("esto no es audio\n", encoding="utf-8")

    with pytest.raises(soundfile.LibsndfileError, match="Format not recognised"):
        load_wav(path)


def write_known_values(fmt, path):
    """Escribe en path un WAV mono del formato fmt con valores conocidos,
    incluidos los extremos del formato, y devuelve lo que load_wav debe
    entregar al leerlo.

    Los formatos de scipy se escriben con scipy.io.wavfile.write. PCM_24
    se escribe con soundfile.write: los 24 bits se pasan alineados a la
    izquierda en int32 (x << 8), que es también como scipy los devuelve
    al leer, y el valor esperado es ese int32 dividido entre 2^31."""
    if fmt == "int16":
        values = np.array([-32768, -16384, -1, 0, 1, 16384, 32767], dtype=np.int16)
        wav.write(path, SAMPLE_RATE, values)
        return values / 32768
    if fmt == "int32":
        values = np.array([-2**31, -2**30, -1, 0, 1, 2**30, 2**31 - 1], dtype=np.int32)
        wav.write(path, SAMPLE_RATE, values)
        return values / 2**31
    if fmt == "uint8":
        # Sin signo: el silencio es 128.
        values = np.array([0, 1, 64, 127, 128, 129, 192, 255], dtype=np.uint8)
        wav.write(path, SAMPLE_RATE, values)
        return (values.astype(np.float64) - 128) / 128
    if fmt == "float32":
        # Incluye valores fuera de [-1, 1]: en punto flotante no se recortan.
        values = np.array([-1.0, -0.5, 0.0, 0.1, 0.25, 1.0, 1.5, -2.0], dtype=np.float32)
        wav.write(path, SAMPLE_RATE, values)
        return values.astype(np.float64)
    if fmt == "PCM_24":
        values = np.array([-2**23, -2**22, -1, 0, 1, 2**22, 2**23 - 1], dtype=np.int32)
        soundfile.write(path, values << 8, SAMPLE_RATE, subtype="PCM_24")
        _, as_int32 = wav.read(path)
        # Comprueba que el archivo guardó exactamente los valores pedidos, extremos incluidos.
        np.testing.assert_array_equal(as_int32, values << 8)
        return as_int32 / 2**31
    raise ValueError(f"formato desconocido: {fmt}")


@pytest.mark.parametrize("fmt", ["int16", "int32", "uint8", "float32", "PCM_24"])
def test_load_wav_normalizes_each_format(tmp_path, fmt):
    """load_wav entrega exactamente el valor normalizado de cada formato:
    int16 -> x / 32768, int32 -> x / 2^31, uint8 -> (x - 128) / 128,
    float32 -> x y PCM_24 -> (int32 de scipy) / 2^31, en float64 y con la
    frecuencia de muestreo del archivo."""
    path = tmp_path / f"{fmt}.wav"
    expected = write_known_values(fmt, path)

    sample_rate, audio = load_wav(path)

    assert sample_rate == SAMPLE_RATE
    assert audio.dtype == np.float64
    np.testing.assert_array_equal(audio, expected)


def test_load_wav_downmixes_stereo_to_mono(tmp_path, capsys):
    """Un WAV estéreo se lee como mono, promediando los dos canales ya
    normalizados, y se avisa por consola (pasa por _downmix_to_mono). Los
    valores son potencias de dos para que el promedio sea exacto."""
    left = np.array([16384, -32768, 0, 8192], dtype=np.int16)
    right = np.array([0, -32768, 16384, -8192], dtype=np.int16)
    path = tmp_path / "estereo.wav"
    wav.write(path, SAMPLE_RATE, np.column_stack([left, right]))

    sample_rate, audio = load_wav(path)

    assert sample_rate == SAMPLE_RATE
    assert audio.ndim == 1
    np.testing.assert_array_equal(audio, [0.25, -1.0, 0.25, 0.0])
    assert "Advertencia: archivo con 2 canales" in capsys.readouterr().out


def test_silencio_no_rompe():
    audio_silencio = np.zeros(100, dtype="float64")

    resultado = float_to_int16(audio_silencio)

    assert np.all(resultado == 0)


def test_clipping_en_save_wav(tmp_path):
    sample_rate = 16000
    audio_fuera_de_rango = np.array([1.5, -2.0, 0.0])
    ruta = tmp_path / "clip_test.wav"

    save_wav(ruta, sample_rate, audio_fuera_de_rango)
    _, audio_leido = load_wav(ruta)

    assert audio_leido[0] <= 1.0
    assert audio_leido[1] >= -1.0