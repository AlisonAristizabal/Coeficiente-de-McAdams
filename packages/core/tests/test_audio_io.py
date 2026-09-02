import numpy as np
import pytest

from mcadams_core.io.audio_io import (
    load_wav, 
    save_wav,
    float_to_int16,
    _normalize_to_float,
    _downmix_to_mono,
    )

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


def test_dtype_no_soportado_lanza_error():
    audio_int64 = np.array([1, 2, 3], dtype="int64")

    with pytest.raises(ValueError):
        _normalize_to_float(audio_int64)


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