import numpy as np
import pytest
import soundfile

from mcadams_core.io.audio_conversion import convert_mp3_to_wav
from mcadams_core.io.audio_io import load_wav
from mcadams_core.config.audio_defaults import TARGET_SAMPLE_RATE

def test_convert_mp3_to_wav_resamples_to_target(tmp_path):

    source_rate = 48000
    duration_s = 1.0
    t = np.arange(int(source_rate * duration_s)) / source_rate
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)

    mp3_path = tmp_path / "sine.mp3"
    wav_path = tmp_path / "sine.wav"
    soundfile.write(mp3_path, sine, source_rate, format="MP3")

    convert_mp3_to_wav(mp3_path, wav_path)
    sample_rate, audio = load_wav(wav_path)

    # Tolerancia de 50 ms: el peor caso de relleno MP3 (retardo del
    # codificador 576 + retardo del decodificador 529 + relleno del
    # último frame < 1152 muestras ≈ 2257 muestras a 48 kHz ≈ 47 ms)
    expected_length = int(TARGET_SAMPLE_RATE * duration_s)
    tolerance = int(0.05 * TARGET_SAMPLE_RATE)

    assert sample_rate == TARGET_SAMPLE_RATE
    assert audio.ndim == 1
    assert abs(len(audio) - expected_length) <= tolerance

def test_convert_mp3_to_wav_missing_file_raises(tmp_path):

    with pytest.raises(soundfile.LibsndfileError):
        convert_mp3_to_wav(tmp_path / "missing.mp3", tmp_path / "out.wav")
