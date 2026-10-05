import numpy as np
from mcadams_core.batch.utterance import (
    anonymize_utterance
)
from mcadams_core.signal_transform.framing import (
    compute_frame_params
)

def test_anonymize_utterance_length_short():
    sample_rate = 16000
    session_id = "sesion prueba"
    simulated_audio = np.random.default_rng(seed=42).normal(size=100)
    anon_audio, _ = anonymize_utterance(simulated_audio, sample_rate, session_id)

    assert len(anon_audio) == len(simulated_audio)

def test_anonymize_utterance_length_long():
    sample_rate = 16000
    session_id = "sesion prueba"
    simulated_audio = np.random.default_rng(seed=42).normal(size=10000)
    anon_audio, _ = anonymize_utterance(simulated_audio, sample_rate, session_id)

    assert len(anon_audio) == len(simulated_audio)


def test_anonymize_utterance_identity(monkeypatch):
    sample_rate = 16000
    session_id = "sesion prueba"
    frame_length, hop_length = compute_frame_params(sample_rate)
    signal_length = 10000
    t = np.arange(signal_length)
    original = np.sin(2 * np.pi * t / 50)

    monkeypatch.setattr(
        "mcadams_core.streaming.anonymizer_stream.generate_coefficient",
        lambda session_id_simulated: (1.0, 0),
    )

    anon_audio, _ = anonymize_utterance(original, sample_rate, session_id)

    edge = frame_length - hop_length
    assert np.allclose(original[edge:-edge], anon_audio[edge:-edge], atol=1e-6)
