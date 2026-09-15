import numpy as np
from mcadams_core.streaming.anonymizer_stream import (
    AnonymizerStream
)

def test_stream_identity():
    sample_rate = 16000
    session_id = "sesion prueba"
    stream = AnonymizerStream(sample_rate, session_id)
    stream.mcadams_coefficient = 1.0
    frame_length = stream.frame_length
    hop_length = stream.hop_length

    n_frames = 10
    signal_length = (n_frames-1)*hop_length + frame_length
    t = np.arange(signal_length)
    original = np.sin(2 * np.pi * t / 50)

    chunk_output = stream.process_chunk(original)
    flush_output = stream.flush()
    streaming_output = np.concatenate([chunk_output,flush_output])
    edge = frame_length - hop_length

    assert np.allclose(original[edge:], streaming_output[edge:-hop_length],atol=1e-6)

def test_stream_chunked_matches_single_call():
    sample_rate = 16000
    session_id = "sesion prueba"
    stream_single = AnonymizerStream(sample_rate, session_id)
    stream_chunked = AnonymizerStream(sample_rate, session_id)
    frame_length = stream_single.frame_length
    hop_length = stream_single.hop_length

    n_frames = 10
    signal_length = (n_frames-1)*hop_length + frame_length
    t = np.arange(signal_length)
    original = np.sin(2 * np.pi * t / 50)

    chunk_single = stream_single.process_chunk(original)
    flush_single = stream_single.flush()
    complete_audio = np.concatenate([chunk_single,flush_single])

    chunk_size = 137
    result=[]
    for i in range(0, len(original), chunk_size):
        chunk_result = stream_chunked.process_chunk(original[i:i + chunk_size])
        result.append(chunk_result)

    flush_result = stream_chunked.flush()
    result.append(flush_result)
    result_concatenated = np.concatenate(result)

    assert np.allclose(complete_audio,result_concatenated,atol=1e-6)