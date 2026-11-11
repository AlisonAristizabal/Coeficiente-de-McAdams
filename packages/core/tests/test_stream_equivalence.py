"""Equivalencia entre AnonymizerStream y una implementación de referencia
independiente, con anonimización (alfa distinto de 1).

La referencia (reference_pipeline) hace el proceso de forma directa: segmenta
toda la señal con frame_signal, enventana con apply_window(hann_window),
pasa cada trama por process_frame y reconstruye con overlap_add.
AnonymizerStream hace lo mismo de forma incremental, con un búfer de entrada y
un búfer de superposición. Las operaciones por trama son las mismas, así que la
diferencia debe ser solo de redondeo de punto flotante y se exige una
tolerancia estricta.

Alineación (sale de process_chunk). Con F = frame_length y H = hop_length
(F = 2H), cada trama k que el stream completa entrega H muestras: la mitad
derecha de la trama k - 1 (guardada en buffer_overlap, en ceros para k = 0)
más la mitad izquierda de la trama k. Eso es exactamente el tramo
[k * H, (k + 1) * H) de overlap_add, así que la salida del stream empieza en la
muestra 0 de la referencia, sin retraso, y tras n tramas ha entregado n * H
muestras. La cola que entrega flush() no se compara: flush rellena con ceros
una trama extra que la referencia no tiene."""

import numpy as np
import pytest

from mcadams_core.batch.utterance import anonymize_utterance
from mcadams_core.config.lpc_defaults import compute_lpc_order
from mcadams_core.signal_transform.framing import apply_window, hann_window
from mcadams_core.signal_transform.mcadams import process_frame
from mcadams_core.streaming.anonymizer_stream import AnonymizerStream
from mcadams_core.streaming.block_anonymizer import BlockAnonymizer
from reference_pipeline import frame_signal, overlap_add

SAMPLE_RATE = 16000
SESSION_ID = "sesion equivalencia"
ALPHA = 0.8
N_FRAMES = 20
SEED = 20261111
# Tolerancia absoluta estricta; rtol=0 para que la relativa por defecto de
# assert_allclose (1e-7) no la afloje.
ATOL = 1e-10


def make_stream():
    """Crea un AnonymizerStream con alfa fijo en ALPHA, pasado como rango
    [ALPHA, ALPHA] tal como lo hacen los lotes y el modo en vivo."""
    return AnonymizerStream(SAMPLE_RATE, SESSION_ID, alpha_min=ALPHA, alpha_max=ALPHA)


def make_signal(frame_length, hop_length):
    """Señal reproducible de (N_FRAMES - 1) * hop_length + frame_length
    muestras a 16 kHz: tres senoides de frecuencias distintas más ruido
    pequeño con semilla fija. Con ese largo la referencia tiene exactamente
    N_FRAMES tramas."""
    length = (N_FRAMES - 1) * hop_length + frame_length
    t = np.arange(length) / SAMPLE_RATE
    rng = np.random.default_rng(SEED)
    return (0.5 * np.sin(2 * np.pi * 220 * t)
            + 0.3 * np.sin(2 * np.pi * 870 * t + 0.4)
            + 0.15 * np.sin(2 * np.pi * 2350 * t + 1.1)
            + 0.01 * rng.standard_normal(length))


def reference_output(signal, frame_length, hop_length, order, alpha):
    """Anonimiza signal con la implementación directa, no incremental:
    frame_signal -> apply_window(hann_window) -> process_frame -> overlap_add."""
    frames = frame_signal(signal, frame_length, hop_length)
    windowed_frames = apply_window(frames, hann_window(frame_length))
    processed_frames = np.array([process_frame(frame, order, alpha) for frame in windowed_frames])
    return overlap_add(processed_frames, hop_length)


@pytest.fixture(scope="module")
def setup():
    """Señal, referencia y parámetros, tomando frame_length y hop_length del
    propio stream. Verifica además que el stream use el mismo alfa y el
    mismo orden LPC que la referencia."""
    stream = make_stream()
    frame_length, hop_length = stream.frame_length, stream.hop_length
    order = compute_lpc_order(SAMPLE_RATE)
    assert stream.mcadams_coefficient == ALPHA
    assert stream.order == order

    signal = make_signal(frame_length, hop_length)
    reference = reference_output(signal, frame_length, hop_length, order, ALPHA)
    assert len(frame_signal(signal, frame_length, hop_length)) == N_FRAMES
    return signal, reference, hop_length


def run_stream(stream, signal, chunk_size):
    """Pasa signal por process_chunk en bloques de chunk_size muestras (el
    último puede ser más corto), o en una sola llamada si chunk_size es None,
    y concatena lo entregado. No llama a flush."""
    if chunk_size is None:
        return stream.process_chunk(signal)
    outputs = [stream.process_chunk(signal[start:start + chunk_size])
               for start in range(0, len(signal), chunk_size)]
    return np.concatenate(outputs)


@pytest.mark.parametrize("chunk_size", [None, 160, 137], ids=["una_llamada", "bloques_160", "bloques_137"])
def test_stream_matches_reference(setup, chunk_size):
    """La salida de process_chunk es la referencia desde la muestra 0, sin
    retraso, sin importar cómo se parta la entrada: en una sola llamada, en
    bloques de 160 muestras (el bloque del prototipo en vivo, igual al salto)
    o en bloques de 137 (no divide al salto, así que las tramas se completan
    a mitad de bloque)."""
    signal, reference, hop_length = setup
    output = run_stream(make_stream(), signal, chunk_size)

    # Tras N_FRAMES tramas el stream ha entregado N_FRAMES * hop_length
    # muestras; el resto de la referencia (la mitad derecha de la última
    # trama) sigue en buffer_overlap esperando a flush.
    n_compared = N_FRAMES * hop_length
    assert output.size == n_compared
    np.testing.assert_allclose(output, reference[:n_compared], rtol=0, atol=ATOL)
    # La prueba tiene sentido solo si de verdad anonimiza.
    assert not np.allclose(output, signal[:n_compared], atol=1e-3)


def test_anonymize_utterance_matches_reference(setup):
    """anonymize_utterance (el anonimizador de los lotes) coincide con la
    referencia en sus primeras N_FRAMES * hop_length muestras. Después
    vienen muestras que dependen del relleno con ceros de flush, que la
    referencia no tiene."""
    signal, reference, hop_length = setup
    output, coefficient = anonymize_utterance(signal, SAMPLE_RATE, SESSION_ID, alpha_min=ALPHA, alpha_max=ALPHA)

    assert coefficient == ALPHA
    assert output.size == signal.size
    n_compared = N_FRAMES * hop_length
    np.testing.assert_allclose(output[:n_compared], reference[:n_compared], rtol=0, atol=ATOL)


def test_block_anonymizer_is_reference_delayed(setup):
    """BlockAnonymizer con bloques de 160 muestras entrega la referencia
    retrasada exactamente delay_samples muestras: primero delay_samples ceros
    (la cola precargada) y luego la referencia desde su muestra 0. Se compara
    solo el tramo en que ambas existen, que cae dentro de las
    N_FRAMES * hop_length muestras que el stream ya entregó."""
    signal, reference, hop_length = setup
    blocksize = 160
    anonymizer = BlockAnonymizer(SAMPLE_RATE, SESSION_ID, alpha_min=ALPHA, alpha_max=ALPHA, blocksize=blocksize)
    assert anonymizer.coefficient == ALPHA

    n_blocks = len(signal) // blocksize
    output = np.concatenate([anonymizer.process(signal[i * blocksize:(i + 1) * blocksize])
                             for i in range(n_blocks)])
    delay = anonymizer.delay_samples
    n_compared = output.size - delay
    assert 0 < n_compared <= N_FRAMES * hop_length

    np.testing.assert_array_equal(output[:delay], np.zeros(delay))
    np.testing.assert_allclose(output[delay:], reference[:n_compared], rtol=0, atol=ATOL)
