"""Pruebas de BlockAnonymizer: largo de la salida, retraso constante y mínimo, y
equivalencia con anonymize_utterance (el anonimizador que se evaluó con EER y WER)."""
import numpy as np
import pytest

from mcadams_core.batch.utterance import anonymize_utterance
from mcadams_core.config.mcadams_defaults import MAX_MCADAMS_COEFFICIENT, MIN_MCADAMS_COEFFICIENT
from mcadams_core.session.coefficient import generate_coefficient
from mcadams_core.signal_transform.framing import compute_frame_params
from mcadams_core.streaming.anonymizer_stream import AnonymizerStream
from mcadams_core.streaming.block_anonymizer import BlockAnonymizer, minimum_delay

FS = 16000
SESSION = "sesion prueba"
FRAME, HOP = compute_frame_params(FS)
EDGE = FRAME - HOP  # bordes conocidos de la ventana: sin superposición completa
BLOCKSIZES = [160, 320, 480, 100]


def voice_signal(n=16000, seed=7):
    """Señal tipo voz: armónicos con amplitud variable y algo de ruido."""
    t = np.arange(n) / FS
    rng = np.random.default_rng(seed)
    harmonics = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 8))
    return 0.2 * (1 + 0.5 * np.sin(2 * np.pi * 3 * t)) * harmonics + 0.01 * rng.standard_normal(n)


def fixed_blocks(n, blocksize):
    """Parte n muestras en bloques de tamaño fijo, descartando el resto."""
    return [blocksize] * (n // blocksize)


def variable_blocks(n, seed=3):
    """Largos de bloque aleatorios, de 1 a 700 muestras, que suman n."""
    rng = np.random.default_rng(seed)
    sizes = []
    while sum(sizes) < n:
        sizes.append(min(int(rng.integers(1, 701)), n - sum(sizes)))
    return sizes


def run_blocks(anonymizer, audio, sizes):
    """Pasa `audio` por `process` en bloques de los largos dados y concatena la salida."""
    outputs, start = [], 0
    for size in sizes:
        block = audio[start:start + size]
        output = anonymizer.process(block)
        assert output.ndim == 1 and len(output) == len(block)
        outputs.append(output.copy())
        start += size
    return np.concatenate(outputs), audio[:start]


# ---------------------------------------------------------------------------
# Largo de la salida
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("blocksize", BLOCKSIZES)
@pytest.mark.parametrize("declared", [True, False])
def test_process_siempre_devuelve_el_largo_del_bloque(blocksize, declared):
    anonymizer = BlockAnonymizer(FS, SESSION, blocksize=blocksize if declared else None)
    output, used = run_blocks(anonymizer, voice_signal(), fixed_blocks(16000, blocksize))
    assert output.size == used.size


def test_process_con_bloques_de_largo_variable():
    output, used = run_blocks(BlockAnonymizer(FS, SESSION), voice_signal(), variable_blocks(16000))
    assert output.size == used.size == 16000


def test_bloque_de_una_muestra_y_bloque_vacio():
    anonymizer = BlockAnonymizer(FS, SESSION)
    assert len(anonymizer.process(np.zeros(1))) == 1
    assert len(anonymizer.process(np.zeros(0))) == 0


def test_bloque_mas_largo_que_varios_frames():
    output, used = run_blocks(BlockAnonymizer(FS, SESSION, blocksize=4000), voice_signal(), [4000] * 4)
    assert output.size == used.size


def test_bloque_de_otro_largo_con_blocksize_fijo_lanza_error():
    anonymizer = BlockAnonymizer(FS, SESSION, blocksize=160)
    with pytest.raises(ValueError, match="160 muestras.*159"):
        anonymizer.process(np.zeros(159))


@pytest.mark.parametrize("blocksize", [0, -160])
def test_blocksize_invalido_lanza_error(blocksize):
    with pytest.raises(ValueError, match="blocksize"):
        BlockAnonymizer(FS, SESSION, blocksize=blocksize)


def test_no_modifica_el_bloque_de_entrada():
    block = voice_signal(160)
    before = block.copy()
    BlockAnonymizer(FS, SESSION, blocksize=160).process(block)
    np.testing.assert_array_equal(block, before)


def test_silencio_da_silencio():
    output, _ = run_blocks(BlockAnonymizer(FS, SESSION, blocksize=160), np.zeros(3200), fixed_blocks(3200, 160))
    np.testing.assert_array_equal(output, np.zeros(3200))


def test_salida_entregada_no_cambia_con_llamadas_posteriores():
    anonymizer = BlockAnonymizer(FS, SESSION, blocksize=160)
    audio = voice_signal(1600)
    first = anonymizer.process(audio[:160])
    saved = first.copy()
    for start in range(160, 1600, 160):
        anonymizer.process(audio[start:start + 160])
    np.testing.assert_array_equal(first, saved)


# ---------------------------------------------------------------------------
# Retraso: valor, constancia y mínimo
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("blocksize, samples, ms", [
    (160, 160, 10.0), (320, 160, 10.0), (480, 160, 10.0), (640, 160, 10.0),  # múltiplos del salto
    (100, 300, 18.75), (80, 240, 15.0), (240, 240, 15.0), (7, 319, 19.9375), (1, 319, 19.9375),
    (None, 319, 19.9375),
])
def test_delay_a_16_khz(blocksize, samples, ms):
    anonymizer = BlockAnonymizer(FS, SESSION, blocksize=blocksize)
    assert (anonymizer.delay_samples, anonymizer.delay_ms) == (samples, pytest.approx(ms))


def test_delay_sale_de_frame_y_hop_de_la_frecuencia():
    # A 8 kHz: F = 160 y H = 80.
    anonymizer = BlockAnonymizer(8000, SESSION, blocksize=80)
    assert (anonymizer.delay_samples, anonymizer.delay_ms) == (80, pytest.approx(10.0))
    assert BlockAnonymizer(8000, SESSION).delay_samples == 159


@pytest.mark.parametrize("frame, hop, blocksize, expected", [
    (320, 160, 160, 160), (320, 160, 320, 160), (320, 160, 100, 300), (320, 160, None, 319),
    (100, 50, 50, 50), (100, 50, 30, 90), (640, 320, 160, 480),
])
def test_minimum_delay(frame, hop, blocksize, expected):
    assert minimum_delay(frame, hop, blocksize) == expected


def required_delay(blocksize, n_blocks):
    """Mide con AnonymizerStream el mayor faltante acumulado: el retraso mínimo real."""
    stream = AnonymizerStream(FS, SESSION)
    audio = voice_signal(blocksize * n_blocks)
    produced, worst = 0, 0
    for m in range(n_blocks):
        produced += stream.process_chunk(audio[m * blocksize:(m + 1) * blocksize]).size
        worst = max(worst, (m + 1) * blocksize - produced)
    return worst


@pytest.mark.parametrize("blocksize", [160, 320, 480, 640, 100, 80, 240, 7, 1, 170, 1000])
def test_delay_es_exactamente_el_minimo_necesario(blocksize):
    # Suficiente (ningún bloque se queda corto) y necesario (uno menos no alcanzaría).
    n_blocks = max(40, 2 * FRAME // blocksize + 2 * HOP)
    assert BlockAnonymizer(FS, SESSION, blocksize=blocksize).delay_samples == required_delay(blocksize, n_blocks)


def test_multiplo_del_salto_tiene_menos_retraso_que_no_multiplo():
    multiple = BlockAnonymizer(FS, SESSION, blocksize=160).delay_samples
    other = BlockAnonymizer(FS, SESSION, blocksize=100).delay_samples
    assert multiple == FRAME - HOP and other > multiple


@pytest.mark.parametrize("sizes", [fixed_blocks(8000, 160), fixed_blocks(8000, 320), fixed_blocks(8000, 480),
                                   fixed_blocks(8000, 100), variable_blocks(8000), [1] * 2000])
def test_salida_es_el_stream_retrasado_sin_ceros_extra(sizes):
    # Retraso constante: la cola nunca agrega ni pierde muestras después del arranque.
    blocksize = sizes[0] if len(set(sizes)) == 1 else None
    audio = voice_signal(8000)
    anonymizer = BlockAnonymizer(FS, SESSION, blocksize=blocksize)
    output, used = run_blocks(anonymizer, audio, sizes)
    direct = AnonymizerStream(FS, SESSION).process_chunk(used)
    delay = anonymizer.delay_samples
    np.testing.assert_array_equal(output[:delay], np.zeros(delay))
    np.testing.assert_array_equal(output[delay:], direct[:output.size - delay])


def test_la_cola_no_crece():
    anonymizer = BlockAnonymizer(FS, SESSION, blocksize=100)
    sizes = []
    audio = voice_signal(16000)
    for start in range(0, 16000, 100):
        anonymizer.process(audio[start:start + 100])
        sizes.append(anonymizer._queue.size)
    assert 0 <= min(sizes) and max(sizes) < anonymizer.delay_samples + FRAME


# ---------------------------------------------------------------------------
# Con alfa = 1 la salida es la entrada retrasada
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("blocksize", [160, 320, 480, 100, None])
def test_alpha_uno_impulso_sale_en_la_posicion_esperada(blocksize):
    position = 1000
    audio = np.zeros(4000)
    audio[position] = 0.5
    anonymizer = BlockAnonymizer(FS, SESSION, 1.0, 1.0, blocksize=blocksize)
    output, _ = run_blocks(anonymizer, audio, fixed_blocks(4000, blocksize or 160))
    expected = np.zeros(output.size)
    expected[position + anonymizer.delay_samples] = 0.5
    assert np.argmax(np.abs(output)) == position + anonymizer.delay_samples
    np.testing.assert_allclose(output, expected, atol=1e-6)


@pytest.mark.parametrize("blocksize", [160, 320, 480, 100, None])
def test_alpha_uno_senal_retrasada_salvo_bordes(blocksize):
    audio = voice_signal(8000)
    anonymizer = BlockAnonymizer(FS, SESSION, 1.0, 1.0, blocksize=blocksize)
    assert anonymizer.coefficient == 1.0
    output, used = run_blocks(anonymizer, audio, fixed_blocks(8000, blocksize or 160))
    delay = anonymizer.delay_samples
    np.testing.assert_allclose(output[delay + EDGE:], used[EDGE:used.size - delay], atol=1e-6)


# ---------------------------------------------------------------------------
# Equivalencia con la evaluación
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("session_id", ["sesion prueba", "common_voice_es_22238783", "otra-sesion"])
@pytest.mark.parametrize("blocksize", [160, 320, 480, 100, None])
def test_equivalencia_con_anonymize_utterance(session_id, blocksize):
    audio = voice_signal()
    anonymizer = BlockAnonymizer(FS, session_id, blocksize=blocksize)
    sizes = variable_blocks(16000) if blocksize is None else fixed_blocks(16000, blocksize)
    output, used = run_blocks(anonymizer, audio, sizes)
    reference, coefficient = anonymize_utterance(used, FS, session_id)
    delay = anonymizer.delay_samples
    assert anonymizer.coefficient == coefficient
    aligned = output[delay:]
    np.testing.assert_allclose(aligned[EDGE:], reference[EDGE:aligned.size], atol=1e-6)
    assert not np.allclose(aligned[EDGE:], used[EDGE:aligned.size], atol=1e-3)  # sí anonimiza


def test_equivalencia_con_bloques_float32_como_en_vivo():
    audio = voice_signal().astype(np.float32)
    anonymizer = BlockAnonymizer(FS, SESSION, blocksize=160)
    output, used = run_blocks(anonymizer, audio, fixed_blocks(16000, 160))
    reference, _ = anonymize_utterance(used.astype(np.float64), FS, SESSION)
    aligned = output[anonymizer.delay_samples:]
    np.testing.assert_allclose(aligned[EDGE:], reference[EDGE:aligned.size], atol=1e-6)


# ---------------------------------------------------------------------------
# Coeficiente y sesión
# ---------------------------------------------------------------------------

def test_sin_rango_usa_el_rango_por_defecto_del_nucleo():
    for i in range(20):
        session_id = f"sesion_{i}"
        anonymizer = BlockAnonymizer(FS, session_id)
        assert anonymizer.coefficient == generate_coefficient(session_id)[0]
        assert anonymizer.coefficient == generate_coefficient(session_id, MIN_MCADAMS_COEFFICIENT,
                                                              MAX_MCADAMS_COEFFICIENT)[0]
        assert MIN_MCADAMS_COEFFICIENT <= anonymizer.coefficient <= MAX_MCADAMS_COEFFICIENT
        assert anonymizer.session_id == session_id


def test_rango_explicito_llega_al_stream():
    assert BlockAnonymizer(FS, "x", 0.6, 0.8).coefficient == generate_coefficient("x", 0.6, 0.8)[0]
    assert BlockAnonymizer(FS, "x", 0.5, 0.5).coefficient == 0.5


def test_el_retraso_no_depende_del_coeficiente():
    assert BlockAnonymizer(FS, "x", 0.5, 0.5, blocksize=160).delay_samples == \
        BlockAnonymizer(FS, "y", 1.0, 1.0, blocksize=160).delay_samples
