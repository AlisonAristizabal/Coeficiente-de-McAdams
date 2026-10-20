import csv
from pathlib import Path

import numpy as np
import pytest

from mcadams_core.batch.utterance import anonymize_utterance
from mcadams_core.config.mcadams_defaults import MAX_MCADAMS_COEFFICIENT, MIN_MCADAMS_COEFFICIENT
from mcadams_core.session.coefficient import generate_coefficient
from mcadams_core.streaming.anonymizer_stream import AnonymizerStream

# Coeficientes copiados de ..\Datos\anon_k134\anon_manifest_u07_09.tsv (U(0,7; 0,9),
# generados antes de agregar alpha_min y alpha_max). Deben salir idénticos.
MANIFEST_SAMPLE = [
    ("common_voice_es_22238783", "0.7472873670702868"),
    ("common_voice_es_18949963", "0.828135124342608"),
    ("common_voice_es_22145786", "0.7916865893818908"),
    ("common_voice_es_30474475", "0.7738799529525487"),
    ("common_voice_es_19723180", "0.7695240355916385"),
    ("common_voice_es_19620156", "0.8396361766231284"),
]
MANIFEST_PATH = Path(__file__).resolve().parents[4] / "Datos" / "anon_k134" / "anon_manifest_u07_09.tsv"


# ---------------------------------------------------------------------------
# Regresión: el alpha por defecto no cambia
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("session_id, coefficient", MANIFEST_SAMPLE)
def test_regresion_coeficiente_por_defecto_igual_al_manifiesto(session_id, coefficient):
    assert generate_coefficient(session_id)[0] == float(coefficient)
    assert repr(float(generate_coefficient(session_id)[0])) == coefficient


@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="no está el manifiesto real en ..\\Datos")
def test_regresion_todo_el_manifiesto_u07_09():
    with open(MANIFEST_PATH, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE) if r["status"] == "ok"]
    assert len(rows) == 6700
    mismatches = [r["session_id"] for r in rows if generate_coefficient(r["session_id"])[0] != float(r["coefficient"])]
    assert mismatches == []


def test_regresion_rango_explicito_igual_al_por_defecto():
    for i in range(50):
        sid = f"sesion_{i}"
        assert generate_coefficient(sid, MIN_MCADAMS_COEFFICIENT, MAX_MCADAMS_COEFFICIENT) == generate_coefficient(sid)


def test_regresion_semilla_igual_que_antes():
    import hashlib
    sid = "common_voice_es_22238783"
    assert generate_coefficient(sid)[1] == int.from_bytes(hashlib.sha256(sid.encode()).digest(), "big")


# ---------------------------------------------------------------------------
# generate_coefficient con alpha fijo y con rango
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.2, 0.5, 0.75, 1.0])
def test_alpha_fijo_devuelve_exactamente_ese_valor(alpha):
    for i in range(20):
        coefficient, _ = generate_coefficient(f"s{i}", alpha, alpha)
        assert coefficient == alpha and type(coefficient) is float


def test_alpha_fijo_conserva_la_semilla_del_session_id():
    assert generate_coefficient("abc", 0.5, 0.5)[1] == generate_coefficient("abc")[1]


def test_rango_nuevo_dentro_de_los_limites_y_determinista():
    values = [generate_coefficient(f"s{i}", 0.6, 0.8)[0] for i in range(200)]
    assert all(0.6 <= v <= 0.8 for v in values)
    assert values == [generate_coefficient(f"s{i}", 0.6, 0.8)[0] for i in range(200)]
    assert min(values) < 0.62 and max(values) > 0.78  # cubre el rango, no queda pegado a un extremo


def test_rango_nuevo_usa_el_mismo_generador_que_el_por_defecto():
    # Misma semilla y mismo uniforme: el valor se reescala al nuevo rango.
    for i in range(20):
        sid = f"s{i}"
        u = (generate_coefficient(sid)[0] - 0.7) / 0.2
        assert generate_coefficient(sid, 0.6, 0.8)[0] == pytest.approx(0.6 + 0.2 * u)


@pytest.mark.parametrize("alpha_min, alpha_max", [
    (0.0, 0.5), (-0.1, 0.5), (0.5, 1.1), (0.9, 0.7), (0.0, 0.0), (1.2, 1.2),
])
def test_alpha_invalido_lanza_valueerror_claro(alpha_min, alpha_max):
    with pytest.raises(ValueError, match="0 < alpha_min <= alpha_max <= 1"):
        generate_coefficient("s", alpha_min, alpha_max)


def test_limites_validos_extremos():
    assert generate_coefficient("s", 1.0, 1.0)[0] == 1.0
    assert 0.01 <= generate_coefficient("s", 0.01, 1.0)[0] <= 1.0


# ---------------------------------------------------------------------------
# AnonymizerStream y anonymize_utterance
# ---------------------------------------------------------------------------

def test_stream_sin_argumentos_igual_que_antes():
    stream = AnonymizerStream(16000, "common_voice_es_22238783")
    assert stream.mcadams_coefficient == 0.7472873670702868


def test_stream_con_alpha_fijo():
    assert AnonymizerStream(16000, "x", 0.3, 0.3).mcadams_coefficient == 0.3


def test_stream_con_rango():
    stream = AnonymizerStream(16000, "x", 0.6, 0.8)
    assert stream.mcadams_coefficient == generate_coefficient("x", 0.6, 0.8)[0]
    assert stream.seed == generate_coefficient("x")[1]


def test_stream_solo_alpha_min_usa_el_maximo_por_defecto():
    assert AnonymizerStream(16000, "x", alpha_min=0.8).mcadams_coefficient == generate_coefficient("x", 0.8, 0.9)[0]


def test_stream_alpha_invalido_lanza():
    with pytest.raises(ValueError):
        AnonymizerStream(16000, "x", 0.9, 0.5)


def audio(n=8000):
    return np.sin(2 * np.pi * np.arange(n) / 50) * 0.3


def test_utterance_sin_argumentos_igual_que_antes():
    anon_default, coefficient = anonymize_utterance(audio(), 16000, "sesion")
    anon_explicit, _ = anonymize_utterance(audio(), 16000, "sesion", 0.7, 0.9)
    assert coefficient == generate_coefficient("sesion")[0]
    np.testing.assert_array_equal(anon_default, anon_explicit)


def test_utterance_pasa_alpha_fijo_y_lo_devuelve():
    anon, coefficient = anonymize_utterance(audio(), 16000, "sesion", 0.5, 0.5)
    assert coefficient == 0.5
    reference = AnonymizerStream(16000, "sesion", 0.5, 0.5)
    expected = np.concatenate([reference.process_chunk(audio()), reference.flush()])[:8000]
    np.testing.assert_array_equal(anon, expected)


def test_utterance_alpha_distinto_cambia_el_audio():
    a, _ = anonymize_utterance(audio(), 16000, "sesion", 0.5, 0.5)
    b, _ = anonymize_utterance(audio(), 16000, "sesion", 0.9, 0.9)
    assert not np.allclose(a, b)


def test_utterance_alpha_uno_es_casi_la_identidad():
    original = audio(10000)
    anon, _ = anonymize_utterance(original, 16000, "sesion", 1.0, 1.0)
    stream = AnonymizerStream(16000, "sesion")
    edge = stream.frame_length - stream.hop_length
    assert np.allclose(original[edge:-edge], anon[edge:-edge], atol=1e-6)
