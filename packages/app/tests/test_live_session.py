"""Pruebas de LiveSession sin dispositivo de audio: se llama al callback directamente
con arreglos de numpy, como lo haría sounddevice."""
from types import SimpleNamespace

import numpy as np
import pytest

from mcadams_app.live.session import LiveSession

FS = 16000
N = 160  # 10 ms a 16 kHz


def status(overflow=False, underflow=False):
    """Imita las banderas de sounddevice (CallbackFlags)."""
    return SimpleNamespace(input_overflow=overflow, output_underflow=underflow)


def call(session, block, st=None):
    """Llama al callback como sounddevice: indata y outdata con forma (frames, 1)."""
    indata = np.asarray(block, dtype=np.float32).reshape(-1, 1)
    outdata = np.full_like(indata, np.nan)
    session.callback(indata, outdata, len(block), None, st or status())
    return indata, outdata


def fake_clock(step):
    """Reloj falso: cada llamada avanza `step` segundos."""
    t = {"now": 0.0}
    def clock():
        t["now"] += step
        return t["now"]
    return clock


def test_paso_directo_salida_igual_a_entrada():
    s = LiveSession(FS)
    block = np.linspace(-0.5, 0.5, N)
    _, out = call(s, block)
    np.testing.assert_array_equal(out[:, 0], block.astype(np.float32))


def test_llena_outdata_completo():
    s = LiveSession(FS)
    _, out = call(s, np.zeros(N))
    assert not np.isnan(out).any()


def test_procesador_se_aplica_al_bloque():
    s = LiveSession(FS, processor=lambda x: 2 * x)
    block = np.full(N, 0.1)
    _, out = call(s, block)
    np.testing.assert_allclose(out[:, 0], 0.2, rtol=1e-6)


def test_procesador_recibe_un_arreglo_1d():
    received = []
    s = LiveSession(FS, processor=lambda x: received.append(x.shape) or x)
    call(s, np.zeros(N))
    assert received == [(N,)]


def test_guarda_copia_de_la_entrada_no_la_referencia():
    s = LiveSession(FS)
    indata, _ = call(s, np.full(N, 0.3))
    indata[:] = 9.0  # sounddevice reutiliza el arreglo
    np.testing.assert_allclose(s.input_blocks[0], 0.3, rtol=1e-6)


def test_guarda_la_salida_que_se_escribio():
    s = LiveSession(FS, processor=lambda x: -x)
    _, out = call(s, np.full(N, 0.25))
    np.testing.assert_array_equal(s.output_blocks[0], out[:, 0])
    out[:] = 0.0
    np.testing.assert_allclose(s.output_blocks[0], -0.25, rtol=1e-6)


def test_bloques_en_orden():
    s = LiveSession(FS)
    for k in range(3):
        call(s, np.full(N, k / 10))
    assert [float(b[0]) for b in s.input_blocks] == pytest.approx([0.0, 0.1, 0.2])
    assert len(s.output_blocks) == 3


def test_procesador_con_longitud_incorrecta_lanza_error():
    s = LiveSession(FS, processor=lambda x: x[:-1])
    with pytest.raises(ValueError):
        call(s, np.zeros(N))


@pytest.mark.parametrize("flags, expected", [
    ((False, False), (0, 0)), ((True, False), (1, 0)), ((False, True), (0, 1)), ((True, True), (1, 1)),
])
def test_cuenta_overflow_y_underflow(flags, expected):
    s = LiveSession(FS)
    call(s, np.zeros(N), status(*flags))
    assert (s.input_overflows, s.output_underflows) == expected


def test_contadores_se_acumulan():
    s = LiveSession(FS)
    call(s, np.zeros(N), status(overflow=True))
    call(s, np.zeros(N), status(overflow=True, underflow=True))
    call(s, np.zeros(N))
    assert (s.input_overflows, s.output_underflows) == (2, 1)


def test_tiempo_por_bloque_con_reloj_falso():
    # Cada llamada al reloj avanza 2 ms: inicio y fin del callback -> 2 ms por bloque.
    s = LiveSession(FS, clock=fake_clock(0.002))
    for _ in range(4):
        call(s, np.zeros(N))
    assert s.process_times == pytest.approx([0.002] * 4)


def test_resumen_con_reloj_falso():
    s = LiveSession(FS, clock=fake_clock(0.002))
    for _ in range(5):
        call(s, np.zeros(N))
    r = s.summary()
    assert r["n_blocks"] == 5
    assert r["block_ms_median"] == pytest.approx(2.0)
    assert r["block_ms_p99"] == pytest.approx(2.0)
    assert r["block_ms_max"] == pytest.approx(2.0)
    assert r["rtf"] == pytest.approx(0.2)  # 2 ms de proceso por cada 10 ms de audio


def test_rtf_depende_del_tamano_de_bloque():
    s = LiveSession(FS, clock=fake_clock(0.002))
    call(s, np.zeros(2 * N))  # bloque de 20 ms
    assert s.summary()["rtf"] == pytest.approx(0.1)


def test_resumen_incluye_contadores():
    s = LiveSession(FS)
    call(s, np.zeros(N), status(overflow=True, underflow=True))
    r = s.summary()
    assert (r["input_overflows"], r["output_underflows"]) == (1, 1)


def test_resumen_sin_bloques():
    r = LiveSession(FS).summary()
    assert r["n_blocks"] == 0
    assert r["rtf"] is None and r["block_ms_median"] is None
    assert r["block_ms_p99"] is None and r["block_ms_max"] is None


def test_resumen_tiene_las_claves_esperadas():
    s = LiveSession(FS)
    call(s, np.zeros(N))
    assert set(s.summary()) == {"n_blocks", "input_overflows", "output_underflows",
                                "block_ms_median", "block_ms_p99", "block_ms_max", "rtf"}