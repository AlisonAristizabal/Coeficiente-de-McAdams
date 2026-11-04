"""Pruebas de LiveSession sin dispositivo de audio: se llama al callback directamente
con arreglos de numpy, como lo haría sounddevice."""
from types import SimpleNamespace

import numpy as np
import pytest
import sounddevice as sd

from mcadams_app.live.session import LiveSession

FS = 16000
N = 160  # 10 ms a 16 kHz


def status(overflow=False, underflow=False, input_underflow=False, output_overflow=False):
    """Imita las banderas de sounddevice (CallbackFlags)."""
    return SimpleNamespace(input_overflow=overflow, output_underflow=underflow,
                           input_underflow=input_underflow, output_overflow=output_overflow)


def counters(session):
    """Los 4 contadores: overflow y underflow de entrada, overflow y underflow de salida."""
    return (session.input_overflows, session.input_underflows, session.output_overflows, session.output_underflows)


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


def test_procesador_que_modifica_en_su_lugar_no_altera_la_entrada_guardada():
    def in_place(block):
        block *= 2
        return block

    s = LiveSession(FS, processor=in_place)
    indata, out = call(s, np.full(N, 0.25))
    np.testing.assert_allclose(s.input_blocks[0], 0.25, rtol=1e-6)
    np.testing.assert_allclose(s.output_blocks[0], 0.5, rtol=1e-6)
    np.testing.assert_allclose(out[:, 0], 0.5, rtol=1e-6)
    np.testing.assert_allclose(indata[:, 0], 0.25, rtol=1e-6)


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


def test_contadores_empiezan_en_cero():
    assert counters(LiveSession(FS)) == (0, 0, 0, 0)


@pytest.mark.parametrize("flags, expected", [
    ({"overflow": True}, (1, 0, 0, 0)),
    ({"input_underflow": True}, (0, 1, 0, 0)),
    ({"output_overflow": True}, (0, 0, 1, 0)),
    ({"underflow": True}, (0, 0, 0, 1)),
    ({}, (0, 0, 0, 0)),
    ({"overflow": True, "output_overflow": True}, (1, 0, 1, 0)),
    ({"input_underflow": True, "underflow": True}, (0, 1, 0, 1)),
    ({"overflow": True, "underflow": True, "input_underflow": True, "output_overflow": True}, (1, 1, 1, 1)),
])
def test_cuenta_cada_bandera_sola_y_combinada(flags, expected):
    s = LiveSession(FS)
    call(s, np.zeros(N), status(**flags))
    assert counters(s) == expected


def test_las_cuatro_banderas_se_acumulan_por_separado():
    s = LiveSession(FS)
    call(s, np.zeros(N), status(overflow=True, input_underflow=True, output_overflow=True, underflow=True))
    call(s, np.zeros(N), status(input_underflow=True, output_overflow=True, underflow=True))
    call(s, np.zeros(N), status(output_overflow=True, underflow=True))
    call(s, np.zeros(N), status(underflow=True))
    call(s, np.zeros(N))
    assert counters(s) == (1, 2, 3, 4)
    r = s.summary()
    assert (r["input_overflows"], r["input_underflows"], r["output_overflows"], r["output_underflows"]) == (1, 2, 3, 4)


def test_banderas_reales_de_sounddevice():
    flags = sd.CallbackFlags()
    flags.input_underflow = True
    flags.output_overflow = True
    s = LiveSession(FS)
    call(s, np.zeros(N), flags)
    assert counters(s) == (0, 1, 1, 0)


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
    assert (r["input_overflows"], r["input_underflows"], r["output_overflows"], r["output_underflows"]) == (0, 0, 0, 0)


def test_resumen_sin_bloques_tiene_las_mismas_claves():
    s = LiveSession(FS)
    empty = set(s.summary())
    call(s, np.zeros(N))
    assert empty == set(s.summary())


def test_resumen_tiene_las_claves_esperadas():
    s = LiveSession(FS)
    call(s, np.zeros(N))
    assert set(s.summary()) == {"n_blocks", "input_overflows", "input_underflows", "output_overflows",
                                "output_underflows", "block_ms_median", "block_ms_p99", "block_ms_max", "rtf"}