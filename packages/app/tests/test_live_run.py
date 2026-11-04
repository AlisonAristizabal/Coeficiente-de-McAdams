"""Pruebas de la prueba en vivo sin dispositivo de audio: un stream falso llama
al callback de la sesión con bloques conocidos y las funciones de consulta de sounddevice
se reemplazan por versiones falsas. Todo se escribe en tmp_path, nunca en ..\\Datos."""
import csv
import inspect
import json
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.io.wavfile as wav
import sounddevice as sd

from mcadams_app.live import run
from mcadams_app.live.session import LiveSession
from mcadams_core.batch.utterance import anonymize_utterance
from mcadams_core.config.mcadams_defaults import MAX_MCADAMS_COEFFICIENT, MIN_MCADAMS_COEFFICIENT
from mcadams_core.io.audio_io import float_to_int16
from mcadams_core.session.coefficient import generate_coefficient
from mcadams_core.streaming.block_anonymizer import BlockAnonymizer

FS = 16000
BLOCK = 160
N_BLOCKS = 25
LATENCY = (0.02, 0.03)
ARGS = ["--seconds", "0.5", "--input-device", "1", "--output-device", "4"]
RUN_NAME = re.compile(r"^\d{8}_\d{6}_paso_directo$")
ANONYMIZED_NAME = re.compile(r"^\d{8}_\d{6}_anonimizado$")
SESSION_KEYS = {"n_blocks", "input_overflows", "input_underflows", "output_overflows", "output_underflows",
                "block_ms_median", "block_ms_p99", "block_ms_max", "rtf"}


def known_block(k, blocksize=BLOCK):
    """Bloque conocido número k, con forma (blocksize,)."""
    return np.linspace(-0.5, 0.5, blocksize, dtype=np.float32) * ((k % 5 + 1) / 5)


def known_audio(n_blocks=N_BLOCKS, blocksize=BLOCK):
    return np.concatenate([known_block(k, blocksize) for k in range(n_blocks)])


def make_stream(n_blocks=N_BLOCKS, flags=None, latency=LATENCY):
    """Clase de stream falsa: al entrar al `with` llama al callback `n_blocks` veces.

    Igual que sounddevice, reutiliza los mismos arreglos en cada llamada. La
    latencia solo se puede leer con el stream abierto. `flags` es {bloque: banderas}."""
    flags = flags or {}

    class FakeStream:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.entered = False
            self.closed = False
            FakeStream.instances.append(self)

        def __enter__(self):
            self.entered = True
            size = self.kwargs["blocksize"]
            indata = np.empty((size, 1), dtype=np.float32)
            outdata = np.empty((size, 1), dtype=np.float32)
            for k in range(n_blocks):
                indata[:, 0] = known_block(k, size)
                outdata.fill(np.nan)
                self.kwargs["callback"](indata, outdata, size, None, flags.get(k, sd.CallbackFlags()))
            return self

        def __exit__(self, *exc_info):
            self.closed = True
            return False

        @property
        def latency(self):
            if not self.entered or self.closed:
                raise RuntimeError("la latencia se lee con el stream abierto")
            return latency

    return FakeStream


def raise_interrupt(_seconds):
    raise KeyboardInterrupt


class ForbiddenStream:
    """Stream que falla si alguien intenta abrirlo."""

    def __init__(self, **_kwargs):
        raise AssertionError("no se debe abrir ningún stream")


@pytest.fixture
def fake_sd(monkeypatch):
    """Reemplaza las consultas de sounddevice. `state.rejects` es {función: dispositivo que rechaza}."""
    state = SimpleNamespace(rejects={}, checks=[])
    devices = {
        1: {"index": 1, "name": "Micrófono de prueba", "hostapi": 0},
        4: {"index": 4, "name": "Audífonos de prueba", "hostapi": 1},
    }
    defaults = {"input": 1, "output": 4}

    def checker(name):
        def check(**kwargs):
            state.checks.append((name, kwargs))
            if name in state.rejects and state.rejects[name] == kwargs["device"]:
                raise sd.PortAudioError("Invalid sample rate", -9997)
        return check

    def query_devices(device=None, kind=None):
        if device is None and kind is None:
            return "LISTA DE DISPOSITIVOS"
        return devices[defaults[kind] if device is None else device]

    monkeypatch.setattr(run.sd, "check_input_settings", checker("input"))
    monkeypatch.setattr(run.sd, "check_output_settings", checker("output"))
    monkeypatch.setattr(run.sd, "query_devices", query_devices)
    monkeypatch.setattr(run.sd, "query_hostapis", lambda index: {0: {"name": "MME"}, 1: {"name": "WASAPI"}}[index])
    monkeypatch.setattr(run.sd, "get_portaudio_version", lambda: (1246976, "PortAudio de prueba"))
    monkeypatch.setattr(run.sd, "__version__", "9.9.9")
    return state


def correr(root, *extra, stream_class=None, wait=lambda _seconds: None):
    """Corre main con un stream falso y devuelve la clase usada."""
    stream_class = stream_class or make_stream()
    run.main([*ARGS, "--output-root", str(root), *extra], stream_class=stream_class, wait=wait)
    return stream_class


def only_run(root):
    """Devuelve la única carpeta de corrida dentro de `root`."""
    dirs = [p for p in Path(root).iterdir()]
    assert len(dirs) == 1 and dirs[0].is_dir()
    return dirs[0]


def read_summary(run_dir):
    return json.loads((run_dir / "resumen.json").read_text(encoding="utf-8"))


def read_times(run_dir):
    with open(run_dir / "tiempos.tsv", encoding="utf-8", newline="") as f:
        return list(csv.reader(f, delimiter="\t"))


def session_with_clock(n_blocks=4, processor=None, step=0.001):
    """Sesión con reloj falso (cada bloque tarda `step` segundos) y `n_blocks` procesados."""
    ticks = iter(np.arange(0, 2 * n_blocks) * step)
    session = LiveSession(FS, processor=processor, clock=lambda: float(next(ticks)))
    with make_stream(n_blocks)(blocksize=BLOCK, callback=session.callback):
        pass
    return session


SETTINGS = {"mode": "paso_directo", "seconds_requested": 1.0, "blocksize": BLOCK, "latency_requested": "low"}
DEVICES = {"input": {"index": 1, "name": "a", "hostapi": "MME"}, "output": {"index": 4, "name": "b", "hostapi": "MME"}}
VERSIONS = {"python": "3", "sounddevice": "0", "portaudio": "19"}


# ---------------------------------------------------------------------------
# Opciones de línea de comandos
# ---------------------------------------------------------------------------

def test_parse_args_valores_por_defecto():
    args = run.parse_args([])
    assert (args.mode, args.seconds, args.input_device, args.output_device, args.sample_rate, args.blocksize,
            args.latency, args.list_devices) == ("paso_directo", 10.0, None, None, 16000, 160, "low", False)


def test_output_root_por_defecto_es_datos_en_vivo_junto_al_repo():
    repo = Path(__file__).resolve().parents[3]
    assert (repo / "pyproject.toml").exists()
    assert Path(run.parse_args([]).output_root) == repo.parent / "Datos" / "en_vivo"


@pytest.mark.parametrize("text, expected", [("low", "low"), ("high", "high"), ("0.05", 0.05), ("1", 1.0)])
def test_latency_acepta_low_high_y_numeros(text, expected):
    assert run.parse_args(["--latency", text]).latency == expected


@pytest.mark.parametrize("extra", [
    ("--latency", "medium"), ("--latency", "0"), ("--latency", "-0.1"), ("--latency", "nan"), ("--latency", "inf"),
    ("--seconds", "0"), ("--seconds", "-1"), ("--seconds", "x"), ("--seconds", "inf"),
    ("--sample-rate", "0"), ("--sample-rate", "16000.5"), ("--blocksize", "0"), ("--blocksize", "-160"),
    ("--input-device", "x"), ("--output-device", "1.5"), ("--mode", "otro"),
])
def test_parse_args_rechaza_valores_invalidos(extra, capsys):
    with pytest.raises(SystemExit) as info:
        run.parse_args(list(extra))
    assert info.value.code == 2
    assert extra[0] in capsys.readouterr().err


def test_valores_por_defecto_del_stream_y_de_la_espera():
    for function in (run.run_session, run.main):
        parameters = inspect.signature(function).parameters
        assert parameters["stream_class"].default is sd.Stream
        assert parameters["wait"].default is time.sleep


def test_se_ejecuta_como_modulo():
    proc = subprocess.run([sys.executable, "-m", "mcadams_app.live.run", "--help"], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    assert proc.returncode == 0
    assert "--list-devices" in proc.stdout


# ---------------------------------------------------------------------------
# --list-devices
# ---------------------------------------------------------------------------

def test_list_devices_imprime_y_no_abre_ningun_stream(fake_sd, tmp_path, capsys):
    root = tmp_path / "salida"
    run.main(["--list-devices", "--output-root", str(root)], stream_class=ForbiddenStream,
             wait=lambda _seconds: pytest.fail("no debe esperar"))
    assert capsys.readouterr().out.strip() == "LISTA DE DISPOSITIVOS"
    assert fake_sd.checks == [] and not root.exists()


# ---------------------------------------------------------------------------
# Validación de los dispositivos
# ---------------------------------------------------------------------------

def test_validacion_pide_frecuencia_un_canal_y_float32(fake_sd, tmp_path):
    correr(tmp_path, "--sample-rate", "8000", "--blocksize", "80")
    assert fake_sd.checks == [
        ("input", {"device": 1, "channels": 1, "dtype": "float32", "samplerate": 8000}),
        ("output", {"device": 4, "channels": 1, "dtype": "float32", "samplerate": 8000}),
    ]


@pytest.mark.parametrize("failed, device, kind, other", [("input", 1, "entrada", "salida"),
                                                         ("output", 4, "salida", "entrada")])
def test_dispositivo_invalido_da_un_mensaje_claro(fake_sd, tmp_path, failed, device, kind, other):
    fake_sd.rejects = {failed: device}
    root = tmp_path / "salida"
    stream = make_stream()
    with pytest.raises(SystemExit) as info:
        run.main([*ARGS, "--output-root", str(root)], stream_class=stream)
    message = str(info.value.code)
    assert f"dispositivo de {kind} ({device})" in message
    assert f"dispositivo de {other}" not in message
    assert "16000 Hz" in message and "1 canal" in message and "float32" in message
    assert "Invalid sample rate" in message
    assert stream.instances == [] and not root.exists()


def test_los_dos_dispositivos_invalidos_se_reportan_juntos(fake_sd, tmp_path):
    fake_sd.rejects = {"input": 1, "output": 4}
    with pytest.raises(SystemExit) as info:
        correr(tmp_path / "salida")
    assert "dispositivo de entrada (1)" in info.value.code and "dispositivo de salida (4)" in info.value.code


def test_dispositivo_del_sistema_se_nombra_en_el_mensaje(fake_sd):
    fake_sd.rejects = {"input": None}
    with pytest.raises(SystemExit, match="dispositivo de entrada \\(predeterminado del sistema\\)"):
        run.validate_devices(None, 4, FS)


def test_valueerror_de_sounddevice_tambien_da_el_mensaje_claro(monkeypatch):
    def reject(**_kwargs):
        raise ValueError("No input channels")
    monkeypatch.setattr(run.sd, "check_input_settings", reject)
    monkeypatch.setattr(run.sd, "check_output_settings", lambda **_kwargs: None)
    with pytest.raises(SystemExit, match="dispositivo de entrada \\(1\\).*No input channels"):
        run.validate_devices(1, 4, FS)


def test_indice_inexistente_con_sounddevice_real_da_el_mensaje_claro():
    # Sin reemplazos: el índice no existe, así que no se toca ningún dispositivo.
    with pytest.raises(SystemExit) as info:
        run.validate_devices(9999, 9999, FS)
    assert "dispositivo de entrada (9999)" in info.value.code
    assert "dispositivo de salida (9999)" in info.value.code


def test_fallo_al_abrir_el_stream_da_el_mensaje_claro(fake_sd, tmp_path):
    class Failing:
        def __init__(self, **_kwargs):
            raise sd.PortAudioError("Illegal combination of I/O devices", -9993)

    root = tmp_path / "salida"
    with pytest.raises(SystemExit) as info:
        run.main([*ARGS, "--output-root", str(root)], stream_class=Failing)
    assert "Micrófono de prueba" in info.value.code and "Audífonos de prueba" in info.value.code
    assert "Illegal combination" in info.value.code
    assert not root.exists()


# ---------------------------------------------------------------------------
# run_session
# ---------------------------------------------------------------------------

def test_stream_recibe_la_configuracion_y_el_callback_de_la_sesion(fake_sd, tmp_path):
    stream = correr(tmp_path)
    assert len(stream.instances) == 1
    kwargs = dict(stream.instances[0].kwargs)
    callback = kwargs.pop("callback")
    assert kwargs == {"device": (1, 4), "channels": 1, "dtype": "float32", "samplerate": 16000, "blocksize": 160,
                      "latency": "low"}
    assert isinstance(callback.__self__, LiveSession) and callback.__func__ is LiveSession.callback
    assert callback.__self__.processor is None
    # La llamada es válida para la firma del sd.Stream real.
    inspect.signature(sd.Stream.__init__).bind(None, **stream.instances[0].kwargs)


def test_opciones_llegan_al_stream(fake_sd, tmp_path):
    stream = correr(tmp_path, "--sample-rate", "8000", "--blocksize", "80", "--latency", "0.05")
    kwargs = stream.instances[0].kwargs
    assert (kwargs["samplerate"], kwargs["blocksize"], kwargs["latency"]) == (8000, 80, 0.05)


def test_main_espera_los_segundos_pedidos(fake_sd, tmp_path):
    waits = []
    correr(tmp_path, wait=waits.append)
    assert waits == [0.5]


def test_dispositivos_por_defecto_llegan_como_none(fake_sd, tmp_path):
    stream = make_stream()
    run.main(["--output-root", str(tmp_path)], stream_class=stream, wait=lambda _seconds: None)
    assert stream.instances[0].kwargs["device"] == (None, None)
    summary = read_summary(only_run(tmp_path))
    assert (summary["devices"]["input"]["index"], summary["devices"]["output"]["index"]) == (1, 4)


def test_espera_los_segundos_pedidos_con_el_stream_abierto():
    waits = []
    stream = make_stream()

    def wait(seconds):
        instance = stream.instances[0]
        assert instance.entered and not instance.closed
        waits.append(seconds)

    latency = run.run_session(LiveSession(FS), 1, 4, BLOCK, "low", 2.5, stream, wait)
    assert waits == [2.5]
    assert latency == LATENCY
    assert stream.instances[0].closed


def test_ctrl_c_cierra_el_stream_y_conserva_lo_grabado():
    stream = make_stream()
    session = LiveSession(FS)
    latency = run.run_session(session, 1, 4, BLOCK, "low", 2.5, stream, raise_interrupt)
    assert stream.instances[0].closed
    assert latency == LATENCY
    assert len(session.input_blocks) == N_BLOCKS


def test_ctrl_c_antes_de_entrar_al_with_no_rompe():
    class Interrupted:
        def __init__(self, **_kwargs):
            raise KeyboardInterrupt

    session = LiveSession(FS)
    assert run.run_session(session, 1, 4, BLOCK, "low", 1, Interrupted, time.sleep) == (None, None)


def test_otro_error_durante_la_espera_no_se_oculta():
    def fail(_seconds):
        raise RuntimeError("otro error")

    stream = make_stream()
    with pytest.raises(RuntimeError, match="otro error"):
        run.run_session(LiveSession(FS), 1, 4, BLOCK, "low", 1, stream, fail)
    assert stream.instances[0].closed


# ---------------------------------------------------------------------------
# Archivos de salida
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["entrada.wav", "salida.wav"])
def test_wav_tienen_n_por_blocksize_muestras_y_el_contenido_esperado(fake_sd, tmp_path, name):
    correr(tmp_path)
    rate, data = wav.read(only_run(tmp_path) / name)
    assert rate == FS
    assert data.dtype == np.int16 and data.ndim == 1
    assert data.size == N_BLOCKS * BLOCK
    np.testing.assert_array_equal(data, float_to_int16(known_audio()))


def test_wav_usan_la_frecuencia_de_la_sesion(fake_sd, tmp_path):
    correr(tmp_path, "--sample-rate", "8000", "--blocksize", "80")
    for name in ("entrada.wav", "salida.wav"):
        rate, data = wav.read(only_run(tmp_path) / name)
        assert rate == 8000 and data.size == N_BLOCKS * 80


def test_entrada_y_salida_no_se_confunden_con_un_procesador(tmp_path):
    # El diseño ya admite un procesador: basta con configurarlo en la sesión.
    session = session_with_clock(processor=lambda block: -block)
    run_dir, _ = run.save_results(session, LATENCY, SETTINGS, DEVICES, VERSIONS, tmp_path)
    np.testing.assert_array_equal(wav.read(run_dir / "entrada.wav")[1], float_to_int16(known_audio(4)))
    np.testing.assert_array_equal(wav.read(run_dir / "salida.wav")[1], float_to_int16(-known_audio(4)))


def test_el_modo_nombra_la_carpeta_y_queda_en_el_resumen(tmp_path):
    settings = {**SETTINGS, "mode": "otro_modo"}
    run_dir, summary = run.save_results(session_with_clock(), LATENCY, settings, DEVICES, VERSIONS, tmp_path)
    assert run_dir.name.endswith("_otro_modo") and summary["config"]["mode"] == "otro_modo"


def test_tiempos_tsv_tiene_una_fila_por_bloque(fake_sd, tmp_path):
    correr(tmp_path)
    run_dir = only_run(tmp_path)
    header, *rows = read_times(run_dir)
    assert header == ["block_index", "process_ms"]
    assert len(rows) == N_BLOCKS
    assert [int(r[0]) for r in rows] == list(range(N_BLOCKS))
    times = np.array([float(r[1]) for r in rows])
    session = read_summary(run_dir)["session"]
    assert np.all(times >= 0)
    assert session["block_ms_median"] == pytest.approx(np.median(times))
    assert session["block_ms_max"] == pytest.approx(np.max(times))


def test_tiempos_tsv_esta_en_milisegundos(tmp_path):
    run_dir, _ = run.save_results(session_with_clock(step=0.002), LATENCY, SETTINGS, DEVICES, VERSIONS, tmp_path)
    assert [float(r[1]) for r in read_times(run_dir)[1:]] == pytest.approx([2.0] * 4)


def test_resumen_json_tiene_todas_las_claves(fake_sd, tmp_path):
    correr(tmp_path)
    summary = read_summary(only_run(tmp_path))
    assert set(summary) == {"session", "stream_latency_ms", "config", "anonymizer", "devices", "versions"}
    assert set(summary["session"]) == SESSION_KEYS
    assert set(summary["anonymizer"]) == {"session_id", "coefficient", "delay_samples", "delay_ms"}
    assert set(summary["stream_latency_ms"]) == {"input", "output"}
    assert set(summary["config"]) == {"mode", "seconds_requested", "seconds_recorded", "sample_rate", "blocksize",
                                      "latency_requested"}
    assert set(summary["devices"]) == {"input", "output"}
    assert set(summary["devices"]["input"]) == set(summary["devices"]["output"]) == {"index", "name", "hostapi"}
    assert set(summary["versions"]) == {"python", "sounddevice", "portaudio"}


def test_resumen_json_valores_coherentes(fake_sd, tmp_path):
    correr(tmp_path)
    run_dir = only_run(tmp_path)
    assert RUN_NAME.match(run_dir.name)
    summary = read_summary(run_dir)
    session = summary["session"]
    assert (session["n_blocks"], session["input_overflows"], session["input_underflows"],
            session["output_overflows"], session["output_underflows"]) == (N_BLOCKS, 0, 0, 0, 0)
    assert session["block_ms_median"] <= session["block_ms_p99"] <= session["block_ms_max"]
    assert summary["stream_latency_ms"]["input"] == pytest.approx(20.0)
    assert summary["stream_latency_ms"]["output"] == pytest.approx(30.0)
    assert summary["config"] == {
        "mode": "paso_directo", "seconds_requested": 0.5, "seconds_recorded": N_BLOCKS * BLOCK / FS,
        "sample_rate": FS, "blocksize": BLOCK, "latency_requested": "low",
    }
    assert summary["devices"] == {
        "input": {"index": 1, "name": "Micrófono de prueba", "hostapi": "MME"},
        "output": {"index": 4, "name": "Audífonos de prueba", "hostapi": "WASAPI"},
    }
    assert summary["versions"] == {"python": sys.version.split()[0], "sounddevice": "9.9.9",
                                   "portaudio": "PortAudio de prueba"}


def test_resumen_json_coincide_con_la_sesion_y_conserva_los_acentos(tmp_path):
    session = session_with_clock()
    devices = {**DEVICES, "input": {"index": 1, "name": "Micrófono", "hostapi": "MME"}}
    run_dir, summary = run.save_results(session, LATENCY, SETTINGS, devices, VERSIONS, tmp_path)
    assert read_summary(run_dir) == summary
    assert summary["session"] == session.summary()
    assert "Micrófono" in (run_dir / "resumen.json").read_text(encoding="utf-8")


@pytest.mark.parametrize("extra", [(), ("--mode", "anonimizado")])
def test_resumen_y_consola_cuentan_las_cuatro_banderas(fake_sd, tmp_path, capsys, extra):
    flags = {name: sd.CallbackFlags() for name in ("in_over", "in_under", "out_over", "out_under")}
    flags["in_over"].input_overflow = True
    flags["in_under"].input_underflow = True
    flags["out_over"].output_overflow = True
    flags["out_under"].output_underflow = True
    by_block = {0: flags["in_over"], 1: flags["in_under"], 2: flags["in_under"], 3: flags["out_over"],
                4: flags["out_over"], 5: flags["out_over"], 6: flags["out_under"], 7: flags["out_under"],
                8: flags["out_under"], 9: flags["out_under"]}
    correr(tmp_path, *extra, stream_class=make_stream(flags=by_block))
    session = read_summary(only_run(tmp_path))["session"]
    assert (session["input_overflows"], session["input_underflows"], session["output_overflows"],
            session["output_underflows"]) == (1, 2, 3, 4)
    out = capsys.readouterr().out
    for line in ("Overflows de entrada: 1", "Underflows de entrada: 2", "Overflows de salida: 3",
                 "Underflows de salida: 4"):
        assert line in out


def test_latencia_pedida_numerica_queda_en_el_resumen(fake_sd, tmp_path):
    correr(tmp_path, "--latency", "0.05")
    assert read_summary(only_run(tmp_path))["config"]["latency_requested"] == 0.05


def test_version_info_con_sounddevice_real():
    versions = run.version_info()
    assert versions["python"] == sys.version.split()[0]
    assert versions["sounddevice"] == sd.__version__
    assert versions["portaudio"] == sd.get_portaudio_version()[1]


# ---------------------------------------------------------------------------
# Ctrl+C
# ---------------------------------------------------------------------------

def test_ctrl_c_durante_la_espera_igual_guarda_los_archivos(fake_sd, tmp_path, capsys):
    stream = correr(tmp_path, wait=raise_interrupt)
    assert stream.instances[0].closed
    run_dir = only_run(tmp_path)
    assert sorted(p.name for p in run_dir.iterdir()) == ["entrada.wav", "resumen.json", "salida.wav", "tiempos.tsv"]
    assert wav.read(run_dir / "entrada.wav")[1].size == N_BLOCKS * BLOCK
    assert len(read_times(run_dir)) == N_BLOCKS + 1
    summary = read_summary(run_dir)
    assert summary["session"]["n_blocks"] == N_BLOCKS
    assert (summary["config"]["seconds_requested"], summary["config"]["seconds_recorded"]) == (0.5, 0.25)
    assert summary["stream_latency_ms"]["input"] == pytest.approx(20.0)
    assert "Bloques: 25" in capsys.readouterr().out


def test_ctrl_c_sin_ningun_bloque_guarda_archivos_vacios(fake_sd, tmp_path, capsys):
    correr(tmp_path, stream_class=make_stream(n_blocks=0), wait=raise_interrupt)
    run_dir = only_run(tmp_path)
    for name in ("entrada.wav", "salida.wav"):
        rate, data = wav.read(run_dir / name)
        assert rate == FS and data.size == 0
    assert read_times(run_dir) == [["block_index", "process_ms"]]
    summary = read_summary(run_dir)
    assert summary["session"]["n_blocks"] == 0 and summary["session"]["rtf"] is None
    assert set(summary["session"]) == SESSION_KEYS
    assert summary["config"]["seconds_recorded"] == 0
    out = capsys.readouterr().out
    assert "Bloques: 0" in out and "mediana n/d ms, p99 n/d ms, máximo n/d ms" in out and "RTF: n/d" in out


# ---------------------------------------------------------------------------
# Carpetas de cada corrida
# ---------------------------------------------------------------------------

def test_dos_corridas_seguidas_no_escriben_en_la_misma_carpeta(fake_sd, tmp_path):
    correr(tmp_path)
    first = only_run(tmp_path)
    before = {p.name: p.read_bytes() for p in first.iterdir()}
    correr(tmp_path, "--seconds", "0.75")
    dirs = sorted(p for p in tmp_path.iterdir())
    assert len(dirs) == 2
    assert {p.name: p.read_bytes() for p in first.iterdir()} == before


def test_dos_corridas_en_el_mismo_segundo_usan_carpetas_distintas(tmp_path):
    now = lambda: datetime(2026, 11, 4, 9, 5, 7)
    dirs = [run.save_results(session_with_clock(), LATENCY, SETTINGS, DEVICES, VERSIONS, tmp_path, now)[0]
            for _ in range(3)]
    assert [d.name for d in dirs] == ["20261104_090507_paso_directo", "20261104_090507_paso_directo_2",
                                      "20261104_090507_paso_directo_3"]
    assert all((d / "resumen.json").exists() for d in dirs)


def test_crea_la_carpeta_raiz_si_no_existe(fake_sd, tmp_path):
    root = tmp_path / "a" / "b"
    correr(root)
    assert RUN_NAME.match(only_run(root).name)


def test_no_borra_lo_que_ya_hay_en_la_carpeta_raiz(fake_sd, tmp_path):
    (tmp_path / "otra").mkdir()
    (tmp_path / "otra" / "dato.txt").write_text("x", encoding="utf-8")
    correr(tmp_path)
    assert (tmp_path / "otra" / "dato.txt").read_text(encoding="utf-8") == "x"
    assert len(list(tmp_path.iterdir())) == 2


# ---------------------------------------------------------------------------
# Resumen en la consola
# ---------------------------------------------------------------------------

def test_print_report_exacto(capsys):
    summary = {
        "session": {"n_blocks": 2000, "input_overflows": 3, "input_underflows": 5, "output_overflows": 6,
                    "output_underflows": 4, "block_ms_median": 0.0125, "block_ms_p99": 0.5, "block_ms_max": 1.25,
                    "rtf": 0.00123},
        "stream_latency_ms": {"input": 20.0, "output": 31.5},
        "config": {"blocksize": 160, "sample_rate": 16000},
        "anonymizer": {"session_id": None, "coefficient": None, "delay_samples": 0, "delay_ms": 0.0},
    }
    run.print_report("carpeta", summary)
    assert capsys.readouterr().out.splitlines() == [
        "Carpeta de salida: carpeta",
        "Bloques: 2000",
        "Overflows de entrada: 3",
        "Underflows de entrada: 5",
        "Overflows de salida: 6",
        "Underflows de salida: 4",
        "Tiempo por bloque: mediana 0,013 ms, p99 0,500 ms, máximo 1,250 ms",
        "RTF: 0,0012",
        "Latencia del dispositivo: entrada 20,00 ms, salida 31,50 ms",
        "Latencia estimada (entrada + salida + un bloque de 10,00 ms): 61,50 ms",
    ]


def test_print_report_sin_latencia_del_stream(capsys):
    summary = {
        "session": {"n_blocks": 0, "input_overflows": 0, "input_underflows": 0, "output_overflows": 0,
                    "output_underflows": 0, "block_ms_median": None, "block_ms_p99": None, "block_ms_max": None,
                    "rtf": None},
        "stream_latency_ms": {"input": None, "output": None},
        "config": {"blocksize": 160, "sample_rate": 16000},
        "anonymizer": {"session_id": None, "coefficient": None, "delay_samples": 0, "delay_ms": 0.0},
    }
    run.print_report("carpeta", summary)
    out = capsys.readouterr().out
    assert "entrada n/d ms, salida n/d ms" in out and "10,00 ms): n/d ms" in out


def test_main_imprime_la_carpeta_y_la_latencia_estimada(fake_sd, tmp_path, capsys):
    correr(tmp_path, "--sample-rate", "8000", "--blocksize", "160")
    out = capsys.readouterr().out
    assert f"Carpeta de salida: {only_run(tmp_path)}" in out
    assert "Latencia estimada (entrada + salida + un bloque de 20,00 ms): 70,00 ms" in out


def test_print_report_anonimizado_exacto(capsys):
    summary = {
        "session": {"n_blocks": 2000, "input_overflows": 0, "input_underflows": 0, "output_overflows": 0,
                    "output_underflows": 0, "block_ms_median": 0.5, "block_ms_p99": 1.0, "block_ms_max": 2.0,
                    "rtf": 0.05},
        "stream_latency_ms": {"input": 20.0, "output": 31.5},
        "config": {"blocksize": 160, "sample_rate": 16000},
        "anonymizer": {"session_id": "abc", "coefficient": 0.76543, "delay_samples": 160, "delay_ms": 10.0},
    }
    run.print_report("carpeta", summary)
    lines = capsys.readouterr().out.splitlines()
    assert lines[-3:] == [
        "Coeficiente alfa de la sesión: 0,7654",
        "Retraso algorítmico del adaptador: 160 muestras (10,00 ms)",
        "Latencia estimada (entrada + salida + un bloque de 10,00 ms + retraso del adaptador de 10,00 ms): 71,50 ms",
    ]
    assert "\n".join(lines).encode("cp1252")  # la consola de Windows puede escribirlo


# ---------------------------------------------------------------------------
# Modo paso directo: sigue igual
# ---------------------------------------------------------------------------

def test_paso_directo_no_tiene_procesador_y_la_salida_es_la_entrada(fake_sd, tmp_path, capsys):
    stream = correr(tmp_path)
    assert stream.instances[0].kwargs["callback"].__self__.processor is None
    run_dir = only_run(tmp_path)
    assert RUN_NAME.match(run_dir.name)
    np.testing.assert_array_equal(wav.read(run_dir / "salida.wav")[1], wav.read(run_dir / "entrada.wav")[1])
    summary = read_summary(run_dir)
    assert summary["config"]["mode"] == "paso_directo"
    assert summary["anonymizer"] == {"session_id": None, "coefficient": None, "delay_samples": 0, "delay_ms": 0.0}
    out = capsys.readouterr().out
    assert "alfa" not in out and "adaptador" not in out
    assert "Latencia estimada (entrada + salida + un bloque de 10,00 ms): 60,00 ms" in out


# ---------------------------------------------------------------------------
# Modo anonimizado
# ---------------------------------------------------------------------------

def correr_anonimizado(root, *extra, **kwargs):
    return correr(root, "--mode", "anonimizado", *extra, **kwargs)


def test_anonimizado_escribe_los_archivos_en_su_carpeta(fake_sd, tmp_path):
    correr_anonimizado(tmp_path)
    run_dir = only_run(tmp_path)
    assert ANONYMIZED_NAME.match(run_dir.name)
    assert sorted(p.name for p in run_dir.iterdir()) == ["entrada.wav", "resumen.json", "salida.wav", "tiempos.tsv"]
    assert len(read_times(run_dir)) == N_BLOCKS + 1
    for name in ("entrada.wav", "salida.wav"):
        rate, data = wav.read(run_dir / name)
        assert rate == FS and data.size == N_BLOCKS * BLOCK


def test_anonimizado_usa_block_anonymizer_como_procesador(fake_sd, tmp_path):
    stream = correr_anonimizado(tmp_path)
    processor = stream.instances[0].kwargs["callback"].__self__.processor
    assert isinstance(processor.__self__, BlockAnonymizer) and processor.__func__ is BlockAnonymizer.process
    assert processor.__self__.blocksize == BLOCK
    assert stream.instances[0].kwargs["device"] == (1, 4)


def test_anonimizado_resumen_con_las_claves_nuevas(fake_sd, tmp_path):
    correr_anonimizado(tmp_path)
    summary = read_summary(only_run(tmp_path))
    anonymizer = summary["anonymizer"]
    assert summary["config"]["mode"] == "anonimizado"
    assert set(summary["session"]) == SESSION_KEYS
    assert uuid.UUID(anonymizer["session_id"]).version == 4
    assert anonymizer["coefficient"] == generate_coefficient(anonymizer["session_id"])[0]
    assert MIN_MCADAMS_COEFFICIENT <= anonymizer["coefficient"] <= MAX_MCADAMS_COEFFICIENT
    assert (anonymizer["delay_samples"], anonymizer["delay_ms"]) == (160, pytest.approx(10.0))


def test_anonimizado_el_retraso_depende_del_blocksize(fake_sd, tmp_path):
    correr_anonimizado(tmp_path, "--blocksize", "100")
    anonymizer = read_summary(only_run(tmp_path))["anonymizer"]
    assert (anonymizer["delay_samples"], anonymizer["delay_ms"]) == (300, pytest.approx(18.75))


def test_anonimizado_salida_distinta_de_la_entrada_y_entrada_original(fake_sd, tmp_path):
    correr_anonimizado(tmp_path)
    run_dir = only_run(tmp_path)
    entrada, salida = wav.read(run_dir / "entrada.wav")[1], wav.read(run_dir / "salida.wav")[1]
    assert read_summary(run_dir)["anonymizer"]["coefficient"] < 1
    np.testing.assert_array_equal(entrada, float_to_int16(known_audio()))
    assert np.mean(np.abs(entrada.astype(float) - salida.astype(float))) > 100


def test_anonimizado_salida_es_el_anonimizador_evaluado_retrasado(fake_sd, tmp_path):
    correr_anonimizado(tmp_path)
    run_dir = only_run(tmp_path)
    summary = read_summary(run_dir)["anonymizer"]
    salida = wav.read(run_dir / "salida.wav")[1]
    delay = summary["delay_samples"]
    np.testing.assert_array_equal(salida[:delay], np.zeros(delay, dtype=np.int16))
    audio = known_audio().astype(np.float64)
    reference, _ = anonymize_utterance(audio, FS, summary["session_id"])
    expected = float_to_int16(np.clip(reference[:audio.size - delay], -1.0, 1.0))
    # Tolerancia de 1 paso de cuantización por pasar la salida a float32 antes del WAV.
    assert np.max(np.abs(salida[delay:].astype(int) - expected.astype(int))) <= 1


def test_anonimizado_dos_corridas_tienen_session_id_distinto(fake_sd, tmp_path):
    correr_anonimizado(tmp_path)
    correr_anonimizado(tmp_path)
    ids = {read_summary(d)["anonymizer"]["session_id"] for d in tmp_path.iterdir()}
    assert len(ids) == 2


def test_anonimizado_consola_con_alfa_retraso_y_latencia(fake_sd, tmp_path, capsys):
    correr_anonimizado(tmp_path)
    out = capsys.readouterr().out
    coefficient = read_summary(only_run(tmp_path))["anonymizer"]["coefficient"]
    assert f"Coeficiente alfa de la sesión: {coefficient:.4f}".replace(".", ",") in out
    assert "Retraso algorítmico del adaptador: 160 muestras (10,00 ms)" in out
    assert "Latencia estimada (entrada + salida + un bloque de 10,00 ms + retraso del adaptador de 10,00 ms): 70,00 ms" in out


def test_anonimizado_ctrl_c_igual_guarda(fake_sd, tmp_path):
    correr_anonimizado(tmp_path, wait=raise_interrupt)
    run_dir = only_run(tmp_path)
    assert read_summary(run_dir)["session"]["n_blocks"] == N_BLOCKS
    assert wav.read(run_dir / "salida.wav")[1].size == N_BLOCKS * BLOCK


def test_build_anonymizer():
    assert run.build_anonymizer("paso_directo", FS, BLOCK) is None
    anonymizer = run.build_anonymizer("anonimizado", 8000, 80)
    assert anonymizer.blocksize == 80 and anonymizer.delay_samples == 80
    assert uuid.UUID(anonymizer.session_id).version == 4
