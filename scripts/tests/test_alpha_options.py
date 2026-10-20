"""Pruebas de las opciones nuevas de α: etiquetas y anonymize_subset (--alpha, --alpha-range, --reuse-original)."""
import sys

import numpy as np
import pytest
import soundfile

import anonymize_subset as modulo
from alpha_config import DEFAULT_RANGE, config_tag, validate_range
from anonymize_subset import anonymize_subset, main, parse_args, process_clip
from common_voice_corpus import read_tsv_rows
from mcadams_core.session.coefficient import generate_coefficient


# ---------------------------------------------------------------------------
# alpha_config
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha, tag", [(0.2, "a020"), (0.3, "a030"), (0.5, "a050"), (0.7, "a070"),
                                        (1.0, "a100"), (0.25, "a025")])
def test_config_tag_alpha_fijo(alpha, tag):
    assert config_tag(alpha, alpha) == tag


@pytest.mark.parametrize("low, high, tag", [(0.7, 0.9, "u07_09"), (0.6, 0.8, "u06_08"), (0.65, 0.85, "u065_085"),
                                            (0.7, 1.0, "u07_1")])
def test_config_tag_rango(low, high, tag):
    assert config_tag(low, high) == tag


def test_config_tag_rango_por_defecto_es_el_de_siempre():
    assert config_tag(*DEFAULT_RANGE) == "u07_09"


@pytest.mark.parametrize("low, high", [(0, 0.5), (0.5, 1.2), (0.9, 0.7), (-0.1, 0.2)])
def test_validate_range_rechaza(low, high):
    with pytest.raises(ValueError, match="0 < alpha_min <= alpha_max <= 1"):
        validate_range(low, high)


def test_validate_range_acepta():
    validate_range(0.5, 0.5)
    validate_range(0.01, 1.0)


# ---------------------------------------------------------------------------
# Datos de prueba
# ---------------------------------------------------------------------------

def crear_mp3(ruta, semilla=0, tasa=48000):
    rng = np.random.default_rng(semilla)
    t = np.arange(int(tasa * 0.5)) / tasa
    soundfile.write(ruta, 0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(len(t)), tasa,
                    format="MP3")


@pytest.fixture
def escenario(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    for i in range(3):
        crear_mp3(clips / f"c{i}.mp3", semilla=i)
    subset = tmp_path / "subset.tsv"
    subset.write_text("client_id\tstratum\tpath\taccents\tsentence\tduration_ms\n"
                      + "".join(f"h{i}\tmexico\tc{i}.mp3\tx\tfrase\t500\n" for i in range(3)), encoding="utf-8")
    return subset, clips, tmp_path / "salida"


def correr(monkeypatch, subset, clips, salida, *extra):
    monkeypatch.setattr(sys, "argv", ["anonymize_subset.py", "--subset", str(subset), "--clips-dir", str(clips),
                                      "--output-dir", str(salida), *extra])
    main()


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

def argumentos(monkeypatch, *extra):
    monkeypatch.setattr(sys, "argv", ["x", "--subset", "s", "--clips-dir", "c", "--output-dir", "o", *extra])
    return parse_args()


def test_parse_args_sin_alpha_usa_el_rango_por_defecto(monkeypatch):
    args = argumentos(monkeypatch)
    assert (args.alpha_min, args.alpha_max, args.tag, args.reuse_original) == (None, None, "u07_09", False)


def test_parse_args_alpha_fijo_y_su_etiqueta(monkeypatch):
    args = argumentos(monkeypatch, "--alpha", "0.2")
    assert (args.alpha_min, args.alpha_max, args.tag) == (0.2, 0.2, "a020")


def test_parse_args_rango_y_su_etiqueta(monkeypatch):
    args = argumentos(monkeypatch, "--alpha-range", "0.6", "0.8")
    assert (args.alpha_min, args.alpha_max, args.tag) == (0.6, 0.8, "u06_08")


def test_parse_args_tag_explicito_tiene_prioridad(monkeypatch):
    assert argumentos(monkeypatch, "--alpha", "0.2", "--tag", "prueba").tag == "prueba"


def test_parse_args_alpha_y_alpha_range_son_excluyentes(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        argumentos(monkeypatch, "--alpha", "0.5", "--alpha-range", "0.6", "0.8")
    assert "not allowed with argument" in capsys.readouterr().err


@pytest.mark.parametrize("extra", [("--alpha", "0"), ("--alpha", "1.5"), ("--alpha-range", "0.9", "0.7")])
def test_parse_args_alpha_invalido_error_claro(monkeypatch, capsys, extra):
    with pytest.raises(SystemExit):
        argumentos(monkeypatch, *extra)
    assert "0 < alpha_min <= alpha_max <= 1" in capsys.readouterr().err


def test_parse_args_reuse_original(monkeypatch):
    assert argumentos(monkeypatch, "--reuse-original").reuse_original is True


# ---------------------------------------------------------------------------
# process_clip y anonymize_subset con α y reutilización
# ---------------------------------------------------------------------------

def test_process_clip_alpha_fijo(tmp_path):
    crear_mp3(tmp_path / "c.mp3")
    assert process_clip(tmp_path / "c.mp3", tmp_path / "o.wav", tmp_path / "a.wav", "c", 0.5, 0.5) == 0.5


def test_process_clip_sin_alpha_igual_que_antes(tmp_path):
    crear_mp3(tmp_path / "c.mp3")
    assert process_clip(tmp_path / "c.mp3", tmp_path / "o.wav", tmp_path / "a.wav", "c") == generate_coefficient("c")[0]


def test_process_clip_reutiliza_el_original_sin_convertir(tmp_path, monkeypatch):
    crear_mp3(tmp_path / "c.mp3")
    process_clip(tmp_path / "c.mp3", tmp_path / "o.wav", tmp_path / "a1.wav", "c")
    before = (tmp_path / "o.wav").read_bytes()
    calls = []
    monkeypatch.setattr(modulo, "convert_mp3_to_wav", lambda *a: calls.append(a))
    process_clip(tmp_path / "no_existe.mp3", tmp_path / "o.wav", tmp_path / "a2.wav", "c", reuse_original=True)
    assert calls == [] and (tmp_path / "o.wav").read_bytes() == before
    # El anonimizado es idéntico al de convertir de nuevo.
    assert (tmp_path / "a1.wav").read_bytes() == (tmp_path / "a2.wav").read_bytes()


def test_process_clip_reuse_convierte_si_falta_el_original(tmp_path):
    crear_mp3(tmp_path / "c.mp3")
    process_clip(tmp_path / "c.mp3", tmp_path / "o.wav", tmp_path / "a.wav", "c", reuse_original=True)
    assert (tmp_path / "o.wav").exists()


def test_process_clip_sin_reuse_vuelve_a_convertir(tmp_path, monkeypatch):
    crear_mp3(tmp_path / "c.mp3")
    process_clip(tmp_path / "c.mp3", tmp_path / "o.wav", tmp_path / "a.wav", "c")
    calls = []
    real = modulo.convert_mp3_to_wav
    monkeypatch.setattr(modulo, "convert_mp3_to_wav", lambda *a: calls.append(a) or real(*a))
    process_clip(tmp_path / "c.mp3", tmp_path / "o.wav", tmp_path / "a.wav", "c")
    assert len(calls) == 1


def test_anonymize_subset_pasa_alpha_y_reuse_a_cada_clip(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(modulo, "process_clip", lambda *a, **kw: calls.append(kw) or 0.4)
    filas = [{"client_id": "h", "stratum": "s", "path": f"c{i}.mp3"} for i in range(2)]
    manifiesto = anonymize_subset(filas, tmp_path, tmp_path / "o", tmp_path / "a", 0.4, 0.4, reuse_original=True)
    assert calls == [{"alpha_min": 0.4, "alpha_max": 0.4, "reuse_original": True}] * 2
    assert [m["coefficient"] for m in manifiesto] == [0.4, 0.4]


def test_anonymize_subset_sin_opciones_llama_como_antes(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(modulo, "process_clip", lambda *a, **kw: calls.append(kw) or 0.8)
    anonymize_subset([{"client_id": "h", "stratum": "s", "path": "c.mp3"}], tmp_path, tmp_path / "o", tmp_path / "a")
    assert calls == [{}]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def test_main_alpha_fijo_manifiesto_y_carpeta_propios(monkeypatch, escenario):
    subset, clips, salida = escenario
    correr(monkeypatch, subset, clips, salida, "--alpha", "0.5")
    filas = list(read_tsv_rows(salida / "anon_manifest_a050.tsv"))
    assert [float(f["coefficient"]) for f in filas] == [0.5, 0.5, 0.5]
    assert len(list((salida / "anon_a050").glob("*.wav"))) == 3
    assert not (salida / "anon_manifest_u07_09.tsv").exists()


def test_main_rango_nuevo_coeficientes_dentro_del_rango(monkeypatch, escenario):
    subset, clips, salida = escenario
    correr(monkeypatch, subset, clips, salida, "--alpha-range", "0.6", "0.8")
    filas = list(read_tsv_rows(salida / "anon_manifest_u06_08.tsv"))
    assert [float(f["coefficient"]) for f in filas] == [generate_coefficient(f"c{i}", 0.6, 0.8)[0] for i in range(3)]


def test_main_sin_alpha_igual_que_antes(monkeypatch, escenario):
    subset, clips, salida = escenario
    correr(monkeypatch, subset, clips, salida)
    filas = list(read_tsv_rows(salida / "anon_manifest_u07_09.tsv"))
    assert [float(f["coefficient"]) for f in filas] == [generate_coefficient(f"c{i}")[0] for i in range(3)]


def test_main_reuse_original_no_reescribe_los_originales(monkeypatch, escenario):
    subset, clips, salida = escenario
    correr(monkeypatch, subset, clips, salida)
    originales = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (salida / "original").glob("*.wav")}
    for mp3 in clips.glob("*.mp3"):
        mp3.unlink()  # sin MP3: solo puede funcionar si reutiliza los WAV
    correr(monkeypatch, subset, clips, salida, "--alpha", "0.5", "--reuse-original")
    despues = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (salida / "original").glob("*.wav")}
    assert despues == originales
    assert all(f["status"] == "ok" for f in read_tsv_rows(salida / "anon_manifest_a050.tsv"))


def test_main_alpha_uno_casi_no_cambia_el_audio(monkeypatch, escenario):
    subset, clips, salida = escenario
    correr(monkeypatch, subset, clips, salida, "--alpha", "1.0")
    original, _ = soundfile.read(salida / "original" / "c0.wav")
    anon, _ = soundfile.read(salida / "anon_a100" / "c0.wav")
    assert np.corrcoef(original[1000:-1000], anon[1000:-1000])[0, 1] > 0.999
