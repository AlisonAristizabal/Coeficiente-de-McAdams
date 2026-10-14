import numpy as np
import pytest
import soundfile

import anonymize_subset as modulo
from anonymize_subset import anonymize_subset, process_clip
from mcadams_core.config.audio_defaults import TARGET_SAMPLE_RATE
from mcadams_core.config.mcadams_defaults import (
    MAX_MCADAMS_COEFFICIENT,
    MIN_MCADAMS_COEFFICIENT,
)
from mcadams_core.io.audio_conversion import convert_mp3_to_wav
from mcadams_core.io.audio_io import load_wav
from mcadams_core.session.coefficient import generate_coefficient

COLUMNAS_MANIFIESTO = [
    "client_id", "stratum", "path", "session_id", "coefficient", "status", "error",
]


def crear_mp3(ruta, segundos=0.5, tasa=48000, semilla=0):
    """Escribe un MP3 de prueba (tono más ruido) de la duración pedida."""
    rng = np.random.default_rng(semilla)
    t = np.arange(int(tasa * segundos)) / tasa
    senal = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(len(t))
    soundfile.write(ruta, senal, tasa, format="MP3")
    return ruta


def fila(nombre, client_id="a1", stratum="mexico"):
    return {"client_id": client_id, "stratum": stratum, "path": nombre, "extra": "x"}


@pytest.fixture
def carpetas(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    return clips, tmp_path / "original", tmp_path / "anon"


# --- process_clip ---------------------------------------------------------

def test_process_clip_crea_los_dos_wav_a_16_khz(tmp_path):
    mp3 = crear_mp3(tmp_path / "c.mp3")
    process_clip(mp3, tmp_path / "o.wav", tmp_path / "a.wav", "c")
    for ruta in (tmp_path / "o.wav", tmp_path / "a.wav"):
        tasa, audio = load_wav(ruta)
        assert tasa == TARGET_SAMPLE_RATE
        assert audio.ndim == 1


def test_process_clip_devuelve_el_coeficiente_de_la_sesion(tmp_path):
    mp3 = crear_mp3(tmp_path / "c.mp3")
    coeficiente = process_clip(mp3, tmp_path / "o.wav", tmp_path / "a.wav", "c")
    assert coeficiente == generate_coefficient("c")[0]
    assert MIN_MCADAMS_COEFFICIENT <= coeficiente <= MAX_MCADAMS_COEFFICIENT


def test_process_clip_el_original_es_solo_la_conversion(tmp_path):
    mp3 = crear_mp3(tmp_path / "c.mp3")
    process_clip(mp3, tmp_path / "o.wav", tmp_path / "a.wav", "c")
    convert_mp3_to_wav(mp3, tmp_path / "solo_conversion.wav")
    _, original = load_wav(tmp_path / "o.wav")
    _, esperado = load_wav(tmp_path / "solo_conversion.wav")
    assert np.array_equal(original, esperado)


def test_process_clip_anonimizado_difiere_del_original_con_misma_longitud(tmp_path):
    mp3 = crear_mp3(tmp_path / "c.mp3")
    process_clip(mp3, tmp_path / "o.wav", tmp_path / "a.wav", "c")
    _, original = load_wav(tmp_path / "o.wav")
    _, anonimizado = load_wav(tmp_path / "a.wav")
    assert len(anonimizado) == len(original)
    assert not np.allclose(anonimizado, original, atol=1e-3)


def test_process_clip_misma_sesion_mismo_resultado(tmp_path):
    mp3 = crear_mp3(tmp_path / "c.mp3")
    process_clip(mp3, tmp_path / "o1.wav", tmp_path / "a1.wav", "c")
    process_clip(mp3, tmp_path / "o2.wav", tmp_path / "a2.wav", "c")
    assert (tmp_path / "a1.wav").read_bytes() == (tmp_path / "a2.wav").read_bytes()


def test_process_clip_sesion_distinta_coeficiente_y_audio_distintos(tmp_path):
    mp3 = crear_mp3(tmp_path / "c.mp3")
    c1 = process_clip(mp3, tmp_path / "o1.wav", tmp_path / "a1.wav", "sesion-1")
    c2 = process_clip(mp3, tmp_path / "o2.wav", tmp_path / "a2.wav", "sesion-2")
    assert c1 != c2
    assert (tmp_path / "a1.wav").read_bytes() != (tmp_path / "a2.wav").read_bytes()


def test_process_clip_mp3_faltante_lanza_error_de_soundfile(tmp_path):
    with pytest.raises(soundfile.LibsndfileError):
        process_clip(tmp_path / "no_existe.mp3", tmp_path / "o.wav", tmp_path / "a.wav", "c")


# --- anonymize_subset: caso normal ---------------------------------------

def test_anonymize_subset_procesa_todos_los_clips(carpetas):
    clips, original, anon = carpetas
    for i in range(3):
        crear_mp3(clips / f"clip{i}.mp3", semilla=i)
    filas = [fila(f"clip{i}.mp3", client_id=f"h{i}") for i in range(3)]
    manifiesto = anonymize_subset(filas, clips, original, anon)
    assert [r["status"] for r in manifiesto] == ["ok", "ok", "ok"]
    for i in range(3):
        assert (original / f"clip{i}.wav").exists()
        assert (anon / f"clip{i}.wav").exists()


def test_anonymize_subset_registro_tiene_las_llaves_y_valores_esperados(carpetas):
    clips, original, anon = carpetas
    crear_mp3(clips / "clip0.mp3")
    manifiesto = anonymize_subset([fila("clip0.mp3", "h9", "caribe")], clips, original, anon)
    registro = manifiesto[0]
    assert list(registro.keys()) == COLUMNAS_MANIFIESTO
    assert registro["client_id"] == "h9"
    assert registro["stratum"] == "caribe"
    assert registro["path"] == "clip0.mp3"
    assert registro["session_id"] == "clip0"
    assert registro["status"] == "ok"
    assert registro["error"] == ""
    assert registro["coefficient"] == generate_coefficient("clip0")[0]


def test_anonymize_subset_cada_clip_usa_su_propio_coeficiente(carpetas):
    clips, original, anon = carpetas
    for i in range(3):
        crear_mp3(clips / f"clip{i}.mp3", semilla=i)
    manifiesto = anonymize_subset([fila(f"clip{i}.mp3") for i in range(3)], clips, original, anon)
    coeficientes = [r["coefficient"] for r in manifiesto]
    assert len(set(coeficientes)) == 3


def test_anonymize_subset_conserva_el_orden_de_las_filas(carpetas):
    clips, original, anon = carpetas
    nombres = ["z.mp3", "a.mp3", "m.mp3"]
    for nombre in nombres:
        crear_mp3(clips / nombre)
    manifiesto = anonymize_subset([fila(n) for n in nombres], clips, original, anon)
    assert [r["path"] for r in manifiesto] == nombres


def test_anonymize_subset_crea_las_carpetas_de_salida(carpetas):
    clips, original, anon = carpetas
    assert not original.exists() and not anon.exists()
    anonymize_subset([], clips, original, anon)
    assert original.is_dir() and anon.is_dir()


def test_anonymize_subset_sin_filas_devuelve_lista_vacia(carpetas):
    clips, original, anon = carpetas
    assert anonymize_subset([], clips, original, anon) == []


def test_anonymize_subset_acepta_un_generador_y_rutas_como_texto(carpetas):
    clips, original, anon = carpetas
    crear_mp3(clips / "clip0.mp3")
    filas = (f for f in [fila("clip0.mp3")])
    manifiesto = anonymize_subset(filas, str(clips), str(original), str(anon))
    assert manifiesto[0]["status"] == "ok"


def test_anonymize_subset_no_modifica_las_filas_de_entrada(carpetas):
    clips, original, anon = carpetas
    crear_mp3(clips / "clip0.mp3")
    filas = [fila("clip0.mp3")]
    copia = [dict(f) for f in filas]
    anonymize_subset(filas, clips, original, anon)
    assert filas == copia


# --- anonymize_subset: fallos --------------------------------------------

def test_anonymize_subset_clip_faltante_se_registra_y_el_loop_continua(carpetas):
    clips, original, anon = carpetas
    crear_mp3(clips / "bueno1.mp3")
    crear_mp3(clips / "bueno2.mp3", semilla=1)
    filas = [fila("bueno1.mp3"), fila("falta.mp3"), fila("bueno2.mp3")]
    manifiesto = anonymize_subset(filas, clips, original, anon)
    assert [r["status"] for r in manifiesto] == ["ok", "error", "ok"]
    assert manifiesto[1]["coefficient"] == ""
    assert manifiesto[1]["session_id"] == "falta"
    assert "LibsndfileError" in manifiesto[1]["error"]


def test_anonymize_subset_mp3_corrupto_se_registra_y_no_deja_wav(carpetas):
    clips, original, anon = carpetas
    (clips / "roto.mp3").write_bytes(b"esto no es un mp3" * 100)
    manifiesto = anonymize_subset([fila("roto.mp3")], clips, original, anon)
    assert manifiesto[0]["status"] == "error"
    assert not (anon / "roto.wav").exists()


def test_anonymize_subset_mp3_vacio_se_registra_como_error(carpetas):
    clips, original, anon = carpetas
    (clips / "vacio.mp3").write_bytes(b"")
    manifiesto = anonymize_subset([fila("vacio.mp3")], clips, original, anon)
    assert manifiesto[0]["status"] == "error"


def test_anonymize_subset_el_mensaje_de_error_es_de_una_sola_linea(carpetas, monkeypatch):
    clips, original, anon = carpetas

    def falla(*args, **kwargs):
        raise soundfile.LibsndfileError(1, prefix="Error\tcon\ntabulación y\r\nsaltos")

    monkeypatch.setattr(modulo, "process_clip", falla)
    manifiesto = anonymize_subset([fila("x.mp3")], clips, original, anon)
    mensaje = manifiesto[0]["error"]
    assert manifiesto[0]["status"] == "error"
    assert "\t" not in mensaje and "\n" not in mensaje and "\r" not in mensaje
    assert "  " not in mensaje


def test_anonymize_subset_errores_que_no_son_de_datos_se_propagan(carpetas, monkeypatch):
    clips, original, anon = carpetas

    def defecto(*args, **kwargs):
        raise RuntimeError("defecto del programa")

    monkeypatch.setattr(modulo, "process_clip", defecto)
    with pytest.raises(RuntimeError):
        anonymize_subset([fila("x.mp3")], clips, original, anon)


def test_anonymize_subset_fila_sin_columna_path_lanza_key_error(carpetas):
    clips, original, anon = carpetas
    with pytest.raises(KeyError):
        anonymize_subset([{"client_id": "a1", "stratum": "mexico"}], clips, original, anon)