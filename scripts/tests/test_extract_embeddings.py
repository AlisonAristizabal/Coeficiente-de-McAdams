import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile

import extract_embeddings
from extract_embeddings import (
    TARGET_SAMPLE_RATE,
    embed_clip,
    embed_wavs,
    extract_embeddings as extract,
)

COLUMNAS = ["client_id", "stratum", "path", "session_id", "coefficient", "status", "error"]


# --- apoyos ----------------------------------------------------------------

def vector_de(ruta):
    """Embedding falso de 4 dimensiones que depende de la ruta completa."""
    ruta = Path(ruta)
    semilla = sum(ord(c) for c in f"{ruta.parent.name}/{ruta.name}")
    return np.array([semilla, semilla + 1, semilla + 2, semilla + 3], dtype=np.float64)


class EmbedderEspia:
    def __init__(self, fn=vector_de):
        self.fn = fn
        self.rutas = []

    def __call__(self, ruta):
        self.rutas.append(ruta)
        return self.fn(ruta)


def escribir_manifiesto(ruta, filas):
    with open(ruta, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(COLUMNAS) + "\n")
        for fila in filas:
            f.write("\t".join(fila[c] for c in COLUMNAS) + "\n")


def fila(session_id, status="ok"):
    return {
        "client_id": "h0", "stratum": "caribe", "path": f"{session_id}.mp3",
        "session_id": session_id, "coefficient": "0.8", "status": status,
        "error": "" if status == "ok" else "falló",
    }


# --- embed_wavs -------------------------------------------------------------

def test_embed_wavs_devuelve_matriz_float32_con_una_fila_por_ruta():
    matriz = embed_wavs(["a.wav", "b.wav", "c.wav"], vector_de)
    assert matriz.shape == (3, 4)
    assert matriz.dtype == np.float32


def test_embed_wavs_conserva_el_orden_de_las_rutas():
    rutas = ["z.wav", "a.wav", "m.wav"]
    matriz = embed_wavs(rutas, vector_de)
    for i, ruta in enumerate(rutas):
        assert np.array_equal(matriz[i], vector_de(ruta).astype(np.float32))


def test_embed_wavs_llama_una_vez_a_la_funcion_por_cada_ruta():
    espia = EmbedderEspia()
    embed_wavs(["a.wav", "b.wav"], espia)
    assert espia.rutas == ["a.wav", "b.wav"]


def test_embed_wavs_convierte_float64_a_float32():
    matriz = embed_wavs(["a.wav"], lambda r: np.array([0.1, 0.2], dtype=np.float64))
    assert matriz.dtype == np.float32
    assert matriz[0, 0] == np.float32(0.1)


def test_embed_wavs_con_una_sola_ruta_da_forma_uno_por_dimension():
    assert embed_wavs(["a.wav"], vector_de).shape == (1, 4)


def test_embed_wavs_sin_rutas_devuelve_matriz_vacia_float32():
    matriz = embed_wavs([], vector_de)
    assert matriz.shape == (0, 0)
    assert matriz.dtype == np.float32


def test_embed_wavs_acepta_un_generador():
    matriz = embed_wavs((p for p in ["a.wav", "b.wav"]), vector_de)
    assert matriz.shape == (2, 4)


def test_embed_wavs_no_llama_a_la_funcion_si_no_hay_rutas():
    espia = EmbedderEspia()
    embed_wavs([], espia)
    assert espia.rutas == []


def test_embed_wavs_con_dimensiones_inconsistentes_lanza_value_error():
    dims = {"a.wav": 4, "b.wav": 3}
    with pytest.raises(ValueError):
        embed_wavs(["a.wav", "b.wav"], lambda r: np.zeros(dims[r]))


def test_embed_wavs_propaga_la_excepcion_de_la_funcion():
    def falla(ruta):
        raise RuntimeError("modelo roto")

    with pytest.raises(RuntimeError, match="modelo roto"):
        embed_wavs(["a.wav"], falla)


# --- extract_embeddings -----------------------------------------------------

def test_extract_devuelve_exactamente_las_tres_llaves():
    resultado = extract(["s1"], "orig", "anon", vector_de)
    assert set(resultado.keys()) == {"session_ids", "original", "anonymized"}


def test_extract_busca_session_id_punto_wav_en_cada_carpeta():
    espia = EmbedderEspia()
    extract(["s1", "s2"], "carpeta_o", "carpeta_a", espia)
    esperadas = {
        Path("carpeta_o") / "s1.wav", Path("carpeta_o") / "s2.wav",
        Path("carpeta_a") / "s1.wav", Path("carpeta_a") / "s2.wav",
    }
    assert set(map(Path, espia.rutas)) == esperadas
    assert len(espia.rutas) == 4


def test_extract_no_intercambia_originales_con_anonimizados():
    resultado = extract(["s1", "s2"], "orig", "anon", vector_de)
    for i, s in enumerate(["s1", "s2"]):
        assert np.array_equal(resultado["original"][i], vector_de(Path("orig") / f"{s}.wav").astype(np.float32))
        assert np.array_equal(resultado["anonymized"][i], vector_de(Path("anon") / f"{s}.wav").astype(np.float32))
    assert not np.array_equal(resultado["original"], resultado["anonymized"])


def test_extract_alinea_las_filas_con_session_ids():
    ids = ["c", "a", "b"]
    resultado = extract(ids, "o", "a", vector_de)
    assert list(resultado["session_ids"]) == ids
    assert resultado["original"].shape[0] == resultado["anonymized"].shape[0] == 3
    for i, s in enumerate(ids):
        assert np.array_equal(resultado["original"][i], vector_de(Path("o") / f"{s}.wav").astype(np.float32))


def test_extract_session_ids_es_un_arreglo_de_texto():
    resultado = extract(["s1", "s2"], "o", "a", vector_de)
    assert resultado["session_ids"].dtype.kind == "U"


def test_extract_acepta_un_generador_de_ids():
    resultado = extract((s for s in ["s1", "s2", "s3"]), "o", "a", vector_de)
    assert list(resultado["session_ids"]) == ["s1", "s2", "s3"]
    assert resultado["original"].shape[0] == 3
    assert resultado["anonymized"].shape[0] == 3


def test_extract_acepta_carpetas_como_texto_o_path():
    a = extract(["s1"], "o", "a", vector_de)
    b = extract(["s1"], Path("o"), Path("a"), vector_de)
    assert np.array_equal(a["original"], b["original"])
    assert np.array_equal(a["anonymized"], b["anonymized"])


def test_extract_respeta_los_puntos_dentro_del_session_id():
    espia = EmbedderEspia()
    extract(["sesion.v2"], "o", "a", espia)
    assert {Path(r).name for r in espia.rutas} == {"sesion.v2.wav"}


def test_extract_sin_ids_devuelve_arreglos_vacios():
    resultado = extract([], "o", "a", vector_de)
    assert len(resultado["session_ids"]) == 0
    assert resultado["original"].shape == (0, 0)
    assert resultado["anonymized"].shape == (0, 0)


def test_extract_se_puede_guardar_y_leer_sin_pickle(tmp_path):
    resultado = extract(["s1", "s2"], "o", "a", vector_de)
    destino = tmp_path / "emb.npz"
    np.savez(destino, **resultado)
    with np.load(destino, allow_pickle=False) as datos:
        assert list(datos["session_ids"]) == ["s1", "s2"]
        assert np.array_equal(datos["original"], resultado["original"])
        assert datos["anonymized"].dtype == np.float32


def test_extract_propaga_el_error_si_falta_un_archivo(tmp_path):
    def lee(ruta):
        return np.array(soundfile.read(ruta)[0][:1])

    with pytest.raises(Exception):
        extract(["no_existe"], tmp_path, tmp_path, lee)


# --- embed_clip (con torch y clasificador falsos) ----------------------------

class TensorFalso:
    def __init__(self, array):
        self.array = array

    def unsqueeze(self, dim):
        return TensorFalso(np.expand_dims(self.array, dim))

    def squeeze(self):
        return TensorFalso(np.squeeze(self.array))

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class ClasificadorFalso:
    def __init__(self, dimension=192, eventos=None):
        self.dimension = dimension
        self.entradas = []
        self.eventos = eventos if eventos is not None else []

    def encode_batch(self, lote):
        self.eventos.append("encode")
        self.entradas.append(lote.array)
        salida = np.arange(self.dimension, dtype=np.float32).reshape(1, 1, -1)
        return TensorFalso(salida)


@pytest.fixture
def eventos(monkeypatch):
    """Instala un `torch` falso que registra la entrada y salida de no_grad."""
    registro = []

    class SinGradiente:
        def __enter__(self):
            registro.append("no_grad_on")

        def __exit__(self, *args):
            registro.append("no_grad_off")
            return False

    torch_falso = types.ModuleType("torch")
    torch_falso.no_grad = SinGradiente
    torch_falso.from_numpy = lambda a: TensorFalso(a)
    monkeypatch.setitem(sys.modules, "torch", torch_falso)
    return registro


def escribir_wav(ruta, muestras, sample_rate=TARGET_SAMPLE_RATE):
    soundfile.write(str(ruta), muestras, sample_rate, subtype="FLOAT")
    return ruta


def test_embed_clip_devuelve_vector_1d_float32(tmp_path, eventos):
    ruta = escribir_wav(tmp_path / "a.wav", np.linspace(-0.5, 0.5, 1600, dtype=np.float32))
    resultado = embed_clip(ClasificadorFalso(192), ruta)
    assert resultado.shape == (192,)
    assert resultado.dtype == np.float32
    assert np.array_equal(resultado, np.arange(192, dtype=np.float32))


def test_embed_clip_entrega_al_clasificador_un_lote_de_una_senal(tmp_path, eventos):
    muestras = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
    ruta = escribir_wav(tmp_path / "a.wav", muestras)
    clasificador = ClasificadorFalso()
    embed_clip(clasificador, ruta)
    assert len(clasificador.entradas) == 1
    entrada = clasificador.entradas[0]
    assert entrada.shape == (1, 1600)
    assert entrada.dtype == np.float32
    assert np.array_equal(entrada[0], muestras)


def test_embed_clip_llama_al_clasificador_dentro_de_no_grad(tmp_path, eventos):
    ruta = escribir_wav(tmp_path / "a.wav", np.zeros(1600, dtype=np.float32))
    embed_clip(ClasificadorFalso(eventos=eventos), ruta)
    assert eventos == ["no_grad_on", "encode", "no_grad_off"]


def test_embed_clip_rechaza_otra_frecuencia_de_muestreo(tmp_path, eventos):
    ruta = escribir_wav(tmp_path / "a.wav", np.zeros(800, dtype=np.float32), sample_rate=8000)
    clasificador = ClasificadorFalso()
    with pytest.raises(ValueError, match="16000"):
        embed_clip(clasificador, ruta)
    assert clasificador.entradas == []


def test_embed_clip_rechaza_audio_estereo(tmp_path, eventos):
    ruta = escribir_wav(tmp_path / "a.wav", np.zeros((1600, 2), dtype=np.float32))
    clasificador = ClasificadorFalso()
    with pytest.raises(ValueError, match="mono"):
        embed_clip(clasificador, ruta)
    assert clasificador.entradas == []


def test_embed_clip_el_mensaje_de_error_incluye_la_ruta(tmp_path, eventos):
    ruta = escribir_wav(tmp_path / "raro.wav", np.zeros(800, dtype=np.float32), sample_rate=8000)
    with pytest.raises(ValueError, match="raro.wav"):
        embed_clip(ClasificadorFalso(), ruta)


def test_embed_clip_con_archivo_inexistente_lanza_error(tmp_path, eventos):
    with pytest.raises(Exception):
        embed_clip(ClasificadorFalso(), tmp_path / "no_existe.wav")


# --- parse_args --------------------------------------------------------------

ARGS_BASE = [
    "--manifest", "m.tsv", "--original-dir", "o", "--anon-dir", "a",
    "--model-dir", "modelo", "--output", "salida.npz",
]


def test_parse_args_lee_todos_los_argumentos(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["extract_embeddings.py", *ARGS_BASE, "--limit", "20"])
    args = extract_embeddings.parse_args()
    assert (args.manifest, args.original_dir, args.anon_dir) == ("m.tsv", "o", "a")
    assert (args.model_dir, args.output, args.limit) == ("modelo", "salida.npz", 20)


def test_parse_args_limit_es_opcional_y_vale_none(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["extract_embeddings.py", *ARGS_BASE])
    assert extract_embeddings.parse_args().limit is None


@pytest.mark.parametrize("faltante", ["--manifest", "--original-dir", "--anon-dir", "--model-dir", "--output"])
def test_parse_args_exige_los_argumentos_obligatorios(monkeypatch, faltante):
    argumentos = list(ARGS_BASE)
    i = argumentos.index(faltante)
    del argumentos[i:i + 2]
    monkeypatch.setattr(sys, "argv", ["extract_embeddings.py", *argumentos])
    with pytest.raises(SystemExit):
        extract_embeddings.parse_args()


# --- main --------------------------------------------------------------------

@pytest.fixture
def entorno(tmp_path, monkeypatch):
    """Manifiesto con 6 clips (4 ok, 2 con error) y funciones pesadas sustituidas."""
    filas = [fila("s1"), fila("s2", "error"), fila("s3"), fila("s4"), fila("s5", "error"), fila("s6")]
    manifiesto = tmp_path / "m.tsv"
    escribir_manifiesto(manifiesto, filas)
    cargas = []
    monkeypatch.setattr(extract_embeddings, "load_classifier", lambda d: cargas.append(d) or "clasificador")
    monkeypatch.setattr(
        extract_embeddings, "embed_clip",
        lambda clasificador, ruta: vector_de(ruta) if clasificador == "clasificador" else None,
    )

    def ejecutar(*extra):
        salida = tmp_path / "salida.npz"
        argumentos = [
            "extract_embeddings.py", "--manifest", str(manifiesto),
            "--original-dir", str(tmp_path / "o"), "--anon-dir", str(tmp_path / "a"),
            "--model-dir", str(tmp_path / "modelo"), "--output", str(salida), *extra,
        ]
        monkeypatch.setattr(sys, "argv", argumentos)
        extract_embeddings.main()
        return salida

    return types.SimpleNamespace(ejecutar=ejecutar, cargas=cargas, tmp=tmp_path)


def test_main_procesa_solo_los_clips_con_estado_ok(entorno):
    with np.load(entorno.ejecutar()) as datos:
        assert list(datos["session_ids"]) == ["s1", "s3", "s4", "s6"]
        assert datos["original"].shape == (4, 4)


def test_main_limit_toma_los_primeros_n_clips_validos(entorno):
    with np.load(entorno.ejecutar("--limit", "2")) as datos:
        assert list(datos["session_ids"]) == ["s1", "s3"]
        assert datos["original"].shape[0] == datos["anonymized"].shape[0] == 2


def test_main_limit_mayor_que_los_clips_procesa_todos(entorno):
    with np.load(entorno.ejecutar("--limit", "100")) as datos:
        assert len(datos["session_ids"]) == 4


def test_main_limit_cero_no_falla_y_guarda_arreglos_vacios(entorno):
    with np.load(entorno.ejecutar("--limit", "0")) as datos:
        assert len(datos["session_ids"]) == 0
        assert datos["original"].shape == (0, 0)


def test_main_usa_las_carpetas_original_y_anonimizada_correctas(entorno):
    with np.load(entorno.ejecutar("--limit", "1")) as datos:
        assert np.array_equal(datos["original"][0], vector_de(entorno.tmp / "o" / "s1.wav").astype(np.float32))
        assert np.array_equal(datos["anonymized"][0], vector_de(entorno.tmp / "a" / "s1.wav").astype(np.float32))


def test_main_carga_el_modelo_una_sola_vez_desde_model_dir(entorno):
    entorno.ejecutar()
    assert entorno.cargas == [str(entorno.tmp / "modelo")]


def test_main_imprime_clips_dimension_tiempo_y_destino(entorno, capsys):
    destino = entorno.ejecutar("--limit", "3")      # 3 clips y dimensión 4: no coinciden
    salida = capsys.readouterr().out
    assert "Clips: 3" in salida
    assert "Dimensión del embedding: 4" in salida
    assert "Tiempo:" in salida
    assert str(destino) in salida


def test_main_el_archivo_se_puede_leer_sin_pickle(entorno):
    with np.load(entorno.ejecutar(), allow_pickle=False) as datos:
        assert set(datos.files) == {"session_ids", "original", "anonymized"}