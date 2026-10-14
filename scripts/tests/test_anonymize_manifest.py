import sys

import numpy as np
import pytest
import soundfile

import anonymize_subset as modulo
from anonymize_subset import (
    MANIFEST_COLUMNS,
    anonymize_subset,
    main,
    parse_args,
    with_progress,
    write_manifest,
)
from common_voice_corpus import read_tsv_rows

COLUMNAS = ["client_id", "stratum", "path", "session_id", "coefficient", "status", "error"]
COLUMNAS_SUBSET = ["client_id", "stratum", "path", "accents", "sentence", "duration_ms"]


def registro(**cambios):
    """Un registro de manifiesto válido; `cambios` sobrescribe campos."""
    base = {
        "client_id": "h1",
        "stratum": "mexico",
        "path": "clip1.mp3",
        "session_id": "clip1",
        "coefficient": 0.7997899358383044,
        "status": "ok",
        "error": "",
    }
    base.update(cambios)
    return base


def crear_mp3(ruta, segundos=0.5, tasa=48000, semilla=0):
    """Escribe un MP3 de prueba (tono más ruido)."""
    rng = np.random.default_rng(semilla)
    t = np.arange(int(tasa * segundos)) / tasa
    senal = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(len(t))
    soundfile.write(ruta, senal, tasa, format="MP3")


def escribir_subset(ruta, paths):
    """Escribe un subset.tsv mínimo con una fila por nombre de MP3."""
    with open(ruta, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(COLUMNAS_SUBSET) + "\n")
        for i, path in enumerate(paths):
            f.write(f"h{i}\tmexico\t{path}\tx\tfrase\t3000\n")
    return ruta


def ejecutar(monkeypatch, subset, clips, salida, *extra):
    """Corre `main` con los argumentos de línea de comandos dados."""
    argv = ["anonymize_subset.py", "--subset", str(subset), "--clips-dir", str(clips),
            "--output-dir", str(salida), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    return main()


@pytest.fixture
def escenario(tmp_path):
    """Dos clips buenos, uno corrupto y uno inexistente."""
    clips = tmp_path / "clips"
    clips.mkdir()
    crear_mp3(clips / "a.mp3", semilla=1)
    crear_mp3(clips / "b.mp3", semilla=2)
    (clips / "roto.mp3").write_bytes(b"esto no es un mp3" * 100)
    subset = escribir_subset(tmp_path / "subset.tsv", ["a.mp3", "b.mp3", "roto.mp3", "falta.mp3"])
    return subset, clips, tmp_path / "salida"


# --- MANIFEST_COLUMNS -----------------------------------------------------

def test_manifest_columns_son_las_siete_en_orden():
    assert MANIFEST_COLUMNS == COLUMNAS


# --- write_manifest -------------------------------------------------------

def test_write_manifest_devuelve_la_ruta(tmp_path):
    salida = tmp_path / "m.tsv"
    assert write_manifest([registro()], salida) == salida


def test_write_manifest_encabezado_y_filas_en_orden(tmp_path):
    salida = tmp_path / "m.tsv"
    write_manifest([registro(), registro(client_id="h2", path="clip2.mp3")], salida)
    lineas = salida.read_text(encoding="utf-8").split("\n")
    assert lineas[0] == "\t".join(COLUMNAS)
    assert lineas[1].startswith("h1\tmexico\tclip1.mp3\tclip1\t")
    assert lineas[2].startswith("h2\tmexico\tclip2.mp3\tclip1\t")
    assert lineas[3] == ""  # termina en "\n", sin línea extra


def test_write_manifest_usa_saltos_de_linea_unix(tmp_path):
    salida = tmp_path / "m.tsv"
    write_manifest([registro()], salida)
    assert b"\r" not in salida.read_bytes()


def test_write_manifest_el_orden_de_columnas_no_depende_del_diccionario(tmp_path):
    salida = tmp_path / "m.tsv"
    invertido = dict(reversed(list(registro().items())))
    write_manifest([invertido], salida)
    filas = list(read_tsv_rows(salida))
    assert list(filas[0].keys()) == COLUMNAS
    assert filas[0]["client_id"] == "h1"


def test_write_manifest_se_lee_de_vuelta_igual(tmp_path):
    salida = tmp_path / "m.tsv"
    original = [
        registro(),
        registro(client_id="h2", status="error", coefficient="",
                 error="LibsndfileError: no se pudo abrir"),
    ]
    write_manifest(original, salida)
    leidas = list(read_tsv_rows(salida))
    assert len(leidas) == 2
    for esperado, leida in zip(original, leidas):
        for col in COLUMNAS:
            assert leida[col] == str(esperado[col])


def test_write_manifest_el_coeficiente_se_recupera_con_precision_completa(tmp_path):
    salida = tmp_path / "m.tsv"
    coeficientes = [0.7997899358383044, 0.7090765833275342, 0.9, 0.7, 1 / 3]
    write_manifest([registro(coefficient=c) for c in coeficientes], salida)
    leidos = [float(fila["coefficient"]) for fila in read_tsv_rows(salida)]
    assert leidos == coeficientes


def test_write_manifest_no_agrega_comillas(tmp_path):
    salida = tmp_path / "m.tsv"
    mensaje = 'Error "raro" con \'comillas\' y "'
    write_manifest([registro(error=mensaje)], salida)
    assert '""' not in salida.read_text(encoding="utf-8")
    assert next(iter(read_tsv_rows(salida)))["error"] == mensaje


def test_write_manifest_conserva_tildes_y_eñes(tmp_path):
    salida = tmp_path / "m.tsv"
    write_manifest([registro(error="Ñandú: canción rápida")], salida)
    assert next(iter(read_tsv_rows(salida)))["error"] == "Ñandú: canción rápida"


def test_write_manifest_sin_registros_escribe_solo_el_encabezado(tmp_path):
    salida = tmp_path / "m.tsv"
    write_manifest([], salida)
    assert salida.read_text(encoding="utf-8") == "\t".join(COLUMNAS) + "\n"
    assert list(read_tsv_rows(salida)) == []


def test_write_manifest_acepta_un_generador(tmp_path):
    salida = tmp_path / "m.tsv"
    write_manifest((registro(client_id=f"h{i}") for i in range(3)), salida)
    assert [f["client_id"] for f in read_tsv_rows(salida)] == ["h0", "h1", "h2"]


def test_write_manifest_sobrescribe_un_archivo_existente(tmp_path):
    salida = tmp_path / "m.tsv"
    salida.write_text("contenido viejo", encoding="utf-8")
    write_manifest([registro()], salida)
    assert "contenido viejo" not in salida.read_text(encoding="utf-8")


@pytest.mark.parametrize("columna", COLUMNAS)
@pytest.mark.parametrize("caracter", ["\t", "\n", "\r"])
def test_write_manifest_rechaza_tabuladores_y_saltos_en_cualquier_columna(
    tmp_path, columna, caracter
):
    salida = tmp_path / "m.tsv"
    with pytest.raises(ValueError) as error:
        write_manifest([registro(**{columna: f"a{caracter}b"})], salida)
    assert columna in str(error.value)
    if columna != "path":  # si el culpable es `path`, el mensaje trae su valor
        assert "clip1.mp3" in str(error.value)
    assert not salida.exists()


def test_write_manifest_valida_todo_antes_de_abrir_el_archivo(tmp_path):
    salida = tmp_path / "m.tsv"
    malos = [registro(), registro(client_id="h2"), registro(error="a\tb")]
    with pytest.raises(ValueError):
        write_manifest(malos, salida)
    assert not salida.exists()


def test_write_manifest_con_error_no_toca_el_archivo_previo(tmp_path):
    salida = tmp_path / "m.tsv"
    salida.write_text("contenido previo", encoding="utf-8")
    with pytest.raises(ValueError):
        write_manifest([registro(error="a\nb")], salida)
    assert salida.read_text(encoding="utf-8") == "contenido previo"


def test_write_manifest_llave_faltante_lanza_key_error_sin_crear_archivo(tmp_path):
    salida = tmp_path / "m.tsv"
    incompleto = registro()
    del incompleto["status"]
    with pytest.raises(KeyError):
        write_manifest([incompleto], salida)
    assert not salida.exists()


def test_write_manifest_integra_con_la_salida_de_anonymize_subset(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    crear_mp3(clips / "a.mp3")
    filas = [{"client_id": "h1", "stratum": "caribe", "path": "a.mp3"},
             {"client_id": "h2", "stratum": "caribe", "path": "falta.mp3"}]
    manifiesto = anonymize_subset(filas, clips, tmp_path / "o", tmp_path / "x")
    salida = write_manifest(manifiesto, tmp_path / "m.tsv")
    leidas = list(read_tsv_rows(salida))
    assert [f["status"] for f in leidas] == ["ok", "error"]
    assert float(leidas[0]["coefficient"]) == manifiesto[0]["coefficient"]
    assert leidas[1]["coefficient"] == ""


# --- with_progress --------------------------------------------------------

def test_with_progress_entrega_las_mismas_filas_en_orden(capsys):
    filas = [{"i": i} for i in range(7)]
    resultado = list(with_progress(filas, every=3))
    assert resultado == filas
    assert all(a is b for a, b in zip(resultado, filas))


def test_with_progress_imprime_cada_n_y_al_final(capsys):
    list(with_progress([{}] * 250))
    lineas = capsys.readouterr().out.strip().split("\n")
    assert lineas == ["100/250 clips procesados", "200/250 clips procesados",
                      "250/250 clips procesados"]


def test_with_progress_no_repite_la_linea_final_si_el_total_es_multiplo(capsys):
    list(with_progress([{}] * 200))
    lineas = capsys.readouterr().out.strip().split("\n")
    assert lineas == ["100/200 clips procesados", "200/200 clips procesados"]


def test_with_progress_con_menos_de_n_filas_imprime_solo_el_total(capsys):
    list(with_progress([{}] * 5))
    assert capsys.readouterr().out.strip() == "5/5 clips procesados"


def test_with_progress_sin_filas_no_imprime_nada(capsys):
    assert list(with_progress([])) == []
    assert capsys.readouterr().out == ""


def test_with_progress_cada_uno_imprime_en_todas(capsys):
    list(with_progress([{}] * 3, every=1))
    assert capsys.readouterr().out.strip().split("\n") == [
        "1/3 clips procesados", "2/3 clips procesados", "3/3 clips procesados"]


def test_with_progress_imprime_despues_de_procesar_la_fila(capsys):
    generador = with_progress([{}] * 4, every=2)
    next(generador)
    next(generador)
    # La fila 2 ya se entregó, pero su procesamiento no ha terminado.
    assert capsys.readouterr().out == ""
    next(generador)  # quien consume pidió la fila 3: la 2 ya terminó
    assert capsys.readouterr().out.strip() == "2/4 clips procesados"


# --- parse_args -----------------------------------------------------------

def test_parse_args_lee_los_argumentos_y_el_tag_por_defecto(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--subset", "s.tsv", "--clips-dir", "c",
                                      "--output-dir", "o"])
    args = parse_args()
    assert (args.subset, args.clips_dir, args.output_dir, args.tag) == ("s.tsv", "c", "o", "u07_09")


def test_parse_args_acepta_un_tag_propio(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--subset", "s", "--clips-dir", "c",
                                      "--output-dir", "o", "--tag", "u08_09"])
    assert parse_args().tag == "u08_09"


@pytest.mark.parametrize("faltante", ["--subset", "--clips-dir", "--output-dir"])
def test_parse_args_exige_los_tres_argumentos_obligatorios(monkeypatch, faltante):
    completos = {"--subset": "s", "--clips-dir": "c", "--output-dir": "o"}
    argv = ["x"]
    for llave, valor in completos.items():
        if llave != faltante:
            argv += [llave, valor]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        parse_args()


# --- main -----------------------------------------------------------------

def test_main_crea_la_estructura_de_salida(monkeypatch, escenario):
    subset, clips, salida = escenario
    ejecutar(monkeypatch, subset, clips, salida)
    assert (salida / "original" / "a.wav").exists()
    assert (salida / "original" / "b.wav").exists()
    assert (salida / "anon_u07_09" / "a.wav").exists()
    assert (salida / "anon_u07_09" / "b.wav").exists()
    assert (salida / "anon_manifest_u07_09.tsv").exists()


def test_main_el_tag_nombra_la_carpeta_y_el_manifiesto(monkeypatch, escenario):
    subset, clips, salida = escenario
    ejecutar(monkeypatch, subset, clips, salida, "--tag", "prueba")
    assert (salida / "anon_prueba" / "a.wav").exists()
    assert (salida / "anon_manifest_prueba.tsv").exists()
    assert not (salida / "anon_u07_09").exists()


def test_main_el_manifiesto_refleja_los_resultados(monkeypatch, escenario):
    subset, clips, salida = escenario
    ejecutar(monkeypatch, subset, clips, salida)
    filas = list(read_tsv_rows(salida / "anon_manifest_u07_09.tsv"))
    assert [f["path"] for f in filas] == ["a.mp3", "b.mp3", "roto.mp3", "falta.mp3"]
    assert [f["status"] for f in filas] == ["ok", "ok", "error", "error"]
    assert filas[0]["coefficient"] != "" and filas[2]["coefficient"] == ""
    assert filas[0]["session_id"] == "a"


def test_main_el_informe_cuenta_ok_y_errores(monkeypatch, escenario, capsys):
    subset, clips, salida = escenario
    ejecutar(monkeypatch, subset, clips, salida)
    salida_texto = capsys.readouterr().out
    assert "Clips ok: 2" in salida_texto
    assert "Clips con error: 2" in salida_texto
    assert "roto.mp3: LibsndfileError" in salida_texto
    assert "falta.mp3: LibsndfileError" in salida_texto
    assert f"Manifiesto: {salida / 'anon_manifest_u07_09.tsv'}" in salida_texto
    assert "4/4 clips procesados" in salida_texto


def test_main_sin_errores_no_lista_ninguno(monkeypatch, tmp_path, capsys):
    clips = tmp_path / "clips"
    clips.mkdir()
    crear_mp3(clips / "a.mp3")
    subset = escribir_subset(tmp_path / "subset.tsv", ["a.mp3"])
    ejecutar(monkeypatch, subset, clips, tmp_path / "salida")
    texto = capsys.readouterr().out
    assert "Clips ok: 1" in texto
    assert "Clips con error: 0" in texto
    assert "LibsndfileError" not in texto


def test_main_no_imprime_un_mensaje_por_clip_de_la_conversion(monkeypatch, escenario, capsys):
    subset, clips, salida = escenario
    ejecutar(monkeypatch, subset, clips, salida)
    assert "convirtió" not in capsys.readouterr().out


def test_main_termina_sin_excepcion_aunque_haya_clips_con_error(monkeypatch, escenario):
    subset, clips, salida = escenario
    assert ejecutar(monkeypatch, subset, clips, salida) is None


def test_main_subset_vacio_escribe_manifiesto_con_solo_encabezado(monkeypatch, tmp_path, capsys):
    clips = tmp_path / "clips"
    clips.mkdir()
    subset = escribir_subset(tmp_path / "subset.tsv", [])
    salida = tmp_path / "salida"
    ejecutar(monkeypatch, subset, clips, salida)
    assert list(read_tsv_rows(salida / "anon_manifest_u07_09.tsv")) == []
    assert "Clips ok: 0" in capsys.readouterr().out


def test_main_dos_etiquetas_comparten_los_originales(monkeypatch, escenario):
    subset, clips, salida = escenario
    ejecutar(monkeypatch, subset, clips, salida, "--tag", "uno")
    ejecutar(monkeypatch, subset, clips, salida, "--tag", "dos")
    assert (salida / "anon_uno" / "a.wav").exists()
    assert (salida / "anon_dos" / "a.wav").exists()
    assert sorted(p.name for p in (salida / "original").iterdir()) == ["a.wav", "b.wav"]
    assert (salida / "anon_manifest_uno.tsv").exists()
    assert (salida / "anon_manifest_dos.tsv").exists()


def test_main_es_reproducible(monkeypatch, escenario):
    subset, clips, salida = escenario
    ejecutar(monkeypatch, subset, clips, salida, "--tag", "uno")
    ejecutar(monkeypatch, subset, clips, salida, "--tag", "dos")
    assert ((salida / "anon_uno" / "a.wav").read_bytes()
            == (salida / "anon_dos" / "a.wav").read_bytes())


def test_main_subset_inexistente_falla_antes_de_crear_salidas(monkeypatch, tmp_path):
    salida = tmp_path / "salida"
    with pytest.raises(FileNotFoundError):
        ejecutar(monkeypatch, tmp_path / "no_existe.tsv", tmp_path, salida)
    assert not salida.exists()


def test_main_un_error_del_programa_se_propaga_y_no_deja_manifiesto(monkeypatch, escenario):
    subset, clips, salida = escenario

    def defecto(*args, **kwargs):
        raise RuntimeError("defecto del programa")

    monkeypatch.setattr(modulo, "process_clip", defecto)
    with pytest.raises(RuntimeError):
        ejecutar(monkeypatch, subset, clips, salida)
    assert not (salida / "anon_manifest_u07_09.tsv").exists()