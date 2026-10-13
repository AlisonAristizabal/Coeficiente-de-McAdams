import inspect
import copy
import pytest

from common_voice_corpus import (
    read_tsv_rows, 
    load_durations, 
    load_speakers,
    filter_speakers,
    write_subset,
    SUBSET_COLUMNS,
)


def write_tsv(tmp_path, lines):
    """Escribe un TSV de prueba: cada línea es una lista de campos."""
    path = tmp_path / "datos.tsv"
    contenido = "\n".join("\t".join(campos) for campos in lines) + "\n"
    path.write_text(contenido, encoding="utf-8", newline="")
    return path


def test_read_tsv_rows_archivo_normal(tmp_path):
    path = write_tsv(tmp_path, [
        ["client_id", "path", "sentence"],
        ["a1", "clip1.mp3", "hola mundo"],
        ["b2", "clip2.mp3", "buenos días"],
    ])
    rows = list(read_tsv_rows(path))
    assert rows == [
        {"client_id": "a1", "path": "clip1.mp3", "sentence": "hola mundo"},
        {"client_id": "b2", "path": "clip2.mp3", "sentence": "buenos días"},
    ]


def test_read_tsv_rows_comilla_suelta_no_une_filas(tmp_path):
    path = write_tsv(tmp_path, [
        ["client_id", "path", "sentence"],
        ["a1", "clip1.mp3", '"hola'],
        ["b2", "clip2.mp3", "adiós"],
    ])
    rows = list(read_tsv_rows(path))
    assert len(rows) == 2
    assert rows[0]["sentence"] == '"hola'
    assert rows[1]["client_id"] == "b2"


def test_read_tsv_rows_campo_vacio_es_cadena_vacia(tmp_path):
    path = write_tsv(tmp_path, [
        ["client_id", "path", "accents"],
        ["a1", "clip1.mp3", ""],
    ])
    rows = list(read_tsv_rows(path))
    assert rows[0]["accents"] == ""


def test_read_tsv_rows_fila_corta_deja_none(tmp_path):
    path = write_tsv(tmp_path, [
        ["client_id", "path", "accents"],
        ["a1", "clip1.mp3"],
    ])
    rows = list(read_tsv_rows(path))
    assert rows[0]["accents"] is None


def test_read_tsv_rows_es_generador(tmp_path):
    path = write_tsv(tmp_path, [
        ["client_id", "path"],
        ["a1", "clip1.mp3"],
    ])
    assert inspect.isgenerator(read_tsv_rows(path))


def test_read_tsv_rows_archivo_inexistente_lanza_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(read_tsv_rows(tmp_path / "no_existe.tsv"))

def test_load_durations_archivo_normal(tmp_path):
    path = write_tsv(tmp_path, [
        ["clip", "duration[ms]"],
        ["common_voice_es_1.mp3", "5580"],
        ["common_voice_es_2.mp3", "3168"],
    ])
    assert load_durations(path) == {
        "common_voice_es_1.mp3": 5580,
        "common_voice_es_2.mp3": 3168,
    }


def test_load_durations_valores_son_enteros(tmp_path):
    path = write_tsv(tmp_path, [
        ["clip", "duration[ms]"],
        ["common_voice_es_1.mp3", "5580"],
    ])
    valor = load_durations(path)["common_voice_es_1.mp3"]
    assert isinstance(valor, int)


def test_load_durations_omite_duracion_vacia(tmp_path):
    path = write_tsv(tmp_path, [
        ["clip", "duration[ms]"],
        ["common_voice_es_1.mp3", ""],
        ["common_voice_es_2.mp3", "4000"],
    ])
    assert load_durations(path) == {"common_voice_es_2.mp3": 4000}


def test_load_durations_omite_duracion_no_numerica(tmp_path):
    path = write_tsv(tmp_path, [
        ["clip", "duration[ms]"],
        ["common_voice_es_1.mp3", "abc"],
        ["common_voice_es_2.mp3", "4.5"],
        ["common_voice_es_3.mp3", "4000"],
    ])
    assert load_durations(path) == {"common_voice_es_3.mp3": 4000}


def test_load_durations_omite_fila_corta(tmp_path):
    path = write_tsv(tmp_path, [
        ["clip", "duration[ms]"],
        ["common_voice_es_1.mp3"],
        ["common_voice_es_2.mp3", "4000"],
    ])
    assert load_durations(path) == {"common_voice_es_2.mp3": 4000}


def test_load_durations_clip_repetido_prevalece_el_ultimo(tmp_path):
    path = write_tsv(tmp_path, [
        ["clip", "duration[ms]"],
        ["common_voice_es_1.mp3", "3000"],
        ["common_voice_es_1.mp3", "5000"],
    ])
    assert load_durations(path) == {"common_voice_es_1.mp3": 5000}


def test_load_durations_archivo_solo_con_encabezado(tmp_path):
    path = write_tsv(tmp_path, [["clip", "duration[ms]"]])
    assert load_durations(path) == {}



ENCABEZADO_DATOS = ["client_id", "path", "sentence", "accents"]


def test_load_speakers_agrupa_por_client_id(tmp_path):
    path = write_tsv(tmp_path, [
        ENCABEZADO_DATOS,
        ["a1", "c1.mp3", "hola", "México"],
        ["b2", "c2.mp3", "adiós", ""],
        ["a1", "c3.mp3", "buenas", "México"],
    ])
    durations = {"c1.mp3": 4000, "c2.mp3": 5000, "c3.mp3": 6000}
    speakers, discarded = load_speakers(path, durations, 3000)
    assert set(speakers) == {"a1", "b2"}
    assert [c["path"] for c in speakers["a1"]] == ["c1.mp3", "c3.mp3"]
    assert [c["path"] for c in speakers["b2"]] == ["c2.mp3"]
    assert discarded == {"sin_duracion": 0, "muy_corto": 0, "sin_client_id": 0}


def test_load_speakers_estructura_del_clip(tmp_path):
    path = write_tsv(tmp_path, [
        ENCABEZADO_DATOS,
        ["a1", "c1.mp3", "hola mundo", "México"],
    ])
    speakers, _ = load_speakers(path, {"c1.mp3": 4000}, 3000)
    assert speakers["a1"] == [{
        "path": "c1.mp3",
        "accents": "México",
        "sentence": "hola mundo",
        "duration_ms": 4000,
    }]


def test_load_speakers_descarta_clip_sin_duracion(tmp_path):
    path = write_tsv(tmp_path, [
        ENCABEZADO_DATOS,
        ["a1", "c1.mp3", "hola", ""],
        ["a1", "c2.mp3", "adiós", ""],
    ])
    speakers, discarded = load_speakers(path, {"c2.mp3": 4000}, 3000)
    assert [c["path"] for c in speakers["a1"]] == ["c2.mp3"]
    assert discarded["sin_duracion"] == 1


def test_load_speakers_descarta_clip_muy_corto(tmp_path):
    path = write_tsv(tmp_path, [
        ENCABEZADO_DATOS,
        ["a1", "c1.mp3", "hola", ""],
        ["a1", "c2.mp3", "adiós", ""],
    ])
    durations = {"c1.mp3": 2999, "c2.mp3": 4000}
    speakers, discarded = load_speakers(path, durations, 3000)
    assert [c["path"] for c in speakers["a1"]] == ["c2.mp3"]
    assert discarded["muy_corto"] == 1


def test_load_speakers_conserva_clip_con_duracion_exacta_al_minimo(tmp_path):
    path = write_tsv(tmp_path, [
        ENCABEZADO_DATOS,
        ["a1", "c1.mp3", "hola", ""],
    ])
    speakers, discarded = load_speakers(path, {"c1.mp3": 3000}, 3000)
    assert [c["path"] for c in speakers["a1"]] == ["c1.mp3"]
    assert discarded["muy_corto"] == 0


def test_load_speakers_duracion_cero_es_muy_corto_no_sin_duracion(tmp_path):
    path = write_tsv(tmp_path, [
        ENCABEZADO_DATOS,
        ["a1", "c1.mp3", "hola", ""],
    ])
    speakers, discarded = load_speakers(path, {"c1.mp3": 0}, 3000)
    assert speakers == {}
    assert discarded == {"sin_duracion": 0, "muy_corto": 1, "sin_client_id": 0}


def test_load_speakers_descarta_clip_sin_client_id(tmp_path):
    path = write_tsv(tmp_path, [
        ENCABEZADO_DATOS,
        ["", "c1.mp3", "hola", ""],
        ["a1", "c2.mp3", "adiós", ""],
    ])
    durations = {"c1.mp3": 4000, "c2.mp3": 4000}
    speakers, discarded = load_speakers(path, durations, 3000)
    assert set(speakers) == {"a1"}
    assert discarded["sin_client_id"] == 1


def test_load_speakers_cuenta_cada_clip_en_una_sola_categoria(tmp_path):
    # Sin duración y sin client_id a la vez: cuenta solo como sin_duracion.
    path = write_tsv(tmp_path, [
        ENCABEZADO_DATOS,
        ["", "c1.mp3", "hola", ""],
    ])
    speakers, discarded = load_speakers(path, {}, 3000)
    assert speakers == {}
    assert discarded == {"sin_duracion": 1, "muy_corto": 0, "sin_client_id": 0}


def test_load_speakers_conserva_accents_crudo(tmp_path):
    path = write_tsv(tmp_path, [
        ENCABEZADO_DATOS,
        ["a1", "c1.mp3", "hola", ""],
        ["a1", "c2.mp3", "adiós"],  # fila corta: accents llega como None
    ])
    durations = {"c1.mp3": 4000, "c2.mp3": 4000}
    speakers, _ = load_speakers(path, durations, 3000)
    assert [c["accents"] for c in speakers["a1"]] == ["", None]


def test_load_speakers_hablante_sin_clips_validos_no_aparece(tmp_path):
    path = write_tsv(tmp_path, [
        ENCABEZADO_DATOS,
        ["a1", "c1.mp3", "hola", ""],
        ["b2", "c2.mp3", "adiós", ""],
    ])
    durations = {"c1.mp3": 1000, "c2.mp3": 4000}
    speakers, _ = load_speakers(path, durations, 3000)
    assert "a1" not in speakers
    assert "b2" in speakers


def test_load_speakers_archivo_solo_con_encabezado(tmp_path):
    path = write_tsv(tmp_path, [ENCABEZADO_DATOS])
    speakers, discarded = load_speakers(path, {}, 3000)
    assert speakers == {}
    assert discarded == {"sin_duracion": 0, "muy_corto": 0, "sin_client_id": 0}


def test_load_speakers_filas_totales_igual_a_conservados_mas_descartados(tmp_path):
    filas = [
        ["a1", "c1.mp3", "uno", ""],      # válido
        ["a1", "c2.mp3", "dos", ""],      # válido
        ["b2", "c3.mp3", "tres", ""],     # muy corto
        ["b2", "c4.mp3", "cuatro", ""],   # sin duración
        ["", "c5.mp3", "cinco", ""],      # sin client_id
        ["c3", "c6.mp3", "seis", ""],     # válido
    ]
    path = write_tsv(tmp_path, [ENCABEZADO_DATOS] + filas)
    durations = {"c1.mp3": 4000, "c2.mp3": 4000, "c3.mp3": 500,
                 "c5.mp3": 4000, "c6.mp3": 4000}
    speakers, discarded = load_speakers(path, durations, 3000)
    conservados = sum(len(clips) for clips in speakers.values())
    assert conservados + sum(discarded.values()) == len(filas)
    assert conservados == 3
    assert discarded == {"sin_duracion": 1, "muy_corto": 1, "sin_client_id": 1}


def clips_de_prueba(n):
    """Lista de n clips mínimos, solo para contar."""
    return [{"path": f"c{i}.mp3"} for i in range(n)]


def test_filter_speakers_descarta_hablantes_con_pocos_clips():
    speakers = {"a1": clips_de_prueba(2), "b2": clips_de_prueba(6)}
    assert set(filter_speakers(speakers, 5)) == {"b2"}


def test_filter_speakers_conserva_hablante_con_exactamente_el_minimo():
    speakers = {"a1": clips_de_prueba(5), "b2": clips_de_prueba(4)}
    assert set(filter_speakers(speakers, 5)) == {"a1"}


def test_filter_speakers_conserva_los_clips_intactos():
    clips = clips_de_prueba(5)
    resultado = filter_speakers({"a1": clips}, 5)
    assert resultado["a1"] == clips


def test_filter_speakers_no_modifica_el_diccionario_de_entrada():
    speakers = {"a1": clips_de_prueba(2), "b2": clips_de_prueba(6)}
    copia = {k: list(v) for k, v in speakers.items()}
    filter_speakers(speakers, 5)
    assert speakers == copia


def test_filter_speakers_devuelve_un_diccionario_nuevo():
    speakers = {"a1": clips_de_prueba(6)}
    resultado = filter_speakers(speakers, 5)
    assert resultado is not speakers


def test_filter_speakers_diccionario_vacio():
    assert filter_speakers({}, 5) == {}


def test_filter_speakers_nadie_cumple_el_minimo():
    speakers = {"a1": clips_de_prueba(1), "b2": clips_de_prueba(3)}
    assert filter_speakers(speakers, 5) == {}


def test_filter_speakers_con_minimo_uno_conserva_a_todos():
    speakers = {"a1": clips_de_prueba(1), "b2": clips_de_prueba(3)}
    assert set(filter_speakers(speakers, 1)) == {"a1", "b2"}



def clip(path, accents="", sentence="hola mundo", duration_ms=4200):
    """Clip como los que devuelve `load_speakers` (sin `client_id`)."""
    return {
        "path": path,
        "accents": accents,
        "sentence": sentence,
        "duration_ms": duration_ms,
    }


def seleccion_basica():
    """Dos estratos, el primero con dos hablantes. Cuatro clips en total."""
    return {
        "mexico": {
            "a1": [clip("a1_1.mp3", "México"), clip("a1_2.mp3", "México")],
            "b2": [clip("b2_1.mp3", "México")],
        },
        "caribe": {
            "c3": [clip("c3_1.mp3", "Caribe")],
        },
    }


def leer(path):
    """Lee el manifiesto de vuelta con la misma función que usa el proyecto."""
    return list(read_tsv_rows(path))


def test_write_subset_devuelve_la_ruta(tmp_path):
    salida = tmp_path / "subset.tsv"
    assert write_subset(seleccion_basica(), salida) == salida


def test_write_subset_encabezado_y_orden_de_columnas(tmp_path):
    salida = tmp_path / "subset.tsv"
    write_subset(seleccion_basica(), salida)
    primera_linea = salida.read_text(encoding="utf-8").split("\n")[0]
    assert primera_linea.split("\t") == [
        "client_id", "stratum", "path", "accents", "sentence", "duration_ms",
    ]
    assert SUBSET_COLUMNS == primera_linea.split("\t")


def test_write_subset_una_fila_por_clip(tmp_path):
    salida = tmp_path / "subset.tsv"
    write_subset(seleccion_basica(), salida)
    assert len(leer(salida)) == 4


def test_write_subset_cada_valor_queda_en_su_columna(tmp_path):
    salida = tmp_path / "subset.tsv"
    seleccion = {"mexico": {"a1": [clip("a1_1.mp3", "México", "buenos días", 3500)]}}
    write_subset(seleccion, salida)
    assert leer(salida) == [{
        "client_id": "a1",
        "stratum": "mexico",
        "path": "a1_1.mp3",
        "accents": "México",
        "sentence": "buenos días",
        "duration_ms": "3500",
    }]


def test_write_subset_conserva_el_orden_recibido(tmp_path):
    salida = tmp_path / "subset.tsv"
    # Orden deliberadamente no alfabético en estratos, hablantes y clips.
    seleccion = {
        "mexico": {"z9": [clip("z9_2.mp3"), clip("z9_1.mp3")], "a1": [clip("a1_1.mp3")]},
        "caribe": {"c3": [clip("c3_1.mp3")]},
    }
    write_subset(seleccion, salida)
    filas = leer(salida)
    assert [(f["stratum"], f["client_id"], f["path"]) for f in filas] == [
        ("mexico", "z9", "z9_2.mp3"),
        ("mexico", "z9", "z9_1.mp3"),
        ("mexico", "a1", "a1_1.mp3"),
        ("caribe", "c3", "c3_1.mp3"),
    ]


def test_write_subset_ida_y_vuelta_con_texto_dificil(tmp_path):
    salida = tmp_path / "subset.tsv"
    sentence = '¿Dijo "hola", señor? Sí: \'ñandú\' y comillas "sueltas'
    seleccion = {"andino_pacifico": {"a1": [clip("a1_1.mp3", "Andino-Pacífico", sentence)]}}
    write_subset(seleccion, salida)
    assert leer(salida)[0]["sentence"] == sentence


def test_write_subset_ida_y_vuelta_completa(tmp_path):
    salida = tmp_path / "subset.tsv"
    seleccion = seleccion_basica()
    write_subset(seleccion, salida)
    esperado = [
        {
            "client_id": client_id,
            "stratum": stratum,
            "path": c["path"],
            "accents": c["accents"],
            "sentence": c["sentence"],
            "duration_ms": str(c["duration_ms"]),
        }
        for stratum, hablantes in seleccion.items()
        for client_id, clips in hablantes.items()
        for c in clips
    ]
    assert leer(salida) == esperado


def test_write_subset_none_se_escribe_como_cadena_vacia(tmp_path):
    salida = tmp_path / "subset.tsv"
    seleccion = {"mexico": {"a1": [clip("a1_1.mp3", accents=None)]}}
    write_subset(seleccion, salida)
    assert leer(salida)[0]["accents"] == ""


def test_write_subset_duration_ms_se_escribe_como_texto_entero(tmp_path):
    salida = tmp_path / "subset.tsv"
    seleccion = {"mexico": {"a1": [clip("a1_1.mp3", duration_ms=12345)]}}
    write_subset(seleccion, salida)
    assert leer(salida)[0]["duration_ms"] == "12345"


def test_write_subset_seleccion_vacia_solo_escribe_encabezado(tmp_path):
    salida = tmp_path / "subset.tsv"
    write_subset({}, salida)
    assert salida.read_text(encoding="utf-8") == "\t".join(SUBSET_COLUMNS) + "\n"
    assert leer(salida) == []


def test_write_subset_usa_salto_de_linea_unix_y_utf8(tmp_path):
    salida = tmp_path / "subset.tsv"
    seleccion = {"mexico": {"a1": [clip("a1_1.mp3", sentence="canción")]}}
    write_subset(seleccion, salida)
    crudo = salida.read_bytes()
    assert b"\r" not in crudo
    assert "canción".encode("utf-8") in crudo


def test_write_subset_sobrescribe_un_archivo_existente(tmp_path):
    salida = tmp_path / "subset.tsv"
    salida.write_text("contenido viejo\n", encoding="utf-8")
    write_subset(seleccion_basica(), salida)
    assert "contenido viejo" not in salida.read_text(encoding="utf-8")
    assert len(leer(salida)) == 4


def test_write_subset_no_modifica_la_entrada(tmp_path):
    seleccion = seleccion_basica()
    copia = copy.deepcopy(seleccion)
    write_subset(seleccion, tmp_path / "subset.tsv")
    assert seleccion == copia


@pytest.mark.parametrize("caracter", ["\t", "\n", "\r"])
@pytest.mark.parametrize("campo", ["path", "accents", "sentence"])
def test_write_subset_rechaza_separadores_en_los_campos_del_clip(tmp_path, campo, caracter):
    salida = tmp_path / "subset.tsv"
    mal = clip("a1_1.mp3")
    mal[campo] = f"antes{caracter}despues"
    with pytest.raises(ValueError):
        write_subset({"mexico": {"a1": [mal]}}, salida)


@pytest.mark.parametrize("caracter", ["\t", "\n", "\r"])
def test_write_subset_rechaza_separadores_en_client_id_y_estrato(tmp_path, caracter):
    with pytest.raises(ValueError):
        write_subset({"mexico": {f"a{caracter}1": [clip("a1_1.mp3")]}}, tmp_path / "s1.tsv")
    with pytest.raises(ValueError):
        write_subset({f"mex{caracter}ico": {"a1": [clip("a1_1.mp3")]}}, tmp_path / "s2.tsv")


def test_write_subset_el_error_indica_estrato_hablante_y_campo(tmp_path):
    mal = clip("a1_1.mp3", sentence="una\tfrase")
    seleccion = {"caribe": {"c3": [clip("c3_1.mp3")]}, "mexico": {"hablante_malo": [mal]}}
    with pytest.raises(ValueError) as error:
        write_subset(seleccion, tmp_path / "subset.tsv")
    mensaje = str(error.value)
    assert "sentence" in mensaje
    assert "mexico" in mensaje
    assert "hablante_malo" in mensaje


def test_write_subset_si_falla_no_crea_el_archivo(tmp_path):
    salida = tmp_path / "subset.tsv"
    seleccion = seleccion_basica()
    # El clip malo va al final: las filas anteriores ya son válidas.
    seleccion["caribe"]["c3"].append(clip("c3_2.mp3", sentence="mala\nfrase"))
    with pytest.raises(ValueError):
        write_subset(seleccion, salida)
    assert not salida.exists()


def test_write_subset_si_falla_no_toca_un_archivo_existente(tmp_path):
    salida = tmp_path / "subset.tsv"
    salida.write_text("manifiesto anterior\n", encoding="utf-8")
    mal = clip("a1_1.mp3", sentence="mala\tfrase")
    with pytest.raises(ValueError):
        write_subset({"mexico": {"a1": [mal]}}, salida)
    assert salida.read_text(encoding="utf-8") == "manifiesto anterior\n"


@pytest.mark.parametrize("llave", ["path", "accents", "sentence", "duration_ms"])
def test_write_subset_clip_sin_llave_lanza_key_error(tmp_path, llave):
    incompleto = clip("a1_1.mp3")
    del incompleto[llave]
    with pytest.raises(KeyError):
        write_subset({"mexico": {"a1": [incompleto]}}, tmp_path / "subset.tsv")