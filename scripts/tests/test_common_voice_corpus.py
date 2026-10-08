import inspect

import pytest

from common_voice_corpus import read_tsv_rows, load_durations, load_speakers


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