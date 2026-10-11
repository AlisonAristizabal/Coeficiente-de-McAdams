import pytest
from common_voice_sampling import (
    group_by_stratum,
    sample_speakers,
    select_clips,
)

CHILENO = "Chileno: Chile, Cuyo"


def clips_con_acentos(*acentos):
    """Lista de clips mínimos, uno por cada valor de `accents` dado."""
    return [{"path": f"c{i}.mp3", "accents": a} for i, a in enumerate(acentos)]


def test_group_by_stratum_hablante_sin_etiquetas_es_sin_dato():
    speakers = {"a1": clips_con_acentos("", None, " ")}
    assert group_by_stratum(speakers) == {"sin_dato": ["a1"]}


def test_group_by_stratum_hablante_con_etiqueta_valida():
    speakers = {"a1": clips_con_acentos("México", "México")}
    assert group_by_stratum(speakers) == {"mexico": ["a1"]}


def test_group_by_stratum_une_clips_vacios_con_clips_etiquetados():
    speakers = {"a1": clips_con_acentos("", "México", None)}
    assert group_by_stratum(speakers) == {"mexico": ["a1"]}


def test_group_by_stratum_etiquetas_en_conflicto_es_otro():
    speakers = {"a1": clips_con_acentos("México", CHILENO)}
    assert group_by_stratum(speakers) == {"otro": ["a1"]}


def test_group_by_stratum_etiqueta_desconocida_es_otro():
    speakers = {"a1": clips_con_acentos("España: Islas Canarias")}
    assert group_by_stratum(speakers) == {"otro": ["a1"]}


def test_group_by_stratum_agrupa_varios_hablantes_en_el_mismo_estrato():
    speakers = {
        "a1": clips_con_acentos("México"),
        "b2": clips_con_acentos(CHILENO),
        "c3": clips_con_acentos("México"),
    }
    resultado = group_by_stratum(speakers)
    assert resultado == {"mexico": ["a1", "c3"], "chileno": ["b2"]}


def test_group_by_stratum_solo_incluye_estratos_que_aparecen():
    speakers = {"a1": clips_con_acentos("México")}
    assert set(group_by_stratum(speakers)) == {"mexico"}


def test_group_by_stratum_listas_ordenadas_sin_importar_el_orden_de_entrada():
    speakers_a = {
        "z9": clips_con_acentos("México"),
        "a1": clips_con_acentos("México"),
        "m5": clips_con_acentos("México"),
    }
    speakers_b = {
        "a1": clips_con_acentos("México"),
        "m5": clips_con_acentos("México"),
        "z9": clips_con_acentos("México"),
    }
    assert group_by_stratum(speakers_a) == group_by_stratum(speakers_b)
    assert group_by_stratum(speakers_a)["mexico"] == ["a1", "m5", "z9"]


def test_group_by_stratum_no_modifica_el_diccionario_de_entrada():
    speakers = {
        "b2": clips_con_acentos("México"),
        "a1": clips_con_acentos("México"),
    }
    llaves_antes = list(speakers)
    clips_antes = {k: list(v) for k, v in speakers.items()}
    group_by_stratum(speakers)
    assert list(speakers) == llaves_antes
    assert speakers == clips_antes


def test_group_by_stratum_diccionario_vacio():
    assert group_by_stratum({}) == {}




def estratos_de_prueba():
    """Dos estratos con 100 y 60 hablantes, con listas ordenadas."""
    return {
        "mexico": [f"m{i:03d}" for i in range(100)],
        "chileno": [f"c{i:03d}" for i in range(60)],
    }


def test_sample_speakers_devuelve_k_hablantes_por_estrato():
    resultado = sample_speakers(estratos_de_prueba(), ["mexico", "chileno"], 10, 42)
    assert len(resultado["mexico"]) == 10
    assert len(resultado["chileno"]) == 10


def test_sample_speakers_respeta_el_orden_de_strata():
    resultado = sample_speakers(estratos_de_prueba(), ["chileno", "mexico"], 5, 42)
    assert list(resultado) == ["chileno", "mexico"]


def test_sample_speakers_elige_solo_hablantes_de_ese_estrato():
    by_stratum = estratos_de_prueba()
    resultado = sample_speakers(by_stratum, ["mexico", "chileno"], 10, 42)
    assert set(resultado["mexico"]) <= set(by_stratum["mexico"])
    assert set(resultado["chileno"]) <= set(by_stratum["chileno"])


def test_sample_speakers_no_repite_hablantes():
    resultado = sample_speakers(estratos_de_prueba(), ["mexico"], 50, 42)
    assert len(set(resultado["mexico"])) == 50


def test_sample_speakers_misma_semilla_mismo_resultado():
    a = sample_speakers(estratos_de_prueba(), ["mexico", "chileno"], 10, 42)
    b = sample_speakers(estratos_de_prueba(), ["mexico", "chileno"], 10, 42)
    assert a == b


def test_sample_speakers_semilla_distinta_resultado_distinto():
    a = sample_speakers(estratos_de_prueba(), ["mexico"], 10, 42)
    b = sample_speakers(estratos_de_prueba(), ["mexico"], 10, 43)
    assert a != b


def test_sample_speakers_es_anidado():
    chico = sample_speakers(estratos_de_prueba(), ["mexico", "chileno"], 5, 42)
    grande = sample_speakers(estratos_de_prueba(), ["mexico", "chileno"], 20, 42)
    assert grande["mexico"][:5] == chico["mexico"]
    assert grande["chileno"][:5] == chico["chileno"]


def test_sample_speakers_agregar_un_estrato_no_cambia_los_demas():
    solo = sample_speakers(estratos_de_prueba(), ["mexico"], 10, 42)
    ambos = sample_speakers(estratos_de_prueba(), ["chileno", "mexico"], 10, 42)
    assert solo["mexico"] == ambos["mexico"]


def test_sample_speakers_estratos_distintos_usan_mezclas_distintas():
    by_stratum = {
        "a": [f"x{i:03d}" for i in range(50)],
        "b": [f"x{i:03d}" for i in range(50)],
    }
    resultado = sample_speakers(by_stratum, ["a", "b"], 10, 42)
    assert resultado["a"] != resultado["b"]


def test_sample_speakers_con_k_igual_al_tamano_devuelve_todos():
    by_stratum = {"mexico": ["a", "b", "c"]}
    resultado = sample_speakers(by_stratum, ["mexico"], 3, 42)
    assert sorted(resultado["mexico"]) == ["a", "b", "c"]


def test_sample_speakers_error_si_el_estrato_tiene_menos_de_k():
    by_stratum = {"mexico": ["a", "b", "c"]}
    with pytest.raises(ValueError, match="mexico"):
        sample_speakers(by_stratum, ["mexico"], 4, 42)


def test_sample_speakers_error_si_el_estrato_no_existe():
    with pytest.raises(ValueError, match="chileno"):
        sample_speakers({"mexico": ["a", "b"]}, ["chileno"], 1, 42)


def test_sample_speakers_no_modifica_by_stratum():
    by_stratum = estratos_de_prueba()
    copia = {k: list(v) for k, v in by_stratum.items()}
    sample_speakers(by_stratum, ["mexico", "chileno"], 10, 42)
    assert by_stratum == copia


def test_sample_speakers_sin_estratos_devuelve_diccionario_vacio():
    assert sample_speakers(estratos_de_prueba(), [], 10, 42) == {}




def clips_de_hablante(cantidad):
    """Clips mínimos con `path` distinto, ya en orden alfabético."""
    return [{"path": f"c{i:03d}.mp3", "sentence": f"frase {i}"} for i in range(cantidad)]


def test_select_clips_devuelve_n_clips():
    resultado = select_clips(clips_de_hablante(20), 5, 42, "a1")
    assert len(resultado) == 5


def test_select_clips_elige_solo_clips_del_hablante():
    clips = clips_de_hablante(20)
    resultado = select_clips(clips, 5, 42, "a1")
    assert all(clip in clips for clip in resultado)


def test_select_clips_no_repite_clips():
    resultado = select_clips(clips_de_hablante(20), 10, 42, "a1")
    assert len({clip["path"] for clip in resultado}) == 10


def test_select_clips_misma_semilla_mismo_resultado():
    a = select_clips(clips_de_hablante(20), 5, 42, "a1")
    b = select_clips(clips_de_hablante(20), 5, 42, "a1")
    assert a == b


def test_select_clips_semilla_distinta_resultado_distinto():
    a = select_clips(clips_de_hablante(20), 5, 42, "a1")
    b = select_clips(clips_de_hablante(20), 5, 43, "a1")
    assert a != b


def test_select_clips_hablantes_distintos_usan_mezclas_distintas():
    a = select_clips(clips_de_hablante(20), 5, 42, "a1")
    b = select_clips(clips_de_hablante(20), 5, 42, "b2")
    assert a != b


def test_select_clips_no_depende_del_orden_de_entrada():
    clips = clips_de_hablante(20)
    al_reves = list(reversed(clips))
    barajados = clips[10:] + clips[:10]
    esperado = select_clips(clips, 5, 42, "a1")
    assert select_clips(al_reves, 5, 42, "a1") == esperado
    assert select_clips(barajados, 5, 42, "a1") == esperado


def test_select_clips_no_modifica_la_lista_de_entrada():
    clips = list(reversed(clips_de_hablante(20)))
    copia = list(clips)
    select_clips(clips, 5, 42, "a1")
    assert clips == copia


def test_select_clips_con_n_igual_al_total_devuelve_todos():
    clips = clips_de_hablante(5)
    resultado = select_clips(clips, 5, 42, "a1")
    assert sorted(c["path"] for c in resultado) == [c["path"] for c in clips]


def test_select_clips_error_si_hay_menos_clips_que_n():
    with pytest.raises(ValueError, match="a1"):
        select_clips(clips_de_hablante(3), 5, 42, "a1")