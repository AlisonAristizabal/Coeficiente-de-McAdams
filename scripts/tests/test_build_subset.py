import copy

import pytest

from build_subset import STRATA, select_subset
from common_voice_corpus import read_tsv_rows, write_subset
from common_voice_strata import ACCENT_TO_STRATUM, OTRO, SIN_DATO

# Acento crudo que asigna cada estrato; `sin_dato` corresponde a un acento vacío.
STRATUM_TO_ACCENT = {estrato: acento for acento, estrato in ACCENT_TO_STRATUM.items()}
STRATUM_TO_ACCENT[SIN_DATO] = ""


def hablantes_de(estrato, cantidad, clips_por_hablante=6):
    """`cantidad` hablantes de un estrato, con clips de nombre único."""
    acento = STRATUM_TO_ACCENT.get(estrato, "Acento desconocido")
    return {
        f"{estrato}-h{i}": [
            {
                "path": f"{estrato}-h{i}_{j}.mp3",
                "accents": acento,
                "sentence": f"frase {j}",
                "duration_ms": 4000,
            }
            for j in range(clips_por_hablante)
        ]
        for i in range(cantidad)
    }


def corpus_de_prueba(hablantes_por_estrato=8, clips_por_hablante=6):
    """Diccionario {client_id: clips} con los 10 estratos del diseño."""
    speakers = {}
    for estrato in STRATA:
        speakers.update(hablantes_de(estrato, hablantes_por_estrato, clips_por_hablante))
    return speakers


# --- STRATA ---------------------------------------------------------------

def test_strata_son_los_nueve_estratos_mas_sin_dato():
    assert len(STRATA) == 10
    assert len(set(STRATA)) == 10
    assert SIN_DATO in STRATA
    assert set(STRATA) == set(ACCENT_TO_STRATUM.values()) | {SIN_DATO}


def test_strata_no_incluye_otro():
    assert OTRO not in STRATA


# --- select_subset --------------------------------------------------------

def test_select_subset_devuelve_los_estratos_pedidos_en_orden():
    resultado = select_subset(corpus_de_prueba(), STRATA, 3, 5, 42)
    assert list(resultado.keys()) == STRATA


def test_select_subset_k_hablantes_por_estrato():
    resultado = select_subset(corpus_de_prueba(), STRATA, 3, 5, 42)
    assert all(len(hablantes) == 3 for hablantes in resultado.values())


def test_select_subset_n_clips_por_hablante():
    resultado = select_subset(corpus_de_prueba(), STRATA, 3, 4, 42)
    for hablantes in resultado.values():
        assert all(len(clips) == 4 for clips in hablantes.values())


def test_select_subset_cada_hablante_queda_en_su_estrato():
    resultado = select_subset(corpus_de_prueba(), STRATA, 3, 5, 42)
    for estrato, hablantes in resultado.items():
        assert all(client_id.startswith(f"{estrato}-") for client_id in hablantes)


def test_select_subset_los_clips_son_del_hablante_y_sin_repetir():
    resultado = select_subset(corpus_de_prueba(), STRATA, 3, 5, 42)
    for hablantes in resultado.values():
        for client_id, clips in hablantes.items():
            paths = [clip["path"] for clip in clips]
            assert all(path.startswith(f"{client_id}_") for path in paths)
            assert len(set(paths)) == len(paths)


def test_select_subset_misma_semilla_mismo_resultado():
    speakers = corpus_de_prueba()
    assert select_subset(speakers, STRATA, 3, 5, 42) == select_subset(speakers, STRATA, 3, 5, 42)


def test_select_subset_semilla_distinta_resultado_distinto():
    speakers = corpus_de_prueba()
    assert select_subset(speakers, STRATA, 3, 5, 42) != select_subset(speakers, STRATA, 3, 5, 7)


def test_select_subset_la_semilla_cambia_los_hablantes_elegidos():
    speakers = corpus_de_prueba(hablantes_por_estrato=8)
    a = select_subset(speakers, STRATA, 3, 5, 42)
    b = select_subset(speakers, STRATA, 3, 5, 7)
    assert any(set(a[estrato]) != set(b[estrato]) for estrato in STRATA)


def test_select_subset_la_semilla_cambia_los_clips_elegidos():
    # Con k igual al total de hablantes del estrato, ambos resultados
    # eligen a los mismos hablantes; solo pueden diferir los clips.
    speakers = corpus_de_prueba(hablantes_por_estrato=3, clips_por_hablante=20)
    a = select_subset(speakers, STRATA, 3, 5, 42)
    b = select_subset(speakers, STRATA, 3, 5, 7)
    assert all(set(a[estrato]) == set(b[estrato]) for estrato in STRATA)
    assert a != b


def test_select_subset_cada_hablante_mezcla_sus_clips_distinto():
    # Hablantes con clips idénticos en estructura (mismos índices 0..19):
    # si todos usaran la misma mezcla, elegirían los mismos índices.
    speakers = corpus_de_prueba(hablantes_por_estrato=3, clips_por_hablante=20)
    resultado = select_subset(speakers, STRATA, 3, 5, 42)
    for hablantes in resultado.values():
        indices = {
            tuple(clip["path"].split("_")[-1] for clip in clips)
            for clips in hablantes.values()
        }
        assert len(indices) > 1


def test_select_subset_es_anidado_al_subir_k():
    speakers = corpus_de_prueba()
    chico = select_subset(speakers, STRATA, 2, 5, 42)
    grande = select_subset(speakers, STRATA, 4, 5, 42)
    for estrato in STRATA:
        assert set(chico[estrato]) <= set(grande[estrato])
        for client_id in chico[estrato]:
            assert chico[estrato][client_id] == grande[estrato][client_id]


def test_select_subset_solo_muestrea_los_estratos_pedidos():
    resultado = select_subset(corpus_de_prueba(), ["mexico", "caribe"], 3, 5, 42)
    assert list(resultado.keys()) == ["mexico", "caribe"]


def test_select_subset_ignora_los_hablantes_de_otro():
    speakers = corpus_de_prueba()
    speakers.update(hablantes_de(OTRO, 8))
    resultado = select_subset(speakers, STRATA, 3, 5, 42)
    assert OTRO not in resultado
    for hablantes in resultado.values():
        assert not any(client_id.startswith(f"{OTRO}-") for client_id in hablantes)


def test_select_subset_no_modifica_la_entrada():
    speakers = corpus_de_prueba()
    copia = copy.deepcopy(speakers)
    select_subset(speakers, STRATA, 3, 5, 42)
    assert speakers == copia


def test_select_subset_error_si_un_estrato_tiene_menos_de_k_hablantes():
    speakers = {c: v for c, v in corpus_de_prueba().items() if not c.startswith("chileno-")}
    speakers.update(hablantes_de("chileno", 2))  # chileno queda con 2 < k=3
    with pytest.raises(ValueError):
        select_subset(speakers, STRATA, 3, 5, 42)


def test_select_subset_error_si_falta_un_estrato():
    speakers = {c: v for c, v in corpus_de_prueba().items() if not c.startswith("caribe-")}
    with pytest.raises(ValueError):
        select_subset(speakers, STRATA, 3, 5, 42)


def test_select_subset_error_si_un_hablante_tiene_menos_de_n_clips():
    speakers = corpus_de_prueba(clips_por_hablante=3)
    with pytest.raises(ValueError):
        select_subset(speakers, STRATA, 3, 5, 42)


def test_select_subset_alimenta_a_write_subset(tmp_path):
    salida = tmp_path / "subset.tsv"
    resultado = select_subset(corpus_de_prueba(), STRATA, 3, 5, 42)
    write_subset(resultado, salida)
    filas = list(read_tsv_rows(salida))
    assert len(filas) == len(STRATA) * 3 * 5
    assert {fila["stratum"] for fila in filas} == set(STRATA)