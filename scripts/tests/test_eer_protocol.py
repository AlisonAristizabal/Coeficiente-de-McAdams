import copy
import random

import pytest

from eer_protocol import split_enrollment_trial


def clips_de(client_id, cantidad=5, estrato="mexico", estado="ok", inicio=0):
    """`cantidad` filas de manifiesto de un hablante, con session_id únicos."""
    return [
        {
            "client_id": client_id,
            "stratum": estrato,
            "path": f"{client_id}_{i}.mp3",
            "session_id": f"{client_id}_{i}",
            "coefficient": "0.8",
            "status": estado,
            "error": "",
        }
        for i in range(inicio, inicio + cantidad)
    ]


def manifiesto(hablantes=6, estrato="mexico", clips=5):
    filas = []
    for i in range(hablantes):
        filas.extend(clips_de(f"h{i}", cantidad=clips, estrato=estrato))
    return filas


# --- estructura del resultado ---------------------------------------------

def test_devuelve_una_tupla_con_splits_y_excluidos():
    resultado = split_enrollment_trial(manifiesto())
    assert isinstance(resultado, tuple) and len(resultado) == 2
    splits, excluidos = resultado
    assert isinstance(splits, dict) and isinstance(excluidos, dict)


def test_cada_hablante_tiene_las_llaves_esperadas():
    splits, _ = split_enrollment_trial(manifiesto())
    for datos in splits.values():
        assert list(datos.keys()) == ["stratum", "enroll", "trial"]


def test_tres_de_registro_y_dos_de_prueba_por_defecto():
    splits, _ = split_enrollment_trial(manifiesto())
    for datos in splits.values():
        assert len(datos["enroll"]) == 3
        assert len(datos["trial"]) == 2


def test_registro_y_prueba_son_disjuntos_y_del_mismo_hablante():
    splits, _ = split_enrollment_trial(manifiesto())
    for client_id, datos in splits.items():
        todos = datos["enroll"] + datos["trial"]
        assert len(set(todos)) == 5
        assert all(session.startswith(f"{client_id}_") for session in todos)


def test_los_clips_se_entregan_como_session_id_de_tipo_texto():
    splits, _ = split_enrollment_trial(manifiesto())
    assert all(isinstance(s, str) for d in splits.values() for s in d["enroll"] + d["trial"])


def test_los_hablantes_salen_ordenados_por_client_id():
    filas = manifiesto(hablantes=8)
    random.Random(1).shuffle(filas)
    splits, _ = split_enrollment_trial(filas)
    assert list(splits.keys()) == sorted(splits.keys())


def test_el_estrato_de_cada_hablante_se_conserva():
    filas = clips_de("a", estrato="caribe") + clips_de("b", estrato="chileno")
    splits, _ = split_enrollment_trial(filas)
    assert splits["a"]["stratum"] == "caribe"
    assert splits["b"]["stratum"] == "chileno"


# --- reproducibilidad ------------------------------------------------------

def test_misma_semilla_mismo_resultado():
    filas = manifiesto()
    assert split_enrollment_trial(filas, seed=42) == split_enrollment_trial(filas, seed=42)


def test_la_semilla_por_defecto_es_42():
    filas = manifiesto()
    assert split_enrollment_trial(filas) == split_enrollment_trial(filas, seed=42)


def test_semilla_distinta_cambia_algun_reparto():
    filas = manifiesto(hablantes=10)
    a, _ = split_enrollment_trial(filas, seed=42)
    b, _ = split_enrollment_trial(filas, seed=7)
    assert a != b


def test_el_resultado_no_depende_del_orden_de_las_filas():
    filas = manifiesto(hablantes=8)
    barajadas = copy.deepcopy(filas)
    random.Random(3).shuffle(barajadas)
    assert split_enrollment_trial(filas) == split_enrollment_trial(barajadas)


def test_cada_hablante_se_reparte_de_forma_distinta():
    splits, _ = split_enrollment_trial(manifiesto(hablantes=12))
    repartos = {tuple(s.split("_")[-1] for s in d["enroll"]) for d in splits.values()}
    assert len(repartos) > 1


# --- filtrado por estado ---------------------------------------------------

def test_ignora_las_filas_con_error():
    filas = clips_de("h0", cantidad=5) + clips_de("h0", cantidad=2, estado="error", inicio=5)
    splits, excluidos = split_enrollment_trial(filas)
    usados = set(splits["h0"]["enroll"] + splits["h0"]["trial"])
    assert usados == {f"h0_{i}" for i in range(5)}
    assert excluidos == {}


def test_hablante_con_cuatro_validos_se_excluye_con_su_conteo():
    filas = clips_de("h0", cantidad=4) + clips_de("h0", cantidad=1, estado="error", inicio=4)
    splits, excluidos = split_enrollment_trial(filas)
    assert "h0" not in splits
    assert excluidos == {"h0": 4}


def test_hablante_sin_ningun_clip_valido_aparece_en_excluidos_con_cero():
    filas = clips_de("h0", cantidad=5, estado="error") + clips_de("h1", cantidad=5)
    splits, excluidos = split_enrollment_trial(filas)
    assert "h0" not in splits
    assert excluidos == {"h0": 0}
    assert "h1" in splits


def test_exactamente_cinco_validos_se_incluye():
    splits, excluidos = split_enrollment_trial(clips_de("h0", cantidad=5))
    assert "h0" in splits
    assert excluidos == {}


def test_excluir_a_un_hablante_no_cambia_el_reparto_de_los_demas():
    completo = manifiesto(hablantes=5)
    sin_h2 = [f for f in completo if f["client_id"] != "h2"]
    a, _ = split_enrollment_trial(completo)
    b, _ = split_enrollment_trial(sin_h2)
    for client_id in b:
        assert a[client_id] == b[client_id]


# --- parámetros ------------------------------------------------------------

def test_con_mas_clips_de_los_necesarios_elige_cinco_distintos_del_hablante():
    splits, excluidos = split_enrollment_trial(clips_de("h0", cantidad=8))
    datos = splits["h0"]
    todos = datos["enroll"] + datos["trial"]
    assert len(todos) == 5 and len(set(todos)) == 5
    assert set(todos) <= {f"h0_{i}" for i in range(8)}
    assert excluidos == {}


def test_n_enroll_y_n_trial_personalizados():
    splits, _ = split_enrollment_trial(clips_de("h0", cantidad=6), n_enroll=4, n_trial=1)
    assert len(splits["h0"]["enroll"]) == 4
    assert len(splits["h0"]["trial"]) == 1


def test_el_umbral_de_exclusion_depende_de_los_parametros():
    filas = clips_de("h0", cantidad=5)
    splits, excluidos = split_enrollment_trial(filas, n_enroll=4, n_trial=2)
    assert splits == {}
    assert excluidos == {"h0": 5}


# --- entradas --------------------------------------------------------------

def test_sin_filas_devuelve_diccionarios_vacios():
    assert split_enrollment_trial([]) == ({}, {})


def test_acepta_un_generador():
    resultado = split_enrollment_trial(fila for fila in manifiesto())
    assert len(resultado[0]) == 6


def test_no_modifica_las_filas_de_entrada():
    filas = manifiesto()
    copia = copy.deepcopy(filas)
    split_enrollment_trial(filas)
    assert filas == copia


def test_fila_sin_la_llave_status_lanza_key_error():
    filas = manifiesto()
    del filas[0]["status"]
    with pytest.raises(KeyError):
        split_enrollment_trial(filas)