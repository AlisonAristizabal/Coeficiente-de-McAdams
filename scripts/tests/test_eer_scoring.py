import random
from fractions import Fraction

import numpy as np
import pytest

from eer_protocol import split_enrollment_trial
from eer_scoring import (
    build_enrollment,
    build_trials,
    compute_eer,
    normalize_rows,
    score_matrix,
    split_target_impostor,
)


# --- utilidades ------------------------------------------------------------

def hacer_splits(spec):
    """spec: {client_id: (n_registro, n_prueba)} -> splits con session_id únicos."""
    splits = {}
    for client_id, (n_enroll, n_trial) in spec.items():
        splits[client_id] = {
            "stratum": "mexico",
            "enroll": [f"{client_id}_e{i}" for i in range(n_enroll)],
            "trial": [f"{client_id}_t{i}" for i in range(n_trial)],
        }
    return splits


def tabla_de(splits, vectores):
    """Construye (embeddings, row_of) desde {session_id: vector}."""
    ids = list(vectores)
    return np.array([vectores[s] for s in ids], dtype=float), {s: i for i, s in enumerate(ids)}


def eer_referencia(target, impostor):
    """EER exacto con Fraction: umbral observado de menor |FAR-FRR|, empate -> más bajo."""
    umbrales = sorted(set(target) | set(impostor))
    mejor = None
    for t in umbrales:
        far = Fraction(sum(1 for x in impostor if x >= t), len(impostor))
        frr = Fraction(sum(1 for x in target if x < t), len(target))
        brecha = abs(far - frr)
        if mejor is None or brecha < mejor[0]:
            mejor = (brecha, (far + frr) / 2, t)
    return mejor[1], mejor[2]


# --- normalize_rows --------------------------------------------------------

def test_normalize_rows_deja_norma_uno():
    salida = normalize_rows([[3.0, 4.0], [0.0, 2.0], [1.0, 1.0]])
    assert np.allclose(np.linalg.norm(salida, axis=1), 1.0)


def test_normalize_rows_valores_conocidos():
    salida = normalize_rows([[3.0, 4.0], [0.0, -2.0]])
    assert np.allclose(salida, [[0.6, 0.8], [0.0, -1.0]])


def test_normalize_rows_normaliza_por_fila_no_por_columna():
    salida = normalize_rows([[3.0, 0.0], [0.0, 5.0]])
    assert np.allclose(salida, [[1.0, 0.0], [0.0, 1.0]])


def test_normalize_rows_no_modifica_la_entrada():
    entrada = np.array([[3.0, 4.0], [1.0, 0.0]])
    copia = entrada.copy()
    normalize_rows(entrada)
    assert np.array_equal(entrada, copia)


def test_normalize_rows_acepta_listas_y_enteros():
    salida = normalize_rows([[3, 4]])
    assert salida.dtype == np.float64
    assert np.allclose(salida, [[0.6, 0.8]])


def test_normalize_rows_una_sola_fila():
    assert normalize_rows([[0.0, 0.0, 7.0]]).shape == (1, 3)


def test_normalize_rows_fila_cero_lanza_value_error():
    with pytest.raises(ValueError):
        normalize_rows([[1.0, 0.0], [0.0, 0.0]])


# --- build_enrollment ------------------------------------------------------

def test_enrollment_promedia_los_normalizados_no_los_crudos():
    # Normas distintas: promedio de normalizados = [0.5, 0.5]; de crudos = [1.5, 2.5].
    splits = hacer_splits({"a": (2, 0)})
    emb, row_of = tabla_de(splits, {"a_e0": [3.0, 0.0], "a_e1": [0.0, 5.0]})
    salida = build_enrollment(emb, row_of, splits)
    assert np.allclose(salida, [[0.5, 0.5]])


def test_enrollment_no_renormaliza_el_promedio():
    splits = hacer_splits({"a": (2, 0)})
    emb, row_of = tabla_de(splits, {"a_e0": [1.0, 0.0], "a_e1": [0.0, 1.0]})
    salida = build_enrollment(emb, row_of, splits)
    assert np.isclose(np.linalg.norm(salida[0]), np.sqrt(0.5))


def test_enrollment_sigue_el_orden_de_splits():
    splits = hacer_splits({"z": (1, 0), "a": (1, 0)})  # orden de inserción: z, a
    emb, row_of = tabla_de(splits, {"z_e0": [1.0, 0.0], "a_e0": [0.0, 1.0]})
    salida = build_enrollment(emb, row_of, splits)
    assert np.allclose(salida, [[1.0, 0.0], [0.0, 1.0]])


def test_enrollment_usa_row_of_y_no_la_posicion():
    splits = hacer_splits({"a": (1, 0)})
    emb = np.array([[9.0, 9.0], [0.0, 4.0], [7.0, 7.0]])
    salida = build_enrollment(emb, {"a_e0": 1}, splits)
    assert np.allclose(salida, [[0.0, 1.0]])


def test_enrollment_ignora_los_clips_de_prueba():
    splits = hacer_splits({"a": (1, 1)})
    emb, row_of = tabla_de(splits, {"a_e0": [1.0, 0.0], "a_t0": [0.0, 1.0]})
    salida = build_enrollment(emb, row_of, splits)
    assert np.allclose(salida, [[1.0, 0.0]])


def test_enrollment_forma_una_fila_por_hablante():
    splits = hacer_splits({"a": (3, 2), "b": (3, 2), "c": (3, 2)})
    emb = np.random.default_rng(0).normal(size=(15, 6))
    row_of = {s: i for i, s in enumerate(
        [x for c in splits for x in splits[c]["enroll"] + splits[c]["trial"]])}
    assert build_enrollment(emb, row_of, splits).shape == (3, 6)


def test_enrollment_splits_vacio_lanza_value_error():
    with pytest.raises(ValueError):
        build_enrollment(np.zeros((1, 2)), {}, {})


def test_enrollment_embedding_cero_lanza_value_error():
    splits = hacer_splits({"a": (2, 0)})
    emb, row_of = tabla_de(splits, {"a_e0": [1.0, 0.0], "a_e1": [0.0, 0.0]})
    with pytest.raises(ValueError):
        build_enrollment(emb, row_of, splits)


def test_enrollment_no_modifica_embeddings():
    splits = hacer_splits({"a": (2, 0)})
    emb, row_of = tabla_de(splits, {"a_e0": [3.0, 0.0], "a_e1": [0.0, 5.0]})
    copia = emb.copy()
    build_enrollment(emb, row_of, splits)
    assert np.array_equal(emb, copia)


# --- build_trials ----------------------------------------------------------

def test_trials_owner_y_orden():
    splits = hacer_splits({"a": (1, 2), "b": (1, 1), "c": (1, 3)})
    ids = [s for c in splits for s in splits[c]["trial"]]
    emb = np.arange(len(ids) * 2, dtype=float).reshape(-1, 2)
    row_of = {s: i for i, s in enumerate(ids)}
    trials, owner = build_trials(emb, row_of, splits)
    assert owner.tolist() == [0, 0, 1, 2, 2, 2]
    assert np.array_equal(trials, emb)


def test_trials_no_normaliza():
    splits = hacer_splits({"a": (0, 1)})
    trials, _ = build_trials(np.array([[3.0, 4.0]]), {"a_t0": 0}, splits)
    assert np.allclose(trials, [[3.0, 4.0]])


def test_trials_usa_row_of():
    splits = hacer_splits({"a": (0, 1)})
    emb = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    trials, _ = build_trials(emb, {"a_t0": 2}, splits)
    assert np.allclose(trials, [[3.0, 3.0]])


def test_trials_ignora_registro():
    splits = hacer_splits({"a": (1, 1)})
    emb, row_of = tabla_de(splits, {"a_e0": [1.0, 0.0], "a_t0": [0.0, 1.0]})
    trials, owner = build_trials(emb, row_of, splits)
    assert trials.shape == (1, 2) and owner.tolist() == [0]


def test_trials_hablante_sin_pruebas_conserva_indices():
    splits = hacer_splits({"a": (1, 0), "b": (1, 1)})
    emb, row_of = tabla_de(splits, {"a_e0": [1.0], "b_e0": [2.0], "b_t0": [3.0]})
    _, owner = build_trials(emb, row_of, splits)
    assert owner.tolist() == [1]


# --- score_matrix ----------------------------------------------------------

def test_score_matrix_forma_y_orientacion():
    registro = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # 3 hablantes
    pruebas = np.array([[2.0, 0.0], [0.0, 3.0]])                # 2 pruebas
    puntos = score_matrix(registro, pruebas)
    assert puntos.shape == (2, 3)
    assert np.allclose(puntos[0], [1.0, 0.0, 1 / np.sqrt(2)])
    assert np.allclose(puntos[1], [0.0, 1.0, 1 / np.sqrt(2)])


def test_score_matrix_invariante_a_la_escala():
    rng = np.random.default_rng(1)
    r, p = rng.normal(size=(4, 5)), rng.normal(size=(6, 5))
    assert np.allclose(score_matrix(r, p), score_matrix(r * 7.5, p * 0.01))


def test_score_matrix_vectores_opuestos_dan_menos_uno():
    assert np.allclose(score_matrix([[1.0, 0.0]], [[-5.0, 0.0]]), [[-1.0]])


def test_score_matrix_valores_dentro_de_menos_uno_y_uno():
    rng = np.random.default_rng(2)
    puntos = score_matrix(rng.normal(size=(5, 8)), rng.normal(size=(9, 8)))
    assert puntos.min() >= -1 - 1e-12 and puntos.max() <= 1 + 1e-12


def test_score_matrix_cero_lanza_value_error():
    with pytest.raises(ValueError):
        score_matrix([[0.0, 0.0]], [[1.0, 0.0]])
    with pytest.raises(ValueError):
        score_matrix([[1.0, 0.0]], [[0.0, 0.0]])


# --- split_target_impostor -------------------------------------------------

def test_split_separa_objetivo_e_impostor():
    puntos = np.array([[0.9, 0.1, 0.2],
                       [0.3, 0.8, 0.4]])
    objetivo, impostor = split_target_impostor(puntos, [0, 1])
    assert objetivo.tolist() == [0.9, 0.8]
    assert impostor.tolist() == [0.1, 0.2, 0.3, 0.4]


def test_split_owner_no_es_la_diagonal():
    puntos = np.array([[0.1, 0.2, 0.9],
                       [0.8, 0.3, 0.4]])
    objetivo, impostor = split_target_impostor(puntos, [2, 0])
    assert objetivo.tolist() == [0.9, 0.8]
    assert sorted(impostor.tolist()) == [0.1, 0.2, 0.3, 0.4]


def test_split_dos_pruebas_del_mismo_hablante():
    puntos = np.array([[0.9, 0.1], [0.7, 0.2], [0.3, 0.6]])
    objetivo, impostor = split_target_impostor(puntos, [0, 0, 1])
    assert objetivo.tolist() == [0.9, 0.7, 0.6]
    assert impostor.tolist() == [0.1, 0.2, 0.3]


def test_split_conteos():
    puntos = np.random.default_rng(3).random((10, 5))
    owner = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    objetivo, impostor = split_target_impostor(puntos, owner)
    assert objetivo.size == 10 and impostor.size == 40


def test_split_owner_de_otro_largo_lanza_value_error():
    with pytest.raises(ValueError):
        split_target_impostor(np.zeros((3, 2)), [0, 1])
    with pytest.raises(ValueError):
        split_target_impostor(np.zeros((2, 2)), [0, 1, 0])


# --- compute_eer: casos a mano --------------------------------------------

def test_eer_separacion_perfecta_es_cero():
    eer, umbral = compute_eer([0.9, 0.8], [0.1, 0.2])
    assert eer == 0.0 and umbral == 0.8


def test_eer_separacion_invertida():
    # En .8 todos los objetivos se rechazan (FRR=1) y todos los impostores se aceptan (FAR=1).
    eer, umbral = compute_eer([0.1, 0.2], [0.8, 0.9])
    assert eer == 1.0 and umbral == 0.8


def test_eer_un_solo_par_igual():
    eer, umbral = compute_eer([0.5], [0.5])
    assert eer == 0.5 and umbral == 0.5


def test_eer_se_acepta_con_mayor_o_igual():
    # Con ">" el FAR sería 0 y el EER 0; con ">=" el impostor igual al umbral se acepta.
    eer, _ = compute_eer([0.5, 0.5], [0.5])
    assert eer == 0.5


def test_eer_empate_gana_el_umbral_mas_bajo():
    # Umbrales .3/.5/.7: brechas 2,1,1. En .5 el EER es .25; en .7 sería .75.
    eer, umbral = compute_eer([0.5], [0.3, 0.7])
    assert umbral == 0.5 and eer == 0.25


def test_eer_reporta_promedio_de_far_y_frr():
    # target {1,2,3,4}, impostor {2.5,3.5,5}: en 3 la brecha es |2/3-2/4|=1/6;
    # en 3.5 es |2/3-3/4|=1/12 (mínima), con EER=(2/3+3/4)/2=17/24.
    eer, umbral = compute_eer([1, 2, 3, 4], [2.5, 3.5, 5])
    assert umbral == 3.5
    assert eer == pytest.approx(17 / 24)


def test_eer_tamanos_distintos_usan_multiplicacion_cruzada():
    # 1 objetivo vs 4 impostores: FRR en {0,1}, FAR en {0,.25,.5,.75,1}.
    eer, umbral = compute_eer([0.6], [0.1, 0.2, 0.7, 0.8])
    # umbral .7: FRR=1 (0.6<.7), FAR=.5 ; umbral .6: FRR=0, FAR=.5 ; brechas .5 y .5 -> gana .6
    assert umbral == 0.6 and eer == pytest.approx(0.25)


# --- compute_eer: propiedades ---------------------------------------------

def test_eer_no_depende_del_orden_de_entrada():
    objetivo = [0.9, 0.2, 0.6, 0.4]
    impostor = [0.1, 0.5, 0.3, 0.7, 0.8]
    a = compute_eer(objetivo, impostor)
    b = compute_eer(objetivo[::-1], impostor[::-1])
    assert a == b


def test_eer_no_modifica_las_entradas():
    objetivo = np.array([0.9, 0.2, 0.6])
    impostor = np.array([0.5, 0.1, 0.7, 0.3])
    copias = objetivo.copy(), impostor.copy()
    compute_eer(objetivo, impostor)
    assert np.array_equal(objetivo, copias[0]) and np.array_equal(impostor, copias[1])


def test_eer_acepta_listas():
    assert compute_eer([0.9, 0.8], [0.1, 0.2]) == (0.0, 0.8)


def test_eer_rango_y_umbral_observado():
    rng = np.random.default_rng(4)
    objetivo, impostor = rng.normal(1, 1, 200), rng.normal(0, 1, 800)
    eer, umbral = compute_eer(objetivo, impostor)
    assert 0.0 <= eer <= 1.0
    assert umbral in set(objetivo) | set(impostor)


def test_eer_vacios_lanzan_value_error():
    with pytest.raises(ValueError):
        compute_eer([], [0.1])
    with pytest.raises(ValueError):
        compute_eer([0.1], [])


def test_eer_nan_lanza_value_error():
    with pytest.raises(ValueError):
        compute_eer([0.1, float("nan")], [0.2])
    with pytest.raises(ValueError):
        compute_eer([0.1], [0.2, float("nan")])


def test_eer_coincide_con_referencia_exacta_en_casos_aleatorios():
    azar = random.Random(7)
    for _ in range(300):
        objetivo = [round(azar.random(), 1) for _ in range(azar.randint(1, 8))]
        impostor = [round(azar.random(), 1) for _ in range(azar.randint(1, 8))]
        esperado, umbral_esp = eer_referencia(objetivo, impostor)
        eer, umbral = compute_eer(objetivo, impostor)
        assert eer == pytest.approx(float(esperado), abs=1e-12), (objetivo, impostor)
        assert umbral == umbral_esp, (objetivo, impostor)


# --- contrato entre módulos y flujo completo ------------------------------

def manifiesto(hablantes, clips=5):
    return [
        {"client_id": f"h{i}", "stratum": "mexico", "path": f"h{i}_{j}.mp3",
         "session_id": f"h{i}_{j}", "coefficient": "0.8", "status": "ok", "error": ""}
        for i in range(hablantes) for j in range(clips)
    ]


def test_contrato_con_split_enrollment_trial():
    # Fija las llaves de `split_enrollment_trial` que consume `build_enrollment`/`build_trials`.
    filas = manifiesto(3)
    splits, _ = split_enrollment_trial(filas)
    rng = np.random.default_rng(5)
    row_of = {f["session_id"]: i for i, f in enumerate(filas)}
    emb = rng.normal(size=(len(filas), 8))
    assert build_enrollment(emb, row_of, splits).shape == (3, 8)
    trials, owner = build_trials(emb, row_of, splits)
    assert trials.shape == (6, 8) and owner.tolist() == [0, 0, 1, 1, 2, 2]


def test_flujo_completo_hablantes_separables_dan_eer_cero():
    # Cada hablante tiene su propia dirección (una por eje) más ruido pequeño.
    rng = np.random.default_rng(6)
    n = 8
    filas = manifiesto(n)
    row_of = {f["session_id"]: i for i, f in enumerate(filas)}
    emb = np.zeros((len(filas), n))
    for i, fila in enumerate(filas):
        propietario = int(fila["client_id"][1:])
        emb[i, propietario] = 5.0
        emb[i] += rng.normal(scale=0.1, size=n)
    splits, _ = split_enrollment_trial(filas)
    registro = build_enrollment(emb, row_of, splits)
    pruebas, owner = build_trials(emb, row_of, splits)
    objetivo, impostor = split_target_impostor(score_matrix(registro, pruebas), owner)
    assert objetivo.size == 16 and impostor.size == 16 * 7
    eer, _ = compute_eer(objetivo, impostor)
    assert eer == 0.0


def test_flujo_completo_sin_informacion_da_eer_cercano_a_la_mitad():
    rng = np.random.default_rng(8)
    n = 60
    filas = manifiesto(n)
    row_of = {f["session_id"]: i for i, f in enumerate(filas)}
    emb = rng.normal(size=(len(filas), 32))
    splits, _ = split_enrollment_trial(filas)
    registro = build_enrollment(emb, row_of, splits)
    pruebas, owner = build_trials(emb, row_of, splits)
    objetivo, impostor = split_target_impostor(score_matrix(registro, pruebas), owner)
    eer, _ = compute_eer(objetivo, impostor)
    assert 0.35 < eer < 0.65