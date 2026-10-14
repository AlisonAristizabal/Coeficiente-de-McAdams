import csv
import json
import sys

import numpy as np
import pytest

from eer_protocol import split_enrollment_trial
from eer_scoring import (
    build_enrollment,
    build_trials,
    compute_eer,
    normalize_rows,
    split_target_impostor,
)
from run_eer import (
    SCENARIOS,
    build_row_index,
    check_splits,
    load_and_score,
    main,
    run_scenarios,
    score_scenario,
)

DIM = 8


# ---------------------------------------------------------------------------
# Datos sintéticos
# ---------------------------------------------------------------------------

def make_rows(n_speakers, n_clips=5, strata=2):
    """Filas de manifiesto: `n_speakers` hablantes con `n_clips` clips "ok"."""
    rows = []
    for s in range(n_speakers):
        for c in range(n_clips):
            session_id = f"spk{s:02d}_c{c}"
            rows.append({
                "client_id": f"spk{s:02d}",
                "stratum": f"st{s % strata}",
                "path": f"{session_id}.mp3",
                "session_id": session_id,
                "coefficient": "0.8",
                "status": "ok",
                "error": "",
            })
    return rows


def make_embeddings(rows, seed=0, spread=0.05):
    """Embeddings originales agrupados por hablante (un centro por hablante)."""
    rng = np.random.default_rng(seed)
    centers = {}
    session_ids, vectors = [], []
    for row in rows:
        cid = row["client_id"]
        if cid not in centers:
            centers[cid] = rng.normal(size=DIM)
        session_ids.append(row["session_id"])
        vectors.append(centers[cid] + spread * rng.normal(size=DIM))
    return session_ids, np.array(vectors), centers


def setup_case(n_speakers=6, seed=0):
    rows = make_rows(n_speakers)
    session_ids, original, centers = make_embeddings(rows, seed=seed)
    row_of = build_row_index(session_ids)
    splits, _ = split_enrollment_trial(rows)
    return rows, session_ids, original, centers, row_of, splits


def manual_scores(enroll_matrix, trial_matrix, row_of, splits):
    """Coseno calculado a mano: (pruebas x hablantes), registro = media de normalizados."""
    enrollment = []
    for cid in splits:
        e = np.array([enroll_matrix[row_of[s]] for s in splits[cid]["enroll"]], dtype=np.float64)
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        m = e.mean(axis=0)
        enrollment.append(m / np.linalg.norm(m))
    enrollment = np.array(enrollment)
    trials = np.array([trial_matrix[row_of[s]] for cid in splits for s in splits[cid]["trial"]], dtype=np.float64)
    trials = trials / np.linalg.norm(trials, axis=1, keepdims=True)
    return trials @ enrollment.T


# ---------------------------------------------------------------------------
# SCENARIOS
# ---------------------------------------------------------------------------

def test_scenarios_tiene_exactamente_los_tres_escenarios():
    assert set(SCENARIOS) == {"o-o", "o-a", "a-a"}


def test_scenarios_o_o():
    assert SCENARIOS["o-o"] == ("original", "original")


def test_scenarios_o_a_registro_original_y_prueba_anonimizada():
    assert SCENARIOS["o-a"] == ("original", "anonymized")


def test_scenarios_a_a():
    assert SCENARIOS["a-a"] == ("anonymized", "anonymized")


def test_scenarios_usa_llaves_del_npz():
    llaves = {fuente for par in SCENARIOS.values() for fuente in par}
    assert llaves == {"original", "anonymized"}


# ---------------------------------------------------------------------------
# build_row_index
# ---------------------------------------------------------------------------

def test_build_row_index_asigna_la_posicion_de_cada_session_id():
    assert build_row_index(["a", "b", "c"]) == {"a": 0, "b": 1, "c": 2}


def test_build_row_index_lista_vacia():
    assert build_row_index([]) == {}


def test_build_row_index_acepta_un_iterable_no_lista():
    assert build_row_index(iter(["x", "y"])) == {"x": 0, "y": 1}


def test_build_row_index_duplicado_lanza_valueerror_con_el_id():
    with pytest.raises(ValueError, match="clip_77"):
        build_row_index(["clip_10", "clip_77", "clip_12", "clip_77"])


def test_build_row_index_duplicado_al_inicio():
    with pytest.raises(ValueError):
        build_row_index(["a", "a"])


def test_build_row_index_indices_coinciden_con_las_filas_del_npz():
    session_ids = [f"clip_{i}" for i in range(50)]
    row_of = build_row_index(session_ids)
    assert all(session_ids[row_of[s]] == s for s in session_ids)


# ---------------------------------------------------------------------------
# check_splits
# ---------------------------------------------------------------------------

SPLITS = {
    "A": {"stratum": "s", "enroll": ["a1", "a2"], "trial": ["a3"]},
    "B": {"stratum": "s", "enroll": ["b1", "b2"], "trial": ["b3"]},
}
COMPLETO = {s: i for i, s in enumerate(["a1", "a2", "a3", "b1", "b2", "b3"])}


def test_check_splits_completo_no_lanza():
    assert check_splits(SPLITS, COMPLETO) is None


def test_check_splits_falta_un_clip_de_registro():
    row_of = {k: v for k, v in COMPLETO.items() if k != "b2"}
    with pytest.raises(ValueError, match="b2"):
        check_splits(SPLITS, row_of)


def test_check_splits_falta_un_clip_de_prueba():
    row_of = {k: v for k, v in COMPLETO.items() if k != "a3"}
    with pytest.raises(ValueError, match="a3"):
        check_splits(SPLITS, row_of)


def test_check_splits_informa_cuantos_faltan():
    row_of = {"a1": 0, "b1": 3}
    with pytest.raises(ValueError, match="4 clips"):
        check_splits(SPLITS, row_of)


def test_check_splits_ignora_clips_sobrantes_en_row_of():
    row_of = dict(COMPLETO, extra=99)
    assert check_splits(SPLITS, row_of) is None


def test_check_splits_no_modifica_entradas():
    splits = {k: {**v, "enroll": list(v["enroll"]), "trial": list(v["trial"])} for k, v in SPLITS.items()}
    row_of = dict(COMPLETO)
    check_splits(splits, row_of)
    assert splits == SPLITS and row_of == COMPLETO


# ---------------------------------------------------------------------------
# score_scenario
# ---------------------------------------------------------------------------

def test_score_scenario_forma_pruebas_por_hablantes():
    _, _, original, _, row_of, splits = setup_case(n_speakers=6)
    scores, owner = score_scenario(original, original, row_of, splits)
    assert scores.shape == (12, 6)
    assert owner.shape == (12,)


def test_score_scenario_owner_sigue_el_orden_de_splits():
    _, _, original, _, row_of, splits = setup_case(n_speakers=4)
    _, owner = score_scenario(original, original, row_of, splits)
    assert owner.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_score_scenario_coincide_con_coseno_manual():
    rng = np.random.default_rng(1)
    _, _, original, _, row_of, splits = setup_case(n_speakers=5)
    other = rng.normal(size=original.shape)
    scores, _ = score_scenario(original, other, row_of, splits)
    np.testing.assert_allclose(scores, manual_scores(original, other, row_of, splits))


def test_score_scenario_registro_sale_de_la_primera_matriz_y_prueba_de_la_segunda():
    rng = np.random.default_rng(2)
    _, _, original, _, row_of, splits = setup_case(n_speakers=5)
    other = rng.normal(size=original.shape)
    scores, _ = score_scenario(original, other, row_of, splits)
    swapped = manual_scores(other, original, row_of, splits)
    assert not np.allclose(scores, swapped)


def test_score_scenario_usa_build_enrollment_y_build_trials():
    rng = np.random.default_rng(3)
    _, _, original, _, row_of, splits = setup_case(n_speakers=5)
    other = rng.normal(size=original.shape)
    enrollment = build_enrollment(original, row_of, splits)
    trials, _ = build_trials(other, row_of, splits)
    expected = normalize_rows(trials) @ normalize_rows(enrollment).T
    scores, _ = score_scenario(original, other, row_of, splits)
    np.testing.assert_allclose(scores, expected)


def test_score_scenario_valores_en_rango_del_coseno():
    rng = np.random.default_rng(4)
    _, _, original, _, row_of, splits = setup_case(n_speakers=5)
    scores, _ = score_scenario(original, rng.normal(size=original.shape), row_of, splits)
    assert np.all(scores <= 1 + 1e-12) and np.all(scores >= -1 - 1e-12)


def test_score_scenario_no_modifica_embeddings():
    _, _, original, _, row_of, splits = setup_case(n_speakers=4)
    copy = original.copy()
    score_scenario(original, original, row_of, splits)
    np.testing.assert_array_equal(original, copy)


# ---------------------------------------------------------------------------
# run_scenarios
# ---------------------------------------------------------------------------

def anonymized_by_role(original, centers, rows, row_of, splits, seed=9):
    """Anonimizados donde los clips de prueba conservan la identidad y los de
    registro son ruido: o-a sale perfecto y su inverso (a-o) no."""
    rng = np.random.default_rng(seed)
    anon = rng.normal(size=original.shape) * 5
    for cid, sp in splits.items():
        for s in sp["trial"]:
            anon[row_of[s]] = centers[cid]
    return anon


def test_run_scenarios_devuelve_los_tres_escenarios():
    _, _, original, _, row_of, splits = setup_case()
    results = run_scenarios({"original": original, "anonymized": original}, row_of, splits)
    assert set(results) == {"o-o", "o-a", "a-a"}


def test_run_scenarios_cada_resultado_tiene_sus_llaves():
    _, _, original, _, row_of, splits = setup_case()
    results = run_scenarios({"original": original, "anonymized": original}, row_of, splits)
    for r in results.values():
        assert set(r) == {"scores", "owner", "eer", "threshold", "n_target", "n_impostor"}


def test_run_scenarios_cuenta_pares_objetivo_e_impostor():
    _, _, original, _, row_of, splits = setup_case(n_speakers=7)
    results = run_scenarios({"original": original, "anonymized": original}, row_of, splits)
    for r in results.values():
        assert r["n_target"] == 14
        assert r["n_impostor"] == 14 * 6


def test_run_scenarios_conteos_salen_de_los_datos_con_n_trial_distinto():
    rows = make_rows(5)
    session_ids, original, _ = make_embeddings(rows)
    row_of = build_row_index(session_ids)
    splits, _ = split_enrollment_trial(rows, n_enroll=2, n_trial=3)
    r = run_scenarios({"original": original, "anonymized": original}, row_of, splits)["o-o"]
    assert r["n_target"] == 15 and r["n_impostor"] == 15 * 4


def test_run_scenarios_conteos_son_enteros_de_python():
    _, _, original, _, row_of, splits = setup_case()
    r = run_scenarios({"original": original, "anonymized": original}, row_of, splits)["o-o"]
    assert type(r["n_target"]) is int and type(r["n_impostor"]) is int


def test_run_scenarios_o_o_separado_da_eer_cero():
    _, _, original, _, row_of, splits = setup_case()
    rng = np.random.default_rng(5)
    results = run_scenarios({"original": original, "anonymized": rng.normal(size=original.shape)}, row_of, splits)
    assert results["o-o"]["eer"] == 0.0


def test_run_scenarios_o_a_usa_registro_original_y_prueba_anonimizada():
    rows, _, original, centers, row_of, splits = setup_case(n_speakers=8)
    anon = anonymized_by_role(original, centers, rows, row_of, splits)
    results = run_scenarios({"original": original, "anonymized": anon}, row_of, splits)
    assert results["o-a"]["eer"] == 0.0


def test_run_scenarios_inverso_de_o_a_no_es_perfecto():
    rows, _, original, centers, row_of, splits = setup_case(n_speakers=8)
    anon = anonymized_by_role(original, centers, rows, row_of, splits)
    inverse = run_scenarios(
        {"original": original, "anonymized": anon}, row_of, splits,
        scenarios={"a-o": ("anonymized", "original")},
    )
    assert inverse["a-o"]["eer"] > 0.2


def test_run_scenarios_a_a_usa_solo_anonimizados():
    _, _, original, _, row_of, splits = setup_case(n_speakers=8)
    rng = np.random.default_rng(6)
    results = run_scenarios({"original": rng.normal(size=original.shape), "anonymized": original}, row_of, splits)
    assert results["a-a"]["eer"] == 0.0
    assert results["o-o"]["eer"] > 0.0


def test_run_scenarios_eer_coincide_con_el_calculo_manual():
    _, _, original, _, row_of, splits = setup_case(n_speakers=8)
    rng = np.random.default_rng(7)
    anon = original + rng.normal(size=original.shape) * 2
    results = run_scenarios({"original": original, "anonymized": anon}, row_of, splits)
    for name, (ek, tk) in SCENARIOS.items():
        m = {"original": original, "anonymized": anon}
        scores = manual_scores(m[ek], m[tk], row_of, splits)
        _, owner = build_trials(m[tk], row_of, splits)
        eer, thr = compute_eer(*split_target_impostor(scores, owner))
        assert results[name]["eer"] == pytest.approx(eer)
        assert results[name]["threshold"] == pytest.approx(thr)


def test_run_scenarios_devuelve_scores_y_owner_para_el_bootstrap():
    _, _, original, _, row_of, splits = setup_case(n_speakers=6)
    r = run_scenarios({"original": original, "anonymized": original}, row_of, splits)["o-a"]
    assert r["scores"].shape == (12, 6)
    assert r["owner"].tolist() == [i for i in range(6) for _ in range(2)]


def test_run_scenarios_acepta_escenarios_propios():
    _, _, original, _, row_of, splits = setup_case()
    results = run_scenarios({"original": original, "anonymized": original}, row_of, splits,
                            scenarios={"solo": ("original", "original")})
    assert set(results) == {"solo"}


def test_run_scenarios_valida_splits_antes_de_puntuar():
    _, _, original, _, row_of, splits = setup_case()
    incompleto = dict(row_of)
    faltante = next(iter(splits.values()))["trial"][0]
    del incompleto[faltante]
    with pytest.raises(ValueError, match=faltante):
        run_scenarios({"original": original, "anonymized": original}, incompleto, splits)


def test_run_scenarios_no_modifica_embeddings():
    _, _, original, _, row_of, splits = setup_case()
    copy = original.copy()
    run_scenarios({"original": original, "anonymized": original}, row_of, splits)
    np.testing.assert_array_equal(original, copy)


# ---------------------------------------------------------------------------
# main (de punta a punta con archivos temporales)
# ---------------------------------------------------------------------------

def write_inputs(tmp_path, rows, session_ids, original, anonymized):
    npz = tmp_path / "emb.npz"
    np.savez(npz, session_ids=np.array(session_ids, dtype=str),
             original=original.astype(np.float32), anonymized=anonymized.astype(np.float32))
    manifest = tmp_path / "manifest.tsv"
    with open(manifest, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return npz, manifest


def run_main(monkeypatch, npz, manifest, output, *extra):
    argv = ["run_eer.py", "--embeddings", str(npz), "--manifest", str(manifest), "--output", str(output), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    return json.loads(output.read_text(encoding="utf-8"))


@pytest.fixture
def inputs(tmp_path):
    rows = make_rows(6)
    session_ids, original, _ = make_embeddings(rows)
    rng = np.random.default_rng(8)
    anonymized = rng.normal(size=original.shape)
    npz, manifest = write_inputs(tmp_path, rows, session_ids, original, anonymized)
    return rows, session_ids, original, anonymized, npz, manifest


def test_main_escribe_json_con_metadatos_y_valores_por_defecto(monkeypatch, tmp_path, inputs):
    *_, npz, manifest = inputs
    out = run_main(monkeypatch, npz, manifest, tmp_path / "out.json")
    assert out["embeddings"] == str(npz) and out["manifest"] == str(manifest)
    assert (out["seed"], out["n_enroll"], out["n_trial"]) == (42, 3, 2)
    assert out["n_speakers"] == 6 and out["excluded"] == {}
    assert set(out["scenarios"]) == {"o-o", "o-a", "a-a"}


def test_main_cada_escenario_tiene_cifras_y_no_matrices(monkeypatch, tmp_path, inputs):
    *_, npz, manifest = inputs
    out = run_main(monkeypatch, npz, manifest, tmp_path / "out.json")
    for s in out["scenarios"].values():
        assert set(s) == {"eer", "threshold", "n_target", "n_impostor"}
        assert s["n_target"] == 12 and s["n_impostor"] == 60


def test_main_coincide_con_run_scenarios(monkeypatch, tmp_path, inputs):
    rows, session_ids, original, anonymized, npz, manifest = inputs
    out = run_main(monkeypatch, npz, manifest, tmp_path / "out.json")
    splits, _ = split_enrollment_trial(rows)
    expected = run_scenarios({"original": original.astype(np.float32), "anonymized": anonymized.astype(np.float32)},
                             build_row_index(session_ids), splits)
    for name in SCENARIOS:
        assert out["scenarios"][name]["eer"] == pytest.approx(float(expected[name]["eer"]))
        assert out["scenarios"][name]["threshold"] == pytest.approx(float(expected[name]["threshold"]))


def test_main_no_confunde_original_y_anonimizado(monkeypatch, tmp_path, inputs):
    *_, npz, manifest = inputs
    out = run_main(monkeypatch, npz, manifest, tmp_path / "out.json")
    assert out["scenarios"]["o-o"]["eer"] == 0.0
    assert out["scenarios"]["a-a"]["eer"] > 0.0


def test_main_respeta_n_enroll_y_n_trial(monkeypatch, tmp_path, inputs):
    *_, npz, manifest = inputs
    out = run_main(monkeypatch, npz, manifest, tmp_path / "out.json", "--n-enroll", "2", "--n-trial", "3")
    assert (out["n_enroll"], out["n_trial"]) == (2, 3)
    assert out["scenarios"]["o-o"]["n_target"] == 18


def test_main_respeta_la_semilla(monkeypatch, tmp_path, inputs):
    rows, session_ids, original, anonymized, npz, manifest = inputs
    out = run_main(monkeypatch, npz, manifest, tmp_path / "out.json", "--seed", "7")
    splits, _ = split_enrollment_trial(rows, seed=7)
    expected = run_scenarios({"original": original.astype(np.float32), "anonymized": anonymized.astype(np.float32)},
                             build_row_index(session_ids), splits)
    assert out["seed"] == 7
    assert out["scenarios"]["a-a"]["threshold"] == pytest.approx(float(expected["a-a"]["threshold"]))


def test_main_informa_hablantes_excluidos(monkeypatch, tmp_path):
    rows = make_rows(5)
    rows = [r for r in rows if r["session_id"] != "spk03_c4"]
    session_ids, original, _ = make_embeddings(rows)
    npz, manifest = write_inputs(tmp_path, rows, session_ids, original, original)
    out = run_main(monkeypatch, npz, manifest, tmp_path / "out.json")
    assert out["excluded"] == {"spk03": 4}
    assert out["n_speakers"] == 4


def test_main_ignora_filas_con_error(monkeypatch, tmp_path):
    rows = make_rows(5)
    session_ids, original, _ = make_embeddings(rows)
    rows[0] = {**rows[0], "status": "error", "error": "LibsndfileError: x"}
    npz, manifest = write_inputs(tmp_path, rows, session_ids, original, original)
    out = run_main(monkeypatch, npz, manifest, tmp_path / "out.json")
    assert out["excluded"] == {"spk00": 4}


def test_main_falla_si_un_clip_del_manifiesto_no_tiene_embedding(monkeypatch, tmp_path):
    rows = make_rows(4)
    session_ids, original, _ = make_embeddings(rows)
    keep = [i for i, s in enumerate(session_ids) if s != "spk01_c2"]
    npz, manifest = write_inputs(tmp_path, rows, [session_ids[i] for i in keep], original[keep], original[keep])
    monkeypatch.setattr(sys, "argv", ["run_eer.py", "--embeddings", str(npz), "--manifest", str(manifest),
                                      "--output", str(tmp_path / "o.json")])
    with pytest.raises(ValueError, match="no están en los embeddings"):
        main()


def test_main_falla_si_hay_session_ids_duplicados(monkeypatch, tmp_path):
    rows = make_rows(4)
    session_ids, original, _ = make_embeddings(rows)
    session_ids = session_ids[:-1] + [session_ids[0]]
    npz, manifest = write_inputs(tmp_path, rows, session_ids, original, original)
    monkeypatch.setattr(sys, "argv", ["run_eer.py", "--embeddings", str(npz), "--manifest", str(manifest),
                                      "--output", str(tmp_path / "o.json")])
    with pytest.raises(ValueError, match="duplicado"):
        main()


def test_main_imprime_resumen(monkeypatch, tmp_path, inputs, capsys):
    *_, npz, manifest = inputs
    output = tmp_path / "out.json"
    run_main(monkeypatch, npz, manifest, output)
    printed = capsys.readouterr().out
    assert "Hablantes: 6 (excluidos: 0)" in printed
    for name in ("o-o", "o-a", "a-a"):
        assert f"{name}: EER =" in printed
    assert str(output) in printed

# ---------------------------------------------------------------------------
# load_and_score
# ---------------------------------------------------------------------------

def test_load_and_score_coincide_con_los_pasos_por_separado(tmp_path, inputs):
    rows, session_ids, original, anonymized, npz, manifest = inputs
    splits, excluded, results = load_and_score(npz, manifest, 42, 3, 2)
    exp_splits, exp_excluded = split_enrollment_trial(rows)
    expected = run_scenarios({"original": original.astype(np.float32), "anonymized": anonymized.astype(np.float32)},
                             build_row_index(session_ids), exp_splits)
    assert splits == exp_splits and excluded == exp_excluded
    assert list(splits) == list(exp_splits)
    for name in SCENARIOS:
        np.testing.assert_array_equal(results[name]["scores"], expected[name]["scores"])
        np.testing.assert_array_equal(results[name]["owner"], expected[name]["owner"])
        assert results[name]["eer"] == expected[name]["eer"]
        assert results[name]["threshold"] == expected[name]["threshold"]


def test_load_and_score_respeta_semilla_n_enroll_y_n_trial(tmp_path, inputs):
    rows, *_, npz, manifest = inputs
    splits, _, results = load_and_score(npz, manifest, 7, 2, 3)
    assert splits == split_enrollment_trial(rows, n_enroll=2, n_trial=3, seed=7)[0]
    assert results["o-o"]["n_target"] == 18


def test_load_and_score_informa_excluidos(tmp_path):
    rows = [r for r in make_rows(5) if r["session_id"] != "spk02_c1"]
    session_ids, original, _ = make_embeddings(rows)
    npz, manifest = write_inputs(tmp_path, rows, session_ids, original, original)
    splits, excluded, _ = load_and_score(npz, manifest, 42, 3, 2)
    assert excluded == {"spk02": 4} and "spk02" not in splits


def test_load_and_score_acepta_rutas_como_texto(tmp_path, inputs):
    *_, npz, manifest = inputs
    a = load_and_score(npz, manifest, 42, 3, 2)[2]
    b = load_and_score(str(npz), str(manifest), 42, 3, 2)[2]
    np.testing.assert_array_equal(a["a-a"]["scores"], b["a-a"]["scores"])


def test_load_and_score_deja_el_npz_cerrado(tmp_path, inputs):
    *_, npz, manifest = inputs
    _, _, results = load_and_score(npz, manifest, 42, 3, 2)
    npz.unlink()  # En Windows falla si el archivo sigue abierto.
    assert results["o-o"]["scores"].shape == (12, 6)


def test_main_usa_load_and_score(monkeypatch, tmp_path, inputs):
    import run_eer
    *_, npz, manifest = inputs
    calls = []

    def spy(*args):
        calls.append(args)
        return load_and_score(*args)

    monkeypatch.setattr(run_eer, "load_and_score", spy)
    run_main(monkeypatch, npz, manifest, tmp_path / "out.json", "--seed", "7", "--n-enroll", "2", "--n-trial", "3")
    assert calls == [(str(npz), str(manifest), 7, 2, 3)]
