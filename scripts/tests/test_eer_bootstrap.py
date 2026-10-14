import csv
import json
import sys
from collections import Counter

import numpy as np
import pytest

import run_eer
from eer_bootstrap import (
    DIFFERENCES,
    bootstrap_eers,
    difference_cis,
    main,
    parse_args,
    percentile_ci,
    resample_pairs,
    stratified_resample,
)
from eer_protocol import split_enrollment_trial
from eer_scoring import compute_eer, split_target_impostor
from run_eer import build_row_index, run_scenarios

DIM = 8


# ---------------------------------------------------------------------------
# Datos sintéticos
# ---------------------------------------------------------------------------

def make_case(n_speakers=12, n_strata=3, seed=0):
    """Filas de manifiesto y embeddings con solapamiento: el EER no es cero."""
    rng = np.random.default_rng(seed)
    rows, session_ids, original, anonymized = [], [], [], []
    for s in range(n_speakers):
        center = rng.normal(size=DIM)
        for c in range(5):
            sid = f"spk{s:02d}_c{c}"
            session_ids.append(sid)
            original.append(center + 0.6 * rng.normal(size=DIM))
            anonymized.append(center + 1.5 * rng.normal(size=DIM))
            rows.append({"client_id": f"spk{s:02d}", "stratum": f"st{s % n_strata}", "path": sid + ".mp3",
                         "session_id": sid, "coefficient": "0.8", "status": "ok", "error": ""})
    return rows, session_ids, np.array(original, np.float32), np.array(anonymized, np.float32)


def make_results(n_speakers=12, n_strata=3, seed=0):
    rows, session_ids, original, anonymized = make_case(n_speakers, n_strata, seed)
    splits, _ = split_enrollment_trial(rows)
    results = run_scenarios({"original": original, "anonymized": anonymized}, build_row_index(session_ids), splits)
    strata = [sp["stratum"] for sp in splits.values()]
    return results, strata


def brute_force_pairs(scores, owner, sample):
    """Pares de la muestra contados uno por uno, copia contra copia."""
    target, impostor = [], []
    for k, trial_speaker in enumerate(sample):
        for i in np.flatnonzero(np.asarray(owner) == trial_speaker):
            for l, enroll_speaker in enumerate(sample):
                if k == l:
                    target.append(scores[i, enroll_speaker])
                elif trial_speaker != enroll_speaker:
                    impostor.append(scores[i, enroll_speaker])
    return np.array(target), np.array(impostor)


def coded_scores(owner, n_speakers):
    """scores[i, j] = 100 * owner[i] + j + i / 1000: cada valor dice de quién es el par."""
    owner = np.asarray(owner)
    i = np.arange(owner.size)[:, None]
    j = np.arange(n_speakers)[None, :]
    return 100.0 * owner[:, None] + j + i / 1000


def decode(values):
    """Devuelve (dueño de la prueba, hablante del registro) de cada valor de `coded_scores`."""
    values = np.floor(np.asarray(values)).astype(int)
    return values // 100, values % 100


# ---------------------------------------------------------------------------
# stratified_resample
# ---------------------------------------------------------------------------

STRATA = ["a"] * 5 + ["b"] * 3 + ["c"] * 1 + ["b"] * 2 + ["a"] * 4


def test_stratified_resample_largo_y_tipo_entero():
    sample = stratified_resample(STRATA, np.random.default_rng(0))
    assert sample.shape == (len(STRATA),)
    assert np.issubdtype(sample.dtype, np.integer)
    assert sample.min() >= 0 and sample.max() < len(STRATA)


def test_stratified_resample_conserva_el_conteo_por_estrato():
    for seed in range(20):
        sample = stratified_resample(STRATA, np.random.default_rng(seed))
        assert Counter(np.array(STRATA)[sample]) == Counter(STRATA)


def test_stratified_resample_solo_toma_indices_del_propio_estrato():
    strata = np.array(STRATA)
    for seed in range(20):
        sample = stratified_resample(STRATA, np.random.default_rng(seed))
        np.testing.assert_array_equal(strata[sample], strata)


def test_stratified_resample_estrato_de_un_hablante_siempre_lo_repite():
    for seed in range(10):
        sample = stratified_resample(STRATA, np.random.default_rng(seed))
        assert sample[8] == 8


def test_stratified_resample_reproducible_con_la_misma_semilla():
    a = stratified_resample(STRATA, np.random.default_rng(123))
    b = stratified_resample(STRATA, np.random.default_rng(123))
    np.testing.assert_array_equal(a, b)


def test_stratified_resample_cambia_con_otra_semilla():
    strata = [f"s{i % 10}" for i in range(1340)]
    a = stratified_resample(strata, np.random.default_rng(1))
    b = stratified_resample(strata, np.random.default_rng(2))
    assert not np.array_equal(a, b)


def test_stratified_resample_produce_repeticiones():
    strata = [f"s{i % 10}" for i in range(1340)]
    sample = stratified_resample(strata, np.random.default_rng(0))
    # Con reemplazo, en promedio queda ~63 % de hablantes distintos.
    assert len(np.unique(sample)) < 0.75 * len(strata)


def test_stratified_resample_todos_los_miembros_son_igual_de_probables():
    strata = ["x"] * 4 + ["y"] * 4
    rng = np.random.default_rng(0)
    draws = np.concatenate([stratified_resample(strata, rng) for _ in range(2000)])
    freq = np.bincount(draws, minlength=8) / draws.size
    np.testing.assert_allclose(freq, 1 / 8, atol=0.01)


def test_stratified_resample_acepta_un_arreglo_de_numpy():
    sample = stratified_resample(np.array([1, 1, 2, 2]), np.random.default_rng(0))
    assert set(sample[:2]) <= {0, 1} and set(sample[2:]) <= {2, 3}


# ---------------------------------------------------------------------------
# resample_pairs
# ---------------------------------------------------------------------------

def test_resample_pairs_con_arange_coincide_con_split_target_impostor():
    results, _ = make_results()
    for r in results.values():
        n = r["scores"].shape[1]
        target, impostor = resample_pairs(r["scores"], r["owner"], np.arange(n))
        exp_target, exp_impostor = split_target_impostor(r["scores"], r["owner"])
        np.testing.assert_array_equal(target, exp_target)
        np.testing.assert_array_equal(impostor, exp_impostor)


def test_resample_pairs_con_arange_y_owner_desordenado_da_los_mismos_conjuntos():
    owner = np.array([2, 0, 1, 0, 2, 1, 1])
    scores = np.random.default_rng(0).normal(size=(7, 3))
    target, impostor = resample_pairs(scores, owner, np.arange(3))
    exp_target, exp_impostor = split_target_impostor(scores, owner)
    np.testing.assert_array_equal(np.sort(target), np.sort(exp_target))
    np.testing.assert_array_equal(np.sort(impostor), np.sort(exp_impostor))


def test_resample_pairs_eer_con_arange_es_el_de_run_scenarios():
    results, _ = make_results()
    for r in results.values():
        eer, _ = compute_eer(*resample_pairs(r["scores"], r["owner"], np.arange(r["scores"].shape[1])))
        assert eer == r["eer"]


def test_resample_pairs_hablante_repetido_cuenta_a_mano():
    # 4 hablantes con 2 pruebas; la muestra repite al hablante 0.
    # Objetivo: 4 copias x 2 pruebas = 8.
    # Impostor: copias de 0 contra 2 y 3 (2 x 2 x 2 = 8) y copias de 2 y 3
    # contra los otros 3 registros (2 x 2 x 3 = 12) = 20.
    # Excluidos: las 2 copias de 0 entre sí (2 x 2 = 4 pares).
    owner = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    scores = np.random.default_rng(1).normal(size=(8, 4))
    target, impostor = resample_pairs(scores, owner, np.array([0, 0, 2, 3]))
    assert target.size == 8 and impostor.size == 20


def test_resample_pairs_coincide_con_fuerza_bruta():
    owner = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    scores = np.random.default_rng(2).normal(size=(8, 4))
    for sample in ([0, 0, 2, 3], [1, 1, 1, 3], [3, 2, 3, 2], [2, 2, 2, 2], [0, 1, 2, 3]):
        target, impostor = resample_pairs(scores, owner, np.array(sample))
        exp_target, exp_impostor = brute_force_pairs(scores, owner, sample)
        np.testing.assert_array_equal(np.sort(target), np.sort(exp_target))
        np.testing.assert_array_equal(np.sort(impostor), np.sort(exp_impostor))


def test_resample_pairs_fuerza_bruta_con_pruebas_desiguales_por_hablante():
    owner = np.array([0, 1, 1, 1, 2, 0])
    scores = np.random.default_rng(3).normal(size=(6, 3))
    sample = [1, 1, 2]
    target, impostor = resample_pairs(scores, owner, np.array(sample))
    exp_target, exp_impostor = brute_force_pairs(scores, owner, sample)
    assert target.size == 3 + 3 + 1
    np.testing.assert_array_equal(np.sort(target), np.sort(exp_target))
    np.testing.assert_array_equal(np.sort(impostor), np.sort(exp_impostor))


def test_resample_pairs_ningun_impostor_es_del_mismo_hablante_original():
    owner = np.repeat(np.arange(4), 2)
    scores = coded_scores(owner, 4)
    target, impostor = resample_pairs(scores, owner, np.array([0, 0, 2, 2]))
    trial_owner, enroll_speaker = decode(impostor)
    assert np.all(trial_owner != enroll_speaker)
    assert impostor.size == 2 * 2 * 2 * 2


def test_resample_pairs_objetivos_son_prueba_y_registro_del_mismo_hablante():
    owner = np.repeat(np.arange(4), 2)
    scores = coded_scores(owner, 4)
    target, _ = resample_pairs(scores, owner, np.array([0, 0, 2, 3]))
    trial_owner, enroll_speaker = decode(target)
    np.testing.assert_array_equal(trial_owner, enroll_speaker)
    # El hablante 0 aparece dos veces: sus 2 pruebas cuentan dos veces.
    assert Counter(trial_owner) == {0: 4, 2: 2, 3: 2}


def test_resample_pairs_impostores_repetidos_segun_las_copias():
    owner = np.repeat(np.arange(3), 2)
    scores = coded_scores(owner, 3)
    _, impostor = resample_pairs(scores, owner, np.array([0, 1, 1]))
    trial_owner, enroll_speaker = decode(impostor)
    pairs = Counter(zip(trial_owner, enroll_speaker))
    # Pruebas de 0 contra 2 copias del registro de 1 (2 x 2) y pruebas de
    # las 2 copias de 1 contra el registro de 0 (2 x 2 x 1).
    assert pairs == {(0, 1): 4, (1, 0): 4}


def test_resample_pairs_no_modifica_entradas():
    owner = np.repeat(np.arange(4), 2)
    scores = np.random.default_rng(4).normal(size=(8, 4))
    sample = np.array([3, 3, 0, 1])
    copies = scores.copy(), owner.copy(), sample.copy()
    resample_pairs(scores, owner, sample)
    for original, copy in zip((scores, owner, sample), copies):
        np.testing.assert_array_equal(original, copy)


# ---------------------------------------------------------------------------
# bootstrap_eers
# ---------------------------------------------------------------------------

def test_bootstrap_eers_un_arreglo_de_largo_n_boot_por_escenario():
    results, strata = make_results()
    eers = bootstrap_eers(results, strata, n_boot=7, seed=0)
    assert set(eers) == {"o-o", "o-a", "a-a"}
    assert all(e.shape == (7,) for e in eers.values())


def test_bootstrap_eers_valores_entre_cero_y_uno():
    results, strata = make_results()
    for e in bootstrap_eers(results, strata, n_boot=30, seed=0).values():
        assert np.all((e >= 0) & (e <= 1))


def test_bootstrap_eers_las_replicas_varian():
    results, strata = make_results()
    e = bootstrap_eers(results, strata, n_boot=30, seed=0)["a-a"]
    assert e.std() > 0


def test_bootstrap_eers_reproducible():
    results, strata = make_results()
    a = bootstrap_eers(results, strata, n_boot=10, seed=5)
    b = bootstrap_eers(results, strata, n_boot=10, seed=5)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])


def test_bootstrap_eers_cambia_con_otra_semilla():
    results, strata = make_results()
    a = bootstrap_eers(results, strata, n_boot=10, seed=5)["a-a"]
    b = bootstrap_eers(results, strata, n_boot=10, seed=6)["a-a"]
    assert not np.array_equal(a, b)


def test_bootstrap_eers_es_pareado():
    results, strata = make_results()
    same = {name: results["a-a"] for name in ("o-o", "o-a", "a-a")}
    eers = bootstrap_eers(same, strata, n_boot=20, seed=0)
    np.testing.assert_array_equal(eers["o-o"], eers["a-a"])
    np.testing.assert_array_equal(eers["o-a"], eers["a-a"])


def test_bootstrap_eers_coincide_con_el_calculo_replica_por_replica():
    results, strata = make_results()
    eers = bootstrap_eers(results, strata, n_boot=5, seed=11)
    rng = np.random.default_rng(11)
    for b in range(5):
        sample = stratified_resample(strata, rng)
        for name, r in results.items():
            expected, _ = compute_eer(*resample_pairs(r["scores"], r["owner"], sample))
            assert eers[name][b] == expected


def test_bootstrap_eers_imprime_el_avance(capsys):
    results, strata = make_results()
    bootstrap_eers(results, strata, n_boot=5, seed=0, progress_every=2)
    printed = capsys.readouterr().out
    assert "Réplica 2/5" in printed and "Réplica 4/5" in printed
    assert "Réplica 5/5" not in printed


def test_bootstrap_eers_sin_avance(capsys):
    results, strata = make_results()
    bootstrap_eers(results, strata, n_boot=3, seed=0, progress_every=0)
    assert capsys.readouterr().out == ""


def test_bootstrap_eers_strata_de_otro_largo_lanza():
    results, strata = make_results()
    with pytest.raises(ValueError, match="strata"):
        bootstrap_eers(results, strata[:-1], n_boot=2, seed=0)


# ---------------------------------------------------------------------------
# percentile_ci
# ---------------------------------------------------------------------------

def test_percentile_ci_al_95():
    assert percentile_ci(np.arange(101)) == pytest.approx((2.5, 97.5))


def test_percentile_ci_otros_niveles():
    assert percentile_ci(np.arange(101), level=0.90) == pytest.approx((5.0, 95.0))
    assert percentile_ci(np.arange(101), level=0.50) == pytest.approx((25.0, 75.0))


def test_percentile_ci_interpola_entre_valores():
    # 11 valores 0..10: el percentil 2,5 cae en 0,25 y el 97,5 en 9,75.
    assert percentile_ci(np.arange(11)) == pytest.approx((0.25, 9.75))


def test_percentile_ci_no_depende_del_orden():
    values = np.random.default_rng(0).permutation(np.arange(101))
    assert percentile_ci(values) == pytest.approx((2.5, 97.5))


def test_percentile_ci_low_menor_que_high_y_floats_de_python():
    low, high = percentile_ci(np.random.default_rng(1).normal(size=500))
    assert low < high
    assert type(low) is float and type(high) is float


def test_percentile_ci_constante():
    assert percentile_ci(np.full(50, 0.07)) == pytest.approx((0.07, 0.07))


@pytest.mark.parametrize("level", [0, 1, 95, -0.5])
def test_percentile_ci_nivel_invalido(level):
    with pytest.raises(ValueError):
        percentile_ci(np.arange(10), level=level)


# ---------------------------------------------------------------------------
# difference_cis
# ---------------------------------------------------------------------------

def test_differences_son_las_tres_pedidas():
    assert DIFFERENCES == [
        ("o-a menos a-a", "o-a", "a-a"),
        ("a-a menos o-o", "a-a", "o-o"),
        ("o-a menos o-o", "o-a", "o-o"),
    ]


def test_difference_cis_puntual_de_todos_los_datos_e_intervalo_pareado():
    point = {"o-o": 0.01, "o-a": 0.30, "a-a": 0.10}
    base = np.arange(101) / 1000
    replicates = {"o-o": base, "o-a": base + 0.3 + base, "a-a": base + 0.1}
    out = difference_cis(point, replicates)
    assert set(out) == {"o-a menos a-a", "a-a menos o-o", "o-a menos o-o"}
    assert out["o-a menos a-a"]["diff"] == pytest.approx(0.20)
    assert out["a-a menos o-o"]["diff"] == pytest.approx(0.09)
    assert out["o-a menos o-o"]["diff"] == pytest.approx(0.29)
    # o-a - a-a réplica por réplica = 0,2 + base: percentiles 2,5 y 97,5.
    assert (out["o-a menos a-a"]["ci_low"], out["o-a menos a-a"]["ci_high"]) == pytest.approx((0.2025, 0.2975))
    # a-a - o-o es constante 0,1 aunque cada escenario varíe: así se ve el pareo.
    assert (out["a-a menos o-o"]["ci_low"], out["a-a menos o-o"]["ci_high"]) == pytest.approx((0.1, 0.1))


def test_difference_cis_respeta_el_nivel():
    point = {"o-o": 0.0, "o-a": 0.0, "a-a": 0.0}
    replicates = {"o-o": np.zeros(101), "o-a": np.arange(101.0), "a-a": np.zeros(101)}
    out = difference_cis(point, replicates, level=0.9)
    assert (out["o-a menos o-o"]["ci_low"], out["o-a menos o-o"]["ci_high"]) == pytest.approx((5.0, 95.0))


# ---------------------------------------------------------------------------
# main (de punta a punta con archivos temporales)
# ---------------------------------------------------------------------------

def write_inputs(tmp_path, rows, session_ids, original, anonymized):
    npz = tmp_path / "emb.npz"
    np.savez(npz, session_ids=np.array(session_ids, dtype=str), original=original, anonymized=anonymized)
    manifest = tmp_path / "manifest.tsv"
    with open(manifest, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return npz, manifest


@pytest.fixture
def inputs(tmp_path):
    rows, session_ids, original, anonymized = make_case()
    npz, manifest = write_inputs(tmp_path, rows, session_ids, original, anonymized)
    return rows, session_ids, original, anonymized, npz, manifest


def run_main(monkeypatch, tmp_path, npz, manifest, *extra, n_boot="40"):
    output, replicas = tmp_path / "boot.json", tmp_path / "boot_replicas.npz"
    monkeypatch.setattr(sys, "argv", ["eer_bootstrap.py", "--embeddings", str(npz), "--manifest", str(manifest),
                                      "--output", str(output), "--replicates-output", str(replicas),
                                      "--n-boot", n_boot, *extra])
    main()
    with np.load(replicas) as data:
        reps = {key: data[key] for key in data.files}
    return json.loads(output.read_text(encoding="utf-8")), reps


def expected_replicates(rows, session_ids, original, anonymized, n_boot, boot_seed):
    splits, _ = split_enrollment_trial(rows)
    results = run_scenarios({"original": original, "anonymized": anonymized}, build_row_index(session_ids), splits)
    strata = [sp["stratum"] for sp in splits.values()]
    return bootstrap_eers(results, strata, n_boot, boot_seed, progress_every=0)


def test_parse_args_valores_por_defecto(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["eer_bootstrap.py", "--embeddings", "e.npz", "--manifest", "m.tsv",
                                      "--output", "o.json", "--replicates-output", "r.npz"])
    args = parse_args()
    assert (args.seed, args.n_enroll, args.n_trial) == (42, 3, 2)
    assert (args.n_boot, args.boot_seed, args.level) == (1000, 2026, 0.95)
    assert args.replicates_output == "r.npz"


def test_main_claves_del_json(monkeypatch, tmp_path, inputs):
    *_, npz, manifest = inputs
    out, _ = run_main(monkeypatch, tmp_path, npz, manifest)
    assert set(out) == {"embeddings", "manifest", "seed", "n_enroll", "n_trial", "n_boot", "boot_seed", "level",
                        "n_speakers", "speakers_per_stratum", "excluded", "scenarios", "differences"}
    assert out["embeddings"] == str(npz) and out["manifest"] == str(manifest)
    assert (out["seed"], out["n_enroll"], out["n_trial"]) == (42, 3, 2)
    assert (out["n_boot"], out["boot_seed"], out["level"]) == (40, 2026, 0.95)
    assert out["n_speakers"] == 12 and out["excluded"] == {}
    assert out["speakers_per_stratum"] == {"st0": 4, "st1": 4, "st2": 4}
    assert set(out["scenarios"]) == {"o-o", "o-a", "a-a"}
    for s in out["scenarios"].values():
        assert set(s) == {"eer", "ci_low", "ci_high", "n_target", "n_impostor"}
        assert (s["n_target"], s["n_impostor"]) == (24, 24 * 11)
    assert set(out["differences"]) == {"o-a menos a-a", "a-a menos o-o", "o-a menos o-o"}
    for d in out["differences"].values():
        assert set(d) == {"diff", "ci_low", "ci_high"}


def test_main_forma_del_npz(monkeypatch, tmp_path, inputs):
    *_, npz, manifest = inputs
    _, reps = run_main(monkeypatch, tmp_path, npz, manifest)
    assert set(reps) == {"o-o", "o-a", "a-a"}
    assert all(r.shape == (40,) for r in reps.values())


def test_main_eer_puntual_es_el_de_run_eer(monkeypatch, tmp_path, inputs):
    *_, npz, manifest = inputs
    out, reps = run_main(monkeypatch, tmp_path, npz, manifest)
    monkeypatch.setattr(sys, "argv", ["run_eer.py", "--embeddings", str(npz), "--manifest", str(manifest),
                                      "--output", str(tmp_path / "eer.json")])
    run_eer.main()
    reference = json.loads((tmp_path / "eer.json").read_text(encoding="utf-8"))
    for name, s in reference["scenarios"].items():
        assert out["scenarios"][name]["eer"] == s["eer"]
        assert out["scenarios"][name]["n_target"] == s["n_target"]
        assert out["scenarios"][name]["n_impostor"] == s["n_impostor"]
    # Con este caso el promedio de las réplicas no coincide con el puntual.
    assert reps["a-a"].mean() != pytest.approx(out["scenarios"]["a-a"]["eer"], abs=1e-9)


def test_main_diferencias_puntuales_salen_de_los_eer_de_todos_los_datos(monkeypatch, tmp_path, inputs):
    *_, npz, manifest = inputs
    out, _ = run_main(monkeypatch, tmp_path, npz, manifest)
    eer = {name: s["eer"] for name, s in out["scenarios"].items()}
    for label, a, b in DIFFERENCES:
        assert out["differences"][label]["diff"] == pytest.approx(eer[a] - eer[b], abs=1e-15)


def test_main_intervalos_salen_de_las_replicas_guardadas(monkeypatch, tmp_path, inputs):
    *_, npz, manifest = inputs
    out, reps = run_main(monkeypatch, tmp_path, npz, manifest)
    for name, s in out["scenarios"].items():
        assert (s["ci_low"], s["ci_high"]) == pytest.approx(percentile_ci(reps[name], 0.95))
        assert s["ci_low"] <= s["ci_high"]
    for label, a, b in DIFFERENCES:
        d = out["differences"][label]
        assert (d["ci_low"], d["ci_high"]) == pytest.approx(percentile_ci(reps[a] - reps[b], 0.95))


def test_main_replicas_usan_boot_seed_por_defecto(monkeypatch, tmp_path, inputs):
    rows, session_ids, original, anonymized, npz, manifest = inputs
    _, reps = run_main(monkeypatch, tmp_path, npz, manifest)
    expected = expected_replicates(rows, session_ids, original, anonymized, 40, 2026)
    for name in expected:
        np.testing.assert_array_equal(reps[name], expected[name])


def test_main_respeta_boot_seed(monkeypatch, tmp_path, inputs):
    rows, session_ids, original, anonymized, npz, manifest = inputs
    out, reps = run_main(monkeypatch, tmp_path, npz, manifest, "--boot-seed", "7")
    assert out["boot_seed"] == 7
    expected = expected_replicates(rows, session_ids, original, anonymized, 40, 7)
    for name in expected:
        np.testing.assert_array_equal(reps[name], expected[name])


def test_main_respeta_level(monkeypatch, tmp_path, inputs, capsys):
    *_, npz, manifest = inputs
    out, reps = run_main(monkeypatch, tmp_path, npz, manifest, "--level", "0.8")
    assert out["level"] == 0.8
    s = out["scenarios"]["a-a"]
    assert (s["ci_low"], s["ci_high"]) == pytest.approx(percentile_ci(reps["a-a"], 0.8))
    assert "IC 80 %" in capsys.readouterr().out


def test_main_respeta_n_enroll_n_trial_y_seed(monkeypatch, tmp_path, inputs):
    *_, npz, manifest = inputs
    out, _ = run_main(monkeypatch, tmp_path, npz, manifest, "--n-enroll", "2", "--n-trial", "3", "--seed", "7")
    assert (out["seed"], out["n_enroll"], out["n_trial"]) == (7, 2, 3)
    assert out["scenarios"]["o-o"]["n_target"] == 36


def test_main_informa_hablantes_excluidos(monkeypatch, tmp_path):
    rows, session_ids, original, anonymized = make_case()
    rows = [r for r in rows if r["session_id"] != "spk04_c0"]
    npz, manifest = write_inputs(tmp_path, rows, session_ids, original, anonymized)
    out, _ = run_main(monkeypatch, tmp_path, npz, manifest)
    assert out["excluded"] == {"spk04": 4}
    assert out["n_speakers"] == 11
    assert out["speakers_per_stratum"] == {"st0": 4, "st1": 3, "st2": 4}


def test_main_imprime_resumen(monkeypatch, tmp_path, inputs, capsys):
    *_, npz, manifest = inputs
    out, _ = run_main(monkeypatch, tmp_path, npz, manifest)
    printed = capsys.readouterr().out
    s = out["scenarios"]["a-a"]
    pct = lambda v: f"{100 * v:.2f}".replace(".", ",")
    assert f"a-a: EER = {pct(s['eer'])} % (IC 95 %: {pct(s['ci_low'])}–{pct(s['ci_high'])} %)" in printed
    for label in ("o-a menos a-a", "a-a menos o-o", "o-a menos o-o"):
        assert f"{label}:" in printed
    assert "boot.json" in printed and "boot_replicas.npz" in printed
