import csv
import json
import sys

import numpy as np
import pytest

from eer_bootstrap import percentile_ci, resample_pairs, stratified_resample
from eer_protocol import split_enrollment_trial
from eer_scoring import compute_eer, split_target_impostor
from impersonation import (
    POINTS,
    RATES,
    acceptance_rate,
    bootstrap_rates,
    eer_threshold,
    far_threshold,
    main,
    operating_point_rates,
    parse_args,
    point_rates,
)
from run_eer import build_row_index, load_and_score, run_scenarios

DIM = 8


# ---------------------------------------------------------------------------
# Datos sintéticos
# ---------------------------------------------------------------------------

def make_case(n_speakers=12, n_strata=3, seed=0):
    """Manifiesto y embeddings: o-o separa bien, la anonimización mezcla."""
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


def make_results(seed=0):
    rows, session_ids, original, anonymized = make_case(seed=seed)
    splits, _ = split_enrollment_trial(rows)
    results = run_scenarios({"original": original, "anonymized": anonymized}, build_row_index(session_ids), splits)
    return results, [sp["stratum"] for sp in splits.values()]


def brute_far_threshold(target, impostor, max_far):
    """Recorre los umbrales observados de menor a mayor y cuenta el FAR uno por uno."""
    for t in sorted(set(target) | set(impostor)):
        far = sum(1 for s in impostor if s >= t) / len(impostor)
        if far <= max_far:
            return t
    return None


# ---------------------------------------------------------------------------
# eer_threshold
# ---------------------------------------------------------------------------

def test_eer_threshold_es_el_de_compute_eer():
    rng = np.random.default_rng(0)
    for _ in range(20):
        target, impostor = rng.normal(2, 1, 50), rng.normal(0, 1, 400)
        assert eer_threshold(target, impostor) == compute_eer(target, impostor)[1]


def test_eer_threshold_con_empates():
    target, impostor = [0.5, 0.5, 0.7, 0.9], [0.1, 0.5, 0.5, 0.6]
    assert eer_threshold(target, impostor) == compute_eer(target, impostor)[1]


def test_eer_threshold_devuelve_float_de_python():
    assert type(eer_threshold([0.9, 0.8], [0.1, 0.2])) is float


# ---------------------------------------------------------------------------
# far_threshold
# ---------------------------------------------------------------------------

def test_far_threshold_coincide_con_fuerza_bruta():
    rng = np.random.default_rng(1)
    for _ in range(30):
        target = np.round(rng.normal(2, 1, 40), 1)  # redondeado: hay empates
        impostor = np.round(rng.normal(0, 1, 200), 1)
        for max_far in (0.0, 0.01, 0.05, 0.1, 0.5, 1.0):
            expected = brute_far_threshold(target.tolist(), impostor.tolist(), max_far)
            if expected is None:
                with pytest.raises(ValueError):
                    far_threshold(target, impostor, max_far)
            else:
                assert far_threshold(target, impostor, max_far) == expected


def test_far_threshold_empates_en_el_umbral():
    # Umbral 0,5: aceptaría 3 de 4 impostores (los dos 0,5 cuentan, >=).
    # Umbral 0,7: acepta 1 de 4 (el 0,9).
    target, impostor = [0.5, 0.7], [0.1, 0.5, 0.5, 0.9]
    assert far_threshold(target, impostor, 0.25) == 0.7
    assert far_threshold(target, impostor, 0.5) == 0.7
    assert far_threshold(target, impostor, 0.75) == 0.5
    assert far_threshold(target, impostor, 1.0) == 0.1


def test_far_threshold_usa_max_far():
    target, impostor = [0.5, 0.7], [0.1, 0.5, 0.5, 0.9]
    assert far_threshold(target, impostor, 0.25) != far_threshold(target, impostor, 0.75)


def test_far_threshold_es_el_menor_que_cumple():
    rng = np.random.default_rng(2)
    target, impostor = rng.normal(2, 1, 100), rng.normal(0, 1, 1000)
    t = far_threshold(target, impostor, 0.01)
    assert np.mean(impostor >= t) <= 0.01
    candidates = np.unique(np.concatenate([target, impostor]))
    previous = candidates[candidates < t].max()
    assert np.mean(impostor >= previous) > 0.01


def test_far_threshold_es_una_puntuacion_observada():
    rng = np.random.default_rng(3)
    target, impostor = rng.normal(2, 1, 30), rng.normal(0, 1, 300)
    assert far_threshold(target, impostor, 0.02) in set(target) | set(impostor)


def test_far_threshold_cero_con_impostor_arriba_lanza():
    with pytest.raises(ValueError, match="ningún umbral"):
        far_threshold([0.5, 0.6], [0.1, 0.9], 0.0)


def test_far_threshold_cero_con_objetivo_arriba():
    assert far_threshold([0.5, 0.95], [0.1, 0.9], 0.0) == 0.95


@pytest.mark.parametrize("max_far", [-0.1, 1.5])
def test_far_threshold_max_far_invalido(max_far):
    with pytest.raises(ValueError):
        far_threshold([0.5], [0.1], max_far)


def test_far_threshold_conjuntos_vacios():
    with pytest.raises(ValueError):
        far_threshold([], [0.1], 0.1)
    with pytest.raises(ValueError):
        far_threshold([0.5], [], 0.1)


# ---------------------------------------------------------------------------
# acceptance_rate
# ---------------------------------------------------------------------------

def test_acceptance_rate_empate_con_el_umbral_se_acepta():
    assert acceptance_rate([0.5, 0.5, 0.4], 0.5) == pytest.approx(2 / 3)


def test_acceptance_rate_extremos():
    assert acceptance_rate([0.1, 0.2], 0.3) == 0.0
    assert acceptance_rate([0.3, 0.4], 0.3) == 1.0


def test_acceptance_rate_vacio_lanza():
    with pytest.raises(ValueError):
        acceptance_rate([], 0.5)


# ---------------------------------------------------------------------------
# operating_point_rates y point_rates
# ---------------------------------------------------------------------------

def test_operating_point_rates_umbral_solo_depende_de_o_o():
    rng = np.random.default_rng(4)
    target_oo, impostor_oo = rng.normal(2, 1, 60), rng.normal(0, 1, 600)
    a = operating_point_rates(target_oo, impostor_oo, rng.normal(1, 1, 60), 0.01)
    b = operating_point_rates(target_oo, impostor_oo, rng.normal(-3, 1, 60), 0.01)
    for point in POINTS:
        assert a[point]["threshold"] == b[point]["threshold"]
        assert a[point]["genuine_rate"] == b[point]["genuine_rate"]


def test_operating_point_rates_tasas_con_el_umbral_de_o_o():
    rng = np.random.default_rng(5)
    target_oo, impostor_oo, target_oa = rng.normal(2, 1, 60), rng.normal(0, 1, 600), rng.normal(1, 1, 60)
    rates = operating_point_rates(target_oo, impostor_oo, target_oa, 0.01)
    thresholds = {"eer": compute_eer(target_oo, impostor_oo)[1], "far": far_threshold(target_oo, impostor_oo, 0.01)}
    for point, t in thresholds.items():
        assert rates[point]["threshold"] == t
        assert rates[point]["genuine_rate"] == np.mean(target_oo >= t)
        assert rates[point]["impersonation_rate"] == np.mean(target_oa >= t)
        assert rates[point]["far_oo"] == np.mean(impostor_oo >= t)


def test_operating_point_rates_far_respeta_max_far():
    rng = np.random.default_rng(6)
    target_oo, impostor_oo = rng.normal(2, 1, 60), rng.normal(0, 1, 600)
    for max_far in (0.01, 0.05, 0.2):
        assert operating_point_rates(target_oo, impostor_oo, target_oo, max_far)["far"]["far_oo"] <= max_far


def test_point_rates_usa_todos_los_pares_de_o_o_y_o_a():
    results, _ = make_results()
    target_oo, impostor_oo = split_target_impostor(results["o-o"]["scores"], results["o-o"]["owner"])
    target_oa, _ = split_target_impostor(results["o-a"]["scores"], results["o-a"]["owner"])
    assert point_rates(results, 0.01) == operating_point_rates(target_oo, impostor_oo, target_oa, 0.01)


def test_point_rates_umbral_eer_coincide_con_run_scenarios():
    results, _ = make_results()
    assert point_rates(results, 0.01)["eer"]["threshold"] == results["o-o"]["threshold"]


def test_point_rates_umbral_no_sale_de_o_a():
    results, _ = make_results()
    swapped = {"o-o": results["o-a"], "o-a": results["o-o"]}
    assert point_rates(results, 0.01)["eer"]["threshold"] != point_rates(swapped, 0.01)["eer"]["threshold"]


# ---------------------------------------------------------------------------
# bootstrap_rates
# ---------------------------------------------------------------------------

def test_bootstrap_rates_forma_y_rango():
    results, strata = make_results()
    out = bootstrap_rates(results, strata, 15, seed=0, max_far=0.05, progress_every=0)
    assert set(out) == {"eer", "far"}
    for point in out.values():
        assert set(point) == {"threshold", "genuine_rate", "impersonation_rate"}
        for key in RATES:
            assert point[key].shape == (15,) and np.all((point[key] >= 0) & (point[key] <= 1))


def test_bootstrap_rates_pareado_o_a_igual_a_o_o():
    results, strata = make_results()
    same = {"o-o": results["o-o"], "o-a": results["o-o"]}
    out = bootstrap_rates(same, strata, 20, seed=0, max_far=0.05, progress_every=0)
    for point in POINTS:
        np.testing.assert_array_equal(out[point]["impersonation_rate"], out[point]["genuine_rate"])


def test_bootstrap_rates_coincide_replica_por_replica():
    results, strata = make_results()
    out = bootstrap_rates(results, strata, 6, seed=9, max_far=0.05, progress_every=0)
    rng = np.random.default_rng(9)
    for b in range(6):
        sample = stratified_resample(strata, rng)
        target_oo, impostor_oo = resample_pairs(results["o-o"]["scores"], results["o-o"]["owner"], sample)
        target_oa, _ = resample_pairs(results["o-a"]["scores"], results["o-a"]["owner"], sample)
        expected = operating_point_rates(target_oo, impostor_oo, target_oa, 0.05)
        for point in POINTS:
            for key in out[point]:
                assert out[point][key][b] == expected[point][key]


def test_bootstrap_rates_umbral_se_reajusta_en_cada_replica():
    results, strata = make_results()
    out = bootstrap_rates(results, strata, 20, seed=0, max_far=0.05, progress_every=0)
    for point in POINTS:
        assert len(np.unique(out[point]["threshold"])) > 1


def test_bootstrap_rates_reproducible_y_depende_de_la_semilla():
    results, strata = make_results()
    a = bootstrap_rates(results, strata, 10, seed=3, max_far=0.05, progress_every=0)
    b = bootstrap_rates(results, strata, 10, seed=3, max_far=0.05, progress_every=0)
    c = bootstrap_rates(results, strata, 10, seed=4, max_far=0.05, progress_every=0)
    for point in POINTS:
        for key in a[point]:
            np.testing.assert_array_equal(a[point][key], b[point][key])
    assert not np.array_equal(a["eer"]["impersonation_rate"], c["eer"]["impersonation_rate"])


def test_bootstrap_rates_respeta_max_far():
    results, strata = make_results()
    a = bootstrap_rates(results, strata, 10, seed=0, max_far=0.01, progress_every=0)
    b = bootstrap_rates(results, strata, 10, seed=0, max_far=0.3, progress_every=0)
    assert not np.array_equal(a["far"]["threshold"], b["far"]["threshold"])
    np.testing.assert_array_equal(a["eer"]["threshold"], b["eer"]["threshold"])


def test_bootstrap_rates_imprime_el_avance(capsys):
    results, strata = make_results()
    bootstrap_rates(results, strata, 5, seed=0, max_far=0.05, progress_every=2)
    printed = capsys.readouterr().out
    assert "Réplica 2/5" in printed and "Réplica 4/5" in printed and "Réplica 5/5" not in printed


def test_bootstrap_rates_strata_de_otro_largo_lanza():
    results, strata = make_results()
    with pytest.raises(ValueError, match="strata"):
        bootstrap_rates(results, strata[:-1], 2, seed=0, max_far=0.05)


# ---------------------------------------------------------------------------
# main (de punta a punta con archivos temporales)
# ---------------------------------------------------------------------------

@pytest.fixture
def inputs(tmp_path):
    rows, session_ids, original, anonymized = make_case()
    npz = tmp_path / "emb.npz"
    np.savez(npz, session_ids=np.array(session_ids, dtype=str), original=original, anonymized=anonymized)
    manifest = tmp_path / "manifest.tsv"
    with open(manifest, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return npz, manifest


def run_main(monkeypatch, tmp_path, npz, manifest, *extra):
    output = tmp_path / "imp.json"
    monkeypatch.setattr(sys, "argv", ["impersonation.py", "--embeddings", str(npz), "--manifest", str(manifest),
                                      "--output", str(output), "--n-boot", "30", *extra])
    main()
    return json.loads(output.read_text(encoding="utf-8"))


def expected_from(npz, manifest, max_far=0.01, boot_seed=2026, n_boot=30):
    splits, _, results = load_and_score(npz, manifest, 42, 3, 2)
    strata = [sp["stratum"] for sp in splits.values()]
    return point_rates(results, max_far), bootstrap_rates(results, strata, n_boot, boot_seed, max_far, 0)


def test_parse_args_valores_por_defecto(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["impersonation.py", "--embeddings", "e.npz", "--manifest", "m.tsv",
                                      "--output", "o.json"])
    args = parse_args()
    assert (args.seed, args.n_enroll, args.n_trial) == (42, 3, 2)
    assert (args.n_boot, args.boot_seed, args.level, args.max_far) == (1000, 2026, 0.95, 0.01)


def test_main_claves_del_json(monkeypatch, tmp_path, inputs):
    out = run_main(monkeypatch, tmp_path, *inputs)
    assert {"embeddings", "manifest", "seed", "n_enroll", "n_trial", "n_boot", "boot_seed", "level", "max_far",
            "n_speakers", "excluded", "n_target_oo", "n_impostor_oo", "n_target_oa", "operating_points"} == set(out)
    assert (out["n_boot"], out["boot_seed"], out["level"], out["max_far"]) == (30, 2026, 0.95, 0.01)
    assert (out["n_speakers"], out["n_target_oo"], out["n_impostor_oo"], out["n_target_oa"]) == (12, 24, 264, 24)
    assert set(out["operating_points"]) == {"eer", "far"}
    assert out["operating_points"]["far"]["max_far"] == 0.01
    for p in out["operating_points"].values():
        assert {"threshold", "far_oo", "genuine_rate", "impersonation_rate"} <= set(p)
        for rate in RATES:
            assert set(p[rate]) == {"value", "ci_low", "ci_high"}
            assert p[rate]["ci_low"] <= p[rate]["ci_high"]


def test_main_valor_puntual_de_todos_los_datos(monkeypatch, tmp_path, inputs):
    out = run_main(monkeypatch, tmp_path, *inputs)
    point, replicates = expected_from(*inputs)
    for name in POINTS:
        p = out["operating_points"][name]
        assert p["threshold"] == point[name]["threshold"]
        assert p["far_oo"] == point[name]["far_oo"]
        for rate in RATES:
            assert p[rate]["value"] == point[name][rate]
    # En este caso el promedio de las réplicas difiere del valor puntual.
    assert replicates["eer"]["impersonation_rate"].mean() != pytest.approx(
        out["operating_points"]["eer"]["impersonation_rate"]["value"], abs=1e-9)


def test_main_intervalos_salen_del_bootstrap(monkeypatch, tmp_path, inputs):
    out = run_main(monkeypatch, tmp_path, *inputs)
    _, replicates = expected_from(*inputs)
    for name in POINTS:
        for rate in RATES:
            p = out["operating_points"][name][rate]
            assert (p["ci_low"], p["ci_high"]) == pytest.approx(percentile_ci(replicates[name][rate], 0.95))


def test_main_respeta_boot_seed_level_y_max_far(monkeypatch, tmp_path, inputs):
    out = run_main(monkeypatch, tmp_path, *inputs, "--boot-seed", "7", "--level", "0.8", "--max-far", "0.2")
    point, replicates = expected_from(*inputs, max_far=0.2, boot_seed=7)
    assert (out["boot_seed"], out["level"], out["max_far"]) == (7, 0.8, 0.2)
    far = out["operating_points"]["far"]
    assert far["threshold"] == point["far"]["threshold"] and far["max_far"] == 0.2
    for rate in RATES:
        assert (far[rate]["ci_low"], far[rate]["ci_high"]) == pytest.approx(
            percentile_ci(replicates["far"][rate], 0.8))


def test_main_imprime_resumen_con_coma_decimal(monkeypatch, tmp_path, inputs, capsys):
    out = run_main(monkeypatch, tmp_path, *inputs)
    printed = capsys.readouterr().out
    pct = lambda v: f"{100 * v:.2f}".replace(".", ",")
    r = out["operating_points"]["eer"]["impersonation_rate"]
    assert (f"grabación anonimizada (o-a): {pct(r['value'])} % aceptada "
            f"(IC 95 %: {pct(r['ci_low'])}–{pct(r['ci_high'])} %)") in printed
    assert "Umbral del EER de o-o" in printed and "Umbral con FAR <= 1 % en o-o" in printed
    assert "persona real (o-o)" in printed and "imp.json" in printed
