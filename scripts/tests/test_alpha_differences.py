"""Pruebas de alpha_differences con datos de juguete (los JSON salen de los main reales, con pocas réplicas)."""
import csv
import json
import sys

import numpy as np
import pytest

import alpha_differences as ad
import eer_bootstrap
import group_analysis as ga
import impersonation
import wer_bootstrap

DIM = 8
N_BOOT = 20


# ---------------------------------------------------------------------------
# Funciones puras
# ---------------------------------------------------------------------------

def cfg(tag, alpha=None, low=None, high=None):
    fixed = alpha is not None
    return {"tag": tag, "fixed": fixed, "alpha": alpha, "alpha_min": alpha if fixed else low,
            "alpha_max": alpha if fixed else high}


def test_neighbor_pairs_por_alpha_real_y_sin_rangos():
    configs = [cfg("a075", 0.75), cfg("u07_09", low=0.7, high=0.9), cfg("a070", 0.7), cfg("a072", 0.725)]
    assert [(a["tag"], b["tag"]) for a, b in ad.neighbor_pairs(configs)] == [("a070", "a072"), ("a072", "a075")]


def test_neighbor_pairs_menos_de_dos():
    assert ad.neighbor_pairs([cfg("a070", 0.7)]) == []


def metrics(points, reps):
    return {m: {"point": p, "reps": np.asarray(r, float)} for m, p, r in zip(ad.METRICS, points, reps)}


def test_paired_difference_configuraciones_identicas_es_cero():
    m = metrics([0.3, 0.2, 0.1], [np.linspace(0, 1, 50)] * 3)
    assert ad.paired_difference(m, m, 0.95) == {k: (0.0, 0.0, 0.0) for k in ad.METRICS}


def test_paired_difference_destino_menos_origen_y_pareada():
    base = np.arange(101) / 1000
    a = metrics([0.10, 0.20, 0.30], [base, base, base])
    b = metrics([0.15, 0.10, 0.30], [base + 0.05, base * 2, base + 0.0])
    out = ad.paired_difference(a, b, 0.95)
    assert out["eer_oa"][0] == pytest.approx(0.05)
    assert out["eer_oa"][1:] == pytest.approx((0.05, 0.05))  # pareada: la resta réplica a réplica es constante
    assert out["impersonation_far"][0] == pytest.approx(-0.10)
    assert out["impersonation_far"][1:] == pytest.approx((0.0025, 0.0975))
    assert out["wer_anonymized"] == pytest.approx((0.0, 0.0, 0.0))


def test_paired_difference_respeta_el_nivel():
    a = metrics([0, 0, 0], [np.zeros(101)] * 3)
    b = metrics([0, 0, 0], [np.arange(101.0)] * 3)
    assert ad.paired_difference(a, b, 0.9)["eer_oa"][1:] == pytest.approx((5.0, 95.0))


# ---------------------------------------------------------------------------
# Datos de juguete: dos configuraciones de α fijo completas y una incompleta
# ---------------------------------------------------------------------------

def make_case(n_speakers=12, seed=0):
    rng = np.random.default_rng(seed)
    rows, session_ids, original, centers = [], [], [], []
    for s in range(n_speakers):
        center = rng.normal(size=DIM)
        centers.append(center)
        for c in range(5):
            sid = f"cv_{s:02d}_{c}"
            session_ids.append(sid)
            original.append(center + 0.6 * rng.normal(size=DIM))
            rows.append({"client_id": f"spk{s:02d}", "stratum": ["caribe", "mexico", "sin_dato"][s % 3],
                         "path": f"{sid}.mp3", "session_id": sid, "status": "ok", "error": ""})
    return rows, session_ids, np.array(original, np.float32)


def write_tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(monkeypatch, module, *argv):
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv])
    module.main()


def build_config(monkeypatch, data, tag, alpha, noise, rows, session_ids, original, steps=("eer", "imp", "wer")):
    """Manifiesto, embeddings y transcripciones de una configuración y, según `steps`, sus JSON."""
    f = ga.config_files(tag, data)
    write_tsv(f["manifest"], [{**r, "coefficient": str(alpha)} for r in rows])
    rng = np.random.default_rng(int(alpha * 1000))
    anon = (original + noise * rng.normal(size=original.shape)).astype(np.float32)
    np.savez(f["embeddings"], session_ids=np.array(session_ids, dtype=str), original=original, anonymized=anon)
    words = "el gato come pescado en la casa de mi abuela".split()
    transcripts = []
    clients = sorted({r["client_id"] for r in rows})
    for i, cid in enumerate(clients):
        stratum = next(r["stratum"] for r in rows if r["client_id"] == cid)
        ref = np.random.default_rng(i).choice(words, size=6).tolist()
        for version, p in (("original", 0.1), ("anonymized", noise / 5)):
            hyp = [w if np.random.default_rng(i * 7 + len(version)).random() > p else "x" for w in ref]
            transcripts.append({"session_id": f"t{i}", "client_id": cid, "stratum": stratum, "version": version,
                                "reference": " ".join(ref), "hypothesis": " ".join(hyp), "seconds": "1"})
    write_tsv(f["transcripts"], transcripts)
    common = ["--n-boot", str(N_BOOT)]
    if "eer" in steps:
        run(monkeypatch, eer_bootstrap, "--embeddings", str(f["embeddings"]), "--manifest", str(f["manifest"]),
            "--output", str(f["eer_json"]), "--replicates-output", str(f["eer_replicates"]), *common)
    if "imp" in steps:
        run(monkeypatch, impersonation, "--embeddings", str(f["embeddings"]), "--manifest", str(f["manifest"]),
            "--output", str(f["impersonation_json"]), "--max-far", "0.05", *common)
    if "wer" in steps:
        run(monkeypatch, wer_bootstrap, "--transcripts", str(f["transcripts"]), "--output", str(f["wer_json"]), *common)
    return f


@pytest.fixture
def toy(tmp_path, monkeypatch):
    rows, session_ids, original = make_case()
    files = {
        "a050": build_config(monkeypatch, tmp_path, "a050", 0.5, 3.0, rows, session_ids, original),
        "a072": build_config(monkeypatch, tmp_path, "a072", 0.725, 1.5, rows, session_ids, original),
        "a090": build_config(monkeypatch, tmp_path, "a090", 0.9, 0.5, rows, session_ids, original, steps=("eer",)),
    }
    return tmp_path, files


def run_differences(monkeypatch, data, *extra):
    output = data / "dif.tsv"
    run(monkeypatch, ad, "--data-dir", str(data), "--output", str(output), "--n-boot", str(N_BOOT),
        "--max-far", "0.05", *extra)
    with open(output, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_main_salta_incompletas_con_aviso_y_usa_el_alpha_real(monkeypatch, toy, capsys):
    data, _ = toy
    (row,) = run_differences(monkeypatch, data)
    out = capsys.readouterr().out
    assert "Aviso: se salta a090" in out and "impersonation_json" in out and "wer_json" in out
    assert (row["config_from"], row["config_to"], row["alpha_from"], row["alpha_to"]) == ("a050", "a072", "0.5", "0.725")
    assert list(row) == ad.COLUMNS


def test_main_diferencia_puntual_es_la_resta_de_los_json(monkeypatch, toy):
    data, files = toy
    (row,) = run_differences(monkeypatch, data)
    configs = {c["tag"]: c for c in ga.discover_configs(data)}
    a, b = ad.json_points(configs["a050"]), ad.json_points(configs["a072"])
    for m in ad.METRICS:
        assert float(row[f"{m}_difference"]) == b[m] - a[m]
        assert float(row[f"{m}_ci_low"]) <= float(row[f"{m}_ci_high"])


def test_main_ic_pareado_con_las_replicas_de_los_scripts(monkeypatch, toy):
    data, files = toy
    (row,) = run_differences(monkeypatch, data)
    with np.load(files["a050"]["eer_replicates"]) as d0, np.load(files["a072"]["eer_replicates"]) as d1:
        low, high = eer_bootstrap.percentile_ci(d1["o-a"] - d0["o-a"])
    assert (float(row["eer_oa_ci_low"]), float(row["eer_oa_ci_high"])) == pytest.approx((low, high), abs=1e-15)


def test_main_dos_configuraciones_identicas_dan_cero(monkeypatch, tmp_path):
    rows, session_ids, original = make_case()
    files = build_config(monkeypatch, tmp_path, "a050", 0.5, 3.0, rows, session_ids, original)
    configs = ga.discover_configs(tmp_path)
    splits, _, results = ad.load_and_score(files["embeddings"], files["manifest"], 42, 3, 2)
    samples = ga.shared_samples([s["stratum"] for s in splits.values()], N_BOOT, 2026)
    reps = [ga.operating_thresholds(results["o-o"], 0.05, s) for s in samples]
    params = {"seed": 42, "n_enroll": 3, "n_trial": 2, "n_boot": N_BOOT, "boot_seed": 2026}
    m = ad.config_metrics(configs[0], results, samples, reps, list(splits), 0.05, params)
    assert ad.paired_difference(m, m, 0.95) == {k: (0.0, 0.0, 0.0) for k in ad.METRICS}
    ad.check_against_json(configs[0], m)


def test_impersonation_y_wer_reproducen_los_json(monkeypatch, toy):
    data, files = toy
    config = {c["tag"]: c for c in ga.discover_configs(data)}["a072"]
    splits, _, results = ad.load_and_score(files["a072"]["embeddings"], files["a072"]["manifest"], 42, 3, 2)
    samples = ga.shared_samples([s["stratum"] for s in splits.values()], N_BOOT, 2026)
    reps = [ga.operating_thresholds(results["o-o"], 0.05, s) for s in samples]
    params = {"seed": 42, "n_enroll": 3, "n_trial": 2, "n_boot": N_BOOT, "boot_seed": 2026}
    m = ad.config_metrics(config, results, samples, reps, list(splits), 0.05, params)
    imp = json.loads(files["a072"]["impersonation_json"].read_text(encoding="utf-8"))["operating_points"]["far"]
    wer = json.loads(files["a072"]["wer_json"].read_text(encoding="utf-8"))["wer"]["anonymized"]
    assert eer_bootstrap.percentile_ci(m["impersonation_far"]["reps"]) == pytest.approx(
        (imp["impersonation_rate"]["ci_low"], imp["impersonation_rate"]["ci_high"]), abs=1e-15)
    assert eer_bootstrap.percentile_ci(m["wer_anonymized"]["reps"]) == pytest.approx(
        (wer["ci_low"], wer["ci_high"]), abs=1e-15)


def test_eer_oa_replicates_reutiliza_las_guardadas(monkeypatch, toy):
    data, files = toy
    config = {c["tag"]: c for c in ga.discover_configs(data)}["a050"]
    splits, _, results = ad.load_and_score(files["a050"]["embeddings"], files["a050"]["manifest"], 42, 3, 2)
    samples = ga.shared_samples([s["stratum"] for s in splits.values()], N_BOOT, 2026)
    params = {"seed": 42, "n_enroll": 3, "n_trial": 2, "n_boot": N_BOOT, "boot_seed": 2026}
    calls = []
    real = ad.compute_eer
    monkeypatch.setattr(ad, "compute_eer", lambda t, i: calls.append(1) or real(t, i))
    reps = ad.eer_oa_replicates(config, results, samples, params)
    assert len(calls) == 2  # solo la primera y la última, para comprobar
    with np.load(files["a050"]["eer_replicates"]) as d:
        np.testing.assert_array_equal(reps, d["o-a"])


def test_eer_oa_replicates_recalcula_si_no_coinciden(monkeypatch, toy):
    data, files = toy
    config = {c["tag"]: c for c in ga.discover_configs(data)}["a050"]
    with np.load(files["a050"]["eer_replicates"]) as d:
        stored = {k: np.array(d[k]) for k in d.files}
    true_reps = stored["o-a"].copy()
    stored["o-a"][0] += 0.5  # réplica guardada corrupta
    np.savez(files["a050"]["eer_replicates"], **stored)
    splits, _, results = ad.load_and_score(files["a050"]["embeddings"], files["a050"]["manifest"], 42, 3, 2)
    samples = ga.shared_samples([s["stratum"] for s in splits.values()], N_BOOT, 2026)
    params = {"seed": 42, "n_enroll": 3, "n_trial": 2, "n_boot": N_BOOT, "boot_seed": 2026}
    np.testing.assert_array_equal(ad.eer_oa_replicates(config, results, samples, params), true_reps)
    # Parámetros distintos: se recalculan todas con las muestras dadas, no se usan las guardadas.
    other_samples = ga.shared_samples([s["stratum"] for s in splits.values()], N_BOOT, 7)
    expected = [ad.compute_eer(*ad.resample_pairs(results["o-a"]["scores"], results["o-a"]["owner"], s))[0]
                for s in other_samples]
    np.testing.assert_array_equal(ad.eer_oa_replicates(config, results, other_samples, {**params, "boot_seed": 7}),
                                  expected)


def test_check_against_json_detecta_un_valor_distinto(monkeypatch, toy):
    data, _ = toy
    config = {c["tag"]: c for c in ga.discover_configs(data)}["a050"]
    points = ad.json_points(config)
    good = {m: {"point": points[m], "reps": None} for m in ad.METRICS}
    ad.check_against_json(config, good)
    bad = {**good, "wer_anonymized": {"point": points["wer_anonymized"] + 1e-12, "reps": None}}
    with pytest.raises(ValueError, match="wer_anonymized"):
        ad.check_against_json(config, bad)


def test_main_sin_pares_error_claro(monkeypatch, toy):
    data, _ = toy
    with pytest.raises(SystemExit, match="al menos dos"):
        run_differences(monkeypatch, data, "--tags", "a050")


def test_main_ic_pareado_de_suplantacion_con_umbral_por_replica(monkeypatch, toy):
    data, files = toy
    (row,) = run_differences(monkeypatch, data)
    reps = {}
    for tag in ("a050", "a072"):
        splits, _, results = ad.load_and_score(files[tag]["embeddings"], files[tag]["manifest"], 42, 3, 2)
        strata = [s["stratum"] for s in splits.values()]
        reps[tag] = impersonation.bootstrap_rates(results, strata, N_BOOT, 2026, 0.05, progress_every=0)
    diff = reps["a072"]["far"]["impersonation_rate"] - reps["a050"]["far"]["impersonation_rate"]
    assert (float(row["impersonation_far_ci_low"]), float(row["impersonation_far_ci_high"])) == pytest.approx(
        eer_bootstrap.percentile_ci(diff), abs=1e-15)


def test_main_falla_si_un_valor_no_coincide_con_su_json(monkeypatch, toy):
    data, files = toy
    path = files["a072"]["impersonation_json"]
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["operating_points"]["far"]["impersonation_rate"]["value"] += 0.01
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="a072: el valor puntual de impersonation_far"):
        run_differences(monkeypatch, data)


def test_eer_oa_replicates_con_otros_parametros_no_usa_las_guardadas(monkeypatch, toy):
    # Caso límite: la primera y la última muestra coinciden con las guardadas, pero las del medio no.
    # Solo la comprobación de parámetros evita devolver réplicas que no corresponden.
    data, files = toy
    config = {c["tag"]: c for c in ga.discover_configs(data)}["a050"]
    splits, _, results = ad.load_and_score(files["a050"]["embeddings"], files["a050"]["manifest"], 42, 3, 2)
    strata = [s["stratum"] for s in splits.values()]
    stored_samples = ga.shared_samples(strata, N_BOOT, 2026)
    mixed = [stored_samples[0], *ga.shared_samples(strata, N_BOOT, 7)[1:-1], stored_samples[-1]]
    params = {"seed": 42, "n_enroll": 3, "n_trial": 2, "n_boot": N_BOOT, "boot_seed": 7}
    expected = [ad.compute_eer(*ad.resample_pairs(results["o-a"]["scores"], results["o-a"]["owner"], s))[0]
                for s in mixed]
    np.testing.assert_array_equal(ad.eer_oa_replicates(config, results, mixed, params), expected)
