"""Pruebas del barrido de α: run_alpha_sweep (plan, reanudación, dry-run) y plot_alpha_sweep (tabla y figura)."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")

import plot_alpha_sweep  # noqa: E402
import run_alpha_sweep  # noqa: E402
from plot_alpha_sweep import SUMMARY_COLUMNS, collect_rows, plot_alpha_sweep as draw_sweep, write_summary  # noqa: E402
from plot_results import ANONYMIZED_COLOR, ORIGINAL_COLOR  # noqa: E402
from run_alpha_sweep import (  # noqa: E402
    DEFAULT_ALPHAS,
    STEPS,
    build_configs,
    config_paths,
    format_seconds,
    parse_args,
    run_sweep,
    step_done,
    step_plan,
)


def settings_for(tmp_path, *extra):
    return parse_args(["--data-dir", str(tmp_path), "--python", "PY", *extra])


# ---------------------------------------------------------------------------
# build_configs y config_paths
# ---------------------------------------------------------------------------

def test_build_configs_por_defecto_son_los_nueve_alphas(tmp_path):
    args = settings_for(tmp_path)
    configs = build_configs(args.alphas, args.ranges)
    assert [c["tag"] for c in configs] == ["a020", "a030", "a040", "a050", "a060", "a070", "a080", "a090", "a100"]
    assert all(c["alpha_min"] == c["alpha_max"] for c in configs)
    assert tuple(args.alphas) == DEFAULT_ALPHAS


def test_build_configs_alphas_y_rangos():
    configs = build_configs([1.0, 0.5], [(0.6, 0.8)])
    assert [(c["tag"], c["alpha_min"], c["alpha_max"]) for c in configs] == [
        ("a100", 1.0, 1.0), ("a050", 0.5, 0.5), ("u06_08", 0.6, 0.8)]


def test_build_configs_solo_rangos_no_agrega_los_alphas(tmp_path):
    args = settings_for(tmp_path, "--range", "0.6", "0.8")
    assert [c["tag"] for c in build_configs(args.alphas, args.ranges)] == ["u06_08"]


def test_build_configs_rechaza_invalidos_y_repetidos():
    with pytest.raises(ValueError, match="0 < alpha_min"):
        build_configs([1.5])
    with pytest.raises(ValueError, match="misma etiqueta"):
        build_configs([0.5, 0.5])


def test_config_paths_siguen_el_patron_de_los_existentes(tmp_path):
    p = config_paths("a020", settings_for(tmp_path))
    assert p["anon_dir"] == tmp_path / "anon_k134" / "anon_a020"
    assert p["manifest"] == tmp_path / "anon_k134" / "anon_manifest_a020.tsv"
    assert p["original_dir"] == tmp_path / "anon_k134" / "original"
    assert p["embeddings"] == tmp_path / "emb_k134_a020.npz"
    assert p["eer"] == tmp_path / "eer_bootstrap_k134_a020.json"
    assert p["eer_replicates"] == tmp_path / "eer_bootstrap_k134_a020_replicas.npz"
    assert p["impersonation"] == tmp_path / "impersonation_k134_a020.json"
    assert p["transcripts"] == tmp_path / "whisper_turbo_k134_a020.tsv"
    assert p["wer"] == tmp_path / "wer_turbo_k134_a020.json"


def test_config_paths_nunca_apuntan_a_los_archivos_existentes(tmp_path):
    existing = {"anon_u07_09", "anon_manifest_u07_09.tsv", "emb_k134.npz", "eer_bootstrap_k134.json",
                "eer_bootstrap_k134_replicas.npz", "impersonation_k134.json", "whisper_turbo_k134.tsv",
                "wer_turbo_k134.json"}
    for config in build_configs(list(DEFAULT_ALPHAS), [(0.6, 0.8)]):
        names = {Path(p).name for key, p in config_paths(config["tag"], settings_for(tmp_path)).items()
                 if key != "original_dir"}
        assert not names & existing


# ---------------------------------------------------------------------------
# step_plan
# ---------------------------------------------------------------------------

def plan_dict(tmp_path, config, *extra):
    return {step: (outputs, cmd) for step, outputs, cmd in step_plan(config, settings_for(tmp_path, *extra))}


def args_of(cmd, flag, n=1):
    i = cmd.index(flag)
    return cmd[i + 1:i + 1 + n] if n > 1 else cmd[i + 1]


def test_step_plan_pasos_en_orden():
    config = build_configs([0.2])[0]
    assert [s for s, _, _ in step_plan(config, settings_for(Path("d")))] == list(STEPS)


def test_step_plan_anonimizar_con_alpha_fijo_y_reutilizando_el_original(tmp_path):
    outputs, cmd = plan_dict(tmp_path, build_configs([0.2])[0])["anonymize"]
    assert cmd[0] == "PY" and cmd[1].endswith("anonymize_subset.py")
    assert args_of(cmd, "--alpha") == "0.2" and "--alpha-range" not in cmd and "--reuse-original" in cmd
    assert args_of(cmd, "--tag") == "a020" and args_of(cmd, "--output-dir") == str(tmp_path / "anon_k134")
    assert args_of(cmd, "--subset") == str(tmp_path / "subset_k134.tsv")
    assert outputs == [tmp_path / "anon_k134" / "anon_manifest_a020.tsv"]


def test_step_plan_anonimizar_con_rango(tmp_path):
    _, cmd = plan_dict(tmp_path, build_configs([], [(0.6, 0.8)])[0])["anonymize"]
    assert args_of(cmd, "--alpha-range", 2) == ["0.6", "0.8"] and "--alpha" not in cmd


def test_step_plan_embeddings_reutilizan_emb_k134(tmp_path):
    _, cmd = plan_dict(tmp_path, build_configs([0.2])[0])["embeddings"]
    assert args_of(cmd, "--reuse-original-from") == str(tmp_path / "emb_k134.npz")
    assert args_of(cmd, "--anon-dir") == str(tmp_path / "anon_k134" / "anon_a020")
    assert args_of(cmd, "--model-dir") == str(tmp_path / "ecapa")


def test_step_plan_parametros_por_defecto_del_protocolo(tmp_path):
    plan = plan_dict(tmp_path, build_configs([0.2])[0])
    for step in ("eer", "impersonation"):
        _, cmd = plan[step]
        assert (args_of(cmd, "--seed"), args_of(cmd, "--n-enroll"), args_of(cmd, "--n-trial")) == ("42", "3", "2")
        assert (args_of(cmd, "--n-boot"), args_of(cmd, "--boot-seed"), args_of(cmd, "--level")) == ("1000", "2026", "0.95")
    assert args_of(plan["impersonation"][1], "--max-far") == "0.01"
    _, cmd = plan["transcribe"]
    assert (args_of(cmd, "--model"), args_of(cmd, "--per-stratum"), args_of(cmd, "--seed")) == ("turbo", "134", "42")
    assert args_of(cmd, "--reuse-original-from") == str(tmp_path / "whisper_turbo_k134.tsv")
    _, cmd = plan["wer"]
    assert (args_of(cmd, "--normalization"), args_of(cmd, "--model-label"), args_of(cmd, "--n-boot")) == (
        "final", "turbo", "1000")


def test_step_plan_cada_paso_lee_la_salida_del_anterior(tmp_path):
    plan = plan_dict(tmp_path, build_configs([0.2])[0])
    manifest = str(plan["anonymize"][0][0])
    assert args_of(plan["embeddings"][1], "--manifest") == manifest
    assert args_of(plan["eer"][1], "--embeddings") == str(plan["embeddings"][0][0])
    assert args_of(plan["impersonation"][1], "--embeddings") == str(plan["embeddings"][0][0])
    assert args_of(plan["wer"][1], "--transcripts") == str(plan["transcribe"][0][0])


def test_step_plan_respeta_los_parametros_dados(tmp_path):
    plan = plan_dict(tmp_path, build_configs([0.2])[0], "--n-boot", "50", "--max-far", "0.05", "--per-stratum", "5")
    assert args_of(plan["eer"][1], "--n-boot") == "50" and args_of(plan["impersonation"][1], "--max-far") == "0.05"
    assert args_of(plan["transcribe"][1], "--per-stratum") == "5"


# ---------------------------------------------------------------------------
# step_done, reanudación y dry-run
# ---------------------------------------------------------------------------

def test_step_done_todas_las_salidas_deben_existir(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.npz"
    a.write_text("{}")
    assert not step_done("eer", [a, b])
    b.write_text("x")
    assert step_done("eer", [a, b])


def write_transcripts(path, n_complete, partial=False):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("session_id\tclient_id\tstratum\tversion\treference\thypothesis\tseconds\n")
        for i in range(n_complete):
            f.write(f"cv_{i}\th\ts\toriginal\tr\th\t1.0\n")
        if partial:
            f.write("cv_x\th\ts\tanonymized\tr\th\t\n")


def test_step_done_transcripcion_incompleta_no_cuenta(tmp_path):
    out = tmp_path / "t.tsv"
    write_transcripts(out, 3, partial=True)
    assert not step_done("transcribe", [out], expected_rows=4)
    write_transcripts(out, 4)
    assert step_done("transcribe", [out], expected_rows=4)
    assert step_done("transcribe", [out], expected_rows=None)


class FakeRunner:
    """Simula subprocess.run: guarda los comandos y crea las salidas del paso."""

    def __init__(self, settings, fail_on=None):
        self.commands, self.settings, self.fail_on = [], settings, fail_on
        self.outputs = {tuple(cmd): outs for c in build_configs(settings.alphas, settings.ranges)
                        for _, outs, cmd in step_plan(c, settings)}

    def __call__(self, cmd, check, env):
        assert check is True
        self.commands.append(cmd)
        self.env = env
        if self.fail_on and self.fail_on in " ".join(cmd):
            raise subprocess.CalledProcessError(1, cmd)
        for out in self.outputs[tuple(cmd)]:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            if str(out).endswith(".tsv") and "whisper" in str(out):
                write_transcripts(out, 4)
            else:
                Path(out).write_text("hecho")


def fake_clock():
    t = {"now": 0.0}

    def clock():
        t["now"] += 2.5
        return t["now"]
    return clock


def sweep(tmp_path, *extra, runner=None, steps=STEPS, dry_run=False):
    settings = settings_for(tmp_path, "--alphas", "1.0", "0.5", *extra)
    runner = runner or FakeRunner(settings)
    lines = []
    records = run_sweep(build_configs(settings.alphas, settings.ranges), settings, steps, dry_run,
                        runner=runner, clock=fake_clock(), log=lines.append)
    return records, runner, lines, settings


def test_run_sweep_corre_cada_paso_de_cada_alpha_en_orden(tmp_path):
    records, runner, lines, _ = sweep(tmp_path)
    assert [(r["tag"], r["step"]) for r in records] == [(t, s) for t in ("a100", "a050") for s in STEPS]
    assert all(r["status"] == "hecho" and r["seconds"] == 2.5 for r in records)
    assert [Path(c[1]).name for c in runner.commands[:6]] == [
        "anonymize_subset.py", "extract_embeddings.py", "eer_bootstrap.py", "impersonation.py", "transcribe.py",
        "wer_bootstrap.py"]


def test_run_sweep_imprime_el_tiempo_de_cada_paso(tmp_path):
    _, _, lines, _ = sweep(tmp_path)
    assert "[a100] anonymize: terminado en 2,5 s" in lines


def test_run_sweep_es_reanudable(tmp_path):
    sweep(tmp_path)
    records, runner, lines, _ = sweep(tmp_path)
    assert runner.commands == [] and all(r["status"] == "saltado" for r in records)
    assert any("se salta" in line for line in lines)


def test_run_sweep_retoma_desde_el_paso_que_falto(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        sweep(tmp_path, runner=FakeRunner(settings_for(tmp_path, "--alphas", "1.0", "0.5"),
                                          fail_on="impersonation_k134_a050"))
    records, runner, _, _ = sweep(tmp_path)
    assert [Path(c[1]).name for c in runner.commands] == ["impersonation.py", "transcribe.py", "wer_bootstrap.py"]
    assert {r["tag"] for r in records if r["status"] == "hecho"} == {"a050"}


def test_run_sweep_transcripcion_incompleta_se_vuelve_a_correr(tmp_path):
    base = tmp_path / "whisper_turbo_k134.tsv"
    write_transcripts(base, 3)  # 3 originales: un TSV completo tiene 6 filas
    sweep(tmp_path)  # el falso escribe 4 filas: queda "incompleto"
    records, runner, _, _ = sweep(tmp_path, steps=("transcribe",))
    assert [r["status"] for r in records] == ["hecho", "hecho"]


def test_run_sweep_steps_elige_los_pasos(tmp_path):
    records, runner, _, _ = sweep(tmp_path, steps=("anonymize", "embeddings", "eer", "impersonation"))
    assert {r["step"] for r in records} == {"anonymize", "embeddings", "eer", "impersonation"}
    assert not any(Path(c[1]).name in ("transcribe.py", "wer_bootstrap.py") for c in runner.commands)


def test_run_sweep_dry_run_no_ejecuta_e_imprime_los_comandos(tmp_path):
    def never(*a, **k):
        raise AssertionError("no debía ejecutar nada")

    records, _, lines, settings = sweep(tmp_path, runner=never, dry_run=True)
    assert all(r["status"] == "simulado" for r in records) and len(records) == 12
    assert any("anonymize_subset.py" in line and "--alpha 0.5" in line for line in lines)
    assert list(tmp_path.iterdir()) == []


def test_run_sweep_dry_run_marca_lo_ya_hecho(tmp_path):
    sweep(tmp_path)
    records, _, _, _ = sweep(tmp_path, dry_run=True, runner=lambda *a, **k: pytest.fail("ejecutó"))
    assert all(r["status"] == "saltado" for r in records)


def test_run_sweep_error_detiene_el_barrido(tmp_path):
    settings = settings_for(tmp_path, "--alphas", "1.0", "0.5")
    runner = FakeRunner(settings, fail_on="emb_k134_a100")
    with pytest.raises(subprocess.CalledProcessError):
        run_sweep(build_configs(settings.alphas, settings.ranges), settings, STEPS, False, runner=runner,
                  clock=fake_clock(), log=lambda s: None)
    assert len(runner.commands) == 2  # anonimizar y embeddings de a100; nada más


def test_run_sweep_pythonpath_incluye_scripts_y_el_core(tmp_path):
    _, runner, _, _ = sweep(tmp_path, steps=("anonymize",))
    paths = runner.env["PYTHONPATH"].split(os.pathsep)
    assert paths[0].endswith("scripts") and paths[1].endswith(os.path.join("packages", "core", "src"))


def test_format_seconds():
    assert format_seconds(12.34) == "12,3 s"
    assert format_seconds(185) == "3 min 05 s"
    assert format_seconds(3725) == "1 h 02 min"


def test_main_dry_run_de_punta_a_punta(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_alpha_sweep.py", "--data-dir", str(tmp_path), "--alphas", "0.2",
                                      "--steps", "eer", "wer", "--dry-run"])
    run_alpha_sweep.main()
    out = capsys.readouterr().out
    assert "Configuraciones: a020; pasos: eer, wer (simulación)" in out
    assert "eer_bootstrap.py" in out and "wer_bootstrap.py" in out and "anonymize_subset.py" not in out
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# plot_alpha_sweep: tabla resumen
# ---------------------------------------------------------------------------

def iv(v, low, high):
    return {"value": v, "ci_low": low, "ci_high": high}


TOY = {  # α: (EER o-o, o-a, a-a, suplantación EER, suplantación FAR, WER original, WER anonimizado)
    0.5: (0.0172, 0.27, 0.19, 0.06, 0.03, 0.0384, 0.15),
    1.0: (0.0172, 0.019, 0.017, 0.98, 0.93, 0.0384, 0.039),
}


def write_toy_jsons(tmp_path, alphas=(0.5, 1.0), skip_wer=()):
    settings = plot_alpha_sweep.parse_args(["--data-dir", str(tmp_path)])
    for config in build_configs(list(alphas)):
        p = config_paths(config["tag"], settings)
        oo, oa, aa, ie, i_far, wo, wa = TOY[config["alpha_min"]]
        eer = {"scenarios": {name: {"eer": v, "ci_low": v - 0.001, "ci_high": v + 0.002, "n_target": 1}
                             for name, v in (("o-o", oo), ("o-a", oa), ("a-a", aa))}}
        p["eer"].write_text(json.dumps(eer), encoding="utf-8")
        p["impersonation"].write_text(json.dumps({"operating_points": {
            "eer": {"impersonation_rate": iv(ie, ie - 0.01, ie + 0.005)},
            "far": {"impersonation_rate": iv(i_far, i_far - 0.01, i_far + 0.004), "max_far": 0.01}}}), encoding="utf-8")
        if config["alpha_min"] not in skip_wer:
            p["wer"].write_text(json.dumps({"wer": {"original": iv(wo, wo - 0.003, wo + 0.004),
                                                    "anonymized": iv(wa, wa - 0.01, wa + 0.02)}}), encoding="utf-8")
    return settings


def test_collect_rows_una_fila_por_alpha_con_todas_las_metricas(tmp_path):
    settings = write_toy_jsons(tmp_path)
    rows = collect_rows(build_configs([0.5, 1.0]), settings)
    assert [r["tag"] for r in rows] == ["a050", "a100"]
    r = rows[0]
    assert (r["eer_oo"], r["eer_oa"], r["eer_aa"]) == (0.0172, 0.27, 0.19)
    assert (r["eer_oa_ci_low"], r["eer_oa_ci_high"]) == pytest.approx((0.269, 0.272))
    assert (r["impersonation_eer"], r["impersonation_far"]) == (0.06, 0.03)
    assert (r["impersonation_far_ci_low"], r["impersonation_far_ci_high"]) == pytest.approx((0.02, 0.034))
    assert (r["wer_original"], r["wer_anonymized"]) == (0.0384, 0.15)
    assert (r["wer_anonymized_ci_low"], r["wer_anonymized_ci_high"]) == pytest.approx((0.14, 0.17))


def test_collect_rows_json_faltante_queda_vacio(tmp_path):
    settings = write_toy_jsons(tmp_path, skip_wer=(1.0,))
    rows = collect_rows(build_configs([0.5, 1.0, 0.3]), settings)
    assert rows[1]["wer_anonymized"] is None and rows[1]["eer_oa"] == 0.019
    assert all(rows[2][c] is None for c in SUMMARY_COLUMNS[3:])


def test_write_summary_columnas_y_valores(tmp_path):
    settings = write_toy_jsons(tmp_path, skip_wer=(1.0,))
    out = write_summary(collect_rows(build_configs([0.5, 1.0]), settings), tmp_path / "resumen.tsv")
    with open(out, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert list(rows[0]) == SUMMARY_COLUMNS
    assert {"eer_aa", "eer_aa_ci_low", "eer_aa_ci_high", "impersonation_eer", "wer_original"} <= set(SUMMARY_COLUMNS)
    assert (rows[0]["alpha_min"], rows[0]["eer_oa"], rows[0]["wer_anonymized"]) == ("0.5", "0.27", "0.15")
    assert rows[1]["wer_anonymized"] == "" and rows[1]["eer_aa"] == "0.017"


# ---------------------------------------------------------------------------
# plot_alpha_sweep: figura
# ---------------------------------------------------------------------------

def points_of(ax, color=ANONYMIZED_COLOR):
    return [l for l in ax.get_lines() if l.get_marker() == "o"
            and matplotlib.colors.same_color(l.get_color(), color)]


def bars_of(ax):
    return [l for l in ax.get_lines() if l.get_marker() == "_"]


def figure(tmp_path, alphas=(0.5, 1.0), **kw):
    settings = write_toy_jsons(tmp_path, **kw)
    return draw_sweep(collect_rows(build_configs(list(alphas)), settings))


def test_figura_tres_paneles_con_alpha_compartido_y_ancho_de_columna(tmp_path):
    fig = figure(tmp_path)
    assert len(fig.axes) == 3 and fig.get_figwidth() == pytest.approx(3.35)
    assert all(fig.axes[0].get_shared_x_axes().joined(fig.axes[0], ax) for ax in fig.axes[1:])
    assert fig.axes[2].get_xlabel() == "α (coeficiente de McAdams)"
    assert [ax.get_title() for ax in fig.axes] == ["", "", ""]


def test_figura_puntos_e_intervalos_de_cada_panel(tmp_path):
    fig = figure(tmp_path)
    expected = [("eer_oa", 1), ("impersonation_far", 4), ("wer_anonymized", 6)]
    for ax, (_, index) in zip(fig.axes, expected):
        (dots,) = points_of(ax)
        np.testing.assert_allclose(dots.get_xdata(), [0.5, 1.0])
        np.testing.assert_allclose(dots.get_ydata(), [100 * TOY[0.5][index], 100 * TOY[1.0][index]])
    bars = sorted(bars_of(fig.axes[0]), key=lambda l: l.get_xdata()[0])
    np.testing.assert_allclose(sorted(bars[0].get_ydata()), [26.9, 27.2])
    np.testing.assert_allclose(bars[0].get_xdata(), [0.5, 0.5])
    bars = sorted(bars_of(fig.axes[2]), key=lambda l: l.get_xdata()[0])
    np.testing.assert_allclose(sorted(bars[0].get_ydata()), [14, 17])  # intervalo asimétrico


def test_figura_referencias_sin_anonimizar_y_metas(tmp_path):
    fig = figure(tmp_path)
    def hlines(ax, color=None, style=None):
        return [l.get_ydata()[0] for l in ax.get_lines() if len(set(l.get_ydata())) == 1 and len(l.get_ydata()) == 2
                and l.get_marker() in ("None", "", None) and (style is None or l.get_linestyle() == style)
                and (color is None or matplotlib.colors.same_color(l.get_color(), color))]
    assert hlines(fig.axes[0], ORIGINAL_COLOR, "--") == [pytest.approx(1.72)]
    assert hlines(fig.axes[0], style=":") == [pytest.approx(30)]
    assert hlines(fig.axes[2], ORIGINAL_COLOR, "--") == [pytest.approx(3.84)]
    assert hlines(fig.axes[2], style=":") == [pytest.approx(15)]
    assert hlines(fig.axes[1], ORIGINAL_COLOR, "--") == [] and hlines(fig.axes[1], style=":") == []
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert "Meta (30 %)" in texts and "Meta (15 %)" in texts


def test_figura_no_dibuja_el_a_a(tmp_path):
    fig = figure(tmp_path)
    ys = {round(y, 6) for ax in fig.axes for l in ax.get_lines() for y in np.atleast_1d(l.get_ydata())}
    assert round(100 * TOY[0.5][2], 6) not in ys and round(100 * TOY[1.0][2], 6) not in ys


def test_figura_eje_alpha_con_coma_y_rotulos(tmp_path):
    fig = figure(tmp_path)
    ax = fig.axes[2]
    assert list(ax.get_xticks()) == [0.5, 1.0]
    assert ax.xaxis.get_major_formatter()(0.5, 0) == "0,5" and ax.yaxis.get_major_formatter()(2.5, 0) == "2,5"
    assert [a.get_ylabel() for a in fig.axes] == ["EER o-a (%)", "Suplantación\nFAR ≤ 1 % (%)", "WER (%)"]
    legend = [t.get_text() for t in fig.legends[0].get_texts()]
    assert legend == ["Anonimizado (o-a)", "Sin anonimizar (o-o)", "Meta"]


def test_figura_con_un_wer_faltante_salta_ese_punto(tmp_path):
    fig = figure(tmp_path, skip_wer=(1.0,))
    (dots,) = points_of(fig.axes[2])
    np.testing.assert_allclose(dots.get_xdata(), [0.5])
    (dots,) = points_of(fig.axes[0])
    np.testing.assert_allclose(dots.get_xdata(), [0.5, 1.0])


def test_figura_los_rangos_no_entran(tmp_path):
    settings = write_toy_jsons(tmp_path)
    rows = collect_rows(build_configs([0.5, 1.0]), settings)
    rows.append({**rows[0], "tag": "u06_08", "alpha_min": 0.6, "alpha_max": 0.8})
    (dots,) = points_of(draw_sweep(rows).axes[0])
    np.testing.assert_allclose(dots.get_xdata(), [0.5, 1.0])


def test_main_escribe_tabla_y_figura(tmp_path, monkeypatch, capsys):
    write_toy_jsons(tmp_path)
    monkeypatch.setattr(sys, "argv", ["plot_alpha_sweep.py", "--data-dir", str(tmp_path), "--alphas", "0.5", "1.0"])
    with matplotlib.rc_context():
        plot_alpha_sweep.main()
    assert (tmp_path / "barrido_alfa_resumen.tsv").exists()
    assert (tmp_path / "figuras" / "barrido_alfa.pdf").read_bytes()[:4] == b"%PDF"
    assert "barrido_alfa.pdf" in capsys.readouterr().out
