"""
Barrido del coeficiente de McAdams: corre todo el protocolo para cada α.

Para cada configuración (un α fijo o un rango) corre, en orden:
anonimizar → embeddings → bootstrap del EER → suplantación → transcripción →
bootstrap del WER, con los mismos parámetros que se usaron para U(0.7, 0.9).
Reutiliza lo que no depende de α: los WAV originales, los embeddings
originales (emb_k134.npz) y las transcripciones originales
(whisper_turbo_k134.tsv).

Es reanudable: un paso cuya salida ya existe se salta, así que nunca se
sobrescribe nada. Con --dry-run solo imprime los comandos y con --steps se
eligen los pasos (por ejemplo, primero EER y suplantación de todos los α y
el WER después). Cada paso es un subproceso con el mismo Python, con
`scripts` y `packages/core/src` en PYTHONPATH (mcadams_core no está
instalado en venv-eval).
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from alpha_config import config_tag, validate_range
from common_voice_corpus import read_tsv_rows

REPO = Path(__file__).resolve().parents[1]
STEPS = ("anonymize", "embeddings", "eer", "impersonation", "transcribe", "wer")
DEFAULT_ALPHAS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
CLIPS_SUBDIR = Path("Common Voice Scripted Speech 26.0 - Spanish", "1781723056928-cv-corpus-26.0-2026-06-12-es",
                    "cv-corpus-26.0-2026-06-12", "es", "clips")


def build_configs(alphas=None, ranges=None):
    """
    Arma la lista de configuraciones: primero los α fijos y luego los rangos.

    Parámetros:
        alphas: lista de α fijos (None si no se dio ninguno).
        ranges: lista de pares (alpha_min, alpha_max).

    Retorna:
        Lista de diccionarios {"tag", "alpha_min", "alpha_max"}. Lanza
        `ValueError` si algún α es inválido o si dos configuraciones darían
        la misma etiqueta (escribirían en los mismos archivos).
    """
    configs = [(a, a) for a in (alphas or [])] + [tuple(r) for r in (ranges or [])]
    result, seen = [], set()
    for alpha_min, alpha_max in configs:
        validate_range(alpha_min, alpha_max)
        tag = config_tag(alpha_min, alpha_max)
        if tag in seen:
            raise ValueError(f"dos configuraciones tienen la misma etiqueta {tag}")
        seen.add(tag)
        result.append({"tag": tag, "alpha_min": alpha_min, "alpha_max": alpha_max})
    return result


def config_paths(tag, settings):
    """
    Rutas de entrada y salida de una configuración, con el patrón de los archivos existentes.

    Parámetros:
        tag: etiqueta de la configuración (por ejemplo "a020").
        settings: argumentos del barrido (ver `parse_args`).

    Retorna:
        Diccionario de rutas (`Path`).
    """
    anon_root, out, ds, model = Path(settings.anon_root), Path(settings.output_dir), settings.dataset_tag, settings.whisper_model
    return {
        "original_dir": anon_root / "original",
        "anon_dir": anon_root / f"anon_{tag}",
        "manifest": anon_root / f"anon_manifest_{tag}.tsv",
        "embeddings": out / f"emb_{ds}_{tag}.npz",
        "eer": out / f"eer_bootstrap_{ds}_{tag}.json",
        "eer_replicates": out / f"eer_bootstrap_{ds}_{tag}_replicas.npz",
        "impersonation": out / f"impersonation_{ds}_{tag}.json",
        "transcripts": out / f"whisper_{model}_{ds}_{tag}.tsv",
        "wer": out / f"wer_{model}_{ds}_{tag}.json",
    }


def step_plan(config, settings):
    """
    Comandos de cada paso para una configuración.

    Retorna:
        Lista de tuplas (paso, salidas que lo dan por hecho, comando como
        lista de textos), en el orden de `STEPS`.
    """
    p = config_paths(config["tag"], settings)
    py, scripts = settings.python, REPO / "scripts"
    if config["alpha_min"] == config["alpha_max"]:
        alpha_args = ["--alpha", str(config["alpha_min"])]
    else:
        alpha_args = ["--alpha-range", str(config["alpha_min"]), str(config["alpha_max"])]
    boot = ["--n-boot", str(settings.n_boot), "--boot-seed", str(settings.boot_seed), "--level", str(settings.level)]
    split = ["--seed", str(settings.seed), "--n-enroll", str(settings.n_enroll), "--n-trial", str(settings.n_trial)]
    commands = {
        "anonymize": ([p["manifest"]], [
            py, scripts / "anonymize_subset.py", "--subset", settings.subset, "--clips-dir", settings.clips_dir,
            "--output-dir", settings.anon_root, "--tag", config["tag"], *alpha_args, "--reuse-original"]),
        "embeddings": ([p["embeddings"]], [
            py, scripts / "extract_embeddings.py", "--manifest", p["manifest"], "--original-dir", p["original_dir"],
            "--anon-dir", p["anon_dir"], "--model-dir", settings.ecapa_dir, "--output", p["embeddings"],
            "--reuse-original-from", settings.base_embeddings]),
        "eer": ([p["eer"], p["eer_replicates"]], [
            py, scripts / "eer_bootstrap.py", "--embeddings", p["embeddings"], "--manifest", p["manifest"],
            "--output", p["eer"], "--replicates-output", p["eer_replicates"], *split, *boot]),
        "impersonation": ([p["impersonation"]], [
            py, scripts / "impersonation.py", "--embeddings", p["embeddings"], "--manifest", p["manifest"],
            "--output", p["impersonation"], *split, *boot, "--max-far", str(settings.max_far)]),
        "transcribe": ([p["transcripts"]], [
            py, scripts / "transcribe.py", "--manifest", p["manifest"], "--subset", settings.subset,
            "--original-dir", p["original_dir"], "--anon-dir", p["anon_dir"], "--model", settings.whisper_model,
            "--model-dir", settings.whisper_dir, "--per-stratum", str(settings.per_stratum),
            "--seed", str(settings.seed), "--output", p["transcripts"],
            "--reuse-original-from", settings.base_transcripts]),
        "wer": ([p["wer"]], [
            py, scripts / "wer_bootstrap.py", "--transcripts", p["transcripts"], "--output", p["wer"], *boot,
            "--model-label", settings.whisper_model, "--normalization", "final"]),
    }
    return [(step, outputs, [str(c) for c in command]) for step, (outputs, command) in commands.items()]


def count_complete_rows(path):
    """Filas completas (con segundos) de un TSV de transcribe.py; 0 si no existe."""
    if not Path(path).exists():
        return 0
    return sum(1 for row in read_tsv_rows(path) if row.get("seconds") not in (None, ""))


def expected_transcript_rows(settings):
    """Filas que debe tener un TSV completo: dos por cada clip original del TSV base (None si no existe)."""
    base = Path(settings.base_transcripts)
    if not base.exists():
        return None
    return 2 * sum(1 for row in read_tsv_rows(base) if row["version"] == "original")


def step_done(step, outputs, expected_rows=None):
    """
    Dice si un paso ya está hecho.

    Un paso está hecho si existen todas sus salidas. La transcripción,
    además, debe estar completa (`expected_rows` filas), porque su TSV se
    escribe fila a fila y puede quedar a medias.
    """
    if not all(Path(o).exists() for o in outputs):
        return False
    if step == "transcribe" and expected_rows is not None:
        return count_complete_rows(outputs[0]) >= expected_rows
    return True


def subprocess_env():
    """Entorno de los subprocesos: agrega scripts y packages/core/src a PYTHONPATH."""
    paths = [str(REPO / "scripts"), str(REPO / "packages" / "core" / "src")]
    current = os.environ.get("PYTHONPATH")
    return {**os.environ, "PYTHONPATH": os.pathsep.join(paths + ([current] if current else []))}


def log_now(message):
    """Imprime y vacía el búfer, para que el registro quede en orden con la salida de los subprocesos."""
    print(message, flush=True)


def run_sweep(configs, settings, steps=STEPS, dry_run=False, runner=subprocess.run, clock=time.perf_counter,
              log=log_now):
    """
    Corre los pasos elegidos para cada configuración, en orden.

    Parámetros:
        configs: salida de `build_configs`.
        settings: argumentos del barrido.
        steps: pasos a correr (los demás se ignoran).
        dry_run: si es True, imprime los comandos sin ejecutarlos.
        runner: función que ejecuta un comando (inyectable para las pruebas);
            se llama como subprocess.run(comando, check=True, env=...).
        clock: reloj en segundos (inyectable).
        log: función para imprimir.

    Retorna:
        Lista de diccionarios {"tag", "step", "status", "seconds"} con
        status "hecho", "saltado" o "simulado". Si un paso falla, la
        excepción se propaga y el barrido se detiene (lo ya hecho queda).
    """
    expected_rows = expected_transcript_rows(settings)
    env = subprocess_env()
    records = []
    for config in configs:
        for step, outputs, command in step_plan(config, settings):
            if step not in steps:
                continue
            label = f"[{config['tag']}] {step}"
            if step_done(step, outputs, expected_rows):
                log(f"{label}: ya existe {outputs[0]}, se salta")
                records.append({"tag": config["tag"], "step": step, "status": "saltado", "seconds": 0.0})
                continue
            if dry_run:
                log(f"{label}: {subprocess.list2cmdline(command)}")
                records.append({"tag": config["tag"], "step": step, "status": "simulado", "seconds": 0.0})
                continue
            log(f"{label}: empieza")
            start = clock()
            runner(command, check=True, env=env)
            seconds = clock() - start
            log(f"{label}: terminado en {format_seconds(seconds)}")
            records.append({"tag": config["tag"], "step": step, "status": "hecho", "seconds": seconds})
    return records


def format_seconds(seconds):
    """Escribe una duración como "1 h 02 min", "3 min 05 s" o "12,3 s"."""
    if seconds >= 3600:
        return f"{int(seconds // 3600)} h {int(seconds % 3600 // 60):02d} min"
    if seconds >= 60:
        return f"{int(seconds // 60)} min {int(seconds % 60):02d} s"
    return f"{seconds:.1f} s".replace(".", ",")


def parse_args(argv=None):
    """Lee los argumentos; las rutas no dadas salen de --data-dir."""
    parser = argparse.ArgumentParser(description="Barrido del coeficiente de McAdams (α fijo o rangos).")
    parser.add_argument("--alphas", type=float, nargs="+", default=None,
                        help="α fijos (por defecto 0.2 0.3 ... 1.0 si no se da --range).")
    parser.add_argument("--range", type=float, nargs=2, action="append", default=[], metavar=("MIN", "MAX"),
                        dest="ranges", help="Rango aleatorio U(MIN, MAX); se puede repetir.")
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS), help="Pasos a correr.")
    parser.add_argument("--dry-run", action="store_true", help="Imprime los comandos sin ejecutarlos.")
    parser.add_argument("--data-dir", default=str(REPO.parent / "Datos"), help="Carpeta de datos (..\\Datos).")
    for name, help_text in (
        ("--subset", "subset_k134.tsv"), ("--clips-dir", "carpeta de los MP3 de Common Voice"),
        ("--anon-root", "carpeta con original\\ y anon_<etiqueta>\\ (anon_k134)"),
        ("--output-dir", "carpeta de los .npz, .json y .tsv de salida"),
        ("--base-embeddings", "emb_k134.npz, de donde salen los embeddings originales"),
        ("--base-transcripts", "whisper_turbo_k134.tsv, de donde salen las transcripciones originales"),
        ("--ecapa-dir", "modelo ECAPA"), ("--whisper-dir", "modelos de Whisper"),
    ):
        parser.add_argument(name, default=None, help=f"Por defecto, dentro de --data-dir: {help_text}.")
    parser.add_argument("--dataset-tag", default="k134", help="Parte del nombre de los archivos (emb_<tag>_a020.npz).")
    parser.add_argument("--whisper-model", default="turbo")
    parser.add_argument("--per-stratum", type=int, default=134, help="Clips por estrato para el WER.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-enroll", type=int, default=3)
    parser.add_argument("--n-trial", type=int, default=2)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--boot-seed", type=int, default=2026)
    parser.add_argument("--level", type=float, default=0.95)
    parser.add_argument("--max-far", type=float, default=0.01)
    parser.add_argument("--python", default=sys.executable, help="Python con el que se corre cada paso.")
    args = parser.parse_args(argv)

    data = Path(args.data_dir)
    defaults = {
        "subset": data / "subset_k134.tsv", "clips_dir": data / CLIPS_SUBDIR, "anon_root": data / "anon_k134",
        "output_dir": data, "base_embeddings": data / "emb_k134.npz",
        "base_transcripts": data / "whisper_turbo_k134.tsv", "ecapa_dir": data / "ecapa",
        "whisper_dir": data / "whisper",
    }
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, str(value))
    if args.alphas is None and not args.ranges:
        args.alphas = list(DEFAULT_ALPHAS)
    return args


def main():
    """Corre el barrido e imprime el tiempo de cada paso."""
    args = parse_args()
    configs = build_configs(args.alphas, args.ranges)
    print(f"Configuraciones: {', '.join(c['tag'] for c in configs)}; pasos: {', '.join(args.steps)}"
          + (" (simulación)" if args.dry_run else ""), flush=True)
    start = time.perf_counter()
    records = run_sweep(configs, args, args.steps, args.dry_run)
    done = [r for r in records if r["status"] == "hecho"]
    if done:
        print("\nTiempo por paso:")
        for r in done:
            print(f"  [{r['tag']}] {r['step']}: {format_seconds(r['seconds'])}")
    print(f"Total: {format_seconds(time.perf_counter() - start)}; hechos {len(done)}, "
          f"saltados {sum(r['status'] == 'saltado' for r in records)}")


if __name__ == "__main__":
    main()
