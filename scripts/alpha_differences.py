"""
Diferencias pareadas entre α vecinos del barrido.

Para cada par de configuraciones de α fijo consecutivas (por α real) calcula
la diferencia, con IC por percentiles, de:
(a) el EER o-a,
(b) la suplantación o-a con el umbral global de FAR <= max_far (como
    impersonation.py: el umbral se ajusta en o-o y se recalcula por réplica),
(c) el WER anonimizado (normalización final).

La diferencia es la del α mayor menos la del α menor. Es PAREADA: a las dos
configuraciones se les aplican las mismas réplicas bootstrap de hablantes
(las de `group_analysis.shared_samples`, que son las mismas que sacan
eer_bootstrap.py, impersonation.py y wer_bootstrap.py con la misma semilla).
Para el EER o-a se reutilizan las réplicas que ya guardó eer_bootstrap.py
(*_replicas.npz), después de comprobar que se hicieron con los mismos
parámetros y de recalcular la primera y la última; si no coinciden, se
recalculan todas. El valor puntual de cada configuración se compara con su
JSON (igualdad exacta) antes de restar.

Solo entran las configuraciones con todas sus salidas; las demás se saltan
con un aviso.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from eer_bootstrap import percentile_ci, resample_pairs
from eer_scoring import compute_eer
from group_analysis import (
    aligned_wer_counts,
    discover_configs,
    impersonation_counts,
    is_fixed,
    operating_thresholds,
    ratio,
    shared_samples,
    wer_by_group,
)
from run_alpha_sweep import REPO
from run_eer import load_and_score

METRICS = ("eer_oa", "impersonation_far", "wer_anonymized")
REQUIRED_FILES = ("manifest", "embeddings", "transcripts", "eer_json", "eer_replicates", "impersonation_json",
                  "wer_json")
COLUMNS = (["config_from", "config_to", "alpha_from", "alpha_to"]
           + [f"{m}_{s}" for m in METRICS for s in ("from", "to", "difference", "ci_low", "ci_high")])


def missing_outputs(config):
    """Archivos que le faltan a una configuración para entrar en las diferencias."""
    return [name for name in REQUIRED_FILES if not Path(config["files"][name]).exists()]


def neighbor_pairs(configs):
    """Pares de configuraciones de α fijo consecutivas, ordenadas por α real."""
    fixed = sorted((c for c in configs if is_fixed(c)), key=lambda c: c["alpha"])
    return list(zip(fixed[:-1], fixed[1:]))


def read_json(path):
    """Lee un JSON en UTF-8 (con o sin BOM)."""
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def json_points(config):
    """Valores puntuales de las tres métricas según los JSON de la configuración."""
    f = config["files"]
    return {
        "eer_oa": read_json(f["eer_json"])["scenarios"]["o-a"]["eer"],
        "impersonation_far": read_json(f["impersonation_json"])["operating_points"]["far"]["impersonation_rate"]["value"],
        "wer_anonymized": read_json(f["wer_json"])["wer"]["anonymized"]["value"],
    }


def eer_oa_replicates(config, results, samples, params):
    """
    Réplicas del EER o-a con las muestras dadas.

    Si el JSON de eer_bootstrap se hizo con los mismos parámetros y la
    primera y la última réplica guardadas coinciden exactamente con las
    recalculadas, se usan las guardadas; si no, se calculan todas.
    """
    oa = results["o-a"]

    def one(sample):
        return compute_eer(*resample_pairs(oa["scores"], oa["owner"], sample))[0]

    summary = read_json(config["files"]["eer_json"])
    same_params = all(summary.get(k) == v for k, v in params.items())
    if same_params and Path(config["files"]["eer_replicates"]).exists():
        with np.load(config["files"]["eer_replicates"]) as data:
            stored = np.array(data["o-a"])
        if stored.size == len(samples) and all(stored[b] == one(samples[b]) for b in (0, len(samples) - 1)):
            return stored
    return np.array([one(s) for s in samples])


def config_metrics(config, results, samples, thresholds_reps, client_ids, max_far, params):
    """
    Valor puntual y réplicas de las tres métricas para una configuración.

    Parámetros:
        config: configuración de `discover_configs`.
        results: salida de `run_scenarios` para esa configuración.
        samples: muestras compartidas (`shared_samples`).
        thresholds_reps: umbrales globales de cada réplica.
        client_ids: hablantes en el orden de las columnas de puntuaciones.
        max_far: FAR máximo del umbral de suplantación.
        params: parámetros con los que debió hacerse eer_bootstrap (para
            poder reutilizar sus réplicas).

    Retorna:
        Diccionario {métrica: {"point": valor, "reps": arreglo (n_boot,)}}.
    """
    total = np.zeros(len(client_ids), dtype=int)
    thresholds = operating_thresholds(results["o-o"], max_far)
    impersonation = ratio(*impersonation_counts(results, total, 1, thresholds)["impersonation_far"])[0]
    impersonation_reps = np.array([
        ratio(*impersonation_counts(results, total, 1, thresholds_reps[b], s)["impersonation_far"])[0]
        for b, s in enumerate(samples)])
    counts = aligned_wer_counts(config["files"]["transcripts"], client_ids)
    wer = wer_by_group(counts, total, 1)["anonymized"][0]
    wer_reps = np.array([wer_by_group(counts, total, 1, s)["anonymized"][0] for s in samples])
    return {
        "eer_oa": {"point": results["o-a"]["eer"], "reps": eer_oa_replicates(config, results, samples, params)},
        "impersonation_far": {"point": impersonation, "reps": impersonation_reps},
        "wer_anonymized": {"point": wer, "reps": wer_reps},
    }


def check_against_json(config, metrics):
    """Lanza `ValueError` si algún valor puntual no es exactamente el del JSON de la configuración."""
    expected = json_points(config)
    wrong = [m for m in METRICS if metrics[m]["point"] != expected[m]]
    if wrong:
        raise ValueError(f"{config['tag']}: el valor puntual de {', '.join(wrong)} no coincide con su JSON")


def paired_difference(from_metrics, to_metrics, level):
    """
    Diferencia pareada (destino menos origen) de cada métrica.

    El valor es la resta de los puntuales; el IC, los percentiles de la resta
    réplica por réplica (mismas muestras en las dos configuraciones).

    Retorna:
        Diccionario {métrica: (diferencia, ci_low, ci_high)}.
    """
    out = {}
    for m in METRICS:
        low, high = percentile_ci(to_metrics[m]["reps"] - from_metrics[m]["reps"], level)
        out[m] = (to_metrics[m]["point"] - from_metrics[m]["point"], low, high)
    return out


def difference_row(config_from, config_to, metrics_from, metrics_to, level):
    """Fila de la tabla para un par de configuraciones."""
    row = {"config_from": config_from["tag"], "config_to": config_to["tag"],
           "alpha_from": config_from["alpha"], "alpha_to": config_to["alpha"]}
    for m, (difference, low, high) in paired_difference(metrics_from, metrics_to, level).items():
        row.update({f"{m}_from": metrics_from[m]["point"], f"{m}_to": metrics_to[m]["point"],
                    f"{m}_difference": difference, f"{m}_ci_low": low, f"{m}_ci_high": high})
    return row


def write_rows(rows, path):
    """Escribe la tabla en TSV (UTF-8, punto decimal)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([float(row[c]) if isinstance(row[c], np.floating) else row[c] for c in COLUMNS])
    return path


def comma(value, decimals=2):
    """Número con coma decimal."""
    return f"{value:.{decimals}f}".replace(".", ",")


def parse_args(argv=None):
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description="Diferencias pareadas entre α vecinos del barrido.")
    parser.add_argument("--data-dir", default=str(REPO.parent / "Datos"))
    parser.add_argument("--output", default=None, help="Por defecto <data-dir>\\barrido_alfa_diferencias.tsv.")
    parser.add_argument("--tags", nargs="+", default=None, help="Solo estas configuraciones.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-enroll", type=int, default=3)
    parser.add_argument("--n-trial", type=int, default=2)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--boot-seed", type=int, default=2026)
    parser.add_argument("--level", type=float, default=0.95)
    parser.add_argument("--max-far", type=float, default=0.01)
    args = parser.parse_args(argv)
    args.output = args.output or str(Path(args.data_dir) / "barrido_alfa_diferencias.tsv")
    return args


def main():
    """Calcula y guarda las diferencias pareadas entre α vecinos."""
    args = parse_args()
    start = time.perf_counter()
    configs = [c for c in discover_configs(args.data_dir) if is_fixed(c)]
    if args.tags:
        configs = [c for c in configs if c["tag"] in args.tags]
    usable = []
    for c in configs:
        missing = missing_outputs(c)
        if missing:
            print(f"Aviso: se salta {c['tag']} (alfa {c['alpha']:g}): faltan {', '.join(missing)}")
        else:
            usable.append(c)
    pairs = neighbor_pairs(usable)
    if not pairs:
        raise SystemExit("Hacen falta al menos dos configuraciones de alfa fijo completas.")
    print(f"Configuraciones: {', '.join(f'{c['tag']} (alfa {c['alpha']:g})' for c in usable)}", flush=True)

    first_splits, _, first = load_and_score(usable[0]["files"]["embeddings"], usable[0]["files"]["manifest"],
                                            args.seed, args.n_enroll, args.n_trial)
    client_ids = list(first_splits)
    samples = shared_samples([s["stratum"] for s in first_splits.values()], args.n_boot, args.boot_seed)
    thresholds_reps = [operating_thresholds(first["o-o"], args.max_far, s) for s in samples]
    print(f"Umbrales globales de {args.n_boot} réplicas: {time.perf_counter() - start:.0f} s", flush=True)
    params = {"seed": args.seed, "n_enroll": args.n_enroll, "n_trial": args.n_trial, "n_boot": args.n_boot,
              "boot_seed": args.boot_seed}

    metrics = {}
    for c in usable:
        splits, _, results = load_and_score(c["files"]["embeddings"], c["files"]["manifest"], args.seed,
                                            args.n_enroll, args.n_trial)
        if splits != first_splits:
            raise ValueError(f"{c['tag']}: la partición registro/prueba no coincide con la de {usable[0]['tag']}")
        same_oo = np.array_equal(results["o-o"]["scores"], first["o-o"]["scores"])
        reps = thresholds_reps if same_oo else [operating_thresholds(results["o-o"], args.max_far, s) for s in samples]
        metrics[c["tag"]] = config_metrics(c, results, samples, reps, client_ids, args.max_far, params)
        check_against_json(c, metrics[c["tag"]])
        print(f"[{c['tag']}] listo ({time.perf_counter() - start:.0f} s)", flush=True)

    rows = [difference_row(a, b, metrics[a["tag"]], metrics[b["tag"]], args.level) for a, b in pairs]
    write_rows(rows, args.output)
    level = f"{100 * args.level:g}".replace(".", ",")
    names = {"eer_oa": "EER o-a", "impersonation_far": "suplantación FAR <= 1 %", "wer_anonymized": "WER anonimizado"}
    for r in rows:
        parts = [f"{names[m]} {comma(100 * r[f'{m}_difference'])} pp (IC {level} %: "
                 f"{comma(100 * r[f'{m}_ci_low'])} a {comma(100 * r[f'{m}_ci_high'])})" for m in METRICS]
        print(f"{r['config_from']}->{r['config_to']}: " + "; ".join(parts))
    print(f"Guardado en: {args.output} ({time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main()
