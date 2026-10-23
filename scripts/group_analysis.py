"""
Privacidad e inteligibilidad por grupo (estrato de acento o género) y configuración de α.

Usa solo resultados ya generados (embeddings, manifiestos y transcripciones):
no anonimiza ni transcribe nada. Las configuraciones se descubren solas a
partir de los manifiestos anon_manifest_<etiqueta>.tsv de ..\\Datos\\anon_k134
(u07_09 usa los nombres originales, sin etiqueta).

Métricas por grupo, con IC por bootstrap:
1. Suplantación: el umbral es GLOBAL (todo o-o, umbral del EER y FAR <=
   max_far, como impersonation.py); por grupo se cuenta qué fracción de los
   pares objetivo o-o (persona real) y o-a (grabación anonimizada) de sus
   hablantes se acepta.
2. EER dentro del grupo: o-o, o-a y a-a solo con registros y pruebas de
   hablantes del grupo.
3. WER de corpus original, anonimizado y su diferencia, con la
   normalización final.

El bootstrap es el de los scripts existentes (hablantes remuestreados por
estrato con stratified_resample, copias de un mismo hablante excluidas como
impostores con resample_pairs, umbral global recalculado en cada réplica) y
usa las MISMAS réplicas para todos los grupos y configuraciones; son las
mismas muestras, en el mismo orden, que sacan eer_bootstrap.py,
impersonation.py y wer_bootstrap.py con la misma semilla.

Además, una prueba de permutación por configuración: se barajan las
etiquetas de grupo entre hablantes y el estadístico es la desviación
estándar (poblacional) entre grupos de la métrica.
"""
import argparse
import csv
import json
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

from alpha_config import DEFAULT_RANGE, config_tag
from common_voice_corpus import read_tsv_rows
from eer_bootstrap import copy_trial_rows, percentile_ci, resample_pairs, stratified_resample
from eer_scoring import compute_eer, split_target_impostor
from impersonation import eer_threshold, far_threshold
from plot_eer import COLUMN_WIDTH, INK, MUTED_INK, SMALL_FONT, configure_fonts, decimal_comma, percent_label
from run_alpha_sweep import CLIPS_SUBDIR, REPO, config_paths
from run_eer import load_and_score
from transcribe import VERSIONS, normalize_final
from wer_bootstrap import speaker_counts

MISSING = "sin_dato"
CONFLICT = "conflicto"
TOTAL = "total"
GENDER_LABELS = {"male_masculine": "masculino", "female_feminine": "femenino"}
WER_MIN_ALPHA = 0.6
HALLUCINATION_NOTE = "dominado por alucinaciones"
HEATMAP_WIDTH = 1.4 * COLUMN_WIDTH
SCENARIOS = ("o-o", "o-a", "a-a")
POINTS = ("eer", "far")

# Nombre de cada análisis (sufijo de los archivos de salida).
GROUPING_NAMES = {("stratum", None): "estrato", ("gender", None): "genero", ("all", None): "total"}

# Rótulos legibles para las figuras.
GROUP_LABELS = {
    "america_central": "América Central", "andino_pacifico": "Andino-pacífico", "caribe": "Caribe",
    "chileno": "Chileno", "espana_centro_sur": "España centro-sur", "espana_norte": "España norte",
    "espana_sur": "España sur", "mexico": "México", "rioplatense": "Rioplatense", MISSING: "Sin dato",
    "masculino": "Masculino", "femenino": "Femenino", TOTAL: "Total",
}


# ---------------------------------------------------------------------------
# Configuraciones
# ---------------------------------------------------------------------------

def parse_tag(tag):
    """
    Rango de α de una etiqueta: "a070" → (0.7, 0.7), "u07_09" → (0.7, 0.9), "u065_085" → (0.65, 0.85).

    Es la inversa de `alpha_config.config_tag`.
    """
    def digits(text):
        return int(text) / 10 ** (len(text) - 1)

    if tag.startswith("a"):
        alpha = int(tag[1:]) / 100
        return alpha, alpha
    if tag.startswith("u") and "_" in tag:
        low, high = tag[1:].split("_")
        return digits(low), digits(high)
    raise ValueError(f"etiqueta de configuración desconocida: {tag}")


def config_files(tag, data_dir, dataset_tag="k134", whisper_model="turbo"):
    """
    Archivos de una configuración.

    El rango por defecto (u07_09) conserva los nombres originales, sin
    etiqueta (emb_k134.npz, whisper_turbo_k134.tsv, ...); las demás siguen
    `run_alpha_sweep.config_paths`.
    """
    data = Path(data_dir)
    anon_root = data / f"anon_{dataset_tag}"
    if tag == config_tag(*DEFAULT_RANGE):
        return {
            "manifest": anon_root / f"anon_manifest_{tag}.tsv",
            "embeddings": data / f"emb_{dataset_tag}.npz",
            "transcripts": data / f"whisper_{whisper_model}_{dataset_tag}.tsv",
            "eer_json": data / f"eer_bootstrap_{dataset_tag}.json",
            "impersonation_json": data / f"impersonation_{dataset_tag}.json",
        }
    p = config_paths(tag, Namespace(anon_root=anon_root, output_dir=data, dataset_tag=dataset_tag,
                                    whisper_model=whisper_model))
    return {"manifest": p["manifest"], "embeddings": p["embeddings"], "transcripts": p["transcripts"],
            "eer_json": p["eer"], "impersonation_json": p["impersonation"]}


def discover_configs(data_dir, dataset_tag="k134", whisper_model="turbo"):
    """
    Busca todas las configuraciones que tienen manifiesto y embeddings.

    Una configuración nueva (por ejemplo a072) entra sola en cuanto existen
    sus archivos. Se ordenan primero los α fijos de menor a mayor y después
    los rangos.

    Retorna:
        Lista de diccionarios {"tag", "alpha_min", "alpha_max", "files"}.
    """
    anon_root = Path(data_dir) / f"anon_{dataset_tag}"
    configs = []
    for manifest in anon_root.glob("anon_manifest_*.tsv"):
        tag = manifest.name[len("anon_manifest_"):-len(".tsv")]
        try:
            alpha_min, alpha_max = parse_tag(tag)
        except ValueError:
            continue
        files = config_files(tag, data_dir, dataset_tag, whisper_model)
        if files["manifest"].exists() and files["embeddings"].exists():
            configs.append({"tag": tag, "alpha_min": alpha_min, "alpha_max": alpha_max, "files": files})
    return sorted(configs, key=lambda c: (c["alpha_min"] != c["alpha_max"], c["alpha_min"], c["alpha_max"]))


def wer_reliable(config):
    """Si el WER de la configuración entra en las figuras y pruebas (α >= 0.6; en un rango, su mínimo)."""
    return config["alpha_min"] >= WER_MIN_ALPHA - 1e-9


# ---------------------------------------------------------------------------
# Grupos
# ---------------------------------------------------------------------------

def speaker_gender(values):
    """
    Género de un hablante a partir de los valores de sus filas en validated.tsv.

    Regla: el único valor no vacío; "sin_dato" si no tiene ninguno;
    "conflicto" si tiene valores distintos. male_masculine y
    female_feminine se escriben "masculino" y "femenino"; otro valor queda
    tal cual.
    """
    found = {v for v in values if v}
    if not found:
        return MISSING
    if len(found) > 1:
        return CONFLICT
    value = found.pop()
    return GENDER_LABELS.get(value, value)


def load_genders(validated_path, client_ids):
    """Género de cada hablante de `client_ids` según validated.tsv (columna gender)."""
    wanted = set(client_ids)
    values = {c: [] for c in wanted}
    for row in read_tsv_rows(validated_path):
        if row["client_id"] in wanted:
            values[row["client_id"]].append(row.get("gender") or "")
    return {c: speaker_gender(values[c]) for c in client_ids}


def assign_groups(strata, genders, group_by, only_gender=None):
    """
    Etiqueta de grupo de cada hablante.

    Parámetros:
        strata, genders: estrato y género de cada hablante, en el mismo orden.
        group_by: "stratum", "gender" o "all" (un solo grupo, "total").
        only_gender: si se da, los hablantes de otro género quedan fuera (None).

    Retorna:
        Lista de etiquetas (None para los hablantes excluidos).
    """
    labels = []
    for stratum, gender in zip(strata, genders):
        if only_gender is not None and gender != only_gender:
            labels.append(None)
        else:
            labels.append({"stratum": stratum, "gender": gender, "all": TOTAL}[group_by])
    return labels


def group_codes(labels):
    """
    Pasa las etiquetas a códigos enteros.

    Los grupos se ordenan alfabéticamente, con sin_dato y conflicto al final.

    Retorna:
        Una tupla (groups, codes): la lista de grupos y un arreglo con el
        código de cada hablante (-1 para los excluidos).
    """
    groups = sorted({l for l in labels if l is not None}, key=lambda g: (g in (MISSING, CONFLICT), g))
    index = {g: i for i, g in enumerate(groups)}
    return groups, np.array([index[l] if l is not None else -1 for l in labels], dtype=int)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def shared_samples(strata, n_boot, seed):
    """Las muestras de hablantes de cada réplica, en el orden en que las sacan los scripts existentes."""
    rng = np.random.default_rng(seed)
    return [stratified_resample(strata, rng) for _ in range(n_boot)]


def operating_thresholds(oo, max_far, sample=None):
    """Umbrales globales del EER y de FAR <= max_far en o-o, con todos los hablantes o con una muestra."""
    if sample is None:
        target, impostor = split_target_impostor(oo["scores"], oo["owner"])
    else:
        target, impostor = resample_pairs(oo["scores"], oo["owner"], sample)
    return {"eer": eer_threshold(target, impostor), "far": far_threshold(target, impostor, max_far)}


def row_targets(result):
    """Puntuación objetivo de cada prueba (contra el registro de su propio hablante)."""
    owner = np.asarray(result["owner"])
    return np.asarray(result["scores"])[np.arange(owner.size), owner]


def acceptance_by_group(scores, row_codes, n_groups, threshold):
    """Pares aceptados (puntuación >= umbral) y pares totales de cada grupo."""
    valid = row_codes >= 0
    accepted = np.bincount(row_codes[valid], weights=(scores[valid] >= threshold).astype(float), minlength=n_groups)
    return accepted, np.bincount(row_codes[valid], minlength=n_groups).astype(float)


def impersonation_counts(results, codes, n_groups, thresholds, sample=None):
    """
    Aceptación de la persona real (o-o) y de la grabación anonimizada (o-a) por grupo.

    Con `sample`, se usan las pruebas de cada copia de la muestra (un
    hablante repetido cuenta varias veces), como en impersonation.py.

    Retorna:
        Diccionario {"genuine_eer", "impersonation_eer", "genuine_far",
        "impersonation_far": (aceptados, pares)} con un valor por grupo.
    """
    owner = np.asarray(results["o-o"]["owner"])
    if sample is None:
        rows, speakers = np.arange(owner.size), owner
    else:
        rows, row_copy = copy_trial_rows(owner, results["o-o"]["scores"].shape[1], sample)
        speakers = np.asarray(sample)[row_copy]
    row_codes = codes[speakers]
    out = {}
    for point in POINTS:
        for rate, scenario in (("genuine", "o-o"), ("impersonation", "o-a")):
            out[f"{rate}_{point}"] = acceptance_by_group(row_targets(results[scenario])[rows], row_codes, n_groups,
                                                         thresholds[point])
    return out


def ratio(numerator, denominator):
    """Cociente elemento a elemento; NaN donde el denominador es 0."""
    numerator, denominator = np.asarray(numerator, float), np.asarray(denominator, float)
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator > 0)


def eer_by_group(result, codes, n_groups, sample=None):
    """
    EER dentro de cada grupo: objetivos e impostores de hablantes del grupo.

    Usa `resample_pairs` con los hablantes (o las copias de la muestra) del
    grupo, así que las copias de un mismo hablante no cuentan como
    impostores. NaN si el grupo no tiene al menos dos hablantes distintos.
    """
    sample = np.arange(codes.size) if sample is None else np.asarray(sample)
    out = np.full(n_groups, np.nan)
    for g in range(n_groups):
        members = sample[codes[sample] == g]
        if np.unique(members).size >= 2:
            out[g] = compute_eer(*resample_pairs(result["scores"], result["owner"], members))[0]
    return out


def wer_by_group(counts, codes, n_groups, sample=None):
    """
    WER de corpus por grupo: errores y palabras sumados entre los hablantes del grupo.

    Parámetros:
        counts: salida de `wer_bootstrap.speaker_counts`, alineada con `codes`.

    Retorna:
        Diccionario con "errors_original", "errors_anonymized", "words" (por
        grupo) y "original", "anonymized", "difference" (WER por grupo;
        diferencia = anonimizado − original).
    """
    index = np.arange(codes.size) if sample is None else np.asarray(sample)
    group = codes[index]
    valid = group >= 0

    def total(values):
        return np.bincount(group[valid], weights=np.asarray(values, float)[index][valid], minlength=n_groups)

    out = {"words": total(counts["words"]["original"])}
    for version in VERSIONS:
        out[f"errors_{version}"] = total(counts["errors"][version])
        out[version] = ratio(out[f"errors_{version}"], out["words"])
    out["difference"] = out["anonymized"] - out["original"]
    return out


def aligned_wer_counts(transcripts_path, client_ids):
    """Cuenta errores y palabras por hablante (normalización final) de un TSV, en el orden de `client_ids`."""
    return aligned_wer_counts_from_rows(read_tsv_rows(transcripts_path), client_ids, source=transcripts_path)


def aligned_wer_counts_from_rows(rows, client_ids, source="transcripciones"):
    """
    Cuenta errores y palabras por hablante con `wer_bootstrap.speaker_counts` y la normalización final.

    Reordena los hablantes al orden de `client_ids` (el de los embeddings),
    para que los mismos índices de muestra sirvan para el EER y el WER.
    Lanza `ValueError` si los hablantes no son los mismos.
    """
    counts = speaker_counts(rows, normalize_final)
    if counts["client_ids"] != list(client_ids):
        if set(counts["client_ids"]) != set(client_ids):
            raise ValueError(f"{source}: los hablantes no coinciden con los de los embeddings")
        position = {c: i for i, c in enumerate(counts["client_ids"])}
        order = np.array([position[c] for c in client_ids])
        counts = {**counts, "client_ids": list(client_ids), "strata": [counts["strata"][i] for i in order],
                  **{k: {v: counts[k][v][order] for v in VERSIONS} for k in ("errors", "words", "clips")}}
    return counts


# ---------------------------------------------------------------------------
# Pruebas de permutación y Holm
# ---------------------------------------------------------------------------

def group_rate_sd(numerator, denominator, codes, n_groups):
    """Desviación estándar (poblacional) entre grupos de suma(numerador) / suma(denominador)."""
    rates = ratio(np.bincount(codes, weights=numerator, minlength=n_groups),
                  np.bincount(codes, weights=denominator, minlength=n_groups))
    return float(np.std(rates))


def permutation_test(numerator, denominator, codes, n_perm, seed):
    """
    Prueba de permutación a nivel de hablante de la heterogeneidad entre grupos.

    Baraja las etiquetas de grupo entre hablantes (cada grupo conserva su
    tamaño) y compara la desviación estándar entre grupos de la métrica con
    la observada. p = (1 + n.º de permutaciones con estadístico >= observado)
    / (1 + n_perm).

    Parámetros:
        numerator, denominator: valor por hablante (por ejemplo pares o-a
            aceptados y pares totales); la métrica de un grupo es
            suma(numerador) / suma(denominador).
        codes: grupo de cada hablante (todos >= 0: los excluidos se quitan antes).
        n_perm: número de permutaciones.
        seed: semilla del generador.

    Retorna:
        Una tupla (estadístico observado, p).
    """
    numerator, denominator, codes = np.asarray(numerator, float), np.asarray(denominator, float), np.asarray(codes)
    n_groups = int(codes.max()) + 1
    observed = group_rate_sd(numerator, denominator, codes, n_groups)
    rng = np.random.default_rng(seed)
    exceed = sum(group_rate_sd(numerator, denominator, rng.permutation(codes), n_groups) >= observed - 1e-12
                 for _ in range(n_perm))
    return observed, (1 + exceed) / (1 + n_perm)


def holm(pvalues):
    """P ajustadas por Holm (paso a paso descendente), en el orden de entrada."""
    p = np.asarray(pvalues, float)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(np.argsort(p, kind="stable")):
        running = max(running, min(1.0, (p.size - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def adjust_by_metric(rows):
    """
    Agrega "p_holm" a cada fila: Holm entre las configuraciones de una misma métrica.

    Las filas son de una sola agrupación; cada métrica se corrige por
    separado. Modifica las filas y las devuelve.
    """
    for metric in dict.fromkeys(r["metric"] for r in rows):
        same = [r for r in rows if r["metric"] == metric]
        for row, adjusted in zip(same, holm([r["p_value"] for r in same])):
            row["p_holm"] = adjusted
    return rows


def test_codes(groups, codes, group_by):
    """
    Grupos que entran en la prueba de permutación.

    Estrato: todos menos sin_dato (no es un acento). Género: solo masculino
    y femenino. Retorna (grupos incluidos, códigos recodificados 0..k-1 o -1).
    """
    if group_by == "gender":
        keep = [g for g in ("masculino", "femenino") if g in groups]
    else:
        keep = [g for g in groups if g not in (MISSING, CONFLICT)]
    remap = np.full(len(groups) + 1, -1)
    for new, g in enumerate(keep):
        remap[groups.index(g)] = new
    return keep, np.where(codes >= 0, remap[codes], -1)


# ---------------------------------------------------------------------------
# Análisis de una configuración
# ---------------------------------------------------------------------------

def ci(values, level):
    """IC por percentiles ignorando los NaN; (NaN, NaN) si no hay valores."""
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    return percentile_ci(values, level) if values.size else (np.nan, np.nan)


def analyze_config(results, wer_counts, codes, n_groups, samples, thresholds_reps, max_far, oo_eer_cache=None,
                   eer_reps=True):
    """
    Métricas puntuales y réplicas de una configuración para un juego de grupos.

    Parámetros:
        results: salida de `run_scenarios` (o-o, o-a y a-a).
        wer_counts: salida de `aligned_wer_counts`, o None si no hay transcripciones.
        codes, n_groups: grupo de cada hablante (-1 excluido) y número de grupos.
        samples: muestras de `shared_samples`.
        thresholds_reps: umbrales globales de cada réplica.
        max_far: FAR máximo del segundo umbral.
        oo_eer_cache: diccionario opcional para reutilizar el EER o-o por
            grupo (no depende de α); la llave la arma quien llama.
        eer_reps: si es False, del EER solo se calcula el valor puntual
            (sus réplicas quedan en None); sirve para grupos grandes cuyo IC
            ya se conoce o no se necesita.

    Retorna:
        Diccionario con "thresholds", "impersonation" (punto y réplicas de
        cada tasa), "eer" (por escenario) y "wer" (o None).
    """
    thresholds = operating_thresholds(results["o-o"], max_far)
    point_counts = impersonation_counts(results, codes, n_groups, thresholds)
    impersonation = {key: {"point": ratio(*value), "pairs": value[1], "accepted": value[0],
                           "reps": np.empty((len(samples), n_groups))} for key, value in point_counts.items()}
    for b, sample in enumerate(samples):
        for key, value in impersonation_counts(results, codes, n_groups, thresholds_reps[b], sample).items():
            impersonation[key]["reps"][b] = ratio(*value)

    eer = {}
    for scenario in SCENARIOS:
        if not eer_reps:
            eer[scenario] = {"point": eer_by_group(results[scenario], codes, n_groups), "reps": None}
            continue
        if scenario == "o-o" and oo_eer_cache is not None and "o-o" in oo_eer_cache:
            eer[scenario] = oo_eer_cache["o-o"]
            continue
        eer[scenario] = {"point": eer_by_group(results[scenario], codes, n_groups),
                         "reps": np.array([eer_by_group(results[scenario], codes, n_groups, s) for s in samples])
                         .reshape(len(samples), n_groups)}
        if scenario == "o-o" and oo_eer_cache is not None:
            oo_eer_cache["o-o"] = eer[scenario]

    wer = None
    if wer_counts is not None:
        point = wer_by_group(wer_counts, codes, n_groups)
        reps = [wer_by_group(wer_counts, codes, n_groups, s) for s in samples]
        wer = {key: {"point": point[key], "reps": np.array([r[key] for r in reps]).reshape(len(samples), n_groups)}
               for key in ("original", "anonymized", "difference")}
        wer.update({key: point[key] for key in ("words", "errors_original", "errors_anonymized")})
    return {"thresholds": thresholds, "impersonation": impersonation, "eer": eer, "wer": wer}


def speaker_level(results, wer_counts, thresholds, n_speakers):
    """
    Valores por hablante para las pruebas de permutación.

    Retorna:
        Diccionario con (numerador, denominador) por hablante para
        "impersonation_far" (pares o-a aceptados con el umbral global FAR y
        pares totales) y, si hay transcripciones, "wer_difference"
        (errores anonimizado − original y palabras de referencia).
    """
    owner = np.asarray(results["o-a"]["owner"])
    accepted = (row_targets(results["o-a"]) >= thresholds["far"]).astype(float)
    out = {"impersonation_far": (np.bincount(owner, weights=accepted, minlength=n_speakers),
                                 np.bincount(owner, minlength=n_speakers).astype(float))}
    if wer_counts is not None:
        out["wer_difference"] = (wer_counts["errors"]["anonymized"] - wer_counts["errors"]["original"],
                                 wer_counts["words"]["original"])
    return out


# ---------------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------------

METRIC_COLUMNS = ["genuine_eer", "impersonation_eer", "genuine_far", "impersonation_far",
                  "eer_oo", "eer_oa", "eer_aa", "wer_original", "wer_anonymized", "wer_difference"]
SUMMARY_COLUMNS = (["config", "alpha_min", "alpha_max", "group", "n_speakers", "n_target_pairs",
                    "threshold_eer", "threshold_far"]
                   + [f"{m}{s}" for m in METRIC_COLUMNS for s in ("", "_ci_low", "_ci_high")]
                   + ["n_words", "wer_note"])
PERMUTATION_COLUMNS = ["grouping", "metric", "config", "groups", "n_speakers", "observed_sd", "n_permutations",
                       "p_value", "p_holm"]


def summary_rows(config, groups, codes, analysis, level, eer_override=None):
    """
    Filas de la tabla resumen de una configuración: una por grupo.

    `eer_override` permite dar el IC del EER de un grupo desde otro origen
    (el JSON de eer_bootstrap para el total, que usa exactamente las mismas
    réplicas) sin recalcularlo.
    """
    rows = []
    for g, group in enumerate(groups):
        row = {"config": config["tag"], "alpha_min": config["alpha_min"], "alpha_max": config["alpha_max"],
               "group": group, "n_speakers": int(np.sum(codes == g)),
               "n_target_pairs": int(analysis["impersonation"]["genuine_eer"]["pairs"][g]),
               "threshold_eer": analysis["thresholds"]["eer"], "threshold_far": analysis["thresholds"]["far"]}
        for key, values in analysis["impersonation"].items():
            row[key] = values["point"][g]
            row[f"{key}_ci_low"], row[f"{key}_ci_high"] = ci(values["reps"][:, g], level)
        for scenario, name in zip(SCENARIOS, ("eer_oo", "eer_oa", "eer_aa")):
            values = analysis["eer"][scenario]
            row[name] = values["point"][g]
            if eer_override and (group, scenario) in eer_override:
                row[f"{name}_ci_low"], row[f"{name}_ci_high"] = eer_override[(group, scenario)]
            elif values.get("reps") is not None:
                row[f"{name}_ci_low"], row[f"{name}_ci_high"] = ci(values["reps"][:, g], level)
            else:
                row[f"{name}_ci_low"] = row[f"{name}_ci_high"] = None
        wer = analysis["wer"]
        for key in ("original", "anonymized", "difference"):
            name = f"wer_{key}"
            if wer is None:
                row[name] = row[f"{name}_ci_low"] = row[f"{name}_ci_high"] = None
            else:
                row[name] = wer[key]["point"][g]
                row[f"{name}_ci_low"], row[f"{name}_ci_high"] = ci(wer[key]["reps"][:, g], level)
        row["n_words"] = int(wer["words"][g]) if wer is not None else None
        row["wer_note"] = "" if wer is None or wer_reliable(config) else HALLUCINATION_NOTE
        rows.append(row)
    return rows


def write_tsv(rows, columns, path):
    """Escribe una tabla TSV en UTF-8 (punto decimal; None y NaN como celda vacía)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def cell(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        return float(value) if isinstance(value, np.floating) else value

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([cell(row.get(c)) for c in columns])
    return path


def merge_permutation_rows(path, grouping, new_rows):
    """Reemplaza en `path` las filas de `grouping` por `new_rows` y conserva las de otras agrupaciones."""
    kept = []
    if Path(path).exists():
        kept = [r for r in read_tsv_rows(path) if r["grouping"] != grouping]
    return write_tsv(kept + new_rows, PERMUTATION_COLUMNS, path)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def config_label(config):
    """Rótulo de una configuración: "0,6" para un α fijo, "U(0,7; 0,9)" para un rango."""
    if config["alpha_min"] == config["alpha_max"]:
        return f"{config['alpha_min']:g}".replace(".", ",")
    return f"U({config['alpha_min']:g}; {config['alpha_max']:g})".replace(".", ",")


def heatmap_layout(groups, configs):
    """
    Orden de filas y columnas del mapa de calor, con None donde va un hueco.

    Filas: los acentos y, separado abajo, sin_dato. Columnas: los α fijos y,
    separados a la derecha, los rangos.
    """
    accents = [g for g in groups if g != MISSING]
    rows = accents + ([None, MISSING] if MISSING in groups else [])
    fixed = [c for c in configs if c["alpha_min"] == c["alpha_max"]]
    ranges = [c for c in configs if c["alpha_min"] != c["alpha_max"]]
    columns = fixed + ([None] + ranges if ranges and fixed else ranges)
    return rows, columns


def plot_heatmap(summary, groups, configs, metric, colorbar_label, cmap="Blues"):
    """
    Mapa de calor grupo × configuración de una métrica, con el valor en cada celda.

    Parámetros:
        summary: filas de `summary_rows`.
        groups: grupos (filas).
        configs: configuraciones (columnas), ya filtradas.
        metric: columna de la tabla (fracción); se muestra en porcentaje.
        colorbar_label: rótulo de la escala.
        cmap: escala secuencial de un solo tono.

    Retorna:
        Una `Figure` de matplotlib.
    """
    import matplotlib

    rows, columns = heatmap_layout(groups, configs)
    lookup = {(r["config"], r["group"]): r[metric] for r in summary}
    matrix = np.full((len(rows), len(columns)), np.nan)
    for i, group in enumerate(rows):
        for j, config in enumerate(columns):
            if group is not None and config is not None and lookup.get((config["tag"], group)) is not None:
                matrix[i, j] = 100 * lookup[(config["tag"], group)]
    # 1,4 columnas: con el ancho de una sola, los valores de celdas vecinas se pegan.
    fig = Figure(figsize=(HEATMAP_WIDTH, 0.24 * len(rows) + 1.3))
    ax = fig.subplots()
    colormap = matplotlib.colormaps[cmap].with_extremes(bad="white")
    values = matrix[~np.isnan(matrix)]
    vmin, vmax = (float(values.min()), float(values.max())) if values.size else (0.0, 1.0)
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=colormap, vmin=vmin, vmax=vmax, aspect="auto")
    middle = (vmin + vmax) / 2
    for i in range(len(rows)):
        for j in range(len(columns)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, decimal_comma(matrix[i, j], 1), ha="center", va="center", fontsize=5,
                        color="white" if matrix[i, j] > middle else INK)
    ax.set_xticks([j for j, c in enumerate(columns) if c is not None],
                  [config_label(c) for c in columns if c is not None], fontsize=SMALL_FONT - 0.5)
    ax.set_yticks([i for i, g in enumerate(rows) if g is not None],
                  [GROUP_LABELS.get(g, g) for g in rows if g is not None], fontsize=SMALL_FONT)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("α", fontsize=SMALL_FONT)
    bar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.02)
    bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: percent_label(round(v, 6))))
    bar.ax.tick_params(labelsize=SMALL_FONT - 0.5)
    bar.set_label(colorbar_label, fontsize=SMALL_FONT)
    fig.tight_layout()
    return fig


def plot_groups_vs_alpha(summary, configs, series):
    """
    Dos paneles frente a α: suplantación o-a con FAR <= max_far y diferencia de WER.

    Solo entran los α fijos; el panel del WER, solo α >= 0.6. Cada serie es
    una línea con sus intervalos (con un pequeño desplazamiento horizontal
    para que no se tapen).

    Parámetros:
        summary: filas de `summary_rows` (incluida la del total si se pide).
        configs: configuraciones.
        series: lista de (grupo, rótulo, color, estilo de línea).

    Retorna:
        Una `Figure` de matplotlib.
    """
    fixed = [c for c in configs if c["alpha_min"] == c["alpha_max"]]
    lookup = {(r["config"], r["group"]): r for r in summary}
    fig = Figure(figsize=(COLUMN_WIDTH, 3.6))
    axes = fig.subplots(2, 1, sharex=True)
    panels = (("impersonation_far", "Suplantación o-a\nFAR ≤ 1 % (%)", fixed),
              ("wer_difference", "Diferencia de WER\n(pp)", [c for c in fixed if wer_reliable(c)]))
    offsets = np.linspace(-0.012, 0.012, len(series)) if len(series) > 1 else [0.0]
    for ax, (metric, ylabel, cfgs) in zip(axes, panels):
        for (group, _, color, style), dx in zip(series, offsets):
            points = [(c["alpha_min"] + dx, lookup[(c["tag"], group)]) for c in cfgs
                      if (c["tag"], group) in lookup and lookup[(c["tag"], group)][metric] is not None]
            if not points:
                continue
            xs = [x for x, _ in points]
            ys = [100 * r[metric] for _, r in points]
            for x, r in points:
                ax.plot([x, x], [100 * r[f"{metric}_ci_low"], 100 * r[f"{metric}_ci_high"]], color=color,
                        linewidth=1.0, marker="_", markersize=5)
            ax.plot(xs, ys, color=color, linestyle=style, linewidth=1.2, marker="o", markersize=3.5,
                    markeredgewidth=0)
        ax.set_ylabel(ylabel, fontsize=SMALL_FONT)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: percent_label(round(v, 6))))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
        ax.grid(axis="y", color="0.9", linewidth=0.5)
    alphas = sorted({c["alpha_min"] for c in fixed})
    if alphas:
        axes[1].set_xticks(alphas)
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    axes[1].set_xlabel("α (coeficiente de McAdams)")
    handles = [Line2D([], [], color=color, linestyle=style, marker="o", markersize=3.5, linewidth=1.2, label=label)
               for _, label, color, style in series]
    fig.legend(handles=handles, loc="lower center", ncol=len(series), frameon=False, fontsize=SMALL_FONT,
               labelcolor=INK)
    fig.tight_layout(rect=(0, 0.25 / fig.get_figheight(), 1, 1))
    return fig


# ---------------------------------------------------------------------------
# Programa
# ---------------------------------------------------------------------------

def grouping_name(group_by, only_gender):
    """Sufijo de los archivos: estrato, genero, estrato_solo_masculino, ..."""
    base = {"stratum": "estrato", "gender": "genero", "all": "total"}[group_by]
    return base if only_gender is None else f"{base}_solo_{only_gender}"


def load_eer_ci(path, level):
    """IC del EER de cada escenario desde el JSON de eer_bootstrap, si existe y tiene el mismo nivel."""
    if not Path(path).exists():
        return None
    with open(path, encoding="utf-8-sig") as f:
        summary = json.load(f)
    if summary.get("level") != level:
        return None
    return {s: (v["ci_low"], v["ci_high"]) for s, v in summary["scenarios"].items()}


def parse_args(argv=None):
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description="Privacidad e inteligibilidad por estrato o género, por configuración.")
    parser.add_argument("--group-by", choices=("stratum", "gender", "all"), required=True)
    parser.add_argument("--only-gender", default=None, help="Analiza solo los hablantes de ese género (p. ej. masculino).")
    parser.add_argument("--data-dir", default=str(REPO.parent / "Datos"))
    parser.add_argument("--output-dir", default=None, help="Por defecto <data-dir>\\por_grupo.")
    parser.add_argument("--figures-dir", default=None, help="Por defecto <data-dir>\\figuras\\por_grupo.")
    parser.add_argument("--configs", nargs="+", default=None, help="Solo estas etiquetas (por defecto todas las que existan).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-enroll", type=int, default=3)
    parser.add_argument("--n-trial", type=int, default=2)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--boot-seed", type=int, default=2026)
    parser.add_argument("--level", type=float, default=0.95)
    parser.add_argument("--max-far", type=float, default=0.01)
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--perm-seed", type=int, default=2026)
    parser.add_argument("--figures-only", action="store_true",
                        help="Solo redibuja las figuras desde la tabla resumen ya escrita.")
    args = parser.parse_args(argv)
    data = Path(args.data_dir)
    args.output_dir = args.output_dir or str(data / "por_grupo")
    args.figures_dir = args.figures_dir or str(data / "figuras" / "por_grupo")
    return args


def main():
    """Corre el análisis por grupo de todas las configuraciones y guarda tablas, pruebas y figuras."""
    args = parse_args()
    if args.figures_only:
        redraw_figures(args)
        return
    start_all = time.perf_counter()
    configs = discover_configs(args.data_dir)
    if args.configs:
        configs = [c for c in configs if c["tag"] in args.configs]
    if not configs:
        raise SystemExit("No se encontró ninguna configuración con manifiesto y embeddings.")
    name = grouping_name(args.group_by, args.only_gender)
    print(f"Análisis: {name}; configuraciones: {', '.join(c['tag'] for c in configs)}", flush=True)

    first_splits, _, first = load_and_score(configs[0]["files"]["embeddings"], configs[0]["files"]["manifest"],
                                            args.seed, args.n_enroll, args.n_trial)
    client_ids = list(first_splits)
    strata = [s["stratum"] for s in first_splits.values()]
    validated = Path(args.data_dir) / CLIPS_SUBDIR.parent / "validated.tsv"
    genders = load_genders(validated, client_ids)
    counts = {g: list(genders.values()).count(g) for g in sorted(set(genders.values()))}
    print(f"Género por hablante: {counts}", flush=True)

    labels = assign_groups(strata, [genders[c] for c in client_ids], args.group_by, args.only_gender)
    groups, codes = group_codes(labels)
    total_codes = np.where(codes >= 0, 0, -1)
    test_groups, perm_codes = test_codes(groups, codes, args.group_by)

    start = time.perf_counter()
    samples = shared_samples(strata, args.n_boot, args.boot_seed)
    thresholds_reps = [operating_thresholds(first["o-o"], args.max_far, s) for s in samples]
    print(f"Umbrales globales de {args.n_boot} réplicas: {time.perf_counter() - start:.0f} s", flush=True)

    summary, permutation_rows, oo_cache, oo_total_cache = [], [], {}, {}
    for config in configs:
        start = time.perf_counter()
        splits, _, results = load_and_score(config["files"]["embeddings"], config["files"]["manifest"],
                                            args.seed, args.n_enroll, args.n_trial)
        if splits != first_splits:
            raise ValueError(f"{config['tag']}: la partición registro/prueba no coincide con la de {configs[0]['tag']}")
        same_oo = np.array_equal(results["o-o"]["scores"], first["o-o"]["scores"])
        reps = thresholds_reps if same_oo else [operating_thresholds(results["o-o"], args.max_far, s) for s in samples]
        transcripts = config["files"]["transcripts"]
        wer_counts = aligned_wer_counts(transcripts, client_ids) if Path(transcripts).exists() else None

        analysis = analyze_config(results, wer_counts, codes, len(groups), samples, reps, args.max_far,
                                  oo_cache if same_oo else None)
        summary += summary_rows(config, groups, codes, analysis, args.level)
        summary_done = args.group_by == "all"  # el único grupo ya es el total

        # Total. Si son todos los hablantes, el IC del EER es el del JSON de eer_bootstrap, que usa
        # exactamente las mismas réplicas; si no hay JSON, se calcula. Si el total es un subconjunto
        # (--only-gender), su EER va sin IC: es un grupo grande y no es el objeto del análisis.
        everyone = bool(np.all(total_codes == 0))
        json_ci = load_eer_ci(config["files"]["eer_json"], args.level) if everyone else None
        compute_total_eer = everyone and json_ci is None
        if not summary_done:
            total = analyze_config(results, wer_counts, total_codes, 1, samples, reps, args.max_far,
                                   oo_total_cache if same_oo and compute_total_eer else None,
                                   eer_reps=compute_total_eer)
            override = {(TOTAL, s): json_ci[s] for s in SCENARIOS} if json_ci else None
            summary += summary_rows(config, [TOTAL], total_codes, total, args.level, override)

        included = perm_codes >= 0
        values = speaker_level(results, wer_counts, analysis["thresholds"], len(client_ids))
        for metric, (num, den) in values.items():
            if len(test_groups) < 2 or (metric == "wer_difference" and not wer_reliable(config)):
                continue
            observed, p = permutation_test(num[included], den[included], perm_codes[included], args.n_perm,
                                           args.perm_seed)
            permutation_rows.append({"grouping": name, "metric": metric, "config": config["tag"],
                                     "groups": ",".join(test_groups), "n_speakers": int(included.sum()),
                                     "observed_sd": observed, "n_permutations": args.n_perm, "p_value": p})
        print(f"[{config['tag']}] {time.perf_counter() - start:.0f} s", flush=True)

    adjust_by_metric(permutation_rows)

    output_dir, figures_dir = Path(args.output_dir), Path(args.figures_dir)
    summary_path = write_tsv(summary, SUMMARY_COLUMNS, output_dir / f"resumen_{name}.tsv")
    permutation_path = merge_permutation_rows(output_dir / "pruebas_permutacion.tsv", name, permutation_rows)
    print(f"Tabla: {summary_path}\nPruebas: {permutation_path}", flush=True)

    save_figures(summary, groups, configs, args.group_by, args.only_gender, figures_dir)
    print(f"Tiempo total: {time.perf_counter() - start_all:.0f} s")


def save_figures(summary, groups, configs, group_by, only_gender, figures_dir):
    """
    Guarda las figuras de un análisis: los mapas de calor y colombia.pdf por
    estrato, genero.pdf por género; el análisis con --only-gender no tiene
    figuras.
    """
    configure_fonts()
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    if group_by == "stratum" and only_gender is None:
        plot_heatmap(summary, groups, configs, "impersonation_far",
                     "Suplantación o-a, FAR ≤ 1 % (%)").savefig(figures_dir / "suplantacion_estrato.pdf")
        plot_heatmap(summary, groups, [c for c in configs if wer_reliable(c)], "wer_difference",
                     "Diferencia de WER (pp)").savefig(figures_dir / "wer_estrato.pdf")
        plot_groups_vs_alpha(summary, configs, [
            ("andino_pacifico", "Andino-pacífico", "#0072B2", "-"), ("caribe", "Caribe", "#D55E00", "--"),
            (TOTAL, "Total", "#000000", ":")]).savefig(figures_dir / "colombia.pdf")
        print(f"Figuras: suplantacion_estrato.pdf, wer_estrato.pdf, colombia.pdf en {figures_dir}")
    elif group_by == "gender" and only_gender is None:
        plot_groups_vs_alpha(summary, configs, [("masculino", "Masculino", "#009E73", "-"),
                                                ("femenino", "Femenino", "#CC79A7", "--")]
                             ).savefig(figures_dir / "genero.pdf")
        print(f"Figura: genero.pdf en {figures_dir}")


def load_summary(path):
    """Lee una tabla resumen escrita por `write_tsv` (números como float; celdas vacías como None)."""
    rows = []
    for row in read_tsv_rows(path):
        rows.append({k: v if k in ("config", "group", "wer_note") else (float(v) if v != "" else None)
                     for k, v in row.items()})
    return rows


def redraw_figures(args):
    """Redibuja las figuras de un análisis desde su tabla resumen, sin recalcular nada."""
    name = grouping_name(args.group_by, args.only_gender)
    summary = load_summary(Path(args.output_dir) / f"resumen_{name}.tsv")
    tags = list(dict.fromkeys(r["config"] for r in summary))
    configs = [c for c in discover_configs(args.data_dir) if c["tag"] in tags]
    groups = [g for g in dict.fromkeys(r["group"] for r in summary) if g != TOTAL]
    save_figures(summary, groups, configs, args.group_by, args.only_gender, args.figures_dir)


if __name__ == "__main__":
    main()
