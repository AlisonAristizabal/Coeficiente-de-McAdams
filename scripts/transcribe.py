"""
Transcripción de los clips originales y anonimizados con Whisper.

Sirve para medir cuánto se degrada el contenido lingüístico con la
anonimización (WER de la versión anonimizada frente a la original). Este
paso mide tiempos y un WER provisional con una muestra pequeña, antes de
decidir el tamaño del modelo y si se transcribe todo.

Cada WAV se carga con soundfile y se le pasa a Whisper como arreglo, no como
ruta, para no depender de ffmpeg. Los resultados se escriben fila a fila en
un TSV; si la corrida se interrumpe, al volver a lanzarla con el mismo
--output se saltan los clips ya transcritos.
"""
import argparse
import csv
import random
import re
import time
import unicodedata
from pathlib import Path

import soundfile

from common_voice_corpus import read_tsv_rows

TARGET_SAMPLE_RATE = 16000

# Opciones fijas de decodificación: español, sin traducir, decodificación
# voraz (temperatura 0, sin reintentos aleatorios), en float32 porque se
# corre en CPU, y sin condicionar en el texto previo (cada clip es una
# oración independiente).
DECODE_OPTIONS = {
    "language": "es",
    "task": "transcribe",
    "temperature": 0.0,
    "fp16": False,
    "condition_on_previous_text": False,
}

VERSIONS = ("original", "anonymized")
OUTPUT_COLUMNS = ["session_id", "client_id", "stratum", "version", "reference", "hypothesis", "seconds"]


def load_model(name, model_dir):
    """
    Carga un modelo de Whisper en CPU.

    El import de whisper está dentro de la función para poder importar este
    módulo (y probarlo) sin esa biblioteca.

    Parámetros:
        name: nombre del modelo, por ejemplo "small", "medium" o "turbo".
        model_dir: carpeta donde se descarga o ya está el modelo.

    Retorna:
        El modelo de Whisper.
    """
    import whisper
    return whisper.load_model(name, device="cpu", download_root=str(model_dir))


def load_audio(wav_path):
    """
    Carga un WAV como arreglo float32 mono de 16 kHz.

    Lanza `ValueError` si el audio no es mono o no está a 16 kHz, igual que
    `extract_embeddings.embed_clip`: Whisper espera exactamente ese formato
    y aquí no se remuestrea.

    Retorna:
        Arreglo float32 de una dimensión.
    """
    signal, sample_rate = soundfile.read(wav_path, dtype="float32")
    if sample_rate != TARGET_SAMPLE_RATE or signal.ndim != 1:
        raise ValueError(f"{wav_path}: se esperaba audio mono a {TARGET_SAMPLE_RATE} Hz")
    return signal


def transcribe_clip(model, wav_path):
    """
    Transcribe un WAV con las opciones fijas de `DECODE_OPTIONS`.

    Retorna:
        El texto transcrito, sin espacios al inicio ni al final.
    """
    result = model.transcribe(load_audio(wav_path), **DECODE_OPTIONS)
    return result["text"].strip()


def load_references(subset_rows):
    """
    Reúne la oración de referencia de cada clip del subconjunto.

    Parámetros:
        subset_rows: filas de `subset_k134.tsv`, con `path` y `sentence`.

    Retorna:
        Diccionario {path: sentence}. Lanza `ValueError` si un `path` se
        repite, porque la referencia sería ambigua.
    """
    references = {}
    for row in subset_rows:
        if row["path"] in references:
            raise ValueError(f"path repetido en el subconjunto: {row['path']}")
        references[row["path"]] = row["sentence"]
    return references


def build_jobs(manifest_rows, references):
    """
    Une el manifiesto con las referencias por la columna `path`.

    Solo usa las filas con estado "ok" (las que tienen WAV anonimizado).
    Lanza `ValueError` si algún clip no tiene referencia.

    Parámetros:
        manifest_rows: filas del manifiesto de `anonymize_subset`.
        references: salida de `load_references`.

    Retorna:
        Lista de diccionarios {"session_id", "client_id", "stratum",
        "reference"}, en el orden del manifiesto.
    """
    jobs = []
    for row in manifest_rows:
        if row["status"] != "ok":
            continue
        if row["path"] not in references:
            raise ValueError(f"el clip {row['path']} no está en el subconjunto")
        jobs.append({
            "session_id": row["session_id"],
            "client_id": row["client_id"],
            "stratum": row["stratum"],
            "reference": references[row["path"]],
        })
    return jobs


def select_sample(rows, per_stratum, seed):
    """
    Elige de forma reproducible `per_stratum` clips por estrato.

    En cada estrato se mezclan los hablantes (ordenados por `client_id`
    antes de mezclar, para no depender del orden de entrada) con un
    generador sembrado por `seed` y el estrato, y se toman los primeros
    `per_stratum`; de cada uno se elige un clip, también con una semilla
    propia. Así hay a lo sumo un clip por hablante y la muestra cubre más
    voces. Lanza `ValueError` si un estrato tiene menos hablantes que
    `per_stratum`.

    Parámetros:
        rows: diccionarios con al menos `session_id`, `client_id` y
            `stratum` (por ejemplo, la salida de `build_jobs`).
        per_stratum: clips por estrato.
        seed: entero que siembra la elección.

    Retorna:
        Lista de filas de `rows`, ordenada por estrato y `session_id`.
    """
    by_stratum = {}
    for row in rows:
        by_stratum.setdefault(row["stratum"], {}).setdefault(row["client_id"], []).append(row)
    chosen = []
    for stratum, speakers in by_stratum.items():
        if len(speakers) < per_stratum:
            raise ValueError(f"el estrato {stratum} tiene {len(speakers)} hablantes, menos de {per_stratum}")
        order = sorted(speakers)
        random.Random(f"{seed}-sample-{stratum}").shuffle(order)
        for client_id in order[:per_stratum]:
            clips = sorted(speakers[client_id], key=lambda r: r["session_id"])
            chosen.append(random.Random(f"{seed}-clip-{client_id}").choice(clips))
    return sorted(chosen, key=lambda r: (r["stratum"], r["session_id"]))


def transcribe_rows(jobs, original_dir, anon_dir, transcribe_fn, done=frozenset(), clock=time.perf_counter):
    """
    Transcribe la versión original y la anonimizada de cada clip.

    Es un generador: entrega cada fila apenas está lista, para que quien la
    consuma la guarde enseguida. Busca `<session_id>.wav` en `original_dir`
    y en `anon_dir`, como `extract_embeddings`. Las parejas (session_id,
    versión) que estén en `done` se saltan.

    Parámetros:
        jobs: salida de `build_jobs` o de `select_sample`.
        original_dir: carpeta con los WAV originales.
        anon_dir: carpeta con los WAV anonimizados.
        transcribe_fn: función que recibe una ruta de WAV y devuelve el texto.
        done: conjunto de parejas (session_id, versión) ya transcritas.
        clock: función de tiempo en segundos (inyectable para las pruebas).

    Retorna:
        Un generador de diccionarios con las columnas de `OUTPUT_COLUMNS`.
    """
    directories = {"original": Path(original_dir), "anonymized": Path(anon_dir)}
    for job in jobs:
        for version in VERSIONS:
            if (job["session_id"], version) in done:
                continue
            start = clock()
            hypothesis = transcribe_fn(directories[version] / f"{job['session_id']}.wav")
            seconds = clock() - start
            yield {
                "session_id": job["session_id"],
                "client_id": job["client_id"],
                "stratum": job["stratum"],
                "version": version,
                "reference": job["reference"],
                "hypothesis": hypothesis,
                "seconds": seconds,
            }


def single_line(text):
    """Reemplaza tabulaciones y saltos de línea por espacios para que el TSV no se rompa."""
    return " ".join(str(text).split())


def prepare_output(output_path):
    """
    Deja listo el TSV de salida para seguir escribiendo y dice qué ya está hecho.

    Si el archivo no existe o está vacío, no hace nada. Si existe y su última
    línea quedó a medias (la corrida se cortó mientras escribía), la borra.

    Retorna:
        Conjunto de parejas (session_id, versión) que ya tienen fila completa.
    """
    path = Path(output_path)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    data = path.read_bytes()
    if not data.endswith(b"\n"):
        path.write_bytes(data[:data.rfind(b"\n") + 1])
    return {(row["session_id"], row["version"]) for row in read_tsv_rows(path)
            if row.get("seconds") not in (None, "")}


def append_rows(rows, output_path, progress_every=10):
    """
    Escribe las filas en el TSV a medida que llegan.

    Escribe el encabezado solo si el archivo está vacío o no existe, y vacía
    el búfer después de cada fila, de modo que lo ya escrito no se pierde si
    el proceso se interrumpe. Las columnas de texto se pasan a una sola línea.

    Parámetros:
        rows: iterable de diccionarios con las columnas de `OUTPUT_COLUMNS`.
        output_path: ruta del TSV.
        progress_every: cada cuántas filas se imprime el avance; 0 o None
            para no imprimir.

    Retorna:
        El número de filas escritas.
    """
    path = Path(output_path)
    new_file = not path.exists() or path.stat().st_size == 0
    written = 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None, lineterminator="\n")
        if new_file:
            writer.writerow(OUTPUT_COLUMNS)
        for row in rows:
            values = [single_line(row[c]) for c in OUTPUT_COLUMNS[:-1]] + [f"{row['seconds']:.3f}"]
            writer.writerow(values)
            f.flush()
            written += 1
            if progress_every and written % progress_every == 0:
                print(f"{written} transcripciones", flush=True)
    return written


def normalize_provisional(text):
    """
    Normalización provisional para el WER de esta medición.

    Compone los caracteres (NFC, para que "é" escrita de dos formas cuente
    igual), pasa a minúsculas, cambia cada signo de puntuación por un
    espacio y une los espacios. Conserva tildes, ñ, diéresis y cifras. La
    normalización definitiva se decide después.

    Retorna:
        El texto normalizado.
    """
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return " ".join(text.split())


def normalize_final(text):
    """
    Normalización definitiva: la provisional más los números en palabras.

    Antes de quitar la puntuación, cada secuencia de dígitos se escribe en
    palabras en español con `num2words` ("14" → "catorce", "100" → "cien"),
    rodeada de espacios para que no se pegue a lo que tenga al lado. Así no
    cuenta como error que Whisper escriba "4" donde la referencia dice
    "cuatro". Los dígitos separados por puntuación se convierten por
    separado ("3,5" → "tres cinco"). El import de num2words está dentro de
    la función para poder importar este módulo sin esa biblioteca.

    Retorna:
        El texto normalizado.
    """
    from num2words import num2words

    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[0-9]+", lambda m: f" {num2words(int(m.group()), lang='es')} ", text)
    return normalize_provisional(text)


def edit_distance(reference, hypothesis):
    """
    Distancia de edición por palabras (Levenshtein).

    Cuenta el mínimo de sustituciones, borrados e inserciones de palabras,
    cada una con costo 1, para convertir `reference` en `hypothesis`. Usa
    programación dinámica guardando solo la fila anterior.

    Parámetros:
        reference: lista de palabras de la referencia.
        hypothesis: lista de palabras de la hipótesis.

    Retorna:
        El número de errores (entero).
    """
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, start=1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis, start=1):
            current.append(min(
                previous[j] + 1,                             # borrado
                current[j - 1] + 1,                          # inserción
                previous[j - 1] + (ref_word != hyp_word),    # sustitución o acierto
            ))
        previous = current
    return previous[-1]


def corpus_wer(references, hypotheses, normalize=normalize_provisional):
    """
    WER de corpus: errores totales sobre palabras de referencia totales.

    No es el promedio de los WER de cada oración: así una oración corta no
    pesa lo mismo que una larga. Lanza `ValueError` si las listas tienen
    distinto largo o si no hay palabras de referencia.

    Parámetros:
        references: textos de referencia.
        hypotheses: textos transcritos, en el mismo orden.
        normalize: función de normalización que se aplica a ambos.

    Retorna:
        El WER como fracción (puede pasar de 1 si hay muchas inserciones).
    """
    if len(references) != len(hypotheses):
        raise ValueError("references e hypotheses deben tener el mismo largo")
    errors = words = 0
    for reference, hypothesis in zip(references, hypotheses):
        ref_words = normalize(reference).split()
        errors += edit_distance(ref_words, normalize(hypothesis).split())
        words += len(ref_words)
    if words == 0:
        raise ValueError("no hay palabras de referencia")
    return errors / words


def summarize_rows(rows):
    """
    Resume un TSV de transcripciones por versión.

    Parámetros:
        rows: filas del TSV (diccionarios con `version`, `reference`,
            `hypothesis` y `seconds`).

    Retorna:
        Diccionario {versión: {"clips", "seconds_per_clip", "wer"}}.
    """
    by_version = {}
    for row in rows:
        by_version.setdefault(row["version"], []).append(row)
    return {
        version: {
            "clips": len(items),
            "seconds_per_clip": sum(float(r["seconds"]) for r in items) / len(items),
            "wer": corpus_wer([r["reference"] for r in items], [r["hypothesis"] for r in items]),
        }
        for version, items in by_version.items()
    }


def print_summary(paths):
    """Imprime, por archivo, los segundos por clip y el WER provisional de cada versión."""
    for path in paths:
        summary = summarize_rows(read_tsv_rows(path))
        print(Path(path).name)
        for version in VERSIONS:
            if version in summary:
                s = summary[version]
                print(f"  {version}: {s['clips']} clips, {s['seconds_per_clip']:.2f} s por clip, "
                      f"WER provisional {100 * s['wer']:.2f} %".replace(".", ","))


def reuse_original_rows(source_rows, jobs):
    """
    Toma las transcripciones originales de un TSV ya hecho para la misma muestra.

    La versión original no depende de la anonimización, así que no hace
    falta volver a transcribirla. Verifica que la muestra sea exactamente la
    misma: el conjunto de parejas (session_id, referencia) de las filas
    "original" del TSV debe ser igual al de `jobs`. Si no, lanza
    `ValueError` con cuántos clips faltan o sobran.

    Parámetros:
        source_rows: filas de un TSV de `transcribe.py` (por ejemplo
            whisper_turbo_k134.tsv).
        jobs: la muestra a transcribir (salida de `select_sample` o de
            `build_jobs`).

    Retorna:
        Lista de filas "original" con las columnas de `OUTPUT_COLUMNS`, en
        el orden de `jobs`, con la hipótesis y los segundos del TSV.
    """
    source = {}
    for row in source_rows:
        if row["version"] != "original":
            continue
        if row["session_id"] in source:
            raise ValueError(f"el TSV reutilizado repite el clip {row['session_id']} en la versión original")
        source[row["session_id"]] = row
    expected = {(job["session_id"], single_line(job["reference"])) for job in jobs}
    found = {(session_id, row["reference"]) for session_id, row in source.items()}
    if expected != found:
        missing, extra = sorted(expected - found), sorted(found - expected)
        example = (missing or extra)[0][0]
        raise ValueError(
            f"la muestra no coincide con la del TSV reutilizado: {len(missing)} clips de la muestra faltan "
            f"o tienen otra referencia y {len(extra)} del TSV no están en la muestra (p. ej. {example}); "
            "revise --per-stratum y --seed"
        )
    return [{
        "session_id": job["session_id"],
        "client_id": job["client_id"],
        "stratum": job["stratum"],
        "version": "original",
        "reference": job["reference"],
        "hypothesis": source[job["session_id"]]["hypothesis"],
        "seconds": float(source[job["session_id"]]["seconds"]),
    } for job in jobs]


def parse_args():
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description="Transcribe los clips originales y anonimizados con Whisper.")
    parser.add_argument("--manifest", required=True, help="Manifiesto TSV generado por anonymize_subset.")
    parser.add_argument("--subset", required=True, help="subset_k134.tsv, con la oración de referencia (sentence).")
    parser.add_argument("--original-dir", required=True, help="Carpeta con los WAV originales.")
    parser.add_argument("--anon-dir", required=True, help="Carpeta con los WAV anonimizados.")
    parser.add_argument("--model", required=True, help="Modelo de Whisper: small, medium, turbo, ...")
    parser.add_argument("--model-dir", required=True, help="Carpeta donde se descarga o está el modelo.")
    parser.add_argument("--per-stratum", type=int, default=None, help="Clips por estrato; si se omite, todos.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de la muestra.")
    parser.add_argument("--output", required=True, help="TSV de salida (se continúa si ya existe).")
    parser.add_argument("--reuse-original-from", default=None,
                        help="TSV de transcribe.py del que se copian las filas originales (solo se transcribe el anonimizado).")
    return parser.parse_args()


def main():
    """Transcribe la muestra (o todo) y guarda el TSV fila a fila."""
    args = parse_args()
    jobs = build_jobs(read_tsv_rows(args.manifest), load_references(read_tsv_rows(args.subset)))
    if args.per_stratum is not None:
        jobs = select_sample(jobs, args.per_stratum, args.seed)
    done = prepare_output(args.output)
    if args.reuse_original_from:
        # Se verifica y copia antes de cargar Whisper, para fallar rápido si la muestra no coincide.
        reused = reuse_original_rows(read_tsv_rows(args.reuse_original_from), jobs)
        copied = append_rows([r for r in reused if (r["session_id"], "original") not in done], args.output,
                             progress_every=0)
        done = done | {(r["session_id"], "original") for r in reused}
        print(f"Originales reutilizados de {args.reuse_original_from}: {len(reused)} (copiados ahora: {copied})")
    pending = sum((job["session_id"], v) not in done for job in jobs for v in VERSIONS)
    print(f"Clips: {len(jobs)}; transcripciones pendientes: {pending} (ya hechas: {2 * len(jobs) - pending})")
    if pending == 0:
        print(f"Nada pendiente; no se carga el modelo. Archivo: {args.output}")
        return

    start = time.perf_counter()
    model = load_model(args.model, args.model_dir)
    print(f"Modelo {args.model} cargado en {time.perf_counter() - start:.1f} s")

    start = time.perf_counter()
    rows = transcribe_rows(jobs, args.original_dir, args.anon_dir, lambda path: transcribe_clip(model, path), done)
    written = append_rows(rows, args.output)
    elapsed = time.perf_counter() - start
    print(f"Transcripciones nuevas: {written} en {elapsed:.1f} s "
          f"({elapsed / max(written, 1):.2f} s por transcripción)".replace(".", ","))
    print(f"Guardado en: {args.output}")


if __name__ == "__main__":
    main()
