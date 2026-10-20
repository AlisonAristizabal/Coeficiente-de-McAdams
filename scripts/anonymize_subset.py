from pathlib import Path

import soundfile
import csv
import argparse

from alpha_config import DEFAULT_RANGE, config_tag, validate_range
from common_voice_corpus import read_tsv_rows
from mcadams_core.batch.utterance import anonymize_utterance
from mcadams_core.io.audio_conversion import convert_mp3_to_wav
from mcadams_core.io.audio_io import load_wav, save_wav


def process_clip(mp3_path, original_path, anon_path, session_id, alpha_min=None, alpha_max=None,
                 reuse_original=False):
    """
    Convierte un clip, lo anonimiza y guarda ambas versiones.

    Convierte `mp3_path` a WAV de 16 kHz en `original_path`, lo lee de
    vuelta, lo anonimiza con el coeficiente derivado de `session_id` y
    guarda el resultado en `anon_path`. La versión original y la
    anonimizada pasan por la misma conversión, de modo que solo difieren
    en la anonimización. Con `reuse_original`, si `original_path` ya
    existe se lee tal cual y no se vuelve a convertir el MP3 (el archivo
    no se reescribe); como el anonimizado siempre parte de leer ese WAV,
    el resultado es idéntico al de convertir de nuevo. Las excepciones de
    archivo no se capturan aquí: las maneja quien llama.

    Parámetros:
        mp3_path: ruta del MP3 de entrada.
        original_path: ruta del WAV sin anonimizar.
        anon_path: ruta del WAV anonimizado.
        session_id: identificador que fija el coeficiente de McAdams.
        alpha_min, alpha_max: rango del coeficiente (iguales para un α
            fijo); None usa el rango por defecto.
        reuse_original: reutilizar `original_path` si ya existe.

    Retorna:
        El coeficiente de McAdams usado.
    """
    if not (reuse_original and Path(original_path).exists()):
        convert_mp3_to_wav(mp3_path, original_path)
    sample_rate, audio = load_wav(original_path)
    anon_audio, coefficient = anonymize_utterance(audio, sample_rate, session_id, alpha_min, alpha_max)
    save_wav(anon_path, sample_rate, anon_audio)
    return coefficient


def anonymize_subset(rows, clips_dir, original_dir, anon_dir, alpha_min=None, alpha_max=None,
                     reuse_original=False):
    """
    Procesa todos los clips del subconjunto y arma el manifiesto.

    Para cada fila de `rows` (las del `subset.tsv`) busca el MP3 en
    `clips_dir` y lo procesa con `process_clip`, usando como
    `session_id` el nombre del clip sin extensión. Si un clip falla por
    un problema de datos (archivo faltante o MP3 ilegible), no se
    detiene: registra el fallo en el manifiesto y sigue. Otros errores
    se propagan, porque indicarían un defecto del programa.

    Parámetros:
        rows: iterable de diccionarios con `client_id`, `stratum` y
            `path` (nombre del MP3, con extensión).
        clips_dir, original_dir, anon_dir: carpetas de entrada y de
            salida; las de salida se crean si no existen.
        alpha_min, alpha_max, reuse_original: se pasan a `process_clip`.

    Retorna:
        Lista de diccionarios con las llaves `client_id`, `stratum`,
        `path`, `session_id`, `coefficient`, `status` y `error`.
    """
    clips_dir, original_dir, anon_dir = Path(clips_dir), Path(original_dir), Path(anon_dir)
    original_dir.mkdir(parents=True, exist_ok=True)
    anon_dir.mkdir(parents=True, exist_ok=True)

    # Solo se pasan las opciones que se dieron, para que la llamada por defecto
    # sea la misma de siempre.
    clip_options = {}
    if alpha_min is not None or alpha_max is not None:
        clip_options.update(alpha_min=alpha_min, alpha_max=alpha_max)
    if reuse_original:
        clip_options["reuse_original"] = True

    manifest = []
    for row in rows:
        stem = Path(row["path"]).stem      
        record = {
            "client_id": row["client_id"],
            "stratum": row["stratum"],
            "path": row["path"],
            "session_id": stem,
            "coefficient": "",
            "status": "ok",
            "error": "",
        }
        try:
            record["coefficient"] = process_clip(
                clips_dir / row["path"],
                original_dir / f"{stem}.wav",
                anon_dir / f"{stem}.wav",                      
                stem,
                **clip_options,
            )
        except (soundfile.LibsndfileError) as e:           
            record["status"] = "error"
            record["error"] = f"{type(e).__name__}: {' '.join(str(e).split())}"        
        manifest.append(record)
    return manifest


MANIFEST_COLUMNS = [
    "client_id", "stratum", "path", "session_id", "coefficient", "status", "error",
]


def write_manifest(manifest, output_path):
    """
    Escribe el manifiesto de anonimización en un archivo TSV.

    Parámetros:
        manifest: iterable de diccionarios con las llaves de
            `MANIFEST_COLUMNS`, como los que devuelve `anonymize_subset`.
        output_path: ruta del archivo TSV de salida.

    Retorna:
        La ruta del archivo escrito.

    Lanza:
        ValueError: si algún campo contiene tabulaciones o saltos de línea.
            Se valida antes de abrir el archivo, así que no queda uno parcial.
        KeyError: si algún registro no tiene alguna de las llaves.
    """
    rows = []
    for record in manifest:
        values = [record[col] for col in MANIFEST_COLUMNS]
        for col, value in zip(MANIFEST_COLUMNS, values):
            if any(ch in str(value) for ch in ["\t", "\n", "\r"]):
                raise ValueError(
                    f"Campo '{col}' con tabulación o salto de línea en {record['path']}"
                )
        rows.append(values)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(
            f, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None, lineterminator="\n"
        )
        writer.writerow(MANIFEST_COLUMNS)
        writer.writerows(rows)

    return output_path


def parse_args():
    """
    Lee los argumentos de la línea de comandos.

    --alpha y --alpha-range son mutuamente excluyentes; sin ninguno se usa
    el rango por defecto U(0.7, 0.9). Deja en `args.alpha_min` y
    `args.alpha_max` el rango elegido (None si no se dio ninguno). Si no se
    da --tag, la etiqueta sale del rango (`u07_09` por defecto, `a020` para
    --alpha 0.2), de modo que un α nuevo nunca escribe sobre la carpeta ni
    el manifiesto del rango por defecto.
    """
    parser = argparse.ArgumentParser(description="Anonimiza un subconjunto de clips de Common Voice y genera un manifiesto TSV.")
    parser.add_argument("--subset", required=True, help="Ruta al archivo TSV que contiene el subconjunto de clips a anonimizar.")
    parser.add_argument("--clips-dir", required=True, help="Ruta a la carpeta que contiene los clips de audio originales.")
    parser.add_argument("--output-dir", required=True, help="Ruta a la carpeta donde se guardarán los clips anonimizados, los clips originales y el manifiesto.")
    parser.add_argument("--tag", default=None, help="Etiqueta para identificar el manifiesto generado (por defecto sale del rango de α).")
    alpha = parser.add_mutually_exclusive_group()
    alpha.add_argument("--alpha", type=float, help="Coeficiente de McAdams fijo para todos los clips.")
    alpha.add_argument("--alpha-range", type=float, nargs=2, metavar=("MIN", "MAX"),
                       help="Rango del coeficiente aleatorio por clip (por defecto 0.7 0.9).")
    parser.add_argument("--reuse-original", action="store_true",
                        help="Si <output-dir>/original/<clip>.wav ya existe, leerlo en vez de convertir el MP3.")
    args = parser.parse_args()

    if args.alpha is not None:
        args.alpha_min = args.alpha_max = args.alpha
    elif args.alpha_range is not None:
        args.alpha_min, args.alpha_max = args.alpha_range
    else:
        args.alpha_min = args.alpha_max = None
    if args.alpha_min is not None:
        try:
            validate_range(args.alpha_min, args.alpha_max)
        except ValueError as error:
            parser.error(str(error))
    if args.tag is None:
        args.tag = config_tag(*DEFAULT_RANGE) if args.alpha_min is None else config_tag(args.alpha_min, args.alpha_max)
    return args


def with_progress(rows, every=100):
    """Entrega las filas una a una e imprime el avance cada `every` clips."""
    total = len(rows)
    for i, row in enumerate(rows, start=1):
        yield row
        if i % every == 0 or i == total:
            print(f"{i}/{total} clips procesados")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    rows = list(read_tsv_rows(args.subset))
    manifest = anonymize_subset(
        with_progress(rows),
        args.clips_dir,                          
        output_dir / "original",                 
        output_dir / f"anon_{args.tag}",    
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        reuse_original=args.reuse_original,
    )
    manifest_path = write_manifest(manifest, output_dir / f"anon_manifest_{args.tag}.tsv")

    errors = [record for record in manifest if record["status"] == "error"]
    print(f"Clips ok: {len(manifest) - len(errors)}")
    print(f"Clips con error: {len(errors)}")
    for record in errors:
        print(f"  {record['path']}: {record['error']}")
    print(f"Manifiesto: {manifest_path}")


if __name__ == "__main__":
    main()