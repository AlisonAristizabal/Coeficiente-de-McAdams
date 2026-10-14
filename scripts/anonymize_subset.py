from pathlib import Path

import soundfile
import csv
import argparse

from common_voice_corpus import read_tsv_rows
from mcadams_core.batch.utterance import anonymize_utterance
from mcadams_core.io.audio_conversion import convert_mp3_to_wav
from mcadams_core.io.audio_io import load_wav, save_wav


def process_clip(mp3_path, original_path, anon_path, session_id):
    """
    Convierte un clip, lo anonimiza y guarda ambas versiones.

    Convierte `mp3_path` a WAV de 16 kHz en `original_path`, lo lee de
    vuelta, lo anonimiza con el coeficiente derivado de `session_id` y
    guarda el resultado en `anon_path`. La versión original y la
    anonimizada pasan por la misma conversión, de modo que solo difieren
    en la anonimización. Las excepciones de archivo no se capturan aquí:
    las maneja quien llama.

    Parámetros:
        mp3_path: ruta del MP3 de entrada.
        original_path: ruta del WAV sin anonimizar.
        anon_path: ruta del WAV anonimizado.
        session_id: identificador que fija el coeficiente de McAdams.

    Retorna:
        El coeficiente de McAdams usado.
    """
    convert_mp3_to_wav(mp3_path, original_path)
    sample_rate, audio = load_wav(original_path)
    anon_audio, coefficient = anonymize_utterance(audio, sample_rate, session_id)
    save_wav(anon_path, sample_rate, anon_audio)
    return coefficient


def anonymize_subset(rows, clips_dir, original_dir, anon_dir):
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

    Retorna:
        Lista de diccionarios con las llaves `client_id`, `stratum`,
        `path`, `session_id`, `coefficient`, `status` y `error`.
    """
    clips_dir, original_dir, anon_dir = Path(clips_dir), Path(original_dir), Path(anon_dir)
    original_dir.mkdir(parents=True, exist_ok=True)
    anon_dir.mkdir(parents=True, exist_ok=True)

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
    parser = argparse.ArgumentParser(description="Anonimiza un subconjunto de clips de Common Voice y genera un manifiesto TSV.")
    parser.add_argument("--subset", required=True, help="Ruta al archivo TSV que contiene el subconjunto de clips a anonimizar.")
    parser.add_argument("--clips-dir", required=True, help="Ruta a la carpeta que contiene los clips de audio originales.")
    parser.add_argument("--output-dir", required=True, help="Ruta a la carpeta donde se guardarán los clips anonimizados, los clips originales y el manifiesto.")
    parser.add_argument("--tag", default="u07_09", help="Etiqueta para identificar el manifiesto generado.")
    return parser.parse_args()


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