import argparse
import time
from pathlib import Path

import numpy as np
import soundfile

from common_voice_corpus import read_tsv_rows

TARGET_SAMPLE_RATE = 16000


def load_classifier(savedir):
    """
    Carga el modelo ECAPA-TDNN de verificación de hablante en CPU.

    Usa `speechbrain/spkrec-ecapa-voxceleb`. Se copian los archivos en
    `savedir` en vez de enlazarlos (`LocalStrategy.COPY`) porque en Windows
    los enlaces simbólicos requieren privilegios. Los imports de torch y
    SpeechBrain están dentro de la función para poder importar este módulo
    sin esas bibliotecas.
    """
    from speechbrain.inference.classifiers import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(savedir),
        run_opts={"device": "cpu"},
        local_strategy=LocalStrategy.COPY,
    )


def embed_clip(classifier, wav_path):
    """
    Calcula el embedding de hablante de un WAV de 16 kHz mono.

    Retorna un arreglo float32 de una dimensión (192 para ECAPA). Lanza
    `ValueError` si el audio no es mono o no está a 16 kHz.
    """

    signal, sample_rate = soundfile.read(wav_path, dtype="float32")
    if sample_rate != TARGET_SAMPLE_RATE or signal.ndim != 1:
        raise ValueError(f"{wav_path}: se esperaba audio mono a {TARGET_SAMPLE_RATE} Hz")
    import torch
    with torch.no_grad():
        embedding = classifier.encode_batch(torch.from_numpy(signal).unsqueeze(0))
    return embedding.squeeze().cpu().numpy()


def embed_wavs(wav_paths, embed_fn):
    """
    Calcula el embedding de cada WAV, en el mismo orden de `wav_paths`.

    Parámetros:
        wav_paths: iterable de rutas de WAV.
        embed_fn: función que recibe una ruta y devuelve un arreglo 1D.

    Retorna:
        Una matriz float32 de forma (número de rutas, dimensión del
        embedding). Si no hay rutas, una matriz de forma (0, 0).
    """
    vectors = [embed_fn(path) for path in wav_paths]
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(vectors).astype(np.float32)


def extract_embeddings(session_ids, original_dir, anon_dir, embed_fn):
    """
    Calcula los embeddings de la versión original y la anonimizada de cada clip.

    Busca `<session_id>.wav` en `original_dir` y en `anon_dir`.

    Retorna:
        Diccionario con tres arreglos alineados: `session_ids` (texto),
        `original` y `anonymized` (matrices float32, una fila por clip).
    """
    original_dir, anon_dir = Path(original_dir), Path(anon_dir)
    session_ids = list(session_ids)
    return {
        "session_ids": np.array(session_ids, dtype=str),
        "original": embed_wavs([original_dir / f"{s}.wav" for s in session_ids], embed_fn),
        "anonymized": embed_wavs([anon_dir / f"{s}.wav" for s in session_ids], embed_fn),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Extrae embeddings ECAPA de los clips originales y anonimizados.")
    parser.add_argument("--manifest", required=True, help="Manifiesto TSV generado por anonymize_subset.")
    parser.add_argument("--original-dir", required=True, help="Carpeta con los WAV originales.")
    parser.add_argument("--anon-dir", required=True, help="Carpeta con los WAV anonimizados.")
    parser.add_argument("--model-dir", required=True, help="Carpeta donde está (o se copiará) el modelo ECAPA.")
    parser.add_argument("--output", required=True, help="Archivo .npz de salida.")
    parser.add_argument("--limit", type=int, default=None, help="Procesa solo los primeros N clips (para medir tiempos).")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = [row for row in read_tsv_rows(args.manifest) if row["status"] == "ok"]
    if args.limit is not None:
        rows = rows[:args.limit]
    session_ids = [row["session_id"] for row in rows]

    classifier = load_classifier(args.model_dir)
    start = time.perf_counter()
    result = extract_embeddings(
        session_ids, args.original_dir, args.anon_dir,
        lambda path: embed_clip(classifier, path),
    )
    elapsed = time.perf_counter() - start

    np.savez(args.output, **result)
    print(f"Clips: {len(session_ids)}")
    print(f"Dimensión del embedding: {result['original'].shape[1]}")
    print(f"Tiempo: {elapsed:.1f} s ({elapsed / max(len(session_ids), 1):.2f} s por clip, 2 embeddings por clip)")
    print(f"Guardado en: {args.output}")


if __name__ == "__main__":
    main()