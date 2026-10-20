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


def load_reused_original(npz_path, session_ids, prefix=False):
    """
    Toma la matriz "original" de un .npz de embeddings ya calculado.

    Sirve para no recalcular los embeddings originales, que no dependen de
    la anonimización. Verifica que los `session_ids` del .npz sean los
    mismos y en el mismo orden; si no, lanza `ValueError` indicando la
    primera posición donde difieren, porque una fila corrida mezclaría
    hablantes en silencio.

    Parámetros:
        npz_path: .npz de `extract_embeddings` (con `session_ids` y `original`).
        session_ids: lista de session_id del manifiesto, en orden.
        prefix: si es True (con --limit), basta con que coincidan los
            primeros len(session_ids) del .npz.

    Retorna:
        Matriz float32 (número de clips, dimensión) con los embeddings
        originales, alineada con `session_ids`.
    """
    session_ids = list(session_ids)
    with np.load(npz_path) as data:
        base_ids = [str(s) for s in data["session_ids"]]
        original = np.array(data["original"])
    if prefix:
        base_ids, original = base_ids[:len(session_ids)], original[:len(session_ids)]
    if base_ids != session_ids:
        first = next((i for i, (a, b) in enumerate(zip(base_ids, session_ids)) if a != b),
                     min(len(base_ids), len(session_ids)))
        raise ValueError(
            f"los session_ids de {npz_path} no coinciden con el manifiesto en el mismo orden "
            f"({len(base_ids)} frente a {len(session_ids)} clips; primera diferencia en la posición {first})"
        )
    return original


def extract_anonymized(session_ids, anon_dir, embed_fn, original):
    """
    Calcula solo los embeddings anonimizados y reutiliza los originales dados.

    Retorna:
        El mismo diccionario que `extract_embeddings`: `session_ids`,
        `original` (la matriz recibida) y `anonymized`.
    """
    session_ids = list(session_ids)
    return {
        "session_ids": np.array(session_ids, dtype=str),
        "original": np.asarray(original, dtype=np.float32),
        "anonymized": embed_wavs([Path(anon_dir) / f"{s}.wav" for s in session_ids], embed_fn),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Extrae embeddings ECAPA de los clips originales y anonimizados.")
    parser.add_argument("--manifest", required=True, help="Manifiesto TSV generado por anonymize_subset.")
    parser.add_argument("--original-dir", required=True, help="Carpeta con los WAV originales.")
    parser.add_argument("--anon-dir", required=True, help="Carpeta con los WAV anonimizados.")
    parser.add_argument("--model-dir", required=True, help="Carpeta donde está (o se copiará) el modelo ECAPA.")
    parser.add_argument("--output", required=True, help="Archivo .npz de salida.")
    parser.add_argument("--limit", type=int, default=None, help="Procesa solo los primeros N clips (para medir tiempos).")
    parser.add_argument("--reuse-original-from", default=None,
                        help="Archivo .npz del que se toman los embeddings originales (solo se calculan los anonimizados).")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = [row for row in read_tsv_rows(args.manifest) if row["status"] == "ok"]
    if args.limit is not None:
        rows = rows[:args.limit]
    session_ids = [row["session_id"] for row in rows]
    # Se verifica antes de cargar el modelo, para fallar rápido si no coinciden.
    reused = None
    if args.reuse_original_from:
        reused = load_reused_original(args.reuse_original_from, session_ids, prefix=args.limit is not None)

    classifier = load_classifier(args.model_dir)
    embed_fn = lambda path: embed_clip(classifier, path)
    start = time.perf_counter()
    if reused is None:
        result = extract_embeddings(session_ids, args.original_dir, args.anon_dir, embed_fn)
    else:
        result = extract_anonymized(session_ids, args.anon_dir, embed_fn, reused)
    elapsed = time.perf_counter() - start

    np.savez(args.output, **result)
    per_clip = 1 if reused is not None else 2
    print(f"Clips: {len(session_ids)}")
    if reused is not None:
        print(f"Embeddings originales reutilizados de: {args.reuse_original_from}")
    print(f"Dimensión del embedding: {result['original'].shape[1]}")
    print(f"Tiempo: {elapsed:.1f} s ({elapsed / max(len(session_ids), 1):.2f} s por clip, "
          f"{per_clip} embedding{'s' if per_clip > 1 else ''} por clip)")
    print(f"Guardado en: {args.output}")


if __name__ == "__main__":
    main()