"""Pruebas de la reutilización de lo original: embeddings (--reuse-original-from .npz) y transcripciones (.tsv)."""
import csv
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile

import extract_embeddings
import transcribe
from common_voice_corpus import read_tsv_rows
from extract_embeddings import extract_anonymized, extract_embeddings as extract_both, load_reused_original
from transcribe import OUTPUT_COLUMNS, reuse_original_rows
from wer_bootstrap import speaker_counts


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def vector_de(ruta):
    """Embedding falso que depende de la ruta (carpeta y nombre)."""
    texto = f"{Path(ruta).parent.name}/{Path(ruta).name}"
    return np.array([len(texto), sum(map(ord, texto)) % 97, texto.count("a")], dtype=np.float32)


def base_npz(path, ids, seed=0):
    original = np.random.default_rng(seed).normal(size=(len(ids), 3)).astype(np.float32)
    np.savez(path, session_ids=np.array(ids, dtype=str), original=original,
             anonymized=np.zeros_like(original))
    return original


def test_load_reused_original_devuelve_la_matriz_original(tmp_path):
    original = base_npz(tmp_path / "b.npz", ["s1", "s2", "s3"])
    np.testing.assert_array_equal(load_reused_original(tmp_path / "b.npz", ["s1", "s2", "s3"]), original)


@pytest.mark.parametrize("ids, posicion", [(["s2", "s1", "s3"], 0), (["s1", "s2", "s4"], 2), (["s1", "s2"], 2),
                                           (["s1", "s2", "s3", "s4"], 3)])
def test_load_reused_original_rechaza_session_ids_desalineados(tmp_path, ids, posicion):
    base_npz(tmp_path / "b.npz", ["s1", "s2", "s3"])
    with pytest.raises(ValueError, match=f"primera diferencia en la posición {posicion}"):
        load_reused_original(tmp_path / "b.npz", ids)


def test_load_reused_original_con_prefijo(tmp_path):
    original = base_npz(tmp_path / "b.npz", ["s1", "s2", "s3"])
    np.testing.assert_array_equal(load_reused_original(tmp_path / "b.npz", ["s1", "s2"], prefix=True), original[:2])
    with pytest.raises(ValueError):
        load_reused_original(tmp_path / "b.npz", ["s2"], prefix=True)


def test_extract_anonymized_calcula_solo_los_anonimizados(tmp_path):
    original = np.ones((2, 3), dtype=np.float32)
    rutas = []
    result = extract_anonymized(["s1", "s2"], tmp_path / "a", lambda r: rutas.append(r) or vector_de(r), original)
    assert [Path(r).parent.name for r in rutas] == ["a", "a"]
    np.testing.assert_array_equal(result["original"], original)
    np.testing.assert_array_equal(result["anonymized"], np.stack([vector_de(tmp_path / "a" / f"s{i}.wav")
                                                                  for i in (1, 2)]))
    assert list(result["session_ids"]) == ["s1", "s2"]


def test_extract_anonymized_igual_que_calcular_todo_con_la_misma_base(tmp_path):
    full = extract_both(["s1", "s2"], tmp_path / "o", tmp_path / "a", vector_de)
    reused = extract_anonymized(["s1", "s2"], tmp_path / "a", vector_de, full["original"])
    for key in full:
        np.testing.assert_array_equal(full[key], reused[key])


def escribir_manifiesto(path, ids):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("client_id\tstratum\tpath\tsession_id\tcoefficient\tstatus\terror\n")
        for i, s in enumerate(ids):
            f.write(f"h{i}\tmexico\t{s}.mp3\t{s}\t0.5\tok\t\n")


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    escribir_manifiesto(tmp_path / "m.tsv", ["s1", "s2", "s3"])
    llamadas = []
    monkeypatch.setattr(extract_embeddings, "load_classifier", lambda d: "clasificador")
    monkeypatch.setattr(extract_embeddings, "embed_clip", lambda c, ruta: llamadas.append(Path(ruta)) or vector_de(ruta))

    def ejecutar(*extra):
        salida = tmp_path / "salida.npz"
        monkeypatch.setattr(sys, "argv", ["extract_embeddings.py", "--manifest", str(tmp_path / "m.tsv"),
                                          "--original-dir", str(tmp_path / "o"), "--anon-dir", str(tmp_path / "a"),
                                          "--model-dir", str(tmp_path / "modelo"), "--output", str(salida), *extra])
        extract_embeddings.main()
        return salida

    return types.SimpleNamespace(ejecutar=ejecutar, llamadas=llamadas, tmp=tmp_path)


def test_main_reutiliza_original_y_solo_calcula_el_anonimizado(entorno, capsys):
    original = base_npz(entorno.tmp / "base.npz", ["s1", "s2", "s3"])
    salida = entorno.ejecutar("--reuse-original-from", str(entorno.tmp / "base.npz"))
    assert {r.parent.name for r in entorno.llamadas} == {"a"} and len(entorno.llamadas) == 3
    with np.load(salida) as datos:
        np.testing.assert_array_equal(datos["original"], original)
        assert list(datos["session_ids"]) == ["s1", "s2", "s3"]
    assert "1 embedding por clip" in capsys.readouterr().out


def test_main_sin_reuse_calcula_las_dos_versiones(entorno):
    entorno.ejecutar()
    assert sorted({r.parent.name for r in entorno.llamadas}) == ["a", "o"]


def test_main_reuse_desalineado_falla_antes_de_cargar_el_modelo(entorno, monkeypatch):
    base_npz(entorno.tmp / "base.npz", ["s1", "s3", "s2"])
    monkeypatch.setattr(extract_embeddings, "load_classifier", lambda d: pytest.fail("no debía cargar el modelo"))
    with pytest.raises(ValueError, match="no coinciden"):
        entorno.ejecutar("--reuse-original-from", str(entorno.tmp / "base.npz"))
    assert not (entorno.tmp / "salida.npz").exists()


def test_main_reuse_con_limit_usa_el_prefijo(entorno):
    original = base_npz(entorno.tmp / "base.npz", ["s1", "s2", "s3"])
    salida = entorno.ejecutar("--reuse-original-from", str(entorno.tmp / "base.npz"), "--limit", "2")
    with np.load(salida) as datos:
        np.testing.assert_array_equal(datos["original"], original[:2])


# ---------------------------------------------------------------------------
# Transcripciones
# ---------------------------------------------------------------------------

JOBS = [{"session_id": f"cv_{i}", "client_id": f"h{i}", "stratum": "s", "reference": f"Frase {i}."} for i in range(3)]


def fuente(jobs=JOBS, extra=()):
    rows = []
    for j in jobs:
        rows.append({**j, "version": "original", "hypothesis": f"frase {j['session_id']}", "seconds": "1.250"})
        rows.append({**j, "version": "anonymized", "hypothesis": "otra cosa", "seconds": "2.000"})
    return rows + list(extra)


def test_reuse_original_rows_copia_las_originales_en_el_orden_de_jobs():
    rows = reuse_original_rows(list(reversed(fuente())), JOBS)
    assert [r["session_id"] for r in rows] == ["cv_0", "cv_1", "cv_2"]
    assert all(r["version"] == "original" and r["seconds"] == 1.25 for r in rows)
    assert rows[1]["hypothesis"] == "frase cv_1" and set(rows[0]) == set(OUTPUT_COLUMNS)


def test_reuse_original_rows_falta_un_clip_de_la_muestra():
    with pytest.raises(ValueError, match="1 clips de la muestra faltan"):
        reuse_original_rows(fuente(JOBS[:2]), JOBS)


def test_reuse_original_rows_sobra_un_clip_en_el_tsv():
    with pytest.raises(ValueError, match="1 del TSV no están en la muestra"):
        reuse_original_rows(fuente(), JOBS[:2])


def test_reuse_original_rows_referencia_distinta():
    otra = [dict(JOBS[0], reference="Otra frase."), *JOBS[1:]]
    with pytest.raises(ValueError, match="otra referencia"):
        reuse_original_rows(fuente(otra), JOBS)


def test_reuse_original_rows_clip_repetido():
    with pytest.raises(ValueError, match="repite"):
        reuse_original_rows(fuente(extra=[dict(JOBS[0], version="original", hypothesis="x", seconds="1")]), JOBS)


def test_reuse_original_rows_ignora_las_anonimizadas_del_tsv():
    rows = reuse_original_rows(fuente(), JOBS)
    assert "otra cosa" not in [r["hypothesis"] for r in rows]


def write_wav(path):
    soundfile.write(path, (0.1 * np.sin(np.linspace(0, 50, 1600))).astype(np.float32), 16000)


@pytest.fixture
def corpus(tmp_path):
    original_dir, anon_dir = tmp_path / "orig", tmp_path / "anon"
    original_dir.mkdir()
    anon_dir.mkdir()
    subset, manifest = [], []
    for st in range(2):
        for s in range(3):
            sid = f"cv_{st}{s}"
            subset.append({"client_id": f"spk{st}{s}", "stratum": f"st{st}", "path": f"{sid}.mp3", "accents": "",
                           "sentence": f"Frase {sid}.", "duration_ms": "4000"})
            manifest.append({"client_id": f"spk{st}{s}", "stratum": f"st{st}", "path": f"{sid}.mp3",
                             "session_id": sid, "coefficient": "0.5", "status": "ok", "error": ""})
            write_wav(original_dir / f"{sid}.wav")
            write_wav(anon_dir / f"{sid}.wav")
    paths = {}
    for name, rows in (("subset.tsv", subset), ("manifest.tsv", manifest)):
        paths[name] = tmp_path / name
        with open(paths[name], "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    return types.SimpleNamespace(manifest=paths["manifest.tsv"], subset=paths["subset.tsv"], original=original_dir,
                                 anon=anon_dir, tmp=tmp_path)


class EchoModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **options):
        self.calls.append(audio)
        return {"text": f" texto {len(self.calls)} "}


def run_transcribe(monkeypatch, corpus, output, *extra, model=None):
    model = model or EchoModel()
    loaded = []
    monkeypatch.setattr(transcribe, "load_model", lambda name, d: loaded.append(name) or model)
    paths = []
    real = transcribe.transcribe_clip
    monkeypatch.setattr(transcribe, "transcribe_clip", lambda m, p: paths.append(Path(p)) or real(m, p))
    monkeypatch.setattr(sys, "argv", ["transcribe.py", "--manifest", str(corpus.manifest), "--subset", str(corpus.subset),
                                      "--original-dir", str(corpus.original), "--anon-dir", str(corpus.anon),
                                      "--model", "turbo", "--model-dir", "m", "--output", str(output), *extra])
    transcribe.main()
    return types.SimpleNamespace(paths=paths, loaded=loaded)


def test_main_reuse_transcribe_solo_el_anonimizado(monkeypatch, corpus, capsys):
    base = corpus.tmp / "base.tsv"
    run_transcribe(monkeypatch, corpus, base, "--per-stratum", "2")
    out = corpus.tmp / "a050.tsv"
    run = run_transcribe(monkeypatch, corpus, out, "--per-stratum", "2", "--reuse-original-from", str(base))
    assert {p.parent.name for p in run.paths} == {"anon"} and len(run.paths) == 4
    rows = list(read_tsv_rows(out))
    base_orig = {r["session_id"]: r for r in read_tsv_rows(base) if r["version"] == "original"}
    originals = [r for r in rows if r["version"] == "original"]
    assert len(rows) == 8 and len(originals) == 4
    for r in originals:
        assert (r["hypothesis"], r["seconds"], r["reference"]) == (
            base_orig[r["session_id"]]["hypothesis"], base_orig[r["session_id"]]["seconds"],
            base_orig[r["session_id"]]["reference"])
    counts = speaker_counts(rows)  # el formato que espera wer_bootstrap
    assert len(counts["client_ids"]) == 4
    assert "Originales reutilizados" in capsys.readouterr().out


def test_main_reuse_muestra_distinta_falla_antes_de_cargar_whisper(monkeypatch, corpus):
    base = corpus.tmp / "base.tsv"
    run_transcribe(monkeypatch, corpus, base, "--per-stratum", "2")
    out = corpus.tmp / "otro.tsv"
    monkeypatch.setattr(transcribe, "load_model", lambda name, d: pytest.fail("no debía cargar Whisper"))
    monkeypatch.setattr(sys, "argv", ["transcribe.py", "--manifest", str(corpus.manifest), "--subset", str(corpus.subset),
                                      "--original-dir", str(corpus.original), "--anon-dir", str(corpus.anon),
                                      "--model", "turbo", "--model-dir", "m", "--output", str(out),
                                      "--per-stratum", "3", "--reuse-original-from", str(base)])
    with pytest.raises(ValueError, match="la muestra no coincide"):
        transcribe.main()
    assert not out.exists()


def test_main_reuse_reanudable_sin_duplicar_originales(monkeypatch, corpus):
    base = corpus.tmp / "base.tsv"
    run_transcribe(monkeypatch, corpus, base, "--per-stratum", "2")
    out = corpus.tmp / "a050.tsv"
    run_transcribe(monkeypatch, corpus, out, "--per-stratum", "2", "--reuse-original-from", str(base))
    first = out.read_text(encoding="utf-8")
    run = run_transcribe(monkeypatch, corpus, out, "--per-stratum", "2", "--reuse-original-from", str(base))
    assert out.read_text(encoding="utf-8") == first
    assert run.loaded == [] and run.paths == []  # nada pendiente: no se carga el modelo


def test_main_sin_pendientes_no_carga_el_modelo(monkeypatch, corpus, capsys):
    out = corpus.tmp / "t.tsv"
    run_transcribe(monkeypatch, corpus, out, "--per-stratum", "1")
    run = run_transcribe(monkeypatch, corpus, out, "--per-stratum", "1")
    assert run.loaded == []
    assert "Nada pendiente" in capsys.readouterr().out
