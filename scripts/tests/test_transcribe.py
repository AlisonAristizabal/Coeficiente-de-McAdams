import csv
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import soundfile

import transcribe
from common_voice_corpus import read_tsv_rows
from transcribe import (
    DECODE_OPTIONS,
    OUTPUT_COLUMNS,
    VERSIONS,
    append_rows,
    build_jobs,
    corpus_wer,
    edit_distance,
    load_audio,
    load_model,
    load_references,
    normalize_provisional,
    prepare_output,
    print_summary,
    select_sample,
    summarize_rows,
    transcribe_clip,
    transcribe_rows,
)


# ---------------------------------------------------------------------------
# Datos sintéticos
# ---------------------------------------------------------------------------

def write_wav(path, sample_rate=16000, channels=1, seconds=0.1):
    n = int(sample_rate * seconds)
    signal = 0.1 * np.sin(np.linspace(0, 50, n)).astype(np.float32)
    if channels > 1:
        signal = np.stack([signal] * channels, axis=1)
    soundfile.write(path, signal, sample_rate)
    return path


def jobs_for(n_strata=3, speakers=4, clips=3):
    return [{"session_id": f"cv_{st}_{s}_{c}", "client_id": f"spk{st}{s}", "stratum": f"st{st}",
             "reference": f"oración {st} {s} {c}"}
            for st in range(n_strata) for s in range(speakers) for c in range(clips)]


class FakeModel:
    """Modelo falso: guarda con qué se llamó y responde un texto con espacios."""

    def __init__(self, text=" Hola mundo. "):
        self.calls, self.text = [], text

    def transcribe(self, audio, **options):
        self.calls.append((audio, options))
        return {"text": self.text, "segments": []}


def fake_clock(step=0.5):
    t = {"now": 0.0}

    def clock():
        t["now"] += step
        return t["now"]
    return clock


# ---------------------------------------------------------------------------
# Opciones, carga del modelo y del audio
# ---------------------------------------------------------------------------

def test_decode_options_fijas():
    assert DECODE_OPTIONS == {"language": "es", "task": "transcribe", "temperature": 0.0, "fp16": False,
                              "condition_on_previous_text": False}


def test_load_model_en_cpu_y_en_la_carpeta_dada(monkeypatch, tmp_path):
    calls = []
    fake = types.SimpleNamespace(load_model=lambda name, **kw: calls.append((name, kw)) or "modelo")
    monkeypatch.setitem(sys.modules, "whisper", fake)
    assert load_model("small", tmp_path / "whisper") == "modelo"
    assert calls == [("small", {"device": "cpu", "download_root": str(tmp_path / "whisper")})]


def test_load_audio_float32_mono(tmp_path):
    audio = load_audio(write_wav(tmp_path / "a.wav"))
    assert audio.dtype == np.float32 and audio.ndim == 1 and audio.size == 1600


def test_load_audio_rechaza_otra_frecuencia(tmp_path):
    with pytest.raises(ValueError, match="16000"):
        load_audio(write_wav(tmp_path / "a.wav", sample_rate=8000))


def test_load_audio_rechaza_estereo(tmp_path):
    with pytest.raises(ValueError, match="mono"):
        load_audio(write_wav(tmp_path / "a.wav", channels=2))


def test_transcribe_clip_pasa_el_arreglo_y_las_opciones(tmp_path):
    wav = write_wav(tmp_path / "a.wav")
    model = FakeModel()
    assert transcribe_clip(model, wav) == "Hola mundo."
    (audio, options), = model.calls
    assert isinstance(audio, np.ndarray) and audio.dtype == np.float32
    np.testing.assert_array_equal(audio, soundfile.read(wav, dtype="float32")[0])
    assert options == DECODE_OPTIONS


def test_transcribe_clip_no_llama_al_modelo_si_el_audio_es_invalido(tmp_path):
    model = FakeModel()
    with pytest.raises(ValueError):
        transcribe_clip(model, write_wav(tmp_path / "a.wav", sample_rate=22050))
    assert model.calls == []


# ---------------------------------------------------------------------------
# Referencias y unión con el manifiesto
# ---------------------------------------------------------------------------

SUBSET = [{"path": "cv_1.mp3", "sentence": "Hola, ¿qué tal?"}, {"path": "cv_2.mp3", "sentence": "Buenos días."}]
MANIFEST = [
    {"client_id": "a", "stratum": "s1", "path": "cv_1.mp3", "session_id": "cv_1", "status": "ok"},
    {"client_id": "b", "stratum": "s2", "path": "cv_2.mp3", "session_id": "cv_2", "status": "error"},
]


def test_load_references_por_path():
    assert load_references(SUBSET) == {"cv_1.mp3": "Hola, ¿qué tal?", "cv_2.mp3": "Buenos días."}


def test_load_references_path_repetido_lanza():
    with pytest.raises(ValueError, match="cv_1.mp3"):
        load_references(SUBSET + [{"path": "cv_1.mp3", "sentence": "otra"}])


def test_build_jobs_une_por_path_y_solo_toma_ok():
    assert build_jobs(MANIFEST, load_references(SUBSET)) == [
        {"session_id": "cv_1", "client_id": "a", "stratum": "s1", "reference": "Hola, ¿qué tal?"}]


def test_build_jobs_usa_path_y_no_session_id():
    # La referencia se busca por path aunque session_id no coincida con él.
    manifest = [{"client_id": "a", "stratum": "s", "path": "cv_2.mp3", "session_id": "otro", "status": "ok"}]
    assert build_jobs(manifest, load_references(SUBSET))[0]["reference"] == "Buenos días."


def test_build_jobs_clip_sin_referencia_lanza():
    manifest = [{"client_id": "a", "stratum": "s", "path": "cv_9.mp3", "session_id": "cv_9", "status": "ok"}]
    with pytest.raises(ValueError, match="cv_9.mp3"):
        build_jobs(manifest, load_references(SUBSET))


# ---------------------------------------------------------------------------
# select_sample
# ---------------------------------------------------------------------------

def test_select_sample_cantidad_por_estrato():
    sample = select_sample(jobs_for(), 2, seed=42)
    assert Counter(r["stratum"] for r in sample) == {"st0": 2, "st1": 2, "st2": 2}


def test_select_sample_a_lo_sumo_un_clip_por_hablante():
    for seed in range(20):
        sample = select_sample(jobs_for(), 4, seed=seed)
        assert len({r["client_id"] for r in sample}) == len(sample) == 12


def test_select_sample_devuelve_filas_de_la_entrada_y_del_estrato():
    jobs = jobs_for()
    for row in select_sample(jobs, 3, seed=1):
        assert row in jobs and row["client_id"].startswith("spk" + row["stratum"][2:])


def test_select_sample_reproducible_y_no_depende_del_orden():
    jobs = jobs_for()
    a = select_sample(jobs, 2, seed=7)
    assert a == select_sample(jobs, 2, seed=7)
    assert a == select_sample(list(reversed(jobs)), 2, seed=7)


def test_select_sample_cambia_con_otra_semilla():
    jobs = jobs_for(n_strata=3, speakers=10, clips=5)
    assert select_sample(jobs, 3, seed=1) != select_sample(jobs, 3, seed=2)


def test_select_sample_elige_entre_todos_los_clips_del_hablante():
    jobs = jobs_for(n_strata=1, speakers=1, clips=5)
    chosen = {select_sample(jobs, 1, seed=s)[0]["session_id"] for s in range(60)}
    assert len(chosen) == 5


def test_select_sample_orden_por_estrato_y_session_id():
    sample = select_sample(jobs_for(), 3, seed=0)
    assert sample == sorted(sample, key=lambda r: (r["stratum"], r["session_id"]))


def test_select_sample_estrato_con_pocos_hablantes_lanza():
    with pytest.raises(ValueError, match="st0"):
        select_sample(jobs_for(speakers=2), 3, seed=0)


# ---------------------------------------------------------------------------
# normalize_provisional
# ---------------------------------------------------------------------------

def test_normalize_minusculas_y_sin_puntuacion():
    assert normalize_provisional("¿Cómo estás, Señor Núñez?") == "cómo estás señor núñez"


def test_normalize_conserva_tildes_enie_dieresis_y_cifras():
    assert normalize_provisional("Pingüino: año 1999; ÁRBOL.") == "pingüino año 1999 árbol"


def test_normalize_signos_espanoles_comillas_y_guiones():
    assert normalize_provisional("¡Hola!—«dijo» (bien-estar)") == "hola dijo bien estar"


def test_normalize_compone_caracteres():
    assert normalize_provisional("é") == normalize_provisional("é") == "é"


# ---------------------------------------------------------------------------
# edit_distance y corpus_wer (resultados calculados a mano)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ref, hyp, expected", [
    ("el gato come", "el gato come", 0),
    ("el gato come", "el perro come", 1),            # 1 sustitución
    ("el gato come pescado", "el gato pescado", 1),  # 1 borrado
    ("el gato", "el gato negro come", 2),            # 2 inserciones
    ("", "hola mundo", 2),                           # todo inserciones
    ("hola mundo", "", 2),                           # todo borrados
    ("", "", 0),
    ("a b c d", "b c d a", 2),                       # borrar "a" al inicio e insertarla al final
    ("a b c", "x y z", 3),                           # 3 sustituciones
    ("a b c", "c b a", 2),                           # 2 sustituciones (a↔c)
    ("uno dos tres cuatro", "uno tres cuatro cinco", 2),
])
def test_edit_distance_a_mano(ref, hyp, expected):
    assert edit_distance(ref.split(), hyp.split()) == expected


def test_edit_distance_compara_palabras_no_letras():
    assert edit_distance(["casa"], ["casas"]) == 1
    assert edit_distance(["casa", "roja"], ["casaroja"]) == 2


def test_corpus_wer_errores_totales_sobre_palabras_totales():
    # Oración 1: 1 error en 3 palabras; oración 2: 1 error en 1 palabra.
    # Corpus: 2 / 4 = 0,5 (el promedio por oración daría 0,67).
    assert corpus_wer(["el gato come", "hola"], ["el perro come", ""]) == pytest.approx(0.5)


def test_corpus_wer_normaliza_mayusculas_y_puntuacion():
    assert corpus_wer(["¿Cómo estás?"], ["cómo estás"]) == 0.0


def test_corpus_wer_tildes_cuentan():
    assert corpus_wer(["cómo estás"], ["como estas"]) == 1.0


def test_corpus_wer_puede_pasar_de_uno():
    assert corpus_wer(["hola"], ["hola que tal amigo"]) == pytest.approx(3.0)


def test_corpus_wer_acepta_otra_normalizacion():
    assert corpus_wer(["Hola"], ["hola"], normalize=lambda s: s) == 1.0


def test_corpus_wer_errores_de_entrada():
    with pytest.raises(ValueError):
        corpus_wer(["a"], ["a", "b"])
    with pytest.raises(ValueError, match="palabras"):
        corpus_wer(["¿?"], ["hola"])


# ---------------------------------------------------------------------------
# transcribe_rows
# ---------------------------------------------------------------------------

JOBS = [{"session_id": "cv_1", "client_id": "a", "stratum": "s1", "reference": "uno"},
        {"session_id": "cv_2", "client_id": "b", "stratum": "s2", "reference": "dos"}]


def path_echo(path):
    return f"{Path(path).parent.name}/{Path(path).name}"


def test_transcribe_rows_original_y_anonimizado_de_cada_clip(tmp_path):
    rows = list(transcribe_rows(JOBS, tmp_path / "orig", tmp_path / "anon", path_echo, clock=fake_clock()))
    assert [(r["session_id"], r["version"], r["hypothesis"]) for r in rows] == [
        ("cv_1", "original", "orig/cv_1.wav"), ("cv_1", "anonymized", "anon/cv_1.wav"),
        ("cv_2", "original", "orig/cv_2.wav"), ("cv_2", "anonymized", "anon/cv_2.wav")]
    assert all(set(r) == set(OUTPUT_COLUMNS) for r in rows)
    assert [r["reference"] for r in rows] == ["uno", "uno", "dos", "dos"]
    assert rows[0]["client_id"] == "a" and rows[3]["stratum"] == "s2"


def test_transcribe_rows_mide_solo_la_transcripcion():
    ticks = iter([10.0, 12.5, 20.0, 21.0])
    rows = list(transcribe_rows(JOBS[:1], "o", "a", path_echo, clock=lambda: next(ticks)))
    assert [r["seconds"] for r in rows] == [2.5, 1.0]


def test_transcribe_rows_salta_lo_ya_hecho():
    calls = []
    rows = list(transcribe_rows(JOBS, "o", "a", lambda p: calls.append(p) or "x",
                                done={("cv_1", "original"), ("cv_2", "anonymized")}, clock=fake_clock()))
    assert [(r["session_id"], r["version"]) for r in rows] == [("cv_1", "anonymized"), ("cv_2", "original")]
    assert len(calls) == 2


def test_transcribe_rows_es_perezoso():
    calls = []
    rows = transcribe_rows(JOBS, "o", "a", lambda p: calls.append(p) or "x", clock=fake_clock())
    next(rows)
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# append_rows y prepare_output
# ---------------------------------------------------------------------------

def make_rows(n, hypothesis="hola"):
    return [{"session_id": f"cv_{i}", "client_id": "a", "stratum": "s", "version": VERSIONS[i % 2],
             "reference": "Hola.", "hypothesis": hypothesis, "seconds": 1.23456} for i in range(n)]


def test_append_rows_escribe_encabezado_y_filas(tmp_path):
    out = tmp_path / "t.tsv"
    assert append_rows(make_rows(3), out, progress_every=0) == 3
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].split("\t") == OUTPUT_COLUMNS
    assert lines[1].split("\t") == ["cv_0", "a", "s", "original", "Hola.", "hola", "1.235"]
    assert len(list(read_tsv_rows(out))) == 3


def test_append_rows_agrega_sin_repetir_encabezado(tmp_path):
    out = tmp_path / "t.tsv"
    append_rows(make_rows(2), out, progress_every=0)
    append_rows(make_rows(1), out, progress_every=0)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4 and lines.count("\t".join(OUTPUT_COLUMNS)) == 1


def test_append_rows_tabulaciones_y_saltos_no_rompen_el_tsv(tmp_path):
    out = tmp_path / "t.tsv"
    append_rows(make_rows(1, hypothesis="hola\tmundo\ncómo\r\nestás"), out, progress_every=0)
    (row,) = read_tsv_rows(out)
    assert row["hypothesis"] == "hola mundo cómo estás" and row["seconds"] == "1.235"


def test_append_rows_comillas_se_conservan(tmp_path):
    out = tmp_path / "t.tsv"
    append_rows(make_rows(1, hypothesis='dijo "hola" y \'adiós\''), out, progress_every=0)
    assert next(read_tsv_rows(out))["hypothesis"] == 'dijo "hola" y \'adiós\''


def test_append_rows_guarda_lo_hecho_si_se_interrumpe(tmp_path):
    out = tmp_path / "t.tsv"
    calls = {"n": 0}

    def flaky(path):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("corte")
        return "texto"

    with pytest.raises(RuntimeError):
        append_rows(transcribe_rows(JOBS, "o", "a", flaky, clock=fake_clock()), out, progress_every=0)
    assert [(r["session_id"], r["version"]) for r in read_tsv_rows(out)] == [("cv_1", "original"),
                                                                             ("cv_1", "anonymized")]


def test_append_rows_imprime_el_avance(tmp_path, capsys):
    append_rows(make_rows(5), tmp_path / "t.tsv", progress_every=2)
    printed = capsys.readouterr().out
    assert "2 transcripciones" in printed and "4 transcripciones" in printed and "5 transcripciones" not in printed


def test_prepare_output_sin_archivo(tmp_path):
    assert prepare_output(tmp_path / "no.tsv") == set()


def test_prepare_output_devuelve_lo_hecho(tmp_path):
    out = tmp_path / "t.tsv"
    append_rows(make_rows(3), out, progress_every=0)
    assert prepare_output(out) == {("cv_0", "original"), ("cv_1", "anonymized"), ("cv_2", "original")}


def test_prepare_output_borra_la_linea_a_medias(tmp_path):
    out = tmp_path / "t.tsv"
    append_rows(make_rows(2), out, progress_every=0)
    with open(out, "a", encoding="utf-8", newline="") as f:
        f.write("cv_9\ta\ts\toriginal\tHol")
    assert prepare_output(out) == {("cv_0", "original"), ("cv_1", "anonymized")}
    append_rows(make_rows(1), out, progress_every=0)
    rows = list(read_tsv_rows(out))
    assert len(rows) == 3 and all(r["seconds"] == "1.235" for r in rows)


# ---------------------------------------------------------------------------
# summarize_rows y print_summary
# ---------------------------------------------------------------------------

SUMMARY_ROWS = [
    {"version": "original", "reference": "el gato come", "hypothesis": "el gato come", "seconds": "1.0"},
    {"version": "original", "reference": "hola", "hypothesis": "hola", "seconds": "3.0"},
    {"version": "anonymized", "reference": "el gato come", "hypothesis": "el perro come", "seconds": "2.0"},
    {"version": "anonymized", "reference": "hola", "hypothesis": "", "seconds": "4.0"},
]


def test_summarize_rows_por_version():
    summary = summarize_rows(SUMMARY_ROWS)
    assert summary["original"] == {"clips": 2, "seconds_per_clip": 2.0, "wer": 0.0}
    assert summary["anonymized"]["clips"] == 2 and summary["anonymized"]["seconds_per_clip"] == 3.0
    assert summary["anonymized"]["wer"] == pytest.approx(0.5)


def test_print_summary_con_coma_decimal(tmp_path, capsys):
    out = tmp_path / "whisper_prueba_small.tsv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["version", "reference", "hypothesis", "seconds"], delimiter="\t")
        writer.writeheader()
        writer.writerows(SUMMARY_ROWS)
    print_summary([out])
    printed = capsys.readouterr().out
    assert "whisper_prueba_small.tsv" in printed
    assert "original: 2 clips, 2,00 s por clip, WER provisional 0,00 %" in printed
    assert "anonymized: 2 clips, 3,00 s por clip, WER provisional 50,00 %" in printed


# ---------------------------------------------------------------------------
# main (de punta a punta, con un modelo falso)
# ---------------------------------------------------------------------------

@pytest.fixture
def corpus(tmp_path):
    original_dir, anon_dir = tmp_path / "orig", tmp_path / "anon"
    original_dir.mkdir()
    anon_dir.mkdir()
    subset, manifest = [], []
    for st in range(2):
        for s in range(3):
            for c in range(2):
                sid = f"cv_{st}{s}{c}"
                subset.append({"client_id": f"spk{st}{s}", "stratum": f"st{st}", "path": f"{sid}.mp3",
                               "accents": "", "sentence": f"Frase {sid}.", "duration_ms": "4000"})
                manifest.append({"client_id": f"spk{st}{s}", "stratum": f"st{st}", "path": f"{sid}.mp3",
                                 "session_id": sid, "coefficient": "0.8", "status": "ok", "error": ""})
                write_wav(original_dir / f"{sid}.wav")
                write_wav(anon_dir / f"{sid}.wav")
    paths = {}
    for name, rows in (("subset.tsv", subset), ("manifest.tsv", manifest)):
        paths[name] = tmp_path / name
        with open(paths[name], "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    return paths["manifest.tsv"], paths["subset.tsv"], original_dir, anon_dir


class EchoModel:
    """Modelo falso: comprueba que recibe el arreglo y las opciones fijas y
    devuelve un texto distinto en cada llamada (cuenta las llamadas)."""

    def __init__(self):
        self.n = 0

    def transcribe(self, audio, **options):
        assert isinstance(audio, np.ndarray) and options == DECODE_OPTIONS
        self.n += 1
        return {"text": f" texto {self.n} "}


def run_main(monkeypatch, corpus, output, *extra, model=None):
    manifest, subset, original_dir, anon_dir = corpus
    loaded = []
    monkeypatch.setattr(transcribe, "load_model", lambda name, model_dir: loaded.append((name, model_dir)) or (model or EchoModel()))
    monkeypatch.setattr(sys, "argv", ["transcribe.py", "--manifest", str(manifest), "--subset", str(subset),
                                      "--original-dir", str(original_dir), "--anon-dir", str(anon_dir),
                                      "--model", "small", "--model-dir", "dir_modelos", "--output", str(output),
                                      *extra])
    transcribe.main()
    return loaded


def test_main_todos_los_clips_sin_per_stratum(monkeypatch, tmp_path, corpus):
    out = tmp_path / "t.tsv"
    loaded = run_main(monkeypatch, corpus, out)
    rows = list(read_tsv_rows(out))
    assert loaded == [("small", "dir_modelos")]
    assert len(rows) == 12 * 2
    assert Counter(r["version"] for r in rows) == {"original": 12, "anonymized": 12}
    assert all(r["reference"] == f"Frase {r['session_id']}." for r in rows)
    assert all(r["hypothesis"].startswith("texto") for r in rows)


def test_main_con_per_stratum_usa_select_sample(monkeypatch, tmp_path, corpus):
    out = tmp_path / "t.tsv"
    run_main(monkeypatch, corpus, out, "--per-stratum", "2", "--seed", "5")
    rows = list(read_tsv_rows(out))
    manifest, subset = corpus[0], corpus[1]
    expected = select_sample(build_jobs(read_tsv_rows(manifest), load_references(read_tsv_rows(subset))), 2, 5)
    assert [r["session_id"] for r in rows[::2]] == [j["session_id"] for j in expected]
    assert Counter(r["stratum"] for r in rows) == {"st0": 4, "st1": 4}


def test_main_continua_donde_quedo(monkeypatch, tmp_path, corpus, capsys):
    out = tmp_path / "t.tsv"
    run_main(monkeypatch, corpus, out, "--per-stratum", "2")
    first = out.read_text(encoding="utf-8")
    model = EchoModel()
    run_main(monkeypatch, corpus, out, "--per-stratum", "2", model=model)
    assert model.n == 0 and out.read_text(encoding="utf-8") == first
    assert "transcripciones pendientes: 0 (ya hechas: 8)" in capsys.readouterr().out


def test_main_imprime_resumen(monkeypatch, tmp_path, corpus, capsys):
    out = tmp_path / "t.tsv"
    run_main(monkeypatch, corpus, out, "--per-stratum", "1")
    printed = capsys.readouterr().out
    assert "Clips: 2; transcripciones pendientes: 4 (ya hechas: 0)" in printed
    assert "Transcripciones nuevas: 4" in printed and str(out) in printed


def test_select_sample_la_semilla_cambia_los_hablantes_elegidos():
    jobs = jobs_for(n_strata=1, speakers=10, clips=1)
    speaker_sets = {frozenset(r["client_id"] for r in select_sample(jobs, 3, seed=s)) for s in range(6)}
    assert len(speaker_sets) > 1


def test_append_rows_cada_fila_queda_en_disco_antes_de_la_siguiente(tmp_path):
    out = tmp_path / "t.tsv"
    seen = []

    def check_disk(path):
        # Al pedir la transcripción k, las k - 1 anteriores ya deben estar en el archivo.
        seen.append(len(list(read_tsv_rows(out))) if out.exists() and out.stat().st_size else 0)
        return "texto"

    append_rows(transcribe_rows(JOBS, "o", "a", check_disk, clock=fake_clock()), out, progress_every=0)
    assert seen == [0, 1, 2, 3]


def test_corpus_wer_normaliza_tambien_la_hipotesis():
    # Whisper devuelve mayúsculas y puntuación: no deben contar como errores.
    assert corpus_wer(["cómo estás"], ["¿Cómo estás?"]) == 0.0


# ---------------------------------------------------------------------------
# normalize_final
# ---------------------------------------------------------------------------

from transcribe import normalize_final  # noqa: E402


@pytest.mark.parametrize("digits, words", [
    ("7", "siete"), ("4", "cuatro"), ("14", "catorce"), ("10", "diez"), ("0", "cero"), ("100", "cien"),
])
def test_normalize_final_casos_del_corpus(digits, words):
    assert normalize_final(digits) == words


def test_normalize_final_capitulo_4():
    assert normalize_final("Capítulo 4") == "capítulo cuatro"


def test_normalize_final_numeros_de_varias_palabras():
    assert normalize_final("En 1999.") == "en mil novecientos noventa y nueve"


def test_normalize_final_cifras_pegadas_a_letras_o_puntuacion():
    assert normalize_final("4x4, (7)") == "cuatro x cuatro siete"
    assert normalize_final("3,5") == "tres cinco"


def test_normalize_final_igual_que_la_provisional_sin_cifras():
    for text in ("¿Cómo estás, Señor Núñez?", "Pingüino: ÁRBOL.", "¡Hola!—«dijo» (bien-estar)", "é"):
        assert normalize_final(text) == normalize_provisional(text)


def test_normalize_final_iguala_cifras_y_palabras_en_el_wer():
    assert corpus_wer(["capítulo cuatro"], ["Capítulo 4."], normalize=normalize_final) == 0.0
    assert corpus_wer(["capítulo cuatro"], ["Capítulo 4."]) == 0.5
