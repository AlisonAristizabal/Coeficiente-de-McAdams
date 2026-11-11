# Coeficiente de McAdams

Anonimización de voz en tiempo real con el coeficiente de McAdams estocástico:
núcleo de procesamiento, prototipo en vivo y evaluación de privacidad (EER,
suplantación) e inteligibilidad (WER) sobre Common Voice en español.

## Estructura

| Carpeta | Contenido |
|---|---|
| `packages/core/` | `mcadams_core`: análisis LPC y transformación de McAdams por trama, anonimización incremental (`AnonymizerStream`, `BlockAnonymizer`), anonimización de audios completos y coeficiente por sesión. |
| `packages/app/` | `mcadams_app`: prototipo en vivo (micrófono → anonimización → audífonos) en `mcadams_app.live`. |
| `scripts/` | Evaluación: construcción del subconjunto de Common Voice, anonimización, embeddings, EER, suplantación, transcripción, WER, barridos de α y figuras. Sus pruebas están en `scripts/tests/`. |
| `notebooks/` | Análisis exploratorio de Common Voice en español. |
| `articulo/` | `Proyecto_de_grado.tex`. La compilación va en `articulo/build/`, que git ignora. |

Los datos y resultados no están en el repositorio. Los scripts y el prototipo
usan por defecto la carpeta `..\Datos`, al lado del repositorio.

## Entornos

Hay dos entornos de Python 3.12, uno para los paquetes y otro para la
evaluación:

| Entorno | Dónde | Contiene | Se usa para |
|---|---|---|---|
| `.venv` | Raíz del repositorio (ignorado por git) | `mcadams-core` y `mcadams-app` instalados en modo editable, numpy, scipy, soundfile, sounddevice y pytest | Pruebas de `packages/` y prototipo en vivo |
| `venv-eval` | Carpeta padre del repositorio (`..\venv-eval`) | torch y torchaudio (CPU), speechbrain, openai-whisper, num2words, matplotlib, pandas, numpy, scipy, soundfile, pytest y jupyter. Las versiones exactas están en `requirements-eval.txt` | Scripts de evaluación y sus pruebas |

### Recrear `.venv`

`.venv` se maneja con [uv](https://docs.astral.sh/uv/) como un workspace:
`pyproject.toml` (raíz) y `packages/*`, con las versiones fijadas en `uv.lock`.
Desde la raíz del repositorio:

```powershell
uv sync --all-packages    # lo mismo que sync.ps1
```

Los comandos de este README no usan `uv run` sino `.\.venv\Scripts\python.exe`
directamente, porque en el equipo de desarrollo Windows Smart App Control
bloquea `uv.exe`.

Hay que volver a correr `uv sync --all-packages` cuando cambian las
dependencias o los scripts declarados en un `pyproject.toml`. Por ejemplo, el
comando `mcadams-app` solo se actualiza al reinstalar el paquete.

### Recrear `venv-eval`

`mcadams_core` no se instala en `venv-eval`. Los scripts lo encuentran por
`PYTHONPATH` (ver [Pipeline de evaluación](#pipeline-de-evaluación)). Desde la
carpeta padre del repositorio, con Python 3.12:

```powershell
python -m venv venv-eval
venv-eval\Scripts\python.exe -m pip install -r Coeficiente-de-McAdams\requirements-eval.txt
```

Todo sale de PyPI. torch y torchaudio son las builds para CPU:
`torch.__version__` reporta `2.14.1+cpu` aunque el paquete se llama
`torch==2.14.1`. No hace falta el índice de PyTorch (`download.pytorch.org`).
Si se agrega con `--extra-index-url`, pip podría cambiar torch y torchaudio por
sus builds CUDA, porque `torch==2.14.1` también acepta versiones locales como
`2.14.1+cu126`.

## Pruebas

Desde la raíz del repositorio:

```powershell
.\.venv\Scripts\python.exe -m pytest packages        # núcleo y prototipo
..\venv-eval\Scripts\python.exe -m pytest scripts    # scripts de evaluación
```

La configuración de pytest (`pyproject.toml` raíz) ya agrega `scripts`,
`packages/core/src` y `packages/core/tests` a `sys.path`, así que no hace falta
fijar `PYTHONPATH` para las pruebas.

Las pruebas de `scripts/` no corren en `.venv`: falta matplotlib (y torch,
speechbrain y whisper), así que `test_alpha_differences.py` no se puede
importar y otras pruebas se saltan.

## Prototipo en vivo

Con `.venv`, desde la raíz del repositorio:

```powershell
# Lista los dispositivos de audio y sus índices
.\.venv\Scripts\python.exe -m mcadams_app.live.run --list-devices

# Paso directo: la salida es igual a la entrada (mide la latencia del equipo)
.\.venv\Scripts\python.exe -m mcadams_app.live.run --mode paso_directo --seconds 20 --input-device 1 --output-device 4

# Anonimizado con McAdams
.\.venv\Scripts\python.exe -m mcadams_app.live.run --mode anonimizado --seconds 20 --input-device 1 --output-device 4
```

| Opción | Por defecto | Descripción |
|---|---|---|
| `--list-devices` | | Imprime los dispositivos de audio y termina. |
| `--mode` | `paso_directo` | `paso_directo` o `anonimizado`. |
| `--input-device` | el del sistema | Índice del micrófono (de `--list-devices`). |
| `--output-device` | el del sistema | Índice de los audífonos (de `--list-devices`). |
| `--seconds` | `10` | Duración de la prueba. Ctrl+C la termina antes. |
| `--blocksize` | `160` | Muestras por bloque (160 = 10 ms a 16 kHz). |
| `--latency` | `low` | Latencia pedida al dispositivo: `low`, `high` o segundos. |
| `--sample-rate` | `16000` | Frecuencia de muestreo en Hz. |
| `--output-root` | `..\Datos\en_vivo` | Carpeta donde se crea la carpeta de cada corrida. |

Cada corrida crea `<AAAAMMDD_HHMMSS>_<modo>\` con `entrada.wav`,
`salida.wav`, `tiempos.tsv` y `resumen.json`, y muestra en consola la latencia
y el costo por bloque. En modo anonimizado cada corrida usa un `session_id`
nuevo (uuid4), y el α sale del rango por defecto del núcleo.

Con el paquete instalado, `mcadams-app` es el mismo comando:
`.\.venv\Scripts\mcadams-app.exe --list-devices`.

## Pipeline de evaluación

Todos los comandos se corren desde la raíz del repositorio con `venv-eval`.
`anonymize_subset.py` importa `mcadams_core`, que no está instalado en
`venv-eval`, así que primero se fija el `PYTHONPATH` igual que lo hace
`run_alpha_sweep.py`:

```powershell
$py = "..\venv-eval\Scripts\python.exe"
$env:PYTHONPATH = "scripts;packages\core\src"
```

Marcadores usados en los comandos:

- `<ruta_common_voice>`: carpeta `es` del corpus de Common Voice, con
  `validated.tsv`, `clip_durations.tsv` y `clips\` (los MP3).
- `<ruta_datos>`: carpeta de datos y resultados. Los scripts de barridos y
  análisis la reciben como `--data-dir` y por defecto usan `..\Datos`.

Los nombres de archivo de abajo son los que buscan `run_alpha_sweep.py` y los
scripts de análisis para la configuración de referencia U(0.7, 0.9) (etiqueta
`u07_09`, subconjunto `k134`). Las opciones entre corchetes son opcionales y
muestran su valor por defecto.

### 1. Subconjunto de Common Voice: `build_subset.py`

```powershell
& $py scripts\build_subset.py --validated <ruta_common_voice>\validated.tsv --durations <ruta_common_voice>\clip_durations.tsv --output <ruta_datos>\subset_k134.tsv
```

Opciones: `[--k 134]` hablantes por estrato de acento, `[--clips 5]` clips por
hablante, `[--seed 42]`.

### 2. Anonimización: `anonymize_subset.py`

```powershell
& $py scripts\anonymize_subset.py --subset <ruta_datos>\subset_k134.tsv --clips-dir <ruta_common_voice>\clips --output-dir <ruta_datos>\anon_k134
```

Escribe `original\` (los WAV originales a 16 kHz), `anon_u07_09\` y
`anon_manifest_u07_09.tsv` dentro de `--output-dir`. Opciones:

- `[--alpha A | --alpha-range MIN MAX]`: α fijo o rango, mutuamente
  excluyentes. Sin ninguno se usa U(0.7, 0.9).
- `[--tag]`: por defecto sale del rango (`u07_09`, `a080`, ...).
- `[--reuse-original]`: lee los WAV de `original\` en vez de volver a
  convertir los MP3.

### 3. Embeddings ECAPA: `extract_embeddings.py`

```powershell
& $py scripts\extract_embeddings.py --manifest <ruta_datos>\anon_k134\anon_manifest_u07_09.tsv --original-dir <ruta_datos>\anon_k134\original --anon-dir <ruta_datos>\anon_k134\anon_u07_09 --model-dir <ruta_datos>\ecapa --output <ruta_datos>\emb_k134.npz
```

Usa `speechbrain/spkrec-ecapa-voxceleb`, que se copia en `--model-dir`.
Opciones: `[--limit N]` procesa solo N clips, `[--reuse-original-from <npz>]`
toma los embeddings originales de otro `.npz`.

### 4. EER: `run_eer.py` y `eer_bootstrap.py`

`run_eer.py` calcula el EER puntual de los escenarios o-o, o-a y a-a.
`eer_bootstrap.py` agrega los intervalos de confianza por bootstrap sobre
hablantes y guarda las réplicas.

```powershell
& $py scripts\run_eer.py --embeddings <ruta_datos>\emb_k134.npz --manifest <ruta_datos>\anon_k134\anon_manifest_u07_09.tsv --output <ruta_datos>\eer_k134.json

& $py scripts\eer_bootstrap.py --embeddings <ruta_datos>\emb_k134.npz --manifest <ruta_datos>\anon_k134\anon_manifest_u07_09.tsv --output <ruta_datos>\eer_bootstrap_k134.json --replicates-output <ruta_datos>\eer_bootstrap_k134_replicas.npz
```

Opciones de ambos: `[--seed 42] [--n-enroll 3] [--n-trial 2]`. Solo de
`eer_bootstrap.py`: `[--n-boot 1000] [--boot-seed 2026] [--level 0.95]`.

### 5. Suplantación: `impersonation.py`

```powershell
& $py scripts\impersonation.py --embeddings <ruta_datos>\emb_k134.npz --manifest <ruta_datos>\anon_k134\anon_manifest_u07_09.tsv --output <ruta_datos>\impersonation_k134.json
```

Opciones: las mismas de `eer_bootstrap.py` (sin `--replicates-output`), más
`[--max-far 0.01]`.

### 6. Transcripción con Whisper: `transcribe.py`

```powershell
& $py scripts\transcribe.py --manifest <ruta_datos>\anon_k134\anon_manifest_u07_09.tsv --subset <ruta_datos>\subset_k134.tsv --original-dir <ruta_datos>\anon_k134\original --anon-dir <ruta_datos>\anon_k134\anon_u07_09 --model turbo --model-dir <ruta_datos>\whisper --per-stratum 134 --output <ruta_datos>\whisper_turbo_k134.tsv
```

`--model` es el modelo de Whisper (`small`, `medium`, `turbo`, ...) y se
descarga en `--model-dir`. `--per-stratum` es el número de clips por estrato;
si se omite, se transcriben todos (el barrido usa 134). Opciones: `[--seed 42]`
y `[--reuse-original-from <tsv>]`, que copia las filas originales de otro TSV.

Es reanudable: con el mismo `--output` se salta lo ya transcrito.

### 7. WER: `wer_bootstrap.py`

```powershell
& $py scripts\wer_bootstrap.py --transcripts <ruta_datos>\whisper_turbo_k134.tsv --output <ruta_datos>\wer_turbo_k134.json --model-label turbo
```

Opciones: `[--normalization final|provisional]` (por defecto `final`),
`[--n-boot 1000] [--boot-seed 2026] [--level 0.95]`.

### Barridos y análisis

**`run_alpha_sweep.py`** corre los pasos 2, 3, 4 (`eer_bootstrap`), 5, 6 y 7
para cada configuración de α. Reutiliza los WAV, los embeddings y las
transcripciones originales de la referencia. Es reanudable: salta los pasos
cuya salida ya existe. Agrega por su cuenta `scripts` y `packages\core\src` al
`PYTHONPATH` de cada paso.

```powershell
& $py scripts\run_alpha_sweep.py --dry-run                                        # solo imprime los comandos
& $py scripts\run_alpha_sweep.py                                                  # α fijos 0.2, 0.3, ..., 1.0
& $py scripts\run_alpha_sweep.py --alphas 0.6 0.7 0.8
& $py scripts\run_alpha_sweep.py --range 0.65 0.85 --range 0.685 0.865            # rangos U(MIN, MAX)
& $py scripts\run_alpha_sweep.py --steps anonymize embeddings eer impersonation   # el WER después
```

Opciones:

- Rutas. Por defecto salen de `[--data-dir ..\Datos]`: `--subset`,
  `--clips-dir`, `--anon-root`, `--output-dir`, `--base-embeddings`,
  `--base-transcripts`, `--ecapa-dir` y `--whisper-dir`.
- Parámetros de los pasos: `[--dataset-tag k134] [--whisper-model turbo]
  [--per-stratum 134] [--seed 42] [--n-enroll 3] [--n-trial 2]
  [--n-boot 1000] [--boot-seed 2026] [--level 0.95] [--max-far 0.01]`.
- `[--python]`: el Python de cada paso. Por defecto, el mismo que corre el
  barrido.

**Análisis.** Estos scripts solo leen resultados ya generados. Todos aceptan
`[--data-dir ..\Datos]`. Los tres primeros aceptan además los parámetros de
bootstrap de los pasos anteriores (`--seed`, `--n-enroll`, `--n-trial`,
`--n-boot`, `--boot-seed`, `--level`); `group_analysis.py` y
`alpha_differences.py` también aceptan `--max-far`.

| Script | Comando mínimo | Salida por defecto y opciones propias |
|---|---|---|
| `range_comparison.py` | `& $py scripts\range_comparison.py` | Compara los 9 rangos uniformes con U(0.7, 0.9). Escribe en `<data-dir>\rangos` y `<data-dir>\figuras\rangos`. Opciones: `--output-dir`, `--figures-dir`, `--tags`, `--reference u07_09`. |
| `group_analysis.py` | `& $py scripts\group_analysis.py --group-by stratum` | `--group-by` es `stratum`, `gender` o `all`. Escribe en `<data-dir>\por_grupo` y `<data-dir>\figuras\por_grupo`. Opciones: `--only-gender`, `--configs`, `--n-perm 5000`, `--perm-seed 2026`, `--figures-only`. |
| `alpha_differences.py` | `& $py scripts\alpha_differences.py` | Diferencias pareadas entre α vecinos del barrido. Escribe `<data-dir>\barrido_alfa_diferencias.tsv`. Opciones: `--output`, `--tags`. |
| `plot_alpha_sweep.py` | `& $py scripts\plot_alpha_sweep.py` | Escribe `<data-dir>\barrido_alfa_resumen.tsv`, `<data-dir>\figuras\barrido_alfa.pdf` y `barrido_alfa_zoom.pdf`. Opciones: `--tags`, `--anon-root`, `--output-dir`, `--dataset-tag k134`, `--whisper-model turbo`, `--summary`, `--figure`, `--zoom-figure`. |

Las figuras del artículo:

```powershell
# Curvas DET y distribuciones de puntuaciones; con --bootstrap-json también eer_ic.pdf
& $py scripts\plot_eer.py --embeddings <ruta_datos>\emb_k134.npz --manifest <ruta_datos>\anon_k134\anon_manifest_u07_09.tsv --output-dir <ruta_datos>\figuras --bootstrap-json <ruta_datos>\eer_bootstrap_k134.json

# suplantacion.pdf, wer.pdf y compromiso.pdf (--config LABEL EER_JSON WER_JSON, se puede repetir)
& $py scripts\plot_results.py --output-dir <ruta_datos>\figuras --impersonation-json <ruta_datos>\impersonation_k134.json --wer-json <ruta_datos>\wer_turbo_k134.json --config "U(0.7, 0.9)" <ruta_datos>\eer_bootstrap_k134.json <ruta_datos>\wer_turbo_k134.json
```

## Rango del coeficiente α

Hay dos rangos por defecto, a propósito:

- **Evaluación: U(0.7, 0.9)**, en `scripts/alpha_config.py` (`DEFAULT_RANGE`,
  etiqueta `u07_09`). Es el rango de la evaluación original, tomado de Patino
  et al. (2021), y es la referencia con la que se comparan las demás
  configuraciones. `anonymize_subset.py` lo usa cuando no se le da
  `--alpha` ni `--alpha-range`.
- **Núcleo: U(0.685, 0.865)**, en
  `packages/core/src/mcadams_core/config/mcadams_defaults.py`. Es el rango
  elegido en este proyecto comparando nueve rangos con EER y WER. Lo usan
  `AnonymizerStream`, `BlockAnonymizer`, `anonymize_utterance` y el prototipo
  en vivo cuando no reciben un rango.

Para usar otro rango (siempre con `0 < alpha_min <= alpha_max <= 1`):

- **Evaluación**: `anonymize_subset.py --alpha-range 0.685 0.865` (etiqueta
  `u0685_0865`) o `--alpha 0.8` (α fijo, etiqueta `a080`). Con
  `run_alpha_sweep.py`, `--range MIN MAX` (se puede repetir) o
  `--alphas A1 A2 ...`.
- **Código**: los argumentos `alpha_min` y `alpha_max` de `AnonymizerStream`,
  `BlockAnonymizer` y `anonymize_utterance`. Con los dos iguales, el α es
  fijo.
- **Prototipo en vivo**: no tiene opción de línea de comandos para α. Siempre
  usa el rango del núcleo.
