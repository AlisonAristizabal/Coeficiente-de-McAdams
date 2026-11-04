"""Prueba de paso directo (micrófono → audífonos, sin anonimizar).

Mide la latencia propia del equipo y el costo del callback, y guarda lo
grabado en una carpeta nueva dentro de ..\\Datos\\en_vivo. Se ejecuta con:

    python -m mcadams_app.live.run --seconds 20 --input-device 1 --output-device 4

El procesador se configura en la LiveSession y el modo solo nombra la carpeta,
así que el modo con anonimización no requiere cambiar `run_session` ni
`save_results`."""
import argparse
import csv
import itertools
import json
import math
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd

from mcadams_app.live.session import LiveSession
from mcadams_core.io.audio_io import save_wav

REPO = Path(__file__).resolve().parents[5]
DEFAULT_OUTPUT_ROOT = REPO.parent / "Datos" / "en_vivo"

CHANNELS = 1
DTYPE = "float32"
MODE_PASSTHROUGH = "paso_directo"


def positive_int(text):
    """Convierte un argumento a entero mayor que 0."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"se esperaba un entero: {text}") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"debe ser mayor que 0: {text}")
    return value


def positive_float(text):
    """Convierte un argumento a número finito mayor que 0."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"se esperaba un número: {text}") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"debe ser un número finito mayor que 0: {text}")
    return value


def parse_latency(text):
    """Convierte --latency: "low", "high" o un número de segundos mayor que 0."""
    if text in ("low", "high"):
        return text
    return positive_float(text)


def parse_args(argv=None):
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description="Prueba de paso directo: micrófono a audífonos, sin anonimizar.")
    parser.add_argument("--seconds", type=positive_float, default=10.0, help="Segundos de la prueba.")
    parser.add_argument("--input-device", type=int, default=None,
                        help="Índice del micrófono en sounddevice (por defecto, el del sistema).")
    parser.add_argument("--output-device", type=int, default=None,
                        help="Índice de los audífonos en sounddevice (por defecto, el del sistema).")
    parser.add_argument("--sample-rate", type=positive_int, default=16000, help="Frecuencia de muestreo en Hz.")
    parser.add_argument("--blocksize", type=positive_int, default=160,
                        help="Muestras por bloque (160 = 10 ms a 16 kHz).")
    parser.add_argument("--latency", type=parse_latency, default="low",
                        help='Latencia pedida al dispositivo: "low", "high" o un número en segundos.')
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT),
                        help="Carpeta donde se crea la carpeta de cada corrida.")
    parser.add_argument("--list-devices", action="store_true", help="Imprime los dispositivos de audio y termina.")
    return parser.parse_args(argv)


def validate_devices(input_device, output_device, sample_rate):
    """Comprueba que los dispositivos aceptan la frecuencia, un canal y float32.

    Parámetros:
        input_device, output_device: índices de sounddevice; None usa el del
            sistema.
        sample_rate: frecuencia de muestreo en Hz.

    Lanza SystemExit con un mensaje que dice qué dispositivo falló y por qué."""
    checks = (("entrada", input_device, sd.check_input_settings),
              ("salida", output_device, sd.check_output_settings))
    failures = []
    for kind, device, check in checks:
        try:
            check(device=device, channels=CHANNELS, dtype=DTYPE, samplerate=sample_rate)
        except (sd.PortAudioError, ValueError) as error:
            label = "predeterminado del sistema" if device is None else device
            failures.append(f"El dispositivo de {kind} ({label}) no acepta {sample_rate} Hz, "
                            f"{CHANNELS} canal y {DTYPE}: {error}")
    if failures:
        raise SystemExit("\n".join(failures))


def describe_device(device, kind):
    """Devuelve el índice, el nombre y la API de host de un dispositivo.

    Parámetros:
        device: índice de sounddevice; None usa el del sistema.
        kind: "input" u "output"."""
    info = sd.query_devices(device, kind)
    return {"index": info["index"], "name": info["name"], "hostapi": sd.query_hostapis(info["hostapi"])["name"]}


def version_info():
    """Devuelve las versiones de Python, sounddevice y PortAudio."""
    return {
        "python": platform.python_version(),
        "sounddevice": sd.__version__,
        "portaudio": sd.get_portaudio_version()[1],
    }


def run_session(session, input_device, output_device, blocksize, latency, seconds,
                stream_class=sd.Stream, wait=time.sleep):
    """Abre el stream, espera los segundos pedidos y lo cierra.

    Ctrl+C termina la espera: el `with` cierra el stream de forma ordenada y
    lo grabado queda en la sesión. La espera por defecto es time.sleep porque
    Ctrl+C la interrumpe en Windows.

    Parámetros:
        session: LiveSession que recibe los bloques.
        input_device, output_device: índices de sounddevice; None usa el del
            sistema.
        blocksize: muestras por bloque.
        latency: "low", "high" o un número en segundos.
        seconds: segundos de espera.
        stream_class: clase del stream (los tests pasan un stream falso).
        wait: función de espera, recibe los segundos.

    Retorna:
        Una tupla (entrada, salida) con la latencia que reportó el stream, en
        segundos; (None, None) si se interrumpió antes de leerla."""
    stream_latency = (None, None)
    try:
        with stream_class(device=(input_device, output_device), channels=CHANNELS, dtype=DTYPE,
                          samplerate=session.sample_rate, blocksize=blocksize, latency=latency,
                          callback=session.callback) as stream:
            stream_latency = stream.latency
            wait(seconds)
    except KeyboardInterrupt:
        pass
    return stream_latency


def create_run_dir(output_root, mode, now=datetime.now):
    """Crea la carpeta <AAAAMMDD_HHMMSS>_<modo> de una corrida.

    Si ya existe una carpeta con ese nombre (dos corridas en el mismo segundo),
    agrega _2, _3, ... de modo que nunca se escribe sobre una corrida anterior.

    Retorna:
        La ruta de la carpeta creada."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    base = f"{now().strftime('%Y%m%d_%H%M%S')}_{mode}"
    for attempt in itertools.count(1):
        run_dir = output_root / (base if attempt == 1 else f"{base}_{attempt}")
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        return run_dir


def join_blocks(blocks):
    """Concatena los bloques; sin bloques devuelve un arreglo vacío."""
    return np.concatenate(blocks) if blocks else np.empty(0, dtype=DTYPE)


def write_times(path, process_times):
    """Escribe tiempos.tsv: una fila por bloque, con el tiempo de proceso en ms."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["block_index", "process_ms"])
        writer.writerows((index, 1000 * seconds) for index, seconds in enumerate(process_times))


def _to_ms(seconds):
    """Pasa de segundos a milisegundos, conservando None."""
    return None if seconds is None else 1000 * seconds


def save_results(session, stream_latency, settings, devices, versions, output_root, now=datetime.now):
    """Guarda los WAV, tiempos.tsv y resumen.json en una carpeta nueva.

    Se debe llamar con el stream ya cerrado.

    Parámetros:
        session: LiveSession con lo grabado.
        stream_latency: tupla (entrada, salida) de `run_session`, en segundos.
        settings: diccionario con mode, seconds_requested, blocksize y
            latency_requested.
        devices: diccionario {"input": ..., "output": ...} de `describe_device`.
        versions: salida de `version_info`.
        output_root: carpeta donde se crea la carpeta de la corrida.
        now: función que devuelve la fecha y hora actuales.

    Retorna:
        Una tupla (carpeta de la corrida, contenido de resumen.json)."""
    run_dir = create_run_dir(output_root, settings["mode"], now)
    input_audio = join_blocks(session.input_blocks)
    save_wav(run_dir / "entrada.wav", session.sample_rate, input_audio)
    save_wav(run_dir / "salida.wav", session.sample_rate, join_blocks(session.output_blocks))
    write_times(run_dir / "tiempos.tsv", session.process_times)

    summary = {
        "session": session.summary(),
        "stream_latency_ms": {"input": _to_ms(stream_latency[0]), "output": _to_ms(stream_latency[1])},
        "config": {
            "mode": settings["mode"],
            "seconds_requested": settings["seconds_requested"],
            "seconds_recorded": input_audio.size / session.sample_rate,
            "sample_rate": session.sample_rate,
            "blocksize": settings["blocksize"],
            "latency_requested": settings["latency_requested"],
        },
        "devices": devices,
        "versions": versions,
    }
    with open(run_dir / "resumen.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return run_dir, summary


def _number(value, decimals=2):
    """Escribe un número con coma decimal; "n/d" si no hay valor."""
    return "n/d" if value is None else f"{value:.{decimals}f}".replace(".", ",")


def print_report(run_dir, summary):
    """Imprime el resumen corto de la corrida en la consola."""
    session, latency, config = summary["session"], summary["stream_latency_ms"], summary["config"]
    block_ms = 1000 * config["blocksize"] / config["sample_rate"]
    if latency["input"] is None or latency["output"] is None:
        estimated = None
    else:
        estimated = latency["input"] + latency["output"] + block_ms
    print(f"Carpeta de salida: {run_dir}")
    print(f"Bloques: {session['n_blocks']}")
    print(f"Overflows de entrada: {session['input_overflows']}")
    print(f"Underflows de salida: {session['output_underflows']}")
    print(f"Tiempo por bloque: mediana {_number(session['block_ms_median'], 3)} ms, "
          f"p99 {_number(session['block_ms_p99'], 3)} ms, máximo {_number(session['block_ms_max'], 3)} ms")
    print(f"RTF: {_number(session['rtf'], 4)}")
    print(f"Latencia del dispositivo: entrada {_number(latency['input'])} ms, salida {_number(latency['output'])} ms")
    print(f"Latencia estimada (entrada + salida + un bloque de {_number(block_ms)} ms): {_number(estimated)} ms")


def main(argv=None, stream_class=sd.Stream, wait=time.sleep):
    """Corre la prueba de paso directo y guarda los resultados."""
    args = parse_args(argv)
    if args.list_devices:
        print(sd.query_devices())
        return

    validate_devices(args.input_device, args.output_device, args.sample_rate)
    devices = {"input": describe_device(args.input_device, "input"),
               "output": describe_device(args.output_device, "output")}

    session = LiveSession(args.sample_rate)
    try:
        stream_latency = run_session(session, args.input_device, args.output_device, args.blocksize, args.latency,
                                     args.seconds, stream_class, wait)
    except sd.PortAudioError as error:
        raise SystemExit(f"No se pudo usar el stream de audio con entrada {devices['input']['name']} "
                         f"y salida {devices['output']['name']}: {error}") from error

    settings = {"mode": MODE_PASSTHROUGH, "seconds_requested": args.seconds, "blocksize": args.blocksize,
                "latency_requested": args.latency}
    run_dir, summary = save_results(session, stream_latency, settings, devices, version_info(), args.output_root)
    print_report(run_dir, summary)


if __name__ == "__main__":
    main()
