import time
import numpy as np

class LiveSession:
    """Sesión de audio en vivo: recibe cada bloque del micrófono, lo pasa por
    un procesador opcional y entrega el resultado a los audífonos.

    Sin procesador funciona como paso directo (la salida es igual a la
    entrada), lo que sirve para medir la latencia propia del equipo. Con
    procesador (por ejemplo, la anonimización) se mide además el costo del
    procesamiento. En los dos casos guarda los bloques de entrada y salida,
    el tiempo de proceso de cada bloque y cuántas veces se perdió audio, para
    analizarlos cuando termine la sesión."""

    def __init__(self, sample_rate, processor=None, clock=time.perf_counter):
        """Prepara una sesión vacía.

        Parámetros:
            sample_rate: frecuencia de muestreo en Hz (por ejemplo 16000).
            processor: función que recibe un bloque 1D y devuelve otro del
                mismo largo. Con None la salida es igual a la entrada.
            clock: función que devuelve el tiempo en segundos. Por defecto
                time.perf_counter; los tests pasan un reloj falso para
                controlar el tiempo medido."""
        self.sample_rate = sample_rate
        self.processor = processor
        self.clock = clock

        self.input_blocks = []
        self.output_blocks = []
        self.process_times = []

        self.input_overflows = 0
        self.output_underflows = 0

    def callback(self, indata, outdata, frames, time_info, status):
        """Procesa un bloque de audio. La llama sounddevice cada vez que el
        micrófono entrega un bloque (cada 10 ms con bloques de 160 muestras a
        16 kHz).

        Toma la columna del bloque de entrada, la pasa por el procesador (o la
        deja igual si no hay), escribe el resultado en outdata, guarda copias
        de la entrada y la salida, y registra cuánto tardó. Si status indica
        que se perdió audio de entrada (overflow) o que faltó audio de salida
        (underflow), suma 1 al contador correspondiente.

        Reglas, porque esta función corre dentro del hilo de audio:
            - Nada lento adentro (print, escribir archivos, esperar): si
              tarda más que la duración del bloque, el audio se entrecorta.
            - Hay que llenar outdata completo en cada llamada.
            - indata se copia antes de guardarlo, porque sounddevice reutiliza
              el mismo arreglo en la llamada siguiente.

        Parámetros:
            indata: arreglo (frames, 1) con el audio del micrófono.
            outdata: arreglo (frames, 1) que se debe llenar con la salida.
            frames: número de muestras del bloque.
            time_info: tiempos del dispositivo que entrega sounddevice (no se
                usa).
            status: banderas de sounddevice (input_overflow, output_underflow).

        Lanza ValueError si el procesador devuelve un bloque de otro largo."""
        start = self.clock()

        if status.input_overflow:
            self.input_overflows += 1
        if status.output_underflow:
            self.output_underflows += 1

        block = indata[:, 0].copy()

        if self.processor is None:
            output = block
        else:
            output = self.processor(block)

        if len(output) != frames:
            raise ValueError(f"Output length {len(output)} does not match expected length {frames}")

        outdata[:, 0] = output
        self.output_blocks.append(outdata[:, 0].copy())
        self.input_blocks.append(block)
        self.process_times.append(self.clock() - start)

    def summary(self):
        """Resume la sesión en un diccionario.

        Claves:
            n_blocks: número de bloques procesados.
            input_overflows: veces que se perdió audio de entrada.
            output_underflows: veces que faltó audio de salida.
            block_ms_median, block_ms_p99, block_ms_max: tiempo de proceso por
                bloque en milisegundos (mediana, percentil 99 y máximo).
            rtf: factor de tiempo real, el tiempo total de proceso dividido por
                la duración total del audio. Menor es mejor: con RTF < 1 el
                procesamiento alcanza a ir al ritmo del audio, y la meta del
                proyecto es RTF < 0,1.

        Si todavía no hay bloques, n_blocks es 0 y las métricas de tiempo y
        el rtf son None."""
        n_blocks = len(self.input_blocks)

        if n_blocks == 0:
            return {
                "n_blocks": 0,
                "input_overflows": self.input_overflows ,
                "output_underflows": self.output_underflows,
                "block_ms_median": None,
                "block_ms_p99": None,
                "block_ms_max": None,
                "rtf": None,
            }

        times_ms = np.array(self.process_times) * 1000  # process_times como arreglo, en ms
        audio_seconds = sum(len(b) for b in self.input_blocks) / self.sample_rate
        process_seconds = np.sum(times_ms) / 1000  # suma de process_times

        return {
            "n_blocks": n_blocks,
            "input_overflows": self.input_overflows ,
            "output_underflows": self.output_underflows,
            "block_ms_median": float(np.median(times_ms)),
            "block_ms_p99": float(np.percentile(times_ms, 99)),
            "block_ms_max": float(np.max(times_ms)),
            "rtf": float(process_seconds / audio_seconds),
        }
    
    