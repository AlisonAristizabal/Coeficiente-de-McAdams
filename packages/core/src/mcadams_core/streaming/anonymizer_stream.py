import numpy as np
from mcadams_core.signal_transform.framing import (
    compute_frame_params,
    hann_window,
    apply_window
)
from mcadams_core.config.lpc_defaults import (
    compute_lpc_order,
)
from mcadams_core.session.coefficient import (
    generate_coefficient,
)
from mcadams_core.signal_transform.mcadams import (
    process_frame,
)

class AnonymizerStream:
    def __init__(self, sample_rate, session_id):
        """Inicializa el flujo de procesamiento de audio para la
        anonimización de voz. Calcula los parámetros de enventanado y
        orden LPC según la frecuencia de muestreo, y genera un coeficiente
        de McAdams determinístico a partir del identificador de sesión
        (session_id)."""

        self.sample_rate = sample_rate
        self.frame_length, self.hop_length = compute_frame_params(sample_rate)
        self.order = compute_lpc_order(sample_rate)
        self.mcadams_coefficient, self.seed = generate_coefficient(session_id)
        self.window = hann_window(self.frame_length)

        self.buffer_original_audio = np.array([])  # Buffer para almacenar audio entrante
        self.buffer_overlap = np.zeros(self.frame_length - self.hop_length)  # Buffer para almacenar la superposición de audio

    def process_chunk(self, chunk):
        """Procesa un fragmento de audio entrante, aplicando el pipeline de
        anonimización de voz. El fragmento se agrega al buffer de audio
        original, se enventana y se procesa cada fragmento resultante a
        través del pipeline de análisis y resíntesis. Se maneja la
        superposición entre fragmentos para asegurar una transición suave.
        Devuelve el audio reconstruido y anonimizado. Cuando el fragmento 
        entrante no alcanza para completar ni un frame se devulve un arreglo vacio."""

        self.buffer_original_audio = np.concatenate((self.buffer_original_audio, chunk))
        left_overlap = []

        while len(self.buffer_original_audio) >= self.frame_length:
            frame = self.buffer_original_audio[:self.frame_length]
            windowed_frame = apply_window(frame, self.window)
            reconstructed_frame = process_frame(windowed_frame, self.order, self.mcadams_coefficient)

            left_edge = reconstructed_frame[:self.hop_length]
            right_edge = reconstructed_frame[self.hop_length:]

            ready_output = self.buffer_overlap + left_edge
            left_overlap.append(ready_output)
            self.buffer_overlap = right_edge
            self.buffer_original_audio = self.buffer_original_audio[self.hop_length:]

        if left_overlap:
            return np.concatenate(left_overlap)
        else: 
            return np.array([])

    def flush(self):
        """Cierra el stream y libera todo el audio que quedó pendiente sin
        llegar a completar un frame ni a recibir su contribución faltante.

        Si no queda audio crudo sin procesar (buffer_original_audio vacío),
        devuelve directamente el contenido de buffer_overlap, que ya
        representa el tramo final cerrado.

        Si queda audio crudo sin llegar a frame_length, lo completa con
        ceros hasta ese tamaño y lo procesa como un frame final — a
        diferencia de process_chunk, acá no se guarda ninguna porción como
        pendiente, porque no va a llegar un frame siguiente que la
        complete: tanto la mitad combinada con buffer_overlap como la
        mitad restante del frame final se devuelven juntas.

        En ambos casos, deja el stream en su estado inicial (buffers
        vacíos/en cero) — el objeto no debe usarse para procesar más audio
        después de llamar a este método."""
        
        original_length = self.buffer_original_audio.size

        if original_length == 0:
            output = self.buffer_overlap
            self.buffer_overlap = np.zeros(self.frame_length - self.hop_length)
            return output
        else:
            pad = np.zeros(self.frame_length-original_length)
            padded_frame = np.concatenate([self.buffer_original_audio, pad])
            windowed_frame = apply_window(padded_frame, self.window)
            reconstructed_frame = process_frame(windowed_frame, self.order, self.mcadams_coefficient)

            left_edge = reconstructed_frame[:self.hop_length]
            right_edge = reconstructed_frame[self.hop_length:]

            ready_output = self.buffer_overlap + left_edge

            output = np.concatenate([ready_output, right_edge])

            self.buffer_original_audio = np.array([])
            self.buffer_overlap = np.zeros(self.frame_length - self.hop_length)
            
            return output

