from math import gcd

import numpy as np

from mcadams_core.streaming.anonymizer_stream import AnonymizerStream


def minimum_delay(frame_length, hop_length, blocksize=None):
    """Calcula el menor retraso constante, en muestras, con el que siempre
    hay salida disponible para entregar un bloque completo.

    Sea F = frame_length y H = hop_length. AnonymizerStream solo entrega
    audio cuando completa un frame: con n >= F muestras de entrada ya
    procesó floor((n - F) / H) + 1 frames y entregó H muestras por cada uno.
    Escribiendo n = F + q * H + r, con 0 <= r < H, las muestras que faltan
    por entregar son n - (q + 1) * H = F - H + r, y con n < F faltan las n.
    El retraso debe cubrir el mayor faltante, y este depende del valor más
    alto de r que se pueda alcanzar:

    - Bloques de tamaño fijo B: n siempre es múltiplo de g = gcd(B, H), así
      que el mayor r es H - g y el retraso es F - g. Si B es múltiplo del
      salto, g = H y el retraso es F - H, el menor de todos (160 muestras con
      F = 320 y H = 160). Con B = 100, g = 20 y el retraso es 300 muestras.
    - Bloques de largo variable o desconocido (blocksize None): n puede
      caer en cualquier valor, r llega a H - 1 y el retraso es F - 1.

    Parámetros:
        frame_length: longitud del frame en muestras (F).
        hop_length: salto en muestras (H).
        blocksize: tamaño fijo de los bloques; None si no se conoce.

    Retorna:
        El retraso en muestras (entero)."""
    if blocksize is None:
        return frame_length - 1
    return frame_length - gcd(blocksize, hop_length)


class BlockAnonymizer:
    """Adapta AnonymizerStream a una salida del mismo largo que la entrada.

    AnonymizerStream.process_chunk entrega un largo variable (vacío al
    inicio). Este adaptador guarda su salida en una cola precargada con
    delay_samples ceros y saca de ella len(block) muestras en cada llamada,
    de modo que process se puede usar como processor de LiveSession. La
    salida es la de AnonymizerStream retrasada exactamente delay_samples.

    Atributos:
        session_id: identificador de la sesión.
        coefficient: coeficiente de McAdams de la sesión.
        blocksize: tamaño fijo de los bloques, o None.
        delay_samples: retraso constante en muestras; ver `minimum_delay`.
        delay_ms: el mismo retraso en milisegundos."""

    def __init__(self, sample_rate, session_id, alpha_min=None, alpha_max=None, blocksize=None):
        """Prepara el adaptador con la cola precargada.

        Parámetros:
            sample_rate: frecuencia de muestreo en Hz.
            session_id: identificador que fija el coeficiente de McAdams.
            alpha_min, alpha_max: rango del coeficiente (iguales para un
                alpha fijo); None usa el rango por defecto del núcleo.
            blocksize: tamaño fijo de los bloques. Con este dato el retraso
                es el mínimo para ese tamaño y process solo acepta bloques de
                ese largo. Con None acepta cualquier largo, pero el retraso
                es F - 1 (ver `minimum_delay`).

        Lanza ValueError si blocksize no es mayor que 0."""
        if blocksize is not None and blocksize <= 0:
            raise ValueError(f"blocksize debe ser mayor que 0: {blocksize}")

        self._stream = AnonymizerStream(sample_rate, session_id, alpha_min, alpha_max)
        self.session_id = session_id
        self.coefficient = self._stream.mcadams_coefficient
        self.blocksize = blocksize
        self.delay_samples = minimum_delay(self._stream.frame_length, self._stream.hop_length, blocksize)
        self.delay_ms = 1000 * self.delay_samples / sample_rate
        self._queue = np.zeros(self.delay_samples)

    def process(self, block):
        """Anonimiza un bloque y devuelve un arreglo 1D del mismo largo.

        Las primeras delay_samples muestras de la sesión son los ceros de la
        cola. No modifica `block`.

        Lanza ValueError si se fijó blocksize y el bloque tiene otro largo:
        con ese largo el retraso calculado podría no alcanzar."""
        if self.blocksize is not None and len(block) != self.blocksize:
            raise ValueError(f"se esperaba un bloque de {self.blocksize} muestras y llegó uno de {len(block)}")

        self._queue = np.concatenate((self._queue, self._stream.process_chunk(block)))
        output, self._queue = self._queue[:len(block)], self._queue[len(block):]
        return output
