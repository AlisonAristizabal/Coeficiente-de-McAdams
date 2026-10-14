"""Umbral de energía para la compensación de ganancia por fragmento.

Por debajo de este valor un fragmento se considera silencio digital y no se
reescala, para evitar dividir entre cero. No es un detector de voz: es una
guarda numérica elegida por debajo de la energía de un fragmento con ruido de
1 LSB de int16 (aprox. 1e-7). No proviene de la literatura ni se ajustó con
datos reales."""

MIN_FRAME_ENERGY = 1e-12