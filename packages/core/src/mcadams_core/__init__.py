"""Núcleo de procesamiento para anonimización de voz con el coeficiente de
McAdams estocástico.

Subpaquetes:
- signal_transform: enventanado, análisis LPC, transformación de McAdams y
  resíntesis de cada trama.
- streaming: anonimización incremental (AnonymizerStream) y su adaptador a
  bloques de largo fijo para tiempo real (BlockAnonymizer).
- batch: anonimización de un audio completo (anonymize_utterance).
- session: coeficiente de McAdams determinístico por sesión.
- io: lectura, escritura y conversión de audio.
- config: valores por defecto (rango de α, orden LPC, energía mínima)."""
