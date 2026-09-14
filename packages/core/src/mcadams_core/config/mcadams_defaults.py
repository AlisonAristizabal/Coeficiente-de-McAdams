"""Rango del coeficiente estocástico de McAdams (alpha) usado para la
anonimización de voz. El intervalo U(0.7, 0.9) sigue la configuración
validada por Patino et al. (2021), quienes muestrean alpha de esta
distribución uniforme para preservar la naturalidad e inteligibilidad
del habla mientras se degrada significativamente el desempeño de los
sistemas de verificación de hablante."""

MIN_MCADAMS_COEFFICIENT = 0.7
MAX_MCADAMS_COEFFICIENT = 0.9