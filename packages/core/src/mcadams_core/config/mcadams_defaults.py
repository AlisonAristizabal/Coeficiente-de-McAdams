"""cada sesión toma un α de U(0,685; 0,865), fijo durante toda la sesión;
el rango se eligió en este proyecto comparando nueve rangos uniformes 
(tres puntos medios × tres anchos) con EER y WER sobre Common Voice 26.0 en español;
Patino et al. (2021) exploraron U(0,7; 0,9), que fue el punto de partida."""

MIN_MCADAMS_COEFFICIENT = 0.685
MAX_MCADAMS_COEFFICIENT = 0.865