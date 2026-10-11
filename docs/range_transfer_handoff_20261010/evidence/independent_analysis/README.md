# Revisión aritmética de FCWD

Resultados de `scripts/analyze_public.py`, usando el CSV nativo publicado en V12.
No hay entrenamiento ni predicciones nuevas. La evidencia CPU del handoff usa
una ruta numérica distinta y sus últimos decimales pueden diferir.

`bands.csv`: bandas reales para diagnóstico, no features de inferencia.
`gap_concentration.csv`: contribución a la **diferencia neta** con Garl.
`prediction_bins.csv`: solapamiento de targets dentro de rangos de predicción.
`ranked_queries.csv`: consultas emparejadas; no selección de ejemplos de TRAIN.
`ineligible.csv`: las 33 filas no elegibles conservadas.
`ANALYSIS.json`: resumen y hash de entrada.

El objetivo algebraico c≈124,07 supone que todas las demás predicciones se
mantuvieran idénticas. No es una mejora ejecutada ni una receta de calibración.
