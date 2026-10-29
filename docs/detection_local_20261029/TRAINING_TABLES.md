# Resultados medidos del detector RGB

Generado por `scripts/summarize_detection_runs.py` desde `evidence/training_runs/*.json`. AP101 es de desarrollo, no oficial: matching XY estricto a 0,5/1/2/4 m y macro sobre clases con GT admitido. La columna de ocho clases asigna cero a las ausentes. ATE/ASE/AOE usan matches a 2 m. AP se expresa entre 0 y 1.

| Run | Población | Frames | Clases GT | AP101 soportadas | AP101 ocho | ATE m | ASE | AOE rad |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| overfit16_seed7_v1 | train | 16 | 6 | 0.1034 | 0.0776 | 1.4161 | 0.5330 | 0.0886 |
| overfit16_seed7_v2_resume | train | 16 | 6 | 0.2014 | 0.1511 | 1.0812 | 0.1906 | 0.0835 |
| pilot256_seed7_velocity_v1 | train | 256 | 7 | 0.0584 | 0.0511 | 1.2657 | 0.4806 | 0.1743 |
| pilot256_seed7_velocity_v1 | selección | 256 | 5 | 0.0206 | 0.0129 | 1.2215 | 0.5297 | 0.1794 |

| Run | Updates etapa / acumulados | Batch | Entrenamiento s | Mediana s/update | Pico GPU GB | Razón pérdida centro | Gate overfit |
|---|---:|---:|---:|---:|---:|---:|---|
| overfit16_seed7_v1 | 344 / 344 | 1 | 600.67 | 1.6974 | 1.523 | 0.5524 | FAIL |
| overfit16_seed7_v2_resume | 456 / 800 | 1 | 787.16 | 1.7188 | 1.525 | 0.3452 | PASS |
| pilot256_seed7_velocity_v1 | 512 / 512 | 4 | 1096.59 | 1.0223 | 4.545 | 0.3549 | NOT_APPLICABLE_PILOT |

Los dos runs `overfit16` usan las mismas imágenes; el segundo continúa el checkpoint y optimizador del primero. No son dos semillas independientes. La razón de pérdida de esa continuación conserva el denominador inicial del primer run. El piloto es un modelo nuevo con backbone ImageNet y cabeza de velocidad; no es una comparación controlada con el overfit.

El tiempo de entrenamiento excluye preparación, evaluaciones y guardado. El pico GPU es memoria asignada por Torch, no consumo total del dispositivo. La tabla no acredita robustez, ocho clases aprendidas, calidad TTC ni equivalencia con CodaBench. Los recibos conservan pérdidas, métricas por clase, configuración, hashes y resultados negativos.
