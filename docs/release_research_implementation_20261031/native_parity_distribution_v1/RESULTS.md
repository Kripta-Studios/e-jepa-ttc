# Deriva nativa respecto al archivo histórico

Comparación de predicciones, sin etiquetas: estas diferencias no son error TTC frente a verdad.
Consultas: 5805. Fallan la tolerancia histórica de TTC: 504.
Cuantiles p50/p95/p99/máximo de diferencia absoluta (s):
[0.00018548965454101562, 0.0010940551757812503, 0.002796783447265628, 0.4967918395996094]

| Magnitud de TTC local (s) | Consultas | Fallos de tolerancia | p95 diferencia (s) |
|---|---:|---:|---:|
| 0–1 | 0 | 0 | sin casos |
| 1–2 | 48 | 1 | 0.00023419857025146483 |
| 2–4 | 1049 | 35 | 0.0003602981567382811 |
| 4–8 | 3883 | 272 | 0.0008279800415039052 |
| 8–16 | 795 | 175 | 0.002464008331298828 |
| 16–None | 30 | 21 | 0.31057491302490114 |

El último intervalo no tiene cota superior. Las categorías se definen por
la predicción local, no por el TTC verdadero ni por categorías del scorer.
No identifican la causa de la deriva ni recuperan la paridad histórica.
