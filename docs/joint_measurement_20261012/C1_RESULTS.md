# V18 C1: medición relativa geométrica

Estado: **NO_CANDIDATE_C1**. Sin promoción de sistema ni afirmación SOTA.

Endpoint fijo de 12.000 updates, seed7, encoder entrenable y readout de escala/traslación.
PAIR-CORR es el control libre de la matriz primaria: se reutiliza sin reentrenarlo.

| Población | Modelo | Agregado | MiDc | Small | Large | Negativo | MAE (s) | Error de signo | p95 AE (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P | A5-P2-FROZEN | 100.873656 | 141.812711 | 62.952010 | 55.295382 | 55.521592 | 4.750671 | 3.3327% | 25.025708 |
| P | PAIR-CORR-S7 | 75.736062 | 65.709252 | 58.176432 | 78.532249 | 175.752812 | 5.826584 | 17.5747% | 25.180548 |
| P | PAIR-CORR-GEOMETRY-S7 | 71.486690 | 77.501529 | 53.036322 | 58.403691 | 109.846599 | 7.462971 | 9.2021% | 50.085495 |
| H | A5-P2-FROZEN | 112.285382 | 157.266510 | 68.256775 | 66.231260 | 65.519682 | 5.637243 | 5.6943% | 31.597022 |
| H | PAIR-CORR-S7 | 103.510372 | 115.555778 | 70.967522 | 88.420599 | 156.001662 | 5.495565 | 13.8041% | 24.456524 |
| H | PAIR-CORR-GEOMETRY-S7 | 90.946048 | 114.348829 | 56.121852 | 63.124236 | 106.226546 | 6.952719 | 8.4790% | 42.243646 |

P es entrenamiento; H es desarrollo del productor, separado de P.
Las métricas menores son mejores. La mejora del agregado no implica dominancia en cada medida.

C1 mejora el agregado y todas las bandas MiD en H respecto al control libre.
Su MAE y cola de error absoluto empeoran frente al control y A5. En P tampoco resuelve
la banda negativa o el signo frente a A5: el fallo no es sólo transferencia P/H.
Frente a A5, los gates de banda y signo siguen fallando. No se abre V para rescatarlo.

Intervalos descriptivos pareados por grupo de adquisición (C1 menos PAIR-CORR en H):

- delta_crucial: IC95 [-10.318552916140005, 8.762011725450966]; réplicas indefinidas: 0.
- delta_large: IC95 [-31.3761634980602, -19.38993886820736]; réplicas indefinidas: 0.
- delta_negative: IC95 [-86.52304121824771, -25.289450119311095]; réplicas indefinidas: 0.
- delta_overall: IC95 [-17.675972613244916, -7.073707372746182]; réplicas indefinidas: 0.
- delta_small: IC95 [-19.71605166902197, -9.990480336964916]; réplicas indefinidas: 0.

El intervalo crucial incluye cero; no se establece una ganancia crucial
frente al control libre. La mejora del agregado aporta una señal
mecanística para una fusión separada con anchor RGB.
Usar C1 como complemento experimental no lo convierte en candidato standalone admisible.

Predicciones propias: 72,312; referencias reutilizadas: cuatro tablas.
Se conservan recálculo NumPy, bootstrap por grupos y verificación
de 384.000 IDs del sampler, estados Adam/scheduler y cambio de los
18 tensores del encoder; export TorchScript y replay.
Checkpoint: `76ba6e75ce6d877a7ab1de3b7e22585aadc2450b44868bc310ae1413696bbd1f`.

Updates científicos: 12.000; recuperación no durable: 14.
Scopes de entrenamiento: 10959.438 s contabilizados;
1 reservas conservadoras no liquidadas incluidas.
Estos scopes incluyen carga/guardado, no sólo kernels;
no incluyen toda la QA y análisis C1.
El coste de selección mide tensores preparados en GPU, no el pipeline completo.

Réplica/refit no se activan por este resultado standalone. Continúa la prueba autorizada
de complemento event a R_CTX con control ZERO. R_CTX fue supervisado en H: entrenar
otra cabeza en H será ajuste secuencial, no OOF ni evaluación H independiente.

Regeneración: `python -B -m operational.joint_measurement.geometry_report`.
