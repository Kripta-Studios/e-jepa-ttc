# PHASE-TTC: PAIR con conversión temporal explícita

Intervención motivada por el diagnóstico previo de dos consultas. No se leen targets, no se actualizan pesos y no se modifica el modelo entregable. Se reutilizan los tensores sellados de carretera y Slider-500. El baseline original vuelve a reproducirse exactamente antes de introducir la intervención.

PAIR recibe sus mismas representaciones y soportes, con sus tres variables de tiempo reconstruidas a 100 ms. Su fase cruda se convierte a inverso TTC, se escala por 0,1/dt_real y se convierte de nuevo a fase. La política original de emisión actúa después, una sola vez. Las filas cuyo dt es exactamente FP32(0,1) devuelven la fase original sin conversión. Los rechazos de dominio se conservan con trazas y sin sustituir argumentos inválidos por un epsilon.

Tiempo de ejecución, incluida carga y reproducción: **9.793 s**. Actualizaciones de pesos: **0**.

## Predicción limpia

Son predicciones, no errores. El orden es F0 original de nueve componentes, contexto RGB y contexto de eventos. El router adaptado de la entrega no forma parte de este piloto.

| Consulta | Bloque | Original (s) | PAIR corregido (s) | Diferencia (s) |
|---|---|---:|---:|---:|
| road | F0 | 2.790874481 | 2.790883780 | +0.000009298 |
| road | Contexto RGB | 3.015907526 | 3.015897036 | -0.000010490 |
| road | Contexto eventos | 2.831587553 | 2.831587553 | +0.000000000 |
| slider500 | F0 | 2.744687557 | 2.715584755 | -0.029102802 |
| slider500 | Contexto RGB | 2.981079578 | 3.006084919 | +0.025005341 |
| slider500 | Contexto eventos | 2.921912670 | 2.921912670 | +0.000000000 |

## Respuesta temporal completa

El cociente a·T_transformado/T_control debería ser 1 en el caso ideal. Se utiliza el control limpio con las mismas observaciones y lotes. Menor distancia a 1 significa mayor consistencia ante esta transformación, no menor error real.

| Consulta | Factor | Bloque | Original | Corregido | Cambio de distancia a 1 | Mismas observaciones admitidas |
|---|---:|---|---:|---:|---:|---|
| road | 0.5 | F0 | 0.673557 | 1.065466 | -0.260978 | True |
| road | 0.5 | Contexto RGB | 0.710814 | 1.069230 | -0.219956 | True |
| road | 0.5 | Contexto eventos | 0.673790 | 1.891432 | +0.565222 | True |
| road | 2.0 | F0 | 1.180513 | 0.966722 | -0.147235 | True |
| road | 2.0 | Contexto RGB | 1.185316 | 0.946442 | -0.131758 | True |
| road | 2.0 | Contexto eventos | 1.115187 | 1.053957 | -0.061229 | True |
| slider500 | 0.5 | F0 | 0.654640 | 0.938890 | -0.284250 | True |
| slider500 | 0.5 | Contexto RGB | 1.239822 | 0.961132 | -0.200954 | True |
| slider500 | 0.5 | Contexto eventos | 0.795923 | 1.073011 | -0.131066 | True |
| slider500 | 2.0 | F0 | 1.219669 | 0.940922 | -0.160591 | True |
| slider500 | 2.0 | Contexto RGB | 1.242094 | 0.895526 | -0.137621 | True |
| slider500 | 2.0 | Contexto eventos | 1.096336 | 1.066362 | -0.029975 | True |

## Respuesta de PAIR

| Consulta | Factor | PAIR | Intervalo de a·T_transformado/T_control |
|---|---:|---|---|
| road | 0.5 | PAIR_R | 1–1 (2/2) |
| road | 0.5 | PAIR_E_MATCHED | 1.00562–1.00562 (1/1) |
| road | 2.0 | PAIR_R | 0.99998–1.00001 (5/5) |
| road | 2.0 | PAIR_E_MATCHED | 0.993899–1.02201 (4/4) |
| slider500 | 0.5 | PAIR_R | 1–1 (3/3) |
| slider500 | 0.5 | PAIR_E_MATCHED | 0.986174–0.986174 (1/1) |
| slider500 | 2.0 | PAIR_R | 0.999988–1.00001 (7/7) |
| slider500 | 2.0 | PAIR_E_MATCHED | 1.00048–1.01863 (4/4) |

## Interpretación y siguiente prueba

La intervención corrige la sensibilidad temporal de PAIR observada en estas consultas. La tabla completa conserva también las respuestas de los contextos: la consistencia de un bloque intermedio no garantiza la del bloque posterior. En particular, el contexto de eventos de carretera al factor 0,5 empeora; no se oculta esa regresión detrás del comportamiento de la salida final.

La siguiente decisión exige precisión y cobertura en desarrollo agrupado, recalculando las historias afectadas. Las dos consultas no permiten seleccionar un modelo por accuracy. La auditoría eAP de entradas actuales, cuando se informe por separado, tampoco sustituye el control completo de contexto y cabeza ni aporta etiquetas de evaluación. Se mantienen los criterios de promoción previos.

No se exporta un nuevo modelo ni se afirma SOTA. El informe se regenera con `scripts/report_phase_canonical_pair.py`; sus tablas proceden de los artefactos verificados. Tensores y checkpoints permanecen fuera de Git.
