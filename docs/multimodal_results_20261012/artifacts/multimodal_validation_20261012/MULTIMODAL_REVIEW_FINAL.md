# Continuación multimodal V16 — resultados ejecutados

F_TRUE supera a Garl full registrado en las 472 etiquetas de soporte común FCWD y al sistema Garl full con fallback event-only en las 597 etiquetas. Esto cierra la comparación pendiente bajo registro estimado. No demuestra SOTA general: Garl event-only conserva mejor MiDc, la ventaja incremental sobre R_CTX no queda establecida en V/Dev32 y no hay evaluación oficial nueva.

Fecha: 2026-10-12. Base: `62cf4dbe914ea077c68e42983f869cd5422ab71c`. Worktree: `e-jepa-ttc-v16-multimodal-validation`. **Actualizaciones de optimizador de esta extensión: cero**, incluidas pruebas técnicas y recuperaciones. No se ejecutaron los dos fits propuestos de 2.500 updates.

## Comparación FCWD completada

Se ejecutaron los checkpoints públicos Garl visual y full, carga estricta, FP32 CPU, preprocesado nativo, registro sellado y sin etiquetas durante inferencia. Cada ruta conserva 630 registros; las predicciones se sellaron antes del scoring. La pareja nativa exige endpoints dentro de ±1 ms y un intervalo de 100 ms ±1 ms: pasan las 630 consultas, con offsets observados entre −105 y 0 µs. Después del filtro geométrico quedan 495 consultas y 472 etiquetas para ambos modelos. Esta igualdad de cobertura se verificó; no se impuso modificando tolerancias.

El full recibe 6 canales RGB y 40 canales event nativos; el visual recibe 6 RGB. Se normaliza antes del crop, se conserva el padding/interpolación nativos y se usa una transformación coherente por historia. No se proyecta con profundidad o TTC GT. El registro event→RGB sigue siendo aproximado y asistido por eventos; tampoco R_CTX constituye aquí un sistema integral RGB-only.

| Soporte común: 472 etiquetas | MAE (s) | mean_MiD | MiDc |
|---|---:|---:|---:|
| F_TRUE | 0,700261 | 55,969252 | 97,935870 |
| R_CTX | 0,938331 | 71,992994 | 97,155661 |
| Garl event-only | 1,352260 | 62,593365 | 93,912576 |
| Garl visual registrado | 1,311225 | 111,867397 | 229,410815 |
| Garl full registrado | 1,252557 | 76,388385 | 128,132971 |

| Sistema completo: 597 etiquetas | MAE (s) | mean_MiD | MiDc |
|---|---:|---:|---:|
| E_CTX | 1,967150 | 71,027628 | 150,093185 |
| F_TRUE, con fallback E_CTX | 1,429754 | 53,929182 | 102,323162 |
| Garl event-only | 2,146173 | 61,816523 | 95,056826 |
| Garl full registrado, con fallback event-only | 2,067345 | 72,723138 | 128,255717 |

La política Garl cambia a event-only por indisponibilidad de RGB, nunca por error o signo de la predicción. Las 135 consultas rechazadas incluyen 125 etiquetas. Los modelos standalone visual/full siguen figurando como incompletos sobre las 630; no se usa su MAE de casos disponibles como score del sistema completo. F_TRUE conserva exactamente E_CTX en su fallback.

En soporte común, F_TRUE−Garl full da ΔMAE −0,552296 s, IC95% descriptivo [−0,621307; −0,441684], y Δmean_MiD −20,419133 [−24,273914; −17,919077]. En sistemas completos, ΔMAE −0,637591 s [−0,849468; −0,503200] y Δmean_MiD −18,793956 [−21,231750; −14,595444]. Son remuestreos posteriores de **tres secuencias**, no confirmación prospectiva ni réplicas de entrenamiento.

`FCWD_METRICS.csv` separa resultados nativos de sensibilidad ±60 s, incluye bandas crucial/small/large/>10, secuencias, colas, signos, fallos y cobertura. `mean_MiD` no es un `overall_MiD` oficial con negativos: faltan esos targets. `FCWD_MULTIMODAL_PREDICTIONS.csv`, `COVERAGE.csv`, `GARL_PAIRED_REVIEW.json` y los dos directorios `garl_*` conservan datos y sellos.

## Cohortes y aporte incremental de eventos

Los doce endpoints V13 y sus hashes están inventariados en `STATE_BINDING.json`: 150.000 updates científicos históricos completos. Esa suma no sustituye su contabilidad física. V14 permanece `NO_CANDIDATE`; su matriz de masas no se repitió. V15 tiene ejecución independiente, registrada por snapshots y procesos en `FINAL_STATE.json`; esta extensión no cambió su receta ni tomó su supervisor.

Se reconciliaron las 1.024 consultas Dev32 con las 1.024 históricas y con la cohorte ampliada de 3.947 por secuencia y timestamp únicos, verificando contenido. Hay 1.024 identidades físicas coincidentes y sólo 39 query_id coincidentes con la ampliada; la discrepancia máxima numérica es 3,55e−15. Se mantienen las 946 etiquetas y las 3.640 ampliadas separadas. No hubo nueva inferencia completa Dev32: se regeneraron los recálculos públicos y se reinfirieron muestras diagnósticas/de coste.

El contraste nuevo F_TRUE−R_CTX en V usa las 16.432 predicciones y las pérdidas de sus ocho grupos: Δgroup_macro_bucket_MiD **−2,855638**, IC95% posterior por grupos **[−7,694064; +1,008157]**, 4.000 remuestreos, semilla 20261012. F_TRUE tiene mejor agregado (90,649718 frente a 93,505356), pero peor MAE (2,003460 frente a 1,883995 s). El intervalo cruza cero; no se convierte en gate aprobado ni en equivalencia. Dev32 mantiene el diagnóstico previo: ventaja pequeña en mean_MiD y peor MAE frente a R_CTX.

## Atribución ejecutada con paridad

Se trazaron 32 controles H elegidos por hash, cuatro por grupo, primero desde features congeladas y después desde HDF5/RGB tar; además ocho consultas Dev32 problemáticas publicadas y doce FCWD seleccionadas por hash y disponibilidad. Los hooks capturan entradas normalizadas, productores A5/C2F, PAIR, estados, expertos, residuales y fase antes/después de conversión, según los tensores expuestos por cada modelo. Los inputs/manifiestos guardan relojes, ROI, soporte y registro. No se reconstruyen anotaciones geométricas inexistentes.

**1.229 llamadas instrumentadas conservan igualdad bit a bit** con la misma llamada CPU sin instrumentar. Las 54 comparaciones separadas con las predicciones históricas alcanzan un delta TTC máximo de 0,013831 s en FCWD; en la muestra Dev32 el máximo es 0,000207 s. Esta ruta CPU no reproduce bit a bit la histórica. No se modificó known, mediana CUDA, scorer, checkpoint ni normalizador histórico. La carga conserva la validación original del freeze; sólo permite trasladar fuentes cuya única diferencia comprobada es CRLF/LF, documentada en `SOURCE_RELOCATION.json`. La atribución exacta de las diferencias de ruta/dispositivo exige una repetición controlada y queda pendiente; las predicciones históricas no se reemplazaron.

Dos episodios localizan un mecanismo de fallo concreto:

| Consulta Dev32 | R_CTX (s) | F_TRUE (s) | Anchor event | Residual F_TRUE | Fase raw F_TRUE |
|---|---:|---:|---:|---:|---:|
| CCRs-1-medium-100-overlap-100:01 | −8,3213 | −60,0000 | +0,003510 | −0,004199 | −0,000689 |
| CCRs-1-low-100-overlap-100:24 | +1,7266 | +22,6493 | −0,001859 | +0,006284 | +0,004425 |

En la primera, el residual de fusión cruza cero y el clamp de fase produce el límite de TTC negativo; R_CTX ya era negativo. En la segunda, R_CTX es razonable mientras la salida de fusión queda demasiado cerca de fase cero. Las trazas apoyan investigar una corrección sobre el RGB refinado, pero no demuestran que esa arquitectura futura mejore ni autorizan escoger el TTC positivo menor. Véase `FAILURE_EPISODES.md`, `STAGE_TRACE_DETAILS.json`, `STAGE_ATTRIBUTION.csv` y `TRACE_PARITY.json`.

## Registro: alternativas preparadas, sin promoción

Se selló por hash un subconjunto de 12 consultas, dos aceptadas y dos rechazadas por secuencia. Se ejecutaron 36 pruebas con acumulación causal de 20/40/80 ms. La ruta 20 ms reproduce exactamente los campos del registro sellado; las aceptaciones son 6/12, 9/12 y 10/12 respectivamente. Estas cifras describen aceptación interna, **no exactitud geométrica**. El prior sólo considera una transformación anterior aceptada de hasta 200 ms, soporte ≥100 y deriva/inconsistencia ≤5 px; las 524 filas elegibles incluyen consultas ya aceptadas y no son 524 recuperaciones nuevas.

No existen correspondencias independientes anotadas en estos entregables. Se exportaron 12 paneles y 72 filas vacías para anotación, con instrucciones de coordenadas y evaluación. No se sustituyó esa referencia por NCC, se retocaron umbrales según TTC ni se activó un registro alternativo. Los costes de cada ensayo están en `geometry/MULTISCALE_TRIALS.json`. La anotación independiente y sus negativos son la dependencia concreta para evaluar/promover geometría.

## Coste y límites de las mediciones

`COST_REVIEW.md` y `COST_SUMMARY.json` contienen los percentiles medidos. El piloto usa 64 identidades por hash, 32 Dev32 (una por secuencia) y 32 FCWD repartidas entre secuencias, con repetición caliente y orden cronológico dentro de secuencia. CPU FP32, un hilo; carga de modelos separada y registro FCWD recomputado para cronometrarlo, comprobando paridad con el sello sin cambiar transformaciones.

La traza de coste E/R/F comparte productores y ejecuta cabezas adicionales para comparar sus outputs; su TOTAL es **el conjunto de rutas compartidas**, no una medida aislada de F_TRUE. E_CTX tiene su propio tramo sensor→salida. Garl mide full o fallback event con preprocesado nativo vectorizado, sin voxels A5, además de una segunda llamada residente con igualdad bit a bit. Estos perímetros están separados; no se suman medianas ni se declara un ganador de velocidad entre perímetros distintos.

Las dos pasadas CPU E/R/F reproducen exactamente sus salidas serializadas. La muestra de coste, distinta de la diagnóstica, alcanza una diferencia máxima de **0,013831 s** respecto al CSV histórico: `FCWD1:0007`, E_CTX y su fallback F_TRUE, −9,944945 frente a −9,958776 s. `FCWD3:0194` difiere 0,004968 s. Se conservan las 368 comparaciones y los mayores deltas en `cpu_profile/`; no se presenta esta ruta CPU como sustituto bit a bit de la histórica ni se atribuye toda diferencia al dispositivo sin una repetición controlada GPU. Las métricas principales siguen usando las predicciones históricas congeladas E/R/F.

La cabeza canónica V12 con inputs preparados cuesta p50 2,53585 ms en CPU; su mediana3 real cuesta 7,34790 ms. No incluyen lectura, productores ni commit de un servicio. Esta campaña no midió un servicio persistente multiobjeto, resets/cachés de despliegue, transferencias H2D o CUDA graphs con GPU aislada: V15 conservaba su recurso. La primera pasada no equivale a caché de disco vacía. No se comparan estos números directamente con históricos de 37/82 ms o 1.656/2.581 ms obtenidos bajo otros perímetros/cargas.

## Verificación y decisión

El ZIP recibido tiene 230.440 bytes y SHA-256 `051eaba78cf5ffe58dca720d07c02701d1f1b30a190fe8da028e0acb3fa1f2c4`; sus 26 payloads y CRC fueron comprobados. Los 22.483.682 bytes citados por la revisión corresponden al paquete histórico de evidencia, no a este ZIP de continuación. El replay público verificó 29 fuentes y 9.471 comprobaciones escalares. `VALIDATION.json` conserva verificaciones finales, pruebas y hashes de dependencias. Se incluye el código exacto de inferencia previo al formateo junto al recibo de igualdad AST; los sellos originales no se reescriben para hacerlos coincidir con fuentes posteriores.

Hubo una concurrencia CPU de tareas propias que alcanzó aproximadamente 4,011 GiB RSS agregada. Se detuvo únicamente el proceso H raw de esta extensión, con progreso duradero, y se reanudó al terminar Garl full. El perfil final se serializó. No se tocaron procesos V15, prioridades, drivers ni configuración global.

**NEXT_DECISION: mantener la receta V13 como referencia, continuar V15 bajo su autorización propia y preparar una siguiente etapa multimodal pequeña; no entrenarla en esta extensión.** La hipótesis R_ANCHOR_EVENT frente a R_ANCHOR_ZERO merece un contrato nuevo con baseline inicial exactamente R_CTX, estados congelados y declaración explícita de nueva supervisión H. No es una mejora demostrada, no es OOF y no se convierte en autorizada por ahorrar presupuesto.

Q0/Q1/Q2 quedan ejecutadas; Q3 entrega alternativas y material de anotación con referencia independiente pendiente; Q4 entrega el piloto CPU con perímetros explícitos, quedando pendiente GPU aislada/servicio persistente comparable. No se propuso promover un sistema nuevo de despliegue ni se hizo submission. La mejora universal/SOTA sigue sin estar demostrada. `COMMANDS.md`, `DEPENDENCIES.json`, `UPDATE_LEDGER.json` y el manifiesto del bundle permiten continuar sin reiniciar la campaña.
