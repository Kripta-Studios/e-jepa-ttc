# Optimización experimental de streaming — 2026-10-10

Implementación: `operational/streaming_revision`. Los modelos y las tres cabezas H8 conservan sus pesos. Los datos y resultados de esta investigación están en E:.

## Condiciones de interpretación

Los pilotos usan 12 consultas de dos secuencias TRAIN40, reservadas para validar el estudiante pero vistas por el profesor. Son pruebas de ingeniería, no una evaluación externa ni evidencia SOTA. Se mantiene el mismo conjunto de consultas entre variantes. MiD es la media por muestra, no el overall_MiD ponderado oficial.

Se informan las tres cabezas H8. La mediana y p95 excluyen la primera consulta de cada secuencia, pero pueden incluir otras recargas y compilaciones. Son solo diez observaciones de latencia; no constituyen una garantía de tiempo real.

HDF5 y streaming residente son rutas diferentes. El segundo reproduce eventos grabados en un búfer causal: excluye la lectura de disco para ambos modelos e incluye la ingestión del paquete. Los saltos grandes entre consultas se recargan como arranques fríos; no se ha medido un sensor físico ni ingestión continua.

## gpu_warp_pilot

Estado: COMPLETE_PILOT.

| variant | n | warm_median_ms | warm_p95_ms | mean_MiD | MAE_s |
| --- | --- | --- | --- | --- | --- |
| garl_event | 12 | 264.600 | 618.312 | 53.188 | 2.575 |
| h8_reference | 12 | 1245.991 | 2787.938 | 31.613 | 0.620 |
| h8_reuse | 12 | 951.800 | 1605.903 | 25.668 | 0.485 |
| h8_warp | 12 | 545.560 | 1740.536 | 32.672 | 0.618 |
| h8_warp_student | 12 | 448.265 | 1124.496 | 44.473 | 0.792 |

h8_warp: 56.2% menos mediana total que H8_reference dentro de este mismo piloto.

## live_roi_pilot

Estado: COMPLETE_PILOT.

| variant | n | warm_median_ms | warm_p95_ms | mean_MiD | MAE_s |
| --- | --- | --- | --- | --- | --- |
| garl_event | 12 | 430.387 | 532.262 | 53.188 | 2.575 |
| h8_reference | 12 | 1812.207 | 2700.969 | 31.613 | 0.620 |
| h8_warp | 12 | 1045.377 | 1516.952 | 32.672 | 0.618 |
| h8_warp_student | 12 | 700.309 | 1052.913 | 44.473 | 0.792 |

h8_warp: 42.3% menos mediana total que H8_reference dentro de este mismo piloto.

## isolated_roi_graph

Estado: COMPLETE_PILOT.

| variant | n | warm_median_ms | warm_p95_ms | mean_MiD | MAE_s |
| --- | --- | --- | --- | --- | --- |
| garl_event | 12 | 92.128 | 163.704 | 53.188 | 2.575 |
| h8_reference | 12 | 347.289 | 539.236 | 31.613 | 0.620 |
| h8_warp | 12 | 177.076 | 315.711 | 32.672 | 0.618 |
| h8_warp_b2 | 12 | 187.458 | 413.526 | 32.672 | 0.618 |
| h8_warp_graph_b2 | 12 | 171.910 | 990.692 | 32.672 | 0.618 |
| h8_warp_student | 12 | 141.332 | 233.435 | 44.473 | 0.792 |

h8_warp: 49.0% menos mediana total que H8_reference dentro de este mismo piloto.

h8_warp_graph_b2: 50.5% menos mediana total que H8_reference dentro de este mismo piloto.

## isolated_fixed

Estado: COMPLETE_PILOT.

| variant | n | warm_median_ms | warm_p95_ms | mean_MiD | MAE_s |
| --- | --- | --- | --- | --- | --- |
| garl_event | 12 | 40.022 | 57.187 | 53.188 | 2.575 |
| garl_event_graph | 12 | 36.806 | 62.839 | 53.188 | 2.575 |
| garl_event_native_input | 12 | 120.228 | 202.777 | 53.188 | 2.575 |
| h8_reference | 12 | 226.717 | 236.349 | 31.613 | 0.620 |
| h8_warp | 12 | 119.802 | 197.661 | 32.672 | 0.618 |
| h8_warp_graph_fixed | 12 | 82.345 | 203.490 | 32.672 | 0.618 |
| h8_warp_student | 12 | 79.642 | 157.032 | 44.473 | 0.792 |

h8_warp: 47.2% menos mediana total que H8_reference dentro de este mismo piloto.

h8_warp_graph_fixed: 63.7% menos mediana total que H8_reference dentro de este mismo piloto.

## Primera consulta, incluida la compilación

| variant | total_ms |
| --- | --- |
| garl_event | 259.666 |
| garl_event_graph | 1389.929 |
| garl_event_native_input | 122.820 |
| h8_reference | 439.127 |
| h8_warp | 424.414 |
| h8_warp_graph_fixed | 12362.295 |
| h8_warp_student | 674.354 |

Estos arranques se conservan y no se confunden con la mediana de consultas posteriores. El calentamiento previo a uso en vivo aún requiere un protocolo explícito.

## Coste medio por etapa en consultas posteriores al arranque

La preparación test12 se suspendió temporalmente y se reanudó automáticamente; véase CPU_ISOLATION_V2.json. El escritorio WDDM sigue activo. La etiqueta genérica de contención del runner se complementa con ese recibo de aislamiento.

| pilot | variant | n | producer_ms | head_and_commit_ms | total_ms | input_ingest_ms | prep_lookup_warp_ms | prep_read_ms | prep_crop_map_ms | prep_encode_cache_ms | prep_stack_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| isolated_fixed | garl_event | 10 | 6.159 | N/D | 37.898 | 17.576 | N/D | N/D | N/D | N/D | N/D |
| isolated_fixed | garl_event_graph | 10 | 6.805 | N/D | 41.917 | 17.576 | N/D | N/D | N/D | N/D | N/D |
| isolated_fixed | garl_event_native_input | 10 | 6.523 | N/D | 122.186 | 17.576 | N/D | N/D | N/D | N/D | N/D |
| isolated_fixed | h8_reference | 10 | 62.936 | 11.425 | 217.654 | 17.576 | 0.037 | 40.383 | 7.520 | 74.326 | 2.897 |
| isolated_fixed | h8_warp | 10 | 57.496 | 10.785 | 131.375 | 17.576 | 3.323 | 19.013 | 2.927 | 18.657 | 1.127 |
| isolated_fixed | h8_warp_graph_fixed | 10 | 19.240 | 31.594 | 112.253 | 17.576 | 3.135 | 17.969 | 2.865 | 18.372 | 1.075 |
| isolated_fixed | h8_warp_student | 10 | 20.395 | 10.973 | 93.138 | 17.576 | 3.047 | 18.174 | 2.893 | 18.468 | 1.159 |

## Ingestión compacta CPU

| variant | warm_push_median_ms | roi_median_ms | warm_roi_median_ms | warm_tail_roi_median_ms | peak_retained_bytes |
| --- | --- | --- | --- | --- | --- |
| blocks_50ms | 50.296 | 190.491 | 192.205 | 35.835 | 253952748 |
| compact_arithmetic | 49.284 | 207.351 | 208.001 | 37.110 | 253952748 |
| reference | 90.194 | 195.301 | 201.092 | 37.960 | 253952748 |

Doce consultas TRAIN40, una ingestión por consulta y tres lecturas ROI por consulta en orden aleatorio. La inferencia test12 seguía activa: estos tiempos no se combinan con el piloto GPU aislado. Las repeticiones de lectura no son ejecuciones independientes de extremo a extremo.

Igualdad exacta de eventos y voxels: 36/36 comprobaciones. Cero segundos GPU.

La aritmética compacta elimina copias temporales de timestamps y exploraciones duplicadas; está activa por defecto en PacketRing. La división adicional en bloques de 50 ms es opcional: empeora ligeramente la ingestión frente a la aritmética compacta, aunque mejora algunas lecturas. La reducción de ingestión no demuestra la misma reducción de latencia total.

## FCWD completo desde eventos crudos

Estado: COMPLETE. 630 consultas cronológicas, tres secuencias, sin ajustes de pesos ni selección por etiquetas. Las etiquetas se incorporan después de todas las predicciones. La inferencia se ejecuta en CPU; no mide latencia GPU.

| variant | eligible_n | mean_MiD | MiDc | MAE_s | signed_bias_s |
| --- | --- | --- | --- | --- | --- |
| H8_reference | 597 | 79.952 | 177.938 | 1.625 | -0.866 |
| Garl_event | 597 | 61.817 | 95.057 | 2.146 | -1.487 |
| H8_warp | 597 | 77.453 | 167.485 | 1.666 | -0.891 |
| H8_warp_student | 597 | 74.561 | 162.969 | 1.506 | -0.643 |

El CSV incluye resultados por secuencia y FR por banda. Paridad del H8 de referencia con las predicciones congeladas: True; diferencia máxima 0.000928879 s. Los intervalos usan agrupación por secuencia; solo hay tres grupos. FCWD ya es un conjunto expuesto y no sustituye test12.

### Error en TTC crucial y reutilización

| variant | band | n | MAE_s | signed_bias_s | underestimate_fraction | overestimate_fraction | absolute_error_p95_s | invalid_MiD_n | strict_mean_MiD | finite_MiD_p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Garl_event | c | 201 | 0.327 | 0.166 | 0.463 | 0.537 | 1.320 | 0 | 95.057 | 320.693 |
| H8_reference | c | 201 | 0.700 | 0.656 | 0.164 | 0.836 | 2.441 | 0 | 177.938 | 454.688 |
| H8_warp | c | 201 | 0.646 | 0.611 | 0.149 | 0.851 | 2.384 | 0 | 167.485 | 429.193 |
| H8_warp_student | c | 201 | 0.595 | 0.568 | 0.134 | 0.866 | 2.169 | 0 | 162.969 | 397.319 |

| variant | n | reused_mean | computed_mean | approximate_voxel_hits_mean | min_source_roi_iou | max_time_approximation_us |
| --- | --- | --- | --- | --- | --- | --- |
| H8_reference | 630 | 0.000 | 8.000 | 0.000 | 1.000 | 0 |
| H8_warp | 630 | 6.913 | 1.087 | 2.113 | 0.651 | 349 |
| H8_warp_student | 630 | 6.913 | 1.087 | 2.113 | 0.651 | 349 |

El principal déficit frente a Garl se concentra en (0,3] s. Una MAE global menor no implica mejor MiD ni mejor resultado en el rango crucial. FR es cero para todas estas variantes según el scorer utilizado. No hay banda negativa y no se calcula un overall_MiD con pesos redistribuidos.

## Precisión externa disponible

Se evaluaron todas las filas guardadas de DEV32 y FCWD. Estas tablas prueban truncamiento de historia y destilación sobre características preparadas; no prueban la reproyección de voxels ni su combinación con precisión reducida.

| dataset | variant | eligible_n | mean_MiD | MiDc | MAE_s | signed_bias_s |
| --- | --- | --- | --- | --- | --- | --- |
| DEV32 | H8_reference | 3640 | 138.377 | 213.223 | 1.377 | 0.443 |
| DEV32 | H4_truncated | 3640 | 137.879 | 212.982 | 1.310 | 0.278 |
| DEV32 | H2_truncated | 3640 | 141.278 | 219.627 | 1.368 | 0.407 |
| DEV32 | H8_A5_student | 3640 | 150.040 | 232.715 | 1.676 | 0.861 |
| DEV32 | H8_A5_student_int8_cpu | 3640 | 149.641 | 232.213 | 1.658 | 0.867 |
| FCWD | H8_reference | 597 | 79.952 | 177.938 | 1.625 | -0.866 |
| FCWD | H4_truncated | 597 | 83.225 | 175.845 | 1.778 | -1.132 |
| FCWD | H2_truncated | 597 | 87.069 | 178.652 | 1.908 | -1.237 |
| FCWD | H8_A5_student | 597 | 73.736 | 165.020 | 1.453 | -0.650 |
| FCWD | H8_A5_student_int8_cpu | 597 | 73.829 | 165.331 | 1.473 | -0.629 |

No hay banda negativa etiquetada en estos conjuntos: overall_MiD permanece N/D; no se redistribuyen los pesos oficiales. Dev32 ya es un conjunto expuesto.

## Cambios y límites

- Búfer de paquetes con compresión sin pérdida: 9 bytes por evento típico frente a 17; coordenadas fuera de int16 usan fallback exacto. Descarte parcial de paquetes caducados, límite de memoria y errores explícitos ante huecos o retrocesos.

- ROI recortada antes de descomprimir columnas; conserva eventos, orden y voxels. Garl recibe la misma optimización preservando el primer evento del sensor como origen temporal. El piloto exige igualdad exacta de sus tensores nativos.

- Caché de características por objeto, secuencia, ROI y antigüedad; ocho observaciones con timestamps reales. Reproyección bilineal solo de voxels originales, sin cadenas de interpolaciones. No recupera eventos fuera de la ROI original y aproxima conteos, normalización y pequeños desfases temporales.

- La caché de 17 características apenas ocupa memoria. Las prioridades de compresión son eventos y voxels. INT8 dinámico del estudiante en CPU fue más lento que FP32; no se promueve ni equivale a cuantización GPU.

- El estudiante suprime C2F/PAIR en inferencia; mejora FCWD frente a H8 y empeora DEV32. H4/H2 son truncamientos de los pesos H8, no modelos reentrenados.

- La captura completa GPU dinámica recompila con nuevas formas; el piloto live_graph_pilot_v2 agotó su presupuesto. La variante b2 fija el tamaño de los productores, incluye padding descartado y conserva las tres cabezas. graph_fixed añade padding con máscara a las cabezas para conservar ocho posiciones incluso en el arranque. Las búsquedas de timestamps usan uint32 para evitar que NumPy copie una columna completa al promover su tipo.

- garl_event_native_input usa la preparación nativa desde la ventana completa del sensor, con el mismo lector persistente. garl_event añade recorte temprano exacto; garl_event_graph añade captura GPU. Ninguna variante Garl aplica reproyección ni aproximación. Son reproducciones locales, no los tiempos del hardware del artículo.

## Fallos conservados

distill_seed7 falló antes del primer update por una operación in-place; distill_seed7_v2 registra el entrenamiento CPU corregido. live_graph_pilot falló por límite de memoria. live_graph_pilot_v2 es parcial y se reserva íntegramente su presupuesto de 300 s. Ninguno se presenta como piloto completo.

## Pendiente

La reproyección CPU ya se evaluó en todo FCWD. Queda validarla en DEV32 y medir el runtime GPU optimizado en secuencias externas completas; examinar MiDc, FR y colas por secuencia; ampliar repetición temporal y memoria pico. La selección final requiere datos independientes, tres semillas para modelos entrenados y evaluación oficial test12. No hay puntuación privada local.

La repetición adicional de latencia quedó pendiente al terminar la preparación de las 6.762 entradas test12 y comenzar su inferencia GPU congelada. No se ejecutan nuevos pilotos GPU en paralelo con esa campaña.

## Reproducción

`python -m operational.streaming_revision.benchmark --help` describe las rutas. Cada nuevo piloto conserva sus fuentes en source_archive, pesos vinculados por hash, consultas, políticas y ROWS.jsonl. La reserva GPU se registra antes de ejecutar, con watchdog y contabilidad de tiempo de pared conservadora.

`python -m operational.streaming_revision.report --root <resultados> --document <informe.md>` regenera este informe. STAGES.csv contiene los costes medios y NUMERICAL_PARITY.csv las diferencias numéricas medidas.
