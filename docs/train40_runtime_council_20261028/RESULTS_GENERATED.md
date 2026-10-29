# Mediciones regeneradas — reanudación TRAIN40

Generado exclusivamente desde recibos JSON. Cada sección conserva su frontera de medición; los tiempos de secciones distintas no se suman ni se convierten en una afirmación de 50 Hz sostenidos. CPU, GPU preparada y cadena completa son ensayos diferentes. Los resultados fallidos permanecen visibles.

Fuentes y SHA-256: [RESULTS_SOURCES.json](RESULTS_SOURCES.json). `—` significa dato no declarado; no significa cero.

## Lectura rápida

| Frente | Dispositivo | Recibos | Máx. objetos/ROI ensayados | Estados |
| --- | --- | --- | --- | --- |
| compiled_pipeline | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 5 | 1 | COMPLETE: 3; COMPLETE_WITH_PARITY_FAILURES: 2 |
| crop_rgb_admission | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 1 | 2 | COMPLETE: 1 |
| exact_transport | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 1 | 1 | COMPLETE: 1 |
| frame_hash_cpu | — | 1 | — | COMPLETE: 1 |
| frame_sealing_cpu | — | 1 | — | COMPLETE: 1 |
| immutable_rgb_admission | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 1 | 2 | COMPLETE: 1 |
| memory_raw_parity_v1 | — | 1 | — | COMPLETE: 1 |
| multiobject | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 1 | 1 | COMPLETE: 1 |
| paced | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 7 | 1 | COMPLETE: 3; FAILED: 4 |
| rgb_exact | cuda | 1 | 1 | COMPLETE: 1 |
| rgb_exact | — | 1 | — | FAILED: 1 |
| subpacket | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 2 | — | COMPLETE: 1; FAILED: 1 |

El máximo de objetos/ROI es cobertura del tramo indicado, no capacidad admitida de la cadena completa. Los encoders expresan batch de endpoints, no objetos. Cada tiempo y su paridad aparecen separados en el detalle.

## Inventario

| Recibo | Estado | Dispositivo | Preservación |
| --- | --- | --- | --- |
| compiled_pipeline/linspace_attribution_v1/summary.json | COMPLETE | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| compiled_pipeline/native_match_sensitive_v1/summary.json | COMPLETE_WITH_PARITY_FAILURES | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| compiled_pipeline/summary_attribution_v1/summary.json | COMPLETE | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| compiled_pipeline/weighted_mean_attribution_v1/summary.json | COMPLETE | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| event/subpacket_backend_sensitive_v1/summary.json | FAILED | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| event/subpacket_backend_sensitive_v2/summary.json | COMPLETE | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| frame_hash_cpu/idx553_v1/summary.json | COMPLETE | — | sí |
| frame_sealing_cpu/idx553_v1/summary.json | COMPLETE | — | sí |
| memory_raw_parity_v1/summary.json | COMPLETE | — | sí |
| multiobject/crop_rgb_companion_v1/summary.json | COMPLETE | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| multiobject/immutable_rgb_companion_v1/summary.json | COMPLETE | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| multiobject/overlap1_threecases_v1/summary.json | COMPLETE | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| native_log_sensitive_v1/summary.json | COMPLETE_WITH_PARITY_FAILURES | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| paced/baseline1_50_v1/summary.json | FAILED | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| paced/memory1_50_v1/summary.json | FAILED | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| paced/memory_cache_overlap1_50_v1/summary.json | FAILED | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| paced/memory_cache_overlap_reverse1_250_v1/summary.json | COMPLETE | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| paced/memory_cache_overlap_reverse1_50_v1/summary.json | COMPLETE | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| paced/memory_cache_overlap_reverse1_50_v2/summary.json | COMPLETE | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| paced/memory_crops_reverse1_250_v1/summary.json | FAILED | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| producers/exact_transport_sensitive_v1/summary.json | COMPLETE | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| rgb_exact_gpu/idx553_v1/summary.json | FAILED | — | — |
| rgb_exact_gpu/idx553_v2/summary.json | COMPLETE | cuda | sí |

## baseline

Una ROI por consulta original; historial reconstruido antes del reloj. Conserva el gate conjunto de las tres salidas.

Sin recibos disponibles en esta generación.

## producers

Productores preparados en GPU; excluye sensores, H2D, contexto y TTC final.

Sin recibos disponibles en esta generación.

## prefix

Paquete RAW a representaciones de eventos; excluye modelos y TTC.

Sin recibos disponibles en esta generación.

## normalization

Normalización de vóxeles preparados; excluye sensores y modelos.

Sin recibos disponibles en esta generación.

## multiobject

Piloto completo con ROI de capacidad artificiales; no es un stream sostenido.

### `multiobject/overlap1_threecases_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`. Alcance: `ARTIFICIAL_DISTINCT_ROIS_SHARED_REAL_RAW_LAST_PACKET_COMPLETE_UPDATE`.

| Configuración | Valor |
| --- | --- |
| precision | FP32_TF32_OFF |
| rgb_object_chunk_size | 1 |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| piloto completo | 1 | RAW disponible → último TTC | 15 | 26.3989667 | 25.6008 | 33.82142 | 34.103884 | 34.1745 | sí |

## transport

Transporte sobre features sintéticas preparadas; excluye sensores y TTC.

Sin recibos disponibles en esta generación.

## encoders

Encoders de endpoints preparados; excluye transporte, contexto y TTC.

Sin recibos disponibles en esta generación.

## compiled_pipeline

Productores con transporte compilado e inputs GPU preparados; la paridad TTC se comprueba fuera del reloj. Excluye RAW, H2D y contexto del tiempo.

### `compiled_pipeline/linspace_attribution_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`. Alcance: `BOUND_TRANSPORT_SUMMARY_LINSPACE_ONLY_DIAGNOSTIC`.

| Configuración | Valor |
| --- | --- |
| precision | FP32_TF32_OFF |

| Caso | Estado | Error |
| --- | --- | --- |
| índice=488; radio=1 | COMPARED | — |
| índice=488; radio=1 | COMPARED | — |
| índice=488; radio=2 | COMPARED | — |
| índice=488; radio=1 | COMPARED | — |
| índice=488; radio=1 | COMPARED | — |
| índice=488; radio=2 | COMPARED | — |
| índice=505; radio=1 | COMPARED | — |
| índice=505; radio=1 | COMPARED | — |
| índice=505; radio=2 | COMPARED | — |
| índice=505; radio=1 | COMPARED | — |
| índice=505; radio=1 | COMPARED | — |
| índice=505; radio=2 | COMPARED | — |

La columna de paridad corresponde a las tres salidas TTC por ruta. all-replays incluye llamadas iniciales, warmup y cada repetición registrada; last-only conserva el alcance limitado de recibos históricos sin ese registro. El estado del ensayo conserva también fallos de auditoría. La paridad de features intermedias es un diagnóstico separado.

### `compiled_pipeline/native_match_sensitive_v1/summary.json`

Estado: **COMPLETE_WITH_PARITY_FAILURES**. Dispositivo: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`. Alcance: `PREPARED_GPU_PRODUCERS_PAIR_PHASE17_GUARDS_NO_RAW_H2D_CONTEXT`.

| Configuración | Valor |
| --- | --- |
| event_endpoint_reuse | no |
| rgb_endpoint_reuse | sí |
| precision | FP32_TF32_OFF |
| event_object_chunk_size | 1 |
| rgb_object_chunk_size | 1 |
| native_entropy_log | no |
| native_correlation_match | sí |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| índice=0 | 1 | original_graph | 20 | 17.938045 | 18.0097 | 19.089545 | 19.102389 | 19.1056 | sí [all-replays; 24/24] |
| índice=0 | 1 | compiled_transport_graph | 20 | 12.92608 | 12.82105 | 13.88986 | 14.117252 | 14.1741 | sí [all-replays; 24/24] |
| índice=137 | 1 | original_graph | 20 | 17.445135 | 17.4233 | 18.05369 | 18.316498 | 18.3822 | sí [all-replays; 24/24] |
| índice=137 | 1 | compiled_transport_graph | 20 | 12.640075 | 12.52895 | 13.59445 | 13.62257 | 13.6296 | sí [all-replays; 24/24] |
| índice=488 | 1 | original_graph | 20 | 17.627785 | 17.5135 | 18.686605 | 18.776361 | 18.7988 | sí [all-replays; 24/24] |
| índice=488 | 1 | compiled_transport_graph | 20 | 12.67839 | 12.6388 | 13.728775 | 13.845435 | 13.8746 | no [all-replays; 0/24] |
| índice=505 | 1 | original_graph | 20 | 19.281845 | 18.9186 | 22.1247 | 24.14934 | 24.6555 | sí [all-replays; 24/24] |
| índice=505 | 1 | compiled_transport_graph | 20 | 14.016275 | 13.7238 | 15.743025 | 17.503565 | 17.9437 | no [all-replays; 0/24] |
| índice=514 | 1 | original_graph | 20 | 19.08235 | 18.942 | 22.13934 | 22.486508 | 22.5733 | sí [all-replays; 24/24] |
| índice=514 | 1 | compiled_transport_graph | 20 | 13.879195 | 13.60465 | 16.14193 | 16.566466 | 16.6726 | sí [all-replays; 24/24] |
| índice=533 | 1 | original_graph | 20 | 21.35057 | 19.7904 | 32.276525 | 33.698105 | 34.0535 | sí [all-replays; 24/24] |
| índice=533 | 1 | compiled_transport_graph | 20 | 15.47502 | 14.39225 | 21.25534 | 23.529868 | 24.0985 | sí [all-replays; 24/24] |
| índice=553 | 1 | original_graph | 20 | 19.977145 | 17.9654 | 29.53647 | 30.820414 | 31.1414 | sí [all-replays; 24/24] |
| índice=553 | 1 | compiled_transport_graph | 20 | 14.318995 | 13.052 | 22.36145 | 22.96109 | 23.111 | sí [all-replays; 24/24] |

La columna de paridad corresponde a las tres salidas TTC por ruta. all-replays incluye llamadas iniciales, warmup y cada repetición registrada; last-only conserva el alcance limitado de recibos históricos sin ese registro. El estado del ensayo conserva también fallos de auditoría. La paridad de features intermedias es un diagnóstico separado.

| Protocolo de validación | Valor |
| --- | --- |
| scope | EVERY_EXPLICIT_INITIAL_WARMUP_AND_MEASURED_PRODUCER_CALL |
| placement | AFTER_PRODUCER_CLOCK_BEFORE_NEXT_CALL_WITH_ONE_OWNED_FEATURE_COPY |
| latency_comparability | INTERLEAVED_CONTEXT_AND_PARITY_CHANGE_CADENCE_VS_LAST_REPLAY_ONLY_RUNS |
| initial_call_role | FIRST_EXPLICIT_CALL_FOR_QUERY; may build or reuse a shape; producer internally warms/traces/captures before its first returned replay; not cold-start latency |

| Índice | Llamadas esperadas | Registradas | Grilla exacta | Todos TTC | Primeras |
| --- | --- | --- | --- | --- | --- |
| 0 | 48 | 48 | sí | sí | sí |
| 137 | 48 | 48 | sí | sí | sí |
| 488 | 48 | 48 | sí | no | no |
| 505 | 48 | 48 | sí | no | no |
| 514 | 48 | 48 | sí | sí | sí |
| 533 | 48 | 48 | sí | sí | sí |
| 553 | 48 | 48 | sí | sí | sí |

### `compiled_pipeline/summary_attribution_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`. Alcance: `REAL_PRE_ROUTER_TRANSPORT_INPUTS_ONLY_NO_COMPLETE_UPDATE_CLOCK_OR_TTC`.

| Configuración | Valor |
| --- | --- |
| event_endpoint_reuse | no |
| rgb_endpoint_reuse | sí |
| precision | FP32_TF32_OFF |
| rgb_object_chunk_size | 1 |

| Caso | Estado | Error |
| --- | --- | --- |
| índice=488; radio=1 | COMPARED | — |
| índice=488; radio=1 | COMPARED | — |
| índice=488; radio=2 | COMPARED | — |
| índice=488; radio=1 | COMPARED | — |
| índice=488; radio=1 | COMPARED | — |
| índice=488; radio=2 | COMPARED | — |
| índice=505; radio=1 | COMPARED | — |
| índice=505; radio=1 | COMPARED | — |
| índice=505; radio=2 | COMPARED | — |
| índice=505; radio=1 | COMPARED | — |
| índice=505; radio=1 | COMPARED | — |
| índice=505; radio=2 | COMPARED | — |

La columna de paridad corresponde a las tres salidas TTC por ruta. all-replays incluye llamadas iniciales, warmup y cada repetición registrada; last-only conserva el alcance limitado de recibos históricos sin ese registro. El estado del ensayo conserva también fallos de auditoría. La paridad de features intermedias es un diagnóstico separado.

### `compiled_pipeline/weighted_mean_attribution_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`. Alcance: `PRESERVED_PRE_ROUTER_TENSORS_NATIVE_WEIGHTED_MEAN_DIAGNOSTIC`.

| Configuración | Valor |
| --- | --- |
| precision | FP32_TF32_OFF |

| Caso | Estado | Error |
| --- | --- | --- |
| índice=488; radio=1 | COMPARED | — |
| índice=488; radio=1 | COMPARED | — |
| índice=488; radio=2 | COMPARED | — |
| índice=488; radio=1 | COMPARED | — |
| índice=488; radio=1 | COMPARED | — |
| índice=488; radio=2 | COMPARED | — |
| índice=505; radio=1 | COMPARED | — |
| índice=505; radio=1 | COMPARED | — |
| índice=505; radio=2 | COMPARED | — |
| índice=505; radio=1 | COMPARED | — |
| índice=505; radio=1 | COMPARED | — |
| índice=505; radio=2 | COMPARED | — |

La columna de paridad corresponde a las tres salidas TTC por ruta. all-replays incluye llamadas iniciales, warmup y cada repetición registrada; last-only conserva el alcance limitado de recibos históricos sin ese registro. El estado del ensayo conserva también fallos de auditoría. La paridad de features intermedias es un diagnóstico separado.

### `native_log_sensitive_v1/summary.json`

Estado: **COMPLETE_WITH_PARITY_FAILURES**. Dispositivo: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`. Alcance: `PREPARED_GPU_PRODUCERS_PAIR_PHASE17_GUARDS_NO_RAW_H2D_CONTEXT`.

| Configuración | Valor |
| --- | --- |
| event_endpoint_reuse | no |
| rgb_endpoint_reuse | sí |
| precision | FP32_TF32_OFF |
| event_object_chunk_size | 1 |
| rgb_object_chunk_size | 1 |
| native_entropy_log | sí |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| índice=0 | 1 | original_graph | 20 | 20.777055 | 20.81895 | 23.48223 | 23.611886 | 23.6443 | sí [last-only] |
| índice=0 | 1 | compiled_transport_graph | 20 | 11.136565 | 10.97515 | 12.721775 | 12.928115 | 12.9797 | sí [last-only] |
| índice=137 | 1 | original_graph | 20 | 18.9003 | 18.63735 | 21.707235 | 23.480087 | 23.9233 | sí [last-only] |
| índice=137 | 1 | compiled_transport_graph | 20 | 10.90613 | 10.43965 | 14.23809 | 16.147058 | 16.6243 | sí [last-only] |
| índice=488 | 1 | original_graph | 20 | 17.213935 | 17.10655 | 18.35779 | 18.483798 | 18.5153 | sí [last-only] |
| índice=488 | 1 | compiled_transport_graph | 20 | 9.467145 | 9.2927 | 10.16295 | 10.30051 | 10.3349 | no [last-only] |
| índice=505 | 1 | original_graph | 20 | 17.15909 | 17.12535 | 17.985135 | 18.330707 | 18.4171 | sí [last-only] |
| índice=505 | 1 | compiled_transport_graph | 20 | 9.61748 | 9.52285 | 10.41926 | 10.900492 | 11.0208 | no [last-only] |
| índice=514 | 1 | original_graph | 20 | 18.38476 | 18.6516 | 19.193685 | 19.640337 | 19.752 | sí [last-only] |
| índice=514 | 1 | compiled_transport_graph | 20 | 10.3769 | 10.35695 | 11.100995 | 11.290919 | 11.3384 | sí [last-only] |
| índice=533 | 1 | original_graph | 20 | 17.662305 | 17.61325 | 18.448115 | 18.550183 | 18.5757 | sí [last-only] |
| índice=533 | 1 | compiled_transport_graph | 20 | 9.85052 | 9.73685 | 10.680585 | 10.692517 | 10.6955 | sí [last-only] |
| índice=553 | 1 | original_graph | 20 | 17.72048 | 17.4905 | 19.045795 | 19.778359 | 19.9615 | sí [last-only] |
| índice=553 | 1 | compiled_transport_graph | 20 | 9.80992 | 9.51795 | 11.375135 | 11.597587 | 11.6532 | sí [last-only] |

La columna de paridad corresponde a las tres salidas TTC por ruta. all-replays incluye llamadas iniciales, warmup y cada repetición registrada; last-only conserva el alcance limitado de recibos históricos sin ese registro. El estado del ensayo conserva también fallos de auditoría. La paridad de features intermedias es un diagnóstico separado.

## exact_transport

Transporte reutilizado con inputs de un objeto preparados en GPU. Valida todas las llamadas explícitas y features exactas, con paridad/contexto fuera del reloj. Excluye RAW, H2D y contexto; su ahorro no es una reducción de latencia completa.

### `producers/exact_transport_sensitive_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`. Alcance: `PREPARED_ONE_OBJECT_EXACT_TRANSPORT_ALL_EXPLICIT_CALLS_NOT_COMPLETE_UPDATE`.

| Configuración | Valor |
| --- | --- |
| event_endpoint_reuse | no |
| rgb_endpoint_reuse | sí |
| precision | FP32_TF32_OFF |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| índice=0 | 1 | original/serial | 8 | 17.4543 | 17.49145 | 17.931065 | 17.994933 | 18.0109 | sí [all-replays; 11/11] |
| índice=0 | 1 | original/parallel | 8 | 13.287625 | 13.3134 | 14.097545 | 14.194229 | 14.2184 | sí [all-replays; 11/11] |
| índice=0 | 1 | reverse/serial | 8 | 15.39945 | 15.31435 | 16.12027 | 16.138694 | 16.1433 | sí [all-replays; 11/11] |
| índice=0 | 1 | reverse/parallel | 8 | 10.9352375 | 10.9692 | 11.43446 | 11.458092 | 11.464 | sí [all-replays; 11/11] |
| índice=0 | 1 | reverse-summaries/serial | 8 | 14.4820375 | 14.65345 | 14.927595 | 14.934959 | 14.9368 | sí [all-replays; 11/11] |
| índice=0 | 1 | reverse-summaries/parallel | 8 | 10.317025 | 10.2401 | 10.96721 | 11.034802 | 11.0517 | sí [all-replays; 11/11] |
| índice=137 | 1 | original/serial | 8 | 17.92265 | 17.86645 | 18.59974 | 18.712188 | 18.7403 | sí [all-replays; 11/11] |
| índice=137 | 1 | original/parallel | 8 | 13.3066125 | 13.3543 | 13.68358 | 13.716956 | 13.7253 | sí [all-replays; 11/11] |
| índice=137 | 1 | reverse/serial | 8 | 15.577525 | 15.58065 | 16.064975 | 16.106835 | 16.1173 | sí [all-replays; 11/11] |
| índice=137 | 1 | reverse/parallel | 8 | 11.3754875 | 11.43475 | 12.069065 | 12.138533 | 12.1559 | sí [all-replays; 11/11] |
| índice=137 | 1 | reverse-summaries/serial | 8 | 14.179475 | 14.1167 | 14.685945 | 14.766389 | 14.7865 | sí [all-replays; 11/11] |
| índice=137 | 1 | reverse-summaries/parallel | 8 | 10.81615 | 10.60255 | 11.76751 | 11.845182 | 11.8646 | sí [all-replays; 11/11] |
| índice=488 | 1 | original/serial | 8 | 18.138325 | 18.06915 | 18.870505 | 19.052981 | 19.0986 | sí [all-replays; 11/11] |
| índice=488 | 1 | original/parallel | 8 | 12.7817 | 12.53395 | 13.766095 | 13.842339 | 13.8614 | sí [all-replays; 11/11] |
| índice=488 | 1 | reverse/serial | 8 | 15.7078625 | 15.64465 | 16.29947 | 16.308374 | 16.3106 | sí [all-replays; 11/11] |
| índice=488 | 1 | reverse/parallel | 8 | 11.378675 | 11.3497 | 12.11212 | 12.225464 | 12.2538 | sí [all-replays; 11/11] |
| índice=488 | 1 | reverse-summaries/serial | 8 | 14.51015 | 14.475 | 14.88462 | 14.912284 | 14.9192 | sí [all-replays; 11/11] |
| índice=488 | 1 | reverse-summaries/parallel | 8 | 10.60135 | 10.5168 | 11.16195 | 11.16391 | 11.1644 | sí [all-replays; 11/11] |
| índice=505 | 1 | original/serial | 8 | 18.7683125 | 18.7939 | 20.83337 | 20.898274 | 20.9145 | sí [all-replays; 11/11] |
| índice=505 | 1 | original/parallel | 8 | 13.3131625 | 13.05555 | 14.690165 | 15.019473 | 15.1018 | sí [all-replays; 11/11] |
| índice=505 | 1 | reverse/serial | 8 | 15.7549625 | 15.2044 | 17.563385 | 18.053637 | 18.1762 | sí [all-replays; 11/11] |
| índice=505 | 1 | reverse/parallel | 8 | 11.4656875 | 11.27355 | 12.62602 | 12.912964 | 12.9847 | sí [all-replays; 11/11] |
| índice=505 | 1 | reverse-summaries/serial | 8 | 14.9455 | 14.6881 | 16.19052 | 16.489224 | 16.5639 | sí [all-replays; 11/11] |
| índice=505 | 1 | reverse-summaries/parallel | 8 | 10.6377375 | 10.57345 | 11.398775 | 11.526315 | 11.5582 | sí [all-replays; 11/11] |
| índice=514 | 1 | original/serial | 8 | 18.604625 | 18.17235 | 20.77497 | 21.770034 | 22.0188 | sí [all-replays; 11/11] |
| índice=514 | 1 | original/parallel | 8 | 12.9466125 | 12.5295 | 14.76516 | 15.274872 | 15.4023 | sí [all-replays; 11/11] |
| índice=514 | 1 | reverse/serial | 8 | 16.9499125 | 15.92915 | 21.76279 | 23.615438 | 24.0786 | sí [all-replays; 11/11] |
| índice=514 | 1 | reverse/parallel | 8 | 11.52115 | 11.05045 | 13.778315 | 14.724463 | 14.961 | sí [all-replays; 11/11] |
| índice=514 | 1 | reverse-summaries/serial | 8 | 14.9959875 | 14.95625 | 16.59755 | 17.12927 | 17.2622 | sí [all-replays; 11/11] |
| índice=514 | 1 | reverse-summaries/parallel | 8 | 10.581675 | 10.50445 | 11.29153 | 11.366066 | 11.3847 | sí [all-replays; 11/11] |
| índice=533 | 1 | original/serial | 8 | 17.9179375 | 17.9056 | 18.78358 | 18.870156 | 18.8918 | sí [all-replays; 11/11] |
| índice=533 | 1 | original/parallel | 8 | 12.7305 | 12.59475 | 13.423225 | 13.456965 | 13.4654 | sí [all-replays; 11/11] |
| índice=533 | 1 | reverse/serial | 8 | 15.43615 | 15.53405 | 15.72417 | 15.753234 | 15.7605 | sí [all-replays; 11/11] |
| índice=533 | 1 | reverse/parallel | 8 | 11.2115875 | 11.168 | 11.908865 | 11.944733 | 11.9537 | sí [all-replays; 11/11] |
| índice=533 | 1 | reverse-summaries/serial | 8 | 14.4459 | 14.1944 | 15.50987 | 15.577574 | 15.5945 | sí [all-replays; 11/11] |
| índice=533 | 1 | reverse-summaries/parallel | 8 | 10.355525 | 10.1444 | 11.25087 | 11.272094 | 11.2774 | sí [all-replays; 11/11] |
| índice=553 | 1 | original/serial | 8 | 18.0133 | 18.1129 | 18.45343 | 18.465246 | 18.4682 | sí [all-replays; 11/11] |
| índice=553 | 1 | original/parallel | 8 | 13.3447625 | 13.2801 | 14.1523 | 14.16686 | 14.1705 | sí [all-replays; 11/11] |
| índice=553 | 1 | reverse/serial | 8 | 15.7288375 | 15.9238 | 16.477395 | 16.506039 | 16.5132 | sí [all-replays; 11/11] |
| índice=553 | 1 | reverse/parallel | 8 | 11.506775 | 11.6609 | 12.344005 | 12.375281 | 12.3831 | sí [all-replays; 11/11] |
| índice=553 | 1 | reverse-summaries/serial | 8 | 15.066175 | 14.9983 | 15.55854 | 15.598748 | 15.6088 | sí [all-replays; 11/11] |
| índice=553 | 1 | reverse-summaries/parallel | 8 | 11.0211625 | 11.0263 | 11.590675 | 11.652695 | 11.6682 | sí [all-replays; 11/11] |

La columna de paridad corresponde a las tres salidas TTC por ruta. all-replays incluye llamadas iniciales, warmup y cada repetición registrada; last-only conserva el alcance limitado de recibos históricos sin ese registro. El estado del ensayo conserva también fallos de auditoría. La paridad de features intermedias es un diagnóstico separado.

| Índice | Llamadas esperadas | Registradas | Grilla exacta | Todos TTC | Primeras |
| --- | --- | --- | --- | --- | --- |
| 0 | 66 | 66 | sí | sí | sí |
| 137 | 66 | 66 | sí | sí | sí |
| 488 | 66 | 66 | sí | sí | sí |
| 505 | 66 | 66 | sí | sí | sí |
| 514 | 66 | 66 | sí | sí | sí |
| 533 | 66 | 66 | sí | sí | sí |
| 553 | 66 | 66 | sí | sí | sí |

| Índice | Ruta | Llamadas explícitas | Features exactas | Todas exactas |
| --- | --- | --- | --- | --- |
| 0 | original/parallel | 11 | 11 | sí |
| 0 | original/serial | 11 | 11 | sí |
| 0 | reverse-summaries/parallel | 11 | 11 | sí |
| 0 | reverse-summaries/serial | 11 | 11 | sí |
| 0 | reverse/parallel | 11 | 11 | sí |
| 0 | reverse/serial | 11 | 11 | sí |
| 137 | original/parallel | 11 | 11 | sí |
| 137 | original/serial | 11 | 11 | sí |
| 137 | reverse-summaries/parallel | 11 | 11 | sí |
| 137 | reverse-summaries/serial | 11 | 11 | sí |
| 137 | reverse/parallel | 11 | 11 | sí |
| 137 | reverse/serial | 11 | 11 | sí |
| 488 | original/parallel | 11 | 11 | sí |
| 488 | original/serial | 11 | 11 | sí |
| 488 | reverse-summaries/parallel | 11 | 11 | sí |
| 488 | reverse-summaries/serial | 11 | 11 | sí |
| 488 | reverse/parallel | 11 | 11 | sí |
| 488 | reverse/serial | 11 | 11 | sí |
| 505 | original/parallel | 11 | 11 | sí |
| 505 | original/serial | 11 | 11 | sí |
| 505 | reverse-summaries/parallel | 11 | 11 | sí |
| 505 | reverse-summaries/serial | 11 | 11 | sí |
| 505 | reverse/parallel | 11 | 11 | sí |
| 505 | reverse/serial | 11 | 11 | sí |
| 514 | original/parallel | 11 | 11 | sí |
| 514 | original/serial | 11 | 11 | sí |
| 514 | reverse-summaries/parallel | 11 | 11 | sí |
| 514 | reverse-summaries/serial | 11 | 11 | sí |
| 514 | reverse/parallel | 11 | 11 | sí |
| 514 | reverse/serial | 11 | 11 | sí |
| 533 | original/parallel | 11 | 11 | sí |
| 533 | original/serial | 11 | 11 | sí |
| 533 | reverse-summaries/parallel | 11 | 11 | sí |
| 533 | reverse-summaries/serial | 11 | 11 | sí |
| 533 | reverse/parallel | 11 | 11 | sí |
| 533 | reverse/serial | 11 | 11 | sí |
| 553 | original/parallel | 11 | 11 | sí |
| 553 | original/serial | 11 | 11 | sí |
| 553 | reverse-summaries/parallel | 11 | 11 | sí |
| 553 | reverse-summaries/serial | 11 | 11 | sí |
| 553 | reverse/parallel | 11 | 11 | sí |
| 553 | reverse/serial | 11 | 11 | sí |

| Gate preparado declarado | Valor |
| --- | --- |
| all_replay_gates_passed | sí |
| query_coverage_exact | sí |
| weights_unchanged | sí |
| normalizers_unchanged | sí |
| variant_weights_equal_original | sí |
| complete_update_admitted | no |
| sustained_50hz_demonstrated | no |
| two_object_admitted | no |

Cadencia de validación: Per-call context/parity and owned copies outside clock

El ahorro observado en este reloj preparado no se extrapola al servicio completo.

## immutable_rgb_admission

Admisión de inputs y caché RGB inmutable a consulta fija. Incluye MISS y HIT comprobados contra RAW independiente; el sellado precede al reloj interno. No se incluyen sus tiempos en comparaciones de actualizaciones completas.

### `multiobject/immutable_rgb_companion_v1/summary.json`

Estado: **COMPLETE**. Alcance: `IMMUTABLE_RGB_CACHE_INPUT_ADMISSION_FIXED_QUERY_REPEATS_NO_ARRIVAL_TIMING`.

| Recibo de admisión | Valor |
| --- | --- |
| passed | sí |
| expected_updates | 12 |
| validated_updates | 12 |
| schedule_exact | sí |
| rgb_cache_hit_updates | 6 |
| rgb_cache_miss_updates | 6 |
| preservación | sí |

| Índice | ROI | Repetición | Caché | RAW exacto | Argumentos device/HIT | Procedencia features | TTC | Primaria |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 0 | MISS | sí | sí | sí | sí | sí |
| 0 | 1 | 1 | HIT | sí | sí | sí | sí | sí |
| 137 | 1 | 0 | MISS | sí | sí | sí | sí | sí |
| 137 | 1 | 1 | HIT | sí | sí | sí | sí | sí |
| 553 | 1 | 0 | MISS | sí | sí | sí | sí | sí |
| 553 | 1 | 1 | HIT | sí | sí | sí | sí | sí |
| 0 | 2 | 0 | MISS | sí | sí | sí | sí | sí |
| 0 | 2 | 1 | HIT | sí | sí | sí | sí | sí |
| 137 | 2 | 0 | MISS | sí | sí | sí | sí | sí |
| 137 | 2 | 1 | HIT | sí | sí | sí | sí | sí |
| 553 | 2 | 0 | MISS | sí | sí | sí | sí | sí |
| 553 | 2 | 1 | HIT | sí | sí | sí | sí | sí |

Los tiempos internos instrumentados se conservan en el JSON original y no se presentan como latencia de llegada ni como evidencia de 50 Hz. Las repeticiones usan la misma consulta; no validan por sí solas una secuencia paced.

## crop_rgb_admission

Admisión de crops exactos y features RGB sobre RAW independiente. La secuencia MISS, HIT y borrado de features con HIT de crops usa una consulta fija. Sellado y borrado forzado preceden al reloj interno; no admite latencia ni 50 Hz.

### `multiobject/crop_rgb_companion_v1/summary.json`

Estado: **COMPLETE**. Alcance: `IMMUTABLE_RGB_CROP_AND_FEATURE_CACHE_ADMISSION_FORCED_MISS_NO_ARRIVAL_TIMING`.

| Recibo de admisión | Valor |
| --- | --- |
| passed | sí |
| expected_updates | 18 |
| validated_updates | 18 |
| schedule_exact | sí |
| rgb_cache_hit_updates | 6 |
| rgb_cache_miss_updates | 12 |
| preservación | sí |

| Índice | ROI | Repetición | Caché | RAW exacto | Argumentos device/HIT | Procedencia features | TTC | Primaria |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 0 | MISS | sí | sí | sí | sí | sí |
| 0 | 1 | 1 | HIT | sí | sí | sí | sí | sí |
| 0 | 1 | 2 | MISS | sí | sí | sí | sí | sí |
| 137 | 1 | 0 | MISS | sí | sí | sí | sí | sí |
| 137 | 1 | 1 | HIT | sí | sí | sí | sí | sí |
| 137 | 1 | 2 | MISS | sí | sí | sí | sí | sí |
| 553 | 1 | 0 | MISS | sí | sí | sí | sí | sí |
| 553 | 1 | 1 | HIT | sí | sí | sí | sí | sí |
| 553 | 1 | 2 | MISS | sí | sí | sí | sí | sí |
| 0 | 2 | 0 | MISS | sí | sí | sí | sí | sí |
| 0 | 2 | 1 | HIT | sí | sí | sí | sí | sí |
| 0 | 2 | 2 | MISS | sí | sí | sí | sí | sí |
| 137 | 2 | 0 | MISS | sí | sí | sí | sí | sí |
| 137 | 2 | 1 | HIT | sí | sí | sí | sí | sí |
| 137 | 2 | 2 | MISS | sí | sí | sí | sí | sí |
| 553 | 2 | 0 | MISS | sí | sí | sí | sí | sí |
| 553 | 2 | 1 | HIT | sí | sí | sí | sí | sí |
| 553 | 2 | 2 | MISS | sí | sí | sí | sí | sí |

| Matriz de crops | Valor |
| --- | --- |
| passed | sí |
| required_cases | [[1, 0], [1, 137], [1, 553], [2, 0], [2, 137], [2, 553]] |
| missing_cases | [] |
| arrival_or_latency_claim | no |

| Índice | ROI | Repetición | Crop HIT | Crop MISS | Crops guardados | Bypass | Borrado forzado | Crops conservados | Píxeles/deltas reales | Ejercicio completo | Checks no aprobados |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 0 | 0 | 7 | 7 | 0 | — | — | sí | sí | ninguno |
| 0 | 1 | 1 | 0 | 0 | 0 | 0 | — | — | sí | sí | ninguno |
| 0 | 1 | 2 | 7 | 0 | 0 | 0 | CLEAR_FEATURE_CACHE_ONLY | sí | sí | sí | ninguno |
| 137 | 1 | 0 | 0 | 7 | 7 | 0 | — | — | sí | sí | ninguno |
| 137 | 1 | 1 | 0 | 0 | 0 | 0 | — | — | sí | sí | ninguno |
| 137 | 1 | 2 | 7 | 0 | 0 | 0 | CLEAR_FEATURE_CACHE_ONLY | sí | sí | sí | ninguno |
| 553 | 1 | 0 | 0 | 7 | 7 | 0 | — | — | sí | sí | ninguno |
| 553 | 1 | 1 | 0 | 0 | 0 | 0 | — | — | sí | sí | ninguno |
| 553 | 1 | 2 | 7 | 0 | 0 | 0 | CLEAR_FEATURE_CACHE_ONLY | sí | sí | sí | ninguno |
| 0 | 2 | 0 | 0 | 14 | 14 | 0 | — | — | sí | sí | ninguno |
| 0 | 2 | 1 | 0 | 0 | 0 | 0 | — | — | sí | sí | ninguno |
| 0 | 2 | 2 | 14 | 0 | 0 | 0 | CLEAR_FEATURE_CACHE_ONLY | sí | sí | sí | ninguno |
| 137 | 2 | 0 | 0 | 14 | 14 | 0 | — | — | sí | sí | ninguno |
| 137 | 2 | 1 | 0 | 0 | 0 | 0 | — | — | sí | sí | ninguno |
| 137 | 2 | 2 | 14 | 0 | 0 | 0 | CLEAR_FEATURE_CACHE_ONLY | sí | sí | sí | ninguno |
| 553 | 2 | 0 | 0 | 14 | 14 | 0 | — | — | sí | sí | ninguno |
| 553 | 2 | 1 | 0 | 0 | 0 | 0 | — | — | sí | sí | ninguno |
| 553 | 2 | 2 | 14 | 0 | 0 | 0 | CLEAR_FEATURE_CACHE_ONLY | sí | sí | sí | ninguno |

El borrado forzado afecta únicamente a features y se registra fuera del reloj interno. Los HIT completos no ejecutan el productor RGB; sus features se vinculan al MISS verificado. El tercer paso comprueba los argumentos reales del productor tras recuperar crops. Esta matriz no acredita comportamiento paced.

Los tiempos internos instrumentados se conservan en el JSON original y no se presentan como latencia de llegada ni como evidencia de 50 Hz. Las repeticiones usan la misma consulta; no validan por sí solas una secuencia paced.

## subpacket

Diagnóstico RAW real de ingestión 5 ms frente a 20 ms: estado retenido, representaciones en cada frontera completa y TTC sólo al anchor final. No mide latencia ni una fuente paced; no demuestra ahorro de tres cuartos ni servicio a 50 Hz.

### `event/subpacket_backend_sensitive_v1/summary.json`

Estado: **FAILED**. Alcance: `REAL_RAW_SUBPACKET_STATE_AND_REPRESENTATION_DIAGNOSTIC_FINAL_TTC_ONLY`.

| Gate declarado | Valor |
| --- | --- |
| coverage_exact | — |
| all_backend_gates_passed | — |
| all_canonical_inputs_passed | — |
| all_final_ttc_passed | — |
| raw_stats_preserved | sí |
| weights_unchanged | — |
| normalizers_unchanged | — |
| latency_measured | no |
| complete_update_admitted | no |
| target_50hz_demonstrated | no |
| three_quarters_latency_saving_claimed | no |
| preservación | sí |

Error conservado: `ValueError: RAW archive exceeds explicit 268435456 byte budget`.

La paridad TTC corresponde sólo a las consultas finales. No se infiere paridad TTC en las fronteras intermedias ni latencia paced.

### `event/subpacket_backend_sensitive_v2/summary.json`

Estado: **COMPLETE**. Alcance: `REAL_RAW_SUBPACKET_STATE_AND_REPRESENTATION_DIAGNOSTIC_FINAL_TTC_ONLY`.

| Gate declarado | Valor |
| --- | --- |
| coverage_exact | sí |
| all_backend_gates_passed | sí |
| all_canonical_inputs_passed | sí |
| all_final_ttc_passed | sí |
| raw_stats_preserved | sí |
| weights_unchanged | sí |
| normalizers_unchanged | sí |
| latency_measured | no |
| complete_update_admitted | no |
| target_50hz_demonstrated | no |
| three_quarters_latency_saving_claimed | no |
| preservación | sí |

| Índice | Gate | Fronteras | Estados aprobados | Repr. exactas | Cobertura fronteras | Guardas | Guardas aprobadas | Cobertura guardas | Inputs canónicos exactos | TTC ambos backends | TTC entre backends exacto | Graphs sin cambios |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | sí | 31 | 31 | 31 | sí | 6 | 6 | sí | sí | sí | sí | sí |
| 137 | sí | 31 | 31 | 31 | sí | 6 | 6 | sí | sí | sí | sí | sí |
| 553 | sí | 31 | 31 | 31 | sí | 6 | 6 | sí | sí | sí | sí | sí |

La paridad TTC corresponde sólo a las consultas finales. No se infiere paridad TTC en las fronteras intermedias ni latencia paced.

## rgb_uint8

Preparación RGB sintética; incluye el tramo declarado en cada recibo. No incluye modelos ni TTC. Bytes propuestos no equivalen a transferencias ejecutadas.

Sin recibos disponibles en esta generación.

## rgb_exact

RAW RGB disponible en CPU a crops exactos preparados en el dispositivo; incluye sellado de llegadas nuevas, H2D y preparación. Excluye disco, reconstrucción del prefijo causal, modelos y TTC. No es un ensayo sostenido del servicio.

### `rgb_exact_gpu/idx553_v1/summary.json`

Estado: **FAILED**. Dispositivo: `—`. Alcance: `CPU_RAW_READY_TO_EXACT_RGB_DEVICE_READY_NO_MODELS_NO_TTC_NO_DISK`.

| Dato declarado | Valor |
| --- | --- |
| actualizaciones esperadas | — |
| actualizaciones medidas | — |
| cobertura completa | — |
| todos los píxeles/deltas exactos | — |
| compilaciones nuevas tras primación | — |
| preservación | — |
| definición primera preparación | — |
| definición primación | — |
| costes excluidos | — |
| modelos o normalizadores cargados | no |

error: `FileNotFoundError: [WinError 2] El sistema no puede encontrar el archivo especificado: 'C:\\Users\\Álvaro Schwiedop\\Desktop\\KriptaStudios\\EVOCON_JEPA_Codex_Handoff\\e-jepa-ttc-v18-joint-measurement\\_classes.py'`

La exactitud aquí corresponde a píxeles y deltas. No acredita paridad TTC.

### `rgb_exact_gpu/idx553_v2/summary.json`

Estado: **COMPLETE**. Dispositivo: `cuda`. Alcance: `CPU_RAW_READY_TO_EXACT_RGB_DEVICE_READY_NO_MODELS_NO_TTC_NO_DISK`.

| Dato declarado | Valor |
| --- | --- |
| actualizaciones esperadas | 7 |
| actualizaciones medidas | 7 |
| cobertura completa | sí |
| todos los píxeles/deltas exactos | sí |
| compilaciones nuevas tras primación | 0 |
| preservación | sí |
| definición primera preparación | FIRST_REAL_PREPARE_PER_RUNTIME; compiler code/cache may already exist from preceding cases |
| definición primación | EXPLICIT_SYNTHETIC_PRIMING_OF_DECLARED_CASE_SHAPES; all measured RAW misses retained |
| costes excluidos | ['disk', 'prefix_rebuild', 'post_clock_D2H_parity', 'models', 'TTC'] |
| modelos o normalizadores cargados | no |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Exactitud global, incluida primera preparación |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| índice=553 | 1 | full_history_miss | 3 | 37.3359667 | 36.5447 | 39.53126 | 39.796732 | 39.8631 | sí |
| índice=553 | 1 | causal_prefix_rebuilt | 3 | 9.34523334 | 8.7839 | 10.38743 | 10.529966 | 10.5656 | sí |

| Índice | Objetos | Fase | Ruta | Wall ms | Compilaciones | Exacto |
| --- | --- | --- | --- | --- | --- | --- |
| 553 | 1 | primera preparación | full_history_miss | 6631.4242 | 2 | sí |
| 553 | 1 | tras primación | full_history_miss | 36.5447 | 0 | sí |
| 553 | 1 | tras primación | causal_prefix_rebuilt | 8.7839 | 0 | sí |
| 553 | 1 | tras primación | full_history_miss | 35.6001 | 0 | sí |
| 553 | 1 | tras primación | causal_prefix_rebuilt | 8.68620002 | 0 | sí |
| 553 | 1 | tras primación | full_history_miss | 39.8631 | 0 | sí |
| 553 | 1 | tras primación | causal_prefix_rebuilt | 10.5656 | 0 | sí |

La exactitud aquí corresponde a píxeles y deltas. No acredita paridad TTC.

## scheduler

Control del planificador con RAW y predicciones ficticias. No ejecuta modelos ni GPU y no admite ningún objetivo TTC.

Sin recibos disponibles en esta generación.

## paced

Replay RAW con llegadas independientes y todos los TTC en host. Se separan deadline→TTC, disponibilidad RAW→TTC y adquisición; no se suman. Un ensayo completado no implica superar su gate de cadencia ni certificar sensores reales.

### `paced/baseline1_50_v1/summary.json`

Estado: **FAILED**. Dispositivo: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`. Alcance: `PACED_PERSISTENT_RAW_REPLAY_ARTIFICIAL_DISTINCT_ROIS_ALL_HOST_TTC`.

| Configuración/recibo | Valor |
| --- | --- |
| objetos por actualización | 1 |
| rate_hz declarado | 50 |
| actualizaciones ofertadas | 50 |
| duración ofertada s | 1 |
| duración wall s | 4.0608532 |
| wall real incluyendo startup s | — |
| primera oferta establecida | — |
| último TTC válido desde primer deadline s | — |
| wall incluye cierre/join | — |
| capacidad cola | 2 |
| wait_strategy | sleep |
| raw_source_mode | persistent |
| fuente RAW cota numérica bytes | — |
| fuente RAW presupuesto declarado bytes | — |
| service_options | {'max_objects': 1, 'reuse_event_encodings': False, 'rgb_object_chunk_size': 1, 'shared_rgb_images': False, 'parallel_branches': True, 'rgb_workers': 8, 'overlap_event_with_rgb_preparation': False} |
| updates esperados | 50 |
| resultados objeto esperados | 50 |
| cobertura completa | no |
| descartes | 37 |
| errores fuente | 0 |
| paridad posterior realizada | sí |
| paridad posterior pasa | no |
| cache MISS→HIT acreditado | — |
| cache HIT updates | — |
| cache MISS updates | — |
| preservación de originales | sí |
| hito con paridad declarado | no |
| capacidad veinte con paridad declarada | no |

| Updates previos aún pendientes en deadline | Deadlines observados |
| --- | --- |
| 0 | 1 |
| 1 | 1 |
| 2 | 1 |
| 3 | 30 |
| 4 | 5 |
| 5 | 4 |
| 6 | 3 |
| 7 | 4 |
| 8 | 1 |

El conteo describe acumulación observada en este ensayo; no prueba estabilidad fuera de su duración ni sustituye el gate de ausencia de trabajo previo.

| Población derivada de recibos | Recuento |
| --- | --- |
| expected_updates | 50 |
| offered_updates | 50 |
| grouped_updates | 50 |
| arrival_receipts | 50 |
| completed_responses | 13 |
| successful_all_object_ttc | 13 |
| failed_acknowledgements | 0 |
| updates_without_completion | 37 |
| dropped_updates | 37 |
| source_errors | 0 |
| missing_arrival_receipts | 0 |
| unexpected_arrival_receipts | 0 |
| counts_consistent | sí |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad TTC del replay |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| successful_all_object_ttc | 1 | deadline_response_ms | 13 | 999.514923 | 180.424 | 3817.13708 | 3820.08558 | 3820.8227 | no |
| successful_all_object_ttc | 1 | raw_availability_response_ms | 13 | 854.397462 | 34.3478 | 3593.94068 | 3602.57982 | 3604.7396 | no |
| successful_all_object_ttc | 1 | service_span_ms | 13 | 301.425962 | 27.1279 | 1459.97192 | 3170.99806 | 3598.7546 | no |
| ALL_RESPONSES_INCLUDING_FAILURES | 1 | deadline_response_ms | 13 | 999.514923 | 180.424 | 3817.13708 | 3820.08558 | 3820.8227 | — |
| ALL_RESPONSES_INCLUDING_FAILURES | 1 | raw_availability_response_ms | 13 | 854.397462 | 34.3478 | 3593.94068 | 3602.57982 | 3604.7396 | — |
| RAW_ARRIVALS_INCLUDING_DROPS | 1 | raw_availability_lateness_ms | 50 | 643.887388 | 659.2388 | 1256.49988 | 1320.30511 | 1327.1534 | — |

| Gate declarado | Resultado |
| --- | --- |
| capacity_checks.all_scheduled_arrivals_recorded | sí |
| capacity_checks.no_drops_or_source_errors | no |
| capacity_checks.complete_object_coverage | no |
| capacity_checks.actual_availability_before_service | sí |
| capacity_checks.every_deadline_response_under_20ms | no |
| capacity_checks.no_prior_work_at_scheduled_arrivals | no |
| capacity_checks.finished_before_end_of_last_period | no |
| capacity_checks.offered_50hz_for_at_least_60s | no |
| capacity_checks.twenty_distinct_objects_per_update | no |
| capacity_checks.source_lifecycle_succeeded | sí |
| milestone_checks.all_scheduled_arrivals_recorded | sí |
| milestone_checks.no_drops_or_source_errors | no |
| milestone_checks.complete_object_coverage | no |
| milestone_checks.actual_availability_before_service | sí |
| milestone_checks.every_deadline_response_under_20ms | no |
| milestone_checks.no_prior_work_at_scheduled_arrivals | no |
| milestone_checks.finished_before_end_of_last_period | no |
| milestone_checks.offered_50hz_for_at_least_60s | no |
| milestone_checks.source_lifecycle_succeeded | sí |
| milestone_checks.required_object_count | sí |
| milestone_checks.all_three_host_ttc_per_required_object | sí |

| Fuente RAW | Tamaño/mtime preservados |
| --- | --- |
| C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\EVOCON_JEPA_Codex_Handoff\e-jepa-ttc\datasets\evttc\CPLA\low\data.hdf5 | sí |

Los gates se muestran tal como quedaron registrados, sin recalcular admisión desde medias ni suponer una frecuencia ausente. COMPLETE no equivale a tiempo real. La distribución deadline_response_ms incluye adquisición y colas; raw_availability_response_ms comienza después de preparar el payload RAW. Las filas primarias successful_all_object_ttc se derivan de los recibos originales con cuantiles lineales; failed_acknowledgements son respuestas de error. Los campos legacy ALL_RESPONSES_INCLUDING_FAILURES mezclan ambas poblaciones y no representan una distribución de TTC válidos. No se modifica el gate registrado.

### `paced/memory1_50_v1/summary.json`

Estado: **FAILED**. Dispositivo: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`. Alcance: `PACED_PRELOADED_RAW_CPU_ARRIVALS_NOT_PHYSICAL_SENSOR`.

| Configuración/recibo | Valor |
| --- | --- |
| objetos por actualización | 1 |
| rate_hz declarado | 50 |
| actualizaciones ofertadas | 50 |
| duración ofertada s | 1 |
| duración wall s | 1.0147217 |
| wall real incluyendo startup s | — |
| primera oferta establecida | — |
| último TTC válido desde primer deadline s | — |
| wall incluye cierre/join | — |
| capacidad cola | 2 |
| wait_strategy | sleep |
| raw_source_mode | memory |
| fuente RAW cota numérica bytes | 1651465904 |
| fuente RAW presupuesto declarado bytes | 2147483648 |
| service_options | {'max_objects': 1, 'reuse_event_encodings': False, 'rgb_object_chunk_size': 1, 'shared_rgb_images': False, 'parallel_branches': True, 'rgb_workers': 8, 'overlap_event_with_rgb_preparation': False} |
| updates esperados | 50 |
| resultados objeto esperados | 50 |
| cobertura completa | no |
| descartes | 1 |
| errores fuente | 0 |
| paridad posterior realizada | sí |
| paridad posterior pasa | no |
| cache MISS→HIT acreditado | — |
| cache HIT updates | — |
| cache MISS updates | — |
| preservación de originales | sí |
| hito con paridad declarado | no |
| capacidad veinte con paridad declarada | no |

| Updates previos aún pendientes en deadline | Deadlines observados |
| --- | --- |
| 0 | 39 |
| 1 | 3 |
| 2 | 6 |
| 3 | 2 |

El conteo describe acumulación observada en este ensayo; no prueba estabilidad fuera de su duración ni sustituye el gate de ausencia de trabajo previo.

| Población derivada de recibos | Recuento |
| --- | --- |
| expected_updates | 50 |
| offered_updates | 50 |
| grouped_updates | 50 |
| arrival_receipts | 50 |
| completed_responses | 49 |
| successful_all_object_ttc | 9 |
| failed_acknowledgements | 40 |
| updates_without_completion | 1 |
| dropped_updates | 1 |
| source_errors | 0 |
| missing_arrival_receipts | 0 |
| unexpected_arrival_receipts | 0 |
| counts_consistent | sí |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad TTC del replay |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| successful_all_object_ttc | 1 | deadline_response_ms | 9 | 55.6078667 | 54.4659 | 74.72102 | 76.359164 | 76.7687 | no |
| successful_all_object_ttc | 1 | raw_availability_response_ms | 9 | 51.6023889 | 53.6565 | 69.5936 | 70.13664 | 70.2724 | no |
| successful_all_object_ttc | 1 | service_span_ms | 9 | 25.3836 | 24.8054 | 27.37738 | 27.666116 | 27.7383 | no |
| failed_acknowledgements | 1 | deadline_response_ms | 40 | 5.87752 | 2.25305 | 17.482795 | 31.831243 | 37.1104 | — |
| failed_acknowledgements | 1 | raw_availability_response_ms | 40 | 2.0016825 | 0.27195 | 11.275545 | 30.799427 | 36.3767 | — |
| failed_acknowledgements | 1 | service_span_ms | 40 | 0.0141175 | 0.0126 | 0.025695 | 0.032624 | 0.0359 | — |
| ALL_RESPONSES_INCLUDING_FAILURES | 1 | deadline_response_ms | 49 | 15.0116653 | 6.1946 | 64.45218 | 74.311484 | 76.7687 | — |
| ALL_RESPONSES_INCLUDING_FAILURES | 1 | raw_availability_response_ms | 49 | 11.1120163 | 0.3572 | 61.41506 | 69.45784 | 70.2724 | — |
| RAW_ARRIVALS_INCLUDING_DROPS | 1 | raw_availability_lateness_ms | 50 | 3.834812 | 1.5226 | 10.4093 | 13.135295 | 13.7647 | — |

| Gate declarado | Resultado |
| --- | --- |
| capacity_checks.all_scheduled_arrivals_recorded | sí |
| capacity_checks.no_drops_or_source_errors | no |
| capacity_checks.complete_object_coverage | no |
| capacity_checks.actual_availability_before_service | sí |
| capacity_checks.every_deadline_response_under_20ms | no |
| capacity_checks.no_prior_work_at_scheduled_arrivals | no |
| capacity_checks.finished_before_end_of_last_period | no |
| capacity_checks.offered_50hz_for_at_least_60s | no |
| capacity_checks.twenty_distinct_objects_per_update | no |
| capacity_checks.source_lifecycle_succeeded | sí |
| milestone_checks.all_scheduled_arrivals_recorded | sí |
| milestone_checks.no_drops_or_source_errors | no |
| milestone_checks.complete_object_coverage | no |
| milestone_checks.actual_availability_before_service | sí |
| milestone_checks.every_deadline_response_under_20ms | no |
| milestone_checks.no_prior_work_at_scheduled_arrivals | no |
| milestone_checks.finished_before_end_of_last_period | no |
| milestone_checks.offered_50hz_for_at_least_60s | no |
| milestone_checks.source_lifecycle_succeeded | sí |
| milestone_checks.required_object_count | sí |
| milestone_checks.all_three_host_ttc_per_required_object | no |

| Fuente RAW | Tamaño/mtime preservados |
| --- | --- |
| C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\EVOCON_JEPA_Codex_Handoff\e-jepa-ttc\datasets\evttc\CPLA\low\data.hdf5 | sí |

history_failure: `ValueError: RAW_GAP: expected index 9, received 10`

Los gates se muestran tal como quedaron registrados, sin recalcular admisión desde medias ni suponer una frecuencia ausente. COMPLETE no equivale a tiempo real. La distribución deadline_response_ms incluye adquisición y colas; raw_availability_response_ms comienza después de preparar el payload RAW. Las filas primarias successful_all_object_ttc se derivan de los recibos originales con cuantiles lineales; failed_acknowledgements son respuestas de error. Los campos legacy ALL_RESPONSES_INCLUDING_FAILURES mezclan ambas poblaciones y no representan una distribución de TTC válidos. No se modifica el gate registrado.

### `paced/memory_cache_overlap1_50_v1/summary.json`

Estado: **FAILED**. Dispositivo: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`. Alcance: `PACED_PRELOADED_RAW_CPU_ARRIVALS_NOT_PHYSICAL_SENSOR`.

| Configuración/recibo | Valor |
| --- | --- |
| objetos por actualización | 1 |
| rate_hz declarado | 50 |
| actualizaciones ofertadas | 50 |
| duración ofertada s | 1 |
| duración wall s | 0.9958626 |
| wall real incluyendo startup s | — |
| primera oferta establecida | — |
| último TTC válido desde primer deadline s | — |
| wall incluye cierre/join | — |
| capacidad cola | 2 |
| wait_strategy | sleep |
| raw_source_mode | memory |
| fuente RAW cota numérica bytes | 1651465904 |
| fuente RAW presupuesto declarado bytes | 2147483648 |
| service_options | {'max_objects': 1, 'reuse_event_encodings': False, 'rgb_object_chunk_size': 1, 'shared_rgb_images': False, 'parallel_branches': True, 'rgb_workers': 8, 'overlap_event_with_rgb_preparation': True} |
| updates esperados | 50 |
| resultados objeto esperados | 50 |
| cobertura completa | no |
| descartes | 1 |
| errores fuente | 0 |
| paridad posterior realizada | sí |
| paridad posterior pasa | no |
| cache MISS→HIT acreditado | sí |
| cache HIT updates | 12 |
| cache MISS updates | 15 |
| preservación de originales | sí |
| hito con paridad declarado | no |
| capacidad veinte con paridad declarada | no |

| Updates previos aún pendientes en deadline | Deadlines observados |
| --- | --- |
| 0 | 21 |
| 1 | 12 |
| 2 | 15 |
| 3 | 2 |

El conteo describe acumulación observada en este ensayo; no prueba estabilidad fuera de su duración ni sustituye el gate de ausencia de trabajo previo.

| Población derivada de recibos | Recuento |
| --- | --- |
| expected_updates | 50 |
| offered_updates | 50 |
| grouped_updates | 50 |
| arrival_receipts | 50 |
| completed_responses | 49 |
| successful_all_object_ttc | 27 |
| failed_acknowledgements | 22 |
| updates_without_completion | 1 |
| dropped_updates | 1 |
| source_errors | 0 |
| missing_arrival_receipts | 0 |
| unexpected_arrival_receipts | 0 |
| counts_consistent | sí |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad TTC del replay |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| successful_all_object_ttc | 1 | deadline_response_ms | 27 | 45.4657815 | 45.5556 | 69.72315 | 72.574482 | 73.1424 | no |
| successful_all_object_ttc | 1 | raw_availability_response_ms | 27 | 41.7525889 | 40.6584 | 65.07725 | 67.931276 | 68.9225 | no |
| successful_all_object_ttc | 1 | service_span_ms | 27 | 21.4260444 | 22.1533 | 27.54172 | 28.805084 | 28.968 | no |
| failed_acknowledgements | 1 | deadline_response_ms | 22 | 6.21724545 | 4.39035 | 15.736505 | 30.075824 | 33.8633 | — |
| failed_acknowledgements | 1 | raw_availability_response_ms | 22 | 2.81911364 | 0.39245 | 14.825145 | 24.248954 | 26.7317 | — |
| failed_acknowledgements | 1 | service_span_ms | 22 | 0.00996818182 | 0.009 | 0.015365 | 0.022194 | 0.024 | — |
| ALL_RESPONSES_INCLUDING_FAILURES | 1 | deadline_response_ms | 49 | 27.8439898 | 29.4006 | 66.75604 | 72.093936 | 73.1424 | — |
| ALL_RESPONSES_INCLUDING_FAILURES | 1 | raw_availability_response_ms | 49 | 24.2722531 | 25.7268 | 62.7346 | 67.092548 | 68.9225 | — |
| RAW_ARRIVALS_INCLUDING_DROPS | 1 | raw_availability_lateness_ms | 50 | 3.514532 | 1.47395 | 7.374105 | 8.134695 | 8.1418 | — |

| Gate declarado | Resultado |
| --- | --- |
| capacity_checks.all_scheduled_arrivals_recorded | sí |
| capacity_checks.no_drops_or_source_errors | no |
| capacity_checks.complete_object_coverage | no |
| capacity_checks.actual_availability_before_service | sí |
| capacity_checks.every_deadline_response_under_20ms | no |
| capacity_checks.no_prior_work_at_scheduled_arrivals | no |
| capacity_checks.finished_before_end_of_last_period | sí |
| capacity_checks.offered_50hz_for_at_least_60s | no |
| capacity_checks.twenty_distinct_objects_per_update | no |
| capacity_checks.source_lifecycle_succeeded | sí |
| milestone_checks.all_scheduled_arrivals_recorded | sí |
| milestone_checks.no_drops_or_source_errors | no |
| milestone_checks.complete_object_coverage | no |
| milestone_checks.actual_availability_before_service | sí |
| milestone_checks.every_deadline_response_under_20ms | no |
| milestone_checks.no_prior_work_at_scheduled_arrivals | no |
| milestone_checks.finished_before_end_of_last_period | sí |
| milestone_checks.offered_50hz_for_at_least_60s | no |
| milestone_checks.source_lifecycle_succeeded | sí |
| milestone_checks.required_object_count | sí |
| milestone_checks.all_three_host_ttc_per_required_object | no |
| milestone_checks.rgb_cache_stored_miss_then_real_hit | sí |

| Fuente RAW | Tamaño/mtime preservados |
| --- | --- |
| C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\EVOCON_JEPA_Codex_Handoff\e-jepa-ttc\datasets\evttc\CPLA\low\data.hdf5 | sí |

history_failure: `ValueError: RAW_GAP: expected index 27, received 28`

Los gates se muestran tal como quedaron registrados, sin recalcular admisión desde medias ni suponer una frecuencia ausente. COMPLETE no equivale a tiempo real. La distribución deadline_response_ms incluye adquisición y colas; raw_availability_response_ms comienza después de preparar el payload RAW. Las filas primarias successful_all_object_ttc se derivan de los recibos originales con cuantiles lineales; failed_acknowledgements son respuestas de error. Los campos legacy ALL_RESPONSES_INCLUDING_FAILURES mezclan ambas poblaciones y no representan una distribución de TTC válidos. No se modifica el gate registrado.

### `paced/memory_cache_overlap_reverse1_250_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`. Alcance: `PACED_PRELOADED_RAW_CPU_ARRIVALS_NOT_PHYSICAL_SENSOR`.

| Configuración/recibo | Valor |
| --- | --- |
| objetos por actualización | 1 |
| rate_hz declarado | 50 |
| actualizaciones ofertadas | 250 |
| duración ofertada s | 5 |
| duración wall s | 5.0493202 |
| wall real incluyendo startup s | — |
| primera oferta establecida | — |
| último TTC válido desde primer deadline s | 5.0036785 |
| wall incluye cierre/join | sí |
| capacidad cola | 2 |
| wait_strategy | sleep |
| raw_source_mode | memory |
| fuente RAW cota numérica bytes | 2611872720 |
| fuente RAW presupuesto declarado bytes | 3221225472 |
| service_options | {'max_objects': 1, 'reuse_event_encodings': False, 'rgb_object_chunk_size': 1, 'shared_rgb_images': False, 'parallel_branches': True, 'rgb_workers': 8, 'overlap_event_with_rgb_preparation': True, 'transport_reuse': 'reverse-summaries'} |
| updates esperados | 250 |
| resultados objeto esperados | 250 |
| cobertura completa | sí |
| descartes | 0 |
| errores fuente | 0 |
| paridad posterior realizada | sí |
| paridad posterior pasa | sí |
| cache MISS→HIT acreditado | sí |
| cache HIT updates | 143 |
| cache MISS updates | 107 |
| preservación de originales | sí |
| hito con paridad declarado | no |
| capacidad veinte con paridad declarada | no |

| Updates previos aún pendientes en deadline | Deadlines observados |
| --- | --- |
| 0 | 50 |
| 1 | 200 |

El conteo describe acumulación observada en este ensayo; no prueba estabilidad fuera de su duración ni sustituye el gate de ausencia de trabajo previo.

| Lifecycle registrado | Valor |
| --- | --- |
| schema | train40_paced_source_lifecycle_v2 |
| order | OFFERING_DONE_THEN_CONSUMER_DRAIN_OR_ABORT_THEN_FINALIZE |
| offering_ended_ns | 307352786448100 |
| consumer_drain_completed_ns | 307352808565600 |
| finalize_permitted_ns | 307352808566400 |
| producer_finished_ns | 307352854153000 |
| finalize.started_ns | 307352808596700 |
| finalize.ended_ns | 307352854150600 |

| Población derivada de recibos | Recuento |
| --- | --- |
| expected_updates | 250 |
| offered_updates | 250 |
| grouped_updates | 250 |
| arrival_receipts | 250 |
| completed_responses | 250 |
| successful_all_object_ttc | 250 |
| failed_acknowledgements | 0 |
| updates_without_completion | 0 |
| dropped_updates | 0 |
| source_errors | 0 |
| missing_arrival_receipts | 0 |
| unexpected_arrival_receipts | 0 |
| counts_consistent | sí |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad TTC del replay |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| successful_all_object_ttc | 1 | deadline_response_ms | 250 | 23.9071148 | 24.538 | 29.143155 | 32.414209 | 35.3326 | sí |
| successful_all_object_ttc | 1 | raw_availability_response_ms | 250 | 20.7330316 | 20.8045 | 25.25299 | 27.627191 | 29.593 | sí |
| successful_all_object_ttc | 1 | service_span_ms | 250 | 17.610386 | 16.91275 | 21.426025 | 23.212746 | 25.9175 | sí |
| ALL_RESPONSES_INCLUDING_FAILURES | 1 | deadline_response_ms | 250 | 23.9071148 | 24.538 | 29.143155 | 32.414209 | 35.3326 | — |
| ALL_RESPONSES_INCLUDING_FAILURES | 1 | raw_availability_response_ms | 250 | 20.7330316 | 20.8045 | 25.25299 | 27.627191 | 29.593 | — |
| RAW_ARRIVALS_INCLUDING_DROPS | 1 | raw_availability_lateness_ms | 250 | 3.1740832 | 1.53505 | 7.136715 | 8.062873 | 9.3893 | — |

| Gate declarado | Resultado |
| --- | --- |
| capacity_checks.all_scheduled_arrivals_recorded | sí |
| capacity_checks.no_drops_or_source_errors | sí |
| capacity_checks.complete_object_coverage | sí |
| capacity_checks.actual_availability_before_service | sí |
| capacity_checks.every_deadline_response_under_20ms | no |
| capacity_checks.no_prior_work_at_scheduled_arrivals | no |
| capacity_checks.finished_before_end_of_last_period | no |
| capacity_checks.offered_50hz_for_at_least_60s | no |
| capacity_checks.twenty_distinct_objects_per_update | no |
| capacity_checks.source_lifecycle_succeeded | sí |
| milestone_checks.all_scheduled_arrivals_recorded | sí |
| milestone_checks.no_drops_or_source_errors | sí |
| milestone_checks.complete_object_coverage | sí |
| milestone_checks.actual_availability_before_service | sí |
| milestone_checks.every_deadline_response_under_20ms | no |
| milestone_checks.no_prior_work_at_scheduled_arrivals | no |
| milestone_checks.finished_before_end_of_last_period | no |
| milestone_checks.offered_50hz_for_at_least_60s | no |
| milestone_checks.source_lifecycle_succeeded | sí |
| milestone_checks.required_object_count | sí |
| milestone_checks.all_three_host_ttc_per_required_object | sí |
| milestone_checks.rgb_cache_stored_miss_then_real_hit | sí |

| Fuente RAW | Tamaño/mtime preservados |
| --- | --- |
| C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\EVOCON_JEPA_Codex_Handoff\e-jepa-ttc\datasets\evttc\CPLA\low\data.hdf5 | sí |

Los gates se muestran tal como quedaron registrados, sin recalcular admisión desde medias ni suponer una frecuencia ausente. COMPLETE no equivale a tiempo real. La distribución deadline_response_ms incluye adquisición y colas; raw_availability_response_ms comienza después de preparar el payload RAW. Las filas primarias successful_all_object_ttc se derivan de los recibos originales con cuantiles lineales; failed_acknowledgements son respuestas de error. Los campos legacy ALL_RESPONSES_INCLUDING_FAILURES mezclan ambas poblaciones y no representan una distribución de TTC válidos. No se modifica el gate registrado.

### `paced/memory_cache_overlap_reverse1_50_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`. Alcance: `PACED_PRELOADED_RAW_CPU_ARRIVALS_NOT_PHYSICAL_SENSOR`.

| Configuración/recibo | Valor |
| --- | --- |
| objetos por actualización | 1 |
| rate_hz declarado | 50 |
| actualizaciones ofertadas | 50 |
| duración ofertada s | 1 |
| duración wall s | 1.0142091 |
| wall real incluyendo startup s | — |
| primera oferta establecida | — |
| último TTC válido desde primer deadline s | — |
| wall incluye cierre/join | — |
| capacidad cola | 2 |
| wait_strategy | sleep |
| raw_source_mode | memory |
| fuente RAW cota numérica bytes | 1651465904 |
| fuente RAW presupuesto declarado bytes | 2147483648 |
| service_options | {'max_objects': 1, 'reuse_event_encodings': False, 'rgb_object_chunk_size': 1, 'shared_rgb_images': False, 'parallel_branches': True, 'rgb_workers': 8, 'overlap_event_with_rgb_preparation': True, 'transport_reuse': 'reverse-summaries'} |
| updates esperados | 50 |
| resultados objeto esperados | 50 |
| cobertura completa | sí |
| descartes | 0 |
| errores fuente | 0 |
| paridad posterior realizada | sí |
| paridad posterior pasa | sí |
| cache MISS→HIT acreditado | sí |
| cache HIT updates | 23 |
| cache MISS updates | 27 |
| preservación de originales | sí |
| hito con paridad declarado | no |
| capacidad veinte con paridad declarada | no |

| Updates previos aún pendientes en deadline | Deadlines observados |
| --- | --- |
| 0 | 8 |
| 1 | 42 |

El conteo describe acumulación observada en este ensayo; no prueba estabilidad fuera de su duración ni sustituye el gate de ausencia de trabajo previo.

| Población derivada de recibos | Recuento |
| --- | --- |
| expected_updates | 50 |
| offered_updates | 50 |
| grouped_updates | 50 |
| arrival_receipts | 50 |
| completed_responses | 50 |
| successful_all_object_ttc | 50 |
| failed_acknowledgements | 0 |
| updates_without_completion | 0 |
| dropped_updates | 0 |
| source_errors | 0 |
| missing_arrival_receipts | 0 |
| unexpected_arrival_receipts | 0 |
| counts_consistent | sí |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad TTC del replay |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| successful_all_object_ttc | 1 | deadline_response_ms | 50 | 24.488924 | 24.526 | 29.892095 | 36.404291 | 38.5476 | sí |
| successful_all_object_ttc | 1 | raw_availability_response_ms | 50 | 21.401698 | 20.9304 | 24.565245 | 32.931396 | 33.1811 | sí |
| successful_all_object_ttc | 1 | service_span_ms | 50 | 18.226436 | 18.74 | 21.823595 | 28.130135 | 32.2307 | sí |
| ALL_RESPONSES_INCLUDING_FAILURES | 1 | deadline_response_ms | 50 | 24.488924 | 24.526 | 29.892095 | 36.404291 | 38.5476 | — |
| ALL_RESPONSES_INCLUDING_FAILURES | 1 | raw_availability_response_ms | 50 | 21.401698 | 20.9304 | 24.565245 | 32.931396 | 33.1811 | — |
| RAW_ARRIVALS_INCLUDING_DROPS | 1 | raw_availability_lateness_ms | 50 | 3.087226 | 1.204 | 6.95354 | 7.344081 | 7.4387 | — |

| Gate declarado | Resultado |
| --- | --- |
| capacity_checks.all_scheduled_arrivals_recorded | sí |
| capacity_checks.no_drops_or_source_errors | sí |
| capacity_checks.complete_object_coverage | sí |
| capacity_checks.actual_availability_before_service | sí |
| capacity_checks.every_deadline_response_under_20ms | no |
| capacity_checks.no_prior_work_at_scheduled_arrivals | no |
| capacity_checks.finished_before_end_of_last_period | no |
| capacity_checks.offered_50hz_for_at_least_60s | no |
| capacity_checks.twenty_distinct_objects_per_update | no |
| capacity_checks.source_lifecycle_succeeded | sí |
| milestone_checks.all_scheduled_arrivals_recorded | sí |
| milestone_checks.no_drops_or_source_errors | sí |
| milestone_checks.complete_object_coverage | sí |
| milestone_checks.actual_availability_before_service | sí |
| milestone_checks.every_deadline_response_under_20ms | no |
| milestone_checks.no_prior_work_at_scheduled_arrivals | no |
| milestone_checks.finished_before_end_of_last_period | no |
| milestone_checks.offered_50hz_for_at_least_60s | no |
| milestone_checks.source_lifecycle_succeeded | sí |
| milestone_checks.required_object_count | sí |
| milestone_checks.all_three_host_ttc_per_required_object | sí |
| milestone_checks.rgb_cache_stored_miss_then_real_hit | sí |

| Fuente RAW | Tamaño/mtime preservados |
| --- | --- |
| C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\EVOCON_JEPA_Codex_Handoff\e-jepa-ttc\datasets\evttc\CPLA\low\data.hdf5 | sí |

Los gates se muestran tal como quedaron registrados, sin recalcular admisión desde medias ni suponer una frecuencia ausente. COMPLETE no equivale a tiempo real. La distribución deadline_response_ms incluye adquisición y colas; raw_availability_response_ms comienza después de preparar el payload RAW. Las filas primarias successful_all_object_ttc se derivan de los recibos originales con cuantiles lineales; failed_acknowledgements son respuestas de error. Los campos legacy ALL_RESPONSES_INCLUDING_FAILURES mezclan ambas poblaciones y no representan una distribución de TTC válidos. No se modifica el gate registrado.

### `paced/memory_cache_overlap_reverse1_50_v2/summary.json`

Estado: **COMPLETE**. Dispositivo: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`. Alcance: `PACED_PRELOADED_RAW_CPU_ARRIVALS_NOT_PHYSICAL_SENSOR`.

| Configuración/recibo | Valor |
| --- | --- |
| objetos por actualización | 1 |
| rate_hz declarado | 50 |
| actualizaciones ofertadas | 50 |
| duración ofertada s | 1 |
| duración wall s | 1.0187554 |
| wall real incluyendo startup s | — |
| primera oferta establecida | — |
| último TTC válido desde primer deadline s | 1.0019322 |
| wall incluye cierre/join | sí |
| capacidad cola | 2 |
| wait_strategy | sleep |
| raw_source_mode | memory |
| fuente RAW cota numérica bytes | 1651465904 |
| fuente RAW presupuesto declarado bytes | 2147483648 |
| service_options | {'max_objects': 1, 'reuse_event_encodings': False, 'rgb_object_chunk_size': 1, 'shared_rgb_images': False, 'parallel_branches': True, 'rgb_workers': 8, 'overlap_event_with_rgb_preparation': True, 'transport_reuse': 'reverse-summaries'} |
| updates esperados | 50 |
| resultados objeto esperados | 50 |
| cobertura completa | sí |
| descartes | 0 |
| errores fuente | 0 |
| paridad posterior realizada | sí |
| paridad posterior pasa | sí |
| cache MISS→HIT acreditado | sí |
| cache HIT updates | 23 |
| cache MISS updates | 27 |
| preservación de originales | sí |
| hito con paridad declarado | no |
| capacidad veinte con paridad declarada | no |

| Updates previos aún pendientes en deadline | Deadlines observados |
| --- | --- |
| 0 | 5 |
| 1 | 45 |

El conteo describe acumulación observada en este ensayo; no prueba estabilidad fuera de su duración ni sustituye el gate de ausencia de trabajo previo.

| Lifecycle registrado | Valor |
| --- | --- |
| schema | train40_paced_source_lifecycle_v2 |
| order | OFFERING_DONE_THEN_CONSUMER_DRAIN_OR_ABORT_THEN_FINALIZE |
| offering_ended_ns | 307089687171500 |
| consumer_drain_completed_ns | 307089708186200 |
| finalize_permitted_ns | 307089708187000 |
| producer_finished_ns | 307089724961300 |
| finalize.started_ns | 307089708219300 |
| finalize.ended_ns | 307089724959700 |

| Población derivada de recibos | Recuento |
| --- | --- |
| expected_updates | 50 |
| offered_updates | 50 |
| grouped_updates | 50 |
| arrival_receipts | 50 |
| completed_responses | 50 |
| successful_all_object_ttc | 50 |
| failed_acknowledgements | 0 |
| updates_without_completion | 0 |
| dropped_updates | 0 |
| source_errors | 0 |
| missing_arrival_receipts | 0 |
| unexpected_arrival_receipts | 0 |
| counts_consistent | sí |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad TTC del replay |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| successful_all_object_ttc | 1 | deadline_response_ms | 50 | 24.543954 | 24.8449 | 28.85508 | 30.415244 | 31.809 | sí |
| successful_all_object_ttc | 1 | raw_availability_response_ms | 50 | 21.329262 | 21.03015 | 24.36377 | 26.333144 | 27.9719 | sí |
| successful_all_object_ttc | 1 | service_span_ms | 50 | 18.139248 | 18.73265 | 21.293255 | 22.769393 | 23.7417 | sí |
| ALL_RESPONSES_INCLUDING_FAILURES | 1 | deadline_response_ms | 50 | 24.543954 | 24.8449 | 28.85508 | 30.415244 | 31.809 | — |
| ALL_RESPONSES_INCLUDING_FAILURES | 1 | raw_availability_response_ms | 50 | 21.329262 | 21.03015 | 24.36377 | 26.333144 | 27.9719 | — |
| RAW_ARRIVALS_INCLUDING_DROPS | 1 | raw_availability_lateness_ms | 50 | 3.214692 | 1.15785 | 6.95824 | 7.71665 | 8.0082 | — |

| Gate declarado | Resultado |
| --- | --- |
| capacity_checks.all_scheduled_arrivals_recorded | sí |
| capacity_checks.no_drops_or_source_errors | sí |
| capacity_checks.complete_object_coverage | sí |
| capacity_checks.actual_availability_before_service | sí |
| capacity_checks.every_deadline_response_under_20ms | no |
| capacity_checks.no_prior_work_at_scheduled_arrivals | no |
| capacity_checks.finished_before_end_of_last_period | no |
| capacity_checks.offered_50hz_for_at_least_60s | no |
| capacity_checks.twenty_distinct_objects_per_update | no |
| capacity_checks.source_lifecycle_succeeded | sí |
| milestone_checks.all_scheduled_arrivals_recorded | sí |
| milestone_checks.no_drops_or_source_errors | sí |
| milestone_checks.complete_object_coverage | sí |
| milestone_checks.actual_availability_before_service | sí |
| milestone_checks.every_deadline_response_under_20ms | no |
| milestone_checks.no_prior_work_at_scheduled_arrivals | no |
| milestone_checks.finished_before_end_of_last_period | no |
| milestone_checks.offered_50hz_for_at_least_60s | no |
| milestone_checks.source_lifecycle_succeeded | sí |
| milestone_checks.required_object_count | sí |
| milestone_checks.all_three_host_ttc_per_required_object | sí |
| milestone_checks.rgb_cache_stored_miss_then_real_hit | sí |

| Fuente RAW | Tamaño/mtime preservados |
| --- | --- |
| C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\EVOCON_JEPA_Codex_Handoff\e-jepa-ttc\datasets\evttc\CPLA\low\data.hdf5 | sí |

Los gates se muestran tal como quedaron registrados, sin recalcular admisión desde medias ni suponer una frecuencia ausente. COMPLETE no equivale a tiempo real. La distribución deadline_response_ms incluye adquisición y colas; raw_availability_response_ms comienza después de preparar el payload RAW. Las filas primarias successful_all_object_ttc se derivan de los recibos originales con cuantiles lineales; failed_acknowledgements son respuestas de error. Los campos legacy ALL_RESPONSES_INCLUDING_FAILURES mezclan ambas poblaciones y no representan una distribución de TTC válidos. No se modifica el gate registrado.

### `paced/memory_crops_reverse1_250_v1/summary.json`

Estado: **FAILED**. Dispositivo: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`. Alcance: `PACED_PRELOADED_RAW_CPU_ARRIVALS_NOT_PHYSICAL_SENSOR`.

| Configuración/recibo | Valor |
| --- | --- |
| objetos por actualización | 1 |
| rate_hz declarado | 50 |
| actualizaciones ofertadas | 250 |
| duración ofertada s | 5 |
| duración wall s | 5.1799377 |
| wall real incluyendo startup s | 5.2947212 |
| primera oferta establecida | sí |
| último TTC válido desde primer deadline s | 2.3673916 |
| wall incluye cierre/join | sí |
| capacidad cola | 2 |
| wait_strategy | sleep |
| raw_source_mode | memory |
| fuente RAW cota numérica bytes | 2611872720 |
| fuente RAW presupuesto declarado bytes | 3217883136 |
| service_options | {'max_objects': 1, 'reuse_event_encodings': False, 'rgb_object_chunk_size': 1, 'shared_rgb_images': False, 'parallel_branches': True, 'rgb_workers': 8, 'overlap_event_with_rgb_preparation': True, 'transport_reuse': 'reverse-summaries', 'rgb_crop_cache': True} |
| updates esperados | 250 |
| resultados objeto esperados | 250 |
| cobertura completa | no |
| descartes | 2 |
| errores fuente | 0 |
| paridad posterior realizada | sí |
| paridad posterior pasa | no |
| cache MISS→HIT acreditado | sí |
| cache HIT updates | 62 |
| cache MISS updates | 53 |
| preservación de originales | sí |
| hito con paridad declarado | no |
| capacidad veinte con paridad declarada | no |

| Updates previos aún pendientes en deadline | Deadlines observados |
| --- | --- |
| 0 | 141 |
| 1 | 102 |
| 2 | 5 |
| 3 | 2 |

El conteo describe acumulación observada en este ensayo; no prueba estabilidad fuera de su duración ni sustituye el gate de ausencia de trabajo previo.

| Lifecycle registrado | Valor |
| --- | --- |
| schema | train40_paced_source_lifecycle_v2 |
| order | OFFERING_DONE_THEN_CONSUMER_DRAIN_OR_ABORT_THEN_FINALIZE |
| offering_ended_ns | 310238610516800 |
| consumer_drain_completed_ns | 310238612443900 |
| finalize_permitted_ns | 310238612448000 |
| producer_finished_ns | 310238776686800 |
| finalize.started_ns | 310238612595000 |
| finalize.ended_ns | 310238776680600 |

| Población derivada de recibos | Recuento |
| --- | --- |
| expected_updates | 250 |
| offered_updates | 250 |
| grouped_updates | 250 |
| arrival_receipts | 250 |
| completed_responses | 248 |
| successful_all_object_ttc | 115 |
| failed_acknowledgements | 133 |
| updates_without_completion | 2 |
| dropped_updates | 2 |
| source_errors | 0 |
| missing_arrival_receipts | 0 |
| unexpected_arrival_receipts | 0 |
| counts_consistent | sí |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad TTC del replay |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| successful_all_object_ttc | 1 | deadline_response_ms | 115 | 27.5726548 | 25.634 | 37.42381 | 81.608452 | 87.3916 | no |
| successful_all_object_ttc | 1 | raw_availability_response_ms | 115 | 23.9080565 | 22.4141 | 33.08609 | 68.918092 | 84.029 | no |
| successful_all_object_ttc | 1 | service_span_ms | 115 | 18.6707843 | 18.2009 | 24.35041 | 32.843468 | 36.8295 | no |
| failed_acknowledgements | 1 | deadline_response_ms | 133 | 6.27570075 | 2.5131 | 16.08682 | 44.100208 | 48.2454 | — |
| failed_acknowledgements | 1 | raw_availability_response_ms | 133 | 0.810809774 | 0.3325 | 1.21498 | 4.0335 | 37.3243 | — |
| failed_acknowledgements | 1 | service_span_ms | 133 | 0.0129300752 | 0.0121 | 0.0244 | 0.037592 | 0.0389 | — |
| ALL_RESPONSES_INCLUDING_FAILURES | 1 | deadline_response_ms | 248 | 16.1513044 | 15.5039 | 32.99453 | 62.940282 | 87.3916 | — |
| ALL_RESPONSES_INCLUDING_FAILURES | 1 | raw_availability_response_ms | 248 | 11.5212266 | 1.1391 | 27.38207 | 53.383056 | 84.029 | — |
| RAW_ARRIVALS_INCLUDING_DROPS | 1 | raw_availability_lateness_ms | 250 | 4.611494 | 2.08275 | 12.867545 | 17.940935 | 43.7277 | — |

| Gate declarado | Resultado |
| --- | --- |
| capacity_checks.all_scheduled_arrivals_recorded | sí |
| capacity_checks.no_drops_or_source_errors | no |
| capacity_checks.complete_object_coverage | no |
| capacity_checks.actual_availability_before_service | sí |
| capacity_checks.every_deadline_response_under_20ms | no |
| capacity_checks.no_prior_work_at_scheduled_arrivals | no |
| capacity_checks.finished_before_end_of_last_period | no |
| capacity_checks.offered_50hz_for_at_least_60s | no |
| capacity_checks.twenty_distinct_objects_per_update | no |
| capacity_checks.source_lifecycle_succeeded | sí |
| milestone_checks.all_scheduled_arrivals_recorded | sí |
| milestone_checks.no_drops_or_source_errors | no |
| milestone_checks.complete_object_coverage | no |
| milestone_checks.actual_availability_before_service | sí |
| milestone_checks.every_deadline_response_under_20ms | no |
| milestone_checks.no_prior_work_at_scheduled_arrivals | no |
| milestone_checks.finished_before_end_of_last_period | no |
| milestone_checks.offered_50hz_for_at_least_60s | no |
| milestone_checks.source_lifecycle_succeeded | sí |
| milestone_checks.required_object_count | sí |
| milestone_checks.all_three_host_ttc_per_required_object | no |
| milestone_checks.rgb_cache_stored_miss_then_real_hit | sí |

| Fuente RAW | Tamaño/mtime preservados |
| --- | --- |
| C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\EVOCON_JEPA_Codex_Handoff\e-jepa-ttc\datasets\evttc\CPLA\low\data.hdf5 | sí |

history_failure: `ValueError: RAW_GAP: expected index 115, received 116`

Los gates se muestran tal como quedaron registrados, sin recalcular admisión desde medias ni suponer una frecuencia ausente. COMPLETE no equivale a tiempo real. La distribución deadline_response_ms incluye adquisición y colas; raw_availability_response_ms comienza después de preparar el payload RAW. Las filas primarias successful_all_object_ttc se derivan de los recibos originales con cuantiles lineales; failed_acknowledgements son respuestas de error. Los campos legacy ALL_RESPONSES_INCLUDING_FAILURES mezclan ambas poblaciones y no representan una distribución de TTC válidos. No se modifica el gate registrado.

## Recibos sin adaptador de tabla

Se conservan en el manifiesto; no se interpreta su esquema.

- `frame_hash_cpu/idx553_v1/summary.json`: COMPLETE
- `frame_sealing_cpu/idx553_v1/summary.json`: COMPLETE
- `memory_raw_parity_v1/summary.json`: COMPLETE
