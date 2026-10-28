# Mediciones regeneradas — reanudación TRAIN40

Generado exclusivamente desde recibos JSON. Cada sección conserva su frontera de medición; los tiempos de secciones distintas no se suman ni se convierten en una afirmación de 50 Hz sostenidos. CPU, GPU preparada y cadena completa son ensayos diferentes. Los resultados fallidos permanecen visibles.

Fuentes y SHA-256: [RESULTS_SOURCES.json](RESULTS_SOURCES.json). `—` significa dato no declarado; no significa cero.

## Lectura rápida

| Frente | Dispositivo | Recibos | Máx. objetos/ROI ensayados | Estados |
| --- | --- | --- | --- | --- |
| baseline | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 2 | 1 | COMPLETE: 1; FAILED: 1 |
| baseline | — | 1 | — | FAILED_BEFORE_MODEL_LOAD: 1 |
| compiled_pipeline | — | 4 | 1 | COMPLETE_WITH_PARITY_FAILURES: 3; FAILED: 1 |
| encoders | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 1 | — | FAILED_ENCODER_PARITY: 1 |
| multiobject | — | 7 | 20 | COMPLETE: 5; FAILED: 2 |
| normalization | cpu | 1 | 1 | PASSED: 1 |
| normalization | cuda | 2 | 20 | PASSED: 2 |
| prefix | cpu | 4 | 20 | COMPLETE: 4 |
| prefix | cuda | 4 | 20 | COMPLETE: 4 |
| producers | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 11 | 20 | COMPLETE: 8; COMPLETE_WITH_PARITY_FAILURES: 3 |
| rgb_uint8 | cpu | 3 | — | COMPLETE: 3 |
| scheduler | — | 1 | — | COMPLETE_FAKE_CONTROL_ONLY: 1 |
| transport | NVIDIA GeForce RTX 5070 Ti Laptop GPU | 1 | — | PROBE_COMPLETE_NOT_FINAL_TTC_ADMISSION: 1 |

El máximo de objetos/ROI es cobertura del tramo indicado, no capacidad admitida de la cadena completa. Los encoders expresan batch de endpoints, no objetos. Cada tiempo y su paridad aparecen separados en el detalle.

## Inventario

| Recibo | Estado | Dispositivo | Preservación |
| --- | --- | --- | --- |
| baseline_all554_v1/setup_failure.json | FAILED_BEFORE_MODEL_LOAD | — | — |
| baseline_all554_v2/admission_analysis.json | FAILED | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| compiled_pipeline/gpu_idx0_v1/summary.json | FAILED | — | sí |
| compiled_pipeline/gpu_no_fma_exact_div_v4/summary.json | COMPLETE_WITH_PARITY_FAILURES | — | sí |
| compiled_pipeline/gpu_no_fma_v3/summary.json | COMPLETE_WITH_PARITY_FAILURES | — | sí |
| compiled_pipeline/gpu_sensitive_v2/summary.json | COMPLETE_WITH_PARITY_FAILURES | — | sí |
| compute_council/paced_scheduler_ab_v1/summary.json | COMPLETE_FAKE_CONTROL_ONLY | — | sí |
| corrected_all554_v1/summary.json | COMPLETE | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| encoder_ea5_b12_v1/summary.json | FAILED_ENCODER_PARITY | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| multiobject/pilot_1_2_dedicated_stream_v1/summary.json | COMPLETE | — | sí |
| multiobject/pilot_1_2_serial_v1/summary.json | COMPLETE | — | sí |
| multiobject/pilot_1_2_v1/summary.json | FAILED | — | sí |
| multiobject/pilot_20_chunk1_querygraphs_v1/summary.json | COMPLETE | — | sí |
| multiobject/pilot_20_dedicated_stream_v1/summary.json | FAILED | — | sí |
| multiobject/pilot_20_exact_cache_v1/summary.json | COMPLETE | — | sí |
| multiobject/pilot_2_only_parallel_v1/summary.json | COMPLETE | — | sí |
| normalization/cpu_smoke_v1/summary.json | PASSED | cpu | sí |
| normalization/gpu_eager_v1/summary.json | PASSED | cuda | sí |
| normalization/gpu_graph_v1/summary.json | PASSED | cuda | sí |
| prefix/compiled_cpu_smoke_v1/summary.json | COMPLETE | cpu | sí |
| prefix/compiled_cpu_smoke_v2/summary.json | COMPLETE | cpu | sí |
| prefix/compiled_gpu_v1/summary.json | COMPLETE | cuda | sí |
| prefix/cpu_smoke_v1/summary.json | COMPLETE | cpu | sí |
| prefix/cpu_topk_v1/summary.json | COMPLETE | cpu | sí |
| prefix/gpu_query_graph_v1/summary.json | COMPLETE | cuda | sí |
| prefix/gpu_sort_v1/summary.json | COMPLETE | cuda | sí |
| prefix/gpu_topk_v1/summary.json | COMPLETE | cuda | sí |
| producers/index0_grouped_baseline_v1/summary.json | COMPLETE | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| producers/index137_event_no_reuse_v1/summary.json | COMPLETE | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| producers/index137_no_reuse_v1/summary.json | COMPLETE | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| producers/index137_reuse_v1/summary.json | COMPLETE_WITH_PARITY_FAILURES | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| producers/index488_event_no_reuse_v1/summary.json | COMPLETE | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| producers/index488_no_reuse_v1/summary.json | COMPLETE_WITH_PARITY_FAILURES | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| producers/index505_event_no_reuse_v1/summary.json | COMPLETE | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| producers/index514_event_no_reuse_v1/summary.json | COMPLETE | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| producers/index533_event_no_reuse_v1/summary.json | COMPLETE | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| producers/index553_chunk1_channels_last_v1/summary.json | COMPLETE_WITH_PARITY_FAILURES | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| producers/index553_chunk1_contiguous_v1/summary.json | COMPLETE | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |
| rgb_uint8/cpu_1200x1920_roi_first_v1/summary.json | COMPLETE | cpu | sí |
| rgb_uint8/cpu_matrix_v1/summary.json | COMPLETE | cpu | sí |
| rgb_uint8/cpu_roi_first_v1/summary.json | COMPLETE | cpu | sí |
| transport/block_gpu_v1/summary.json | PROBE_COMPLETE_NOT_FINAL_TTC_ADMISSION | NVIDIA GeForce RTX 5070 Ti Laptop GPU | sí |

## baseline

Una ROI por consulta original; historial reconstruido antes del reloj. Conserva el gate conjunto de las tres salidas.

### `baseline_all554_v1/setup_failure.json`

Estado: **FAILED_BEFORE_MODEL_LOAD**.

Frozen oracle source location/bytes differ: absolute source paths are outside the detached worktree

### `baseline_all554_v2/admission_analysis.json`

Estado: **FAILED**.

| Cobertura | Inputs pasan | Inputs exactos | Archivos preservados | Intactos |
| --- | --- | --- | --- | --- |
| 554/554 | 554 | 553 | 603 | sí |

| Salida | Pasan | Pares finitos | Máx. error absoluto s | Máx. relativo |
| --- | --- | --- | --- | --- |
| PHASE_ONLY_TRAIN40_S7 | 554 | 554 | 0.000188827515 | 7.33587318e-06 |
| E_CTX_TRAIN40 | 549 | 554 | 0.000637054443 | 3.84658224e-05 |
| R_CTX_TRAIN40 | 554 | 554 | 0.000312805176 | 8.60476559e-06 |

| Índice rechazado | Salida | Error absoluto s | Límite s |
| --- | --- | --- | --- |
| 137 | E_CTX_TRAIN40 | 0.000144958496 | 0.000103988218 |
| 488 | E_CTX_TRAIN40 | 7.72476196e-05 | 7.13922501e-05 |
| 505 | E_CTX_TRAIN40 | 0.000502586365 | 0.000140657902 |
| 514 | E_CTX_TRAIN40 | 0.00030708313 | 0.000140493994 |
| 533 | E_CTX_TRAIN40 | 0.000637054443 | 0.000353280449 |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cohorte completa | 1 | RAW disponible → TTC | 554 | 44.2483944 | 40.5906 | 63.77779 | 98.415266 | 294.5156 | no |

Other swarm agents performed CPU development; desktop GPU shared. Single-object registered replay with per-query history initialization outside clock. Not sustained and not isolated before/after performance evidence.

### `corrected_all554_v1/summary.json`

Estado: **COMPLETE**.

| Cobertura | Inputs pasan | Inputs exactos | Archivos preservados | Intactos |
| --- | --- | --- | --- | --- |
| 554/554 | 554 | — | 606 | sí |

| Salida | Pasan | Pares finitos | Máx. error absoluto s | Máx. relativo |
| --- | --- | --- | --- | --- |
| PHASE_ONLY_TRAIN40_S7 | 554 | 554 | 6.67572021e-05 | 4.93935321e-06 |
| E_CTX_TRAIN40 | 554 | 554 | 0 | 0 |
| R_CTX_TRAIN40 | 554 | 554 | 0.000312805176 | 8.60476559e-06 |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cohorte completa | 1 | RAW disponible → TTC | 554 | 46.1917847 | 44.1926 | 66.634525 | 76.600801 | 103.7058 | sí |

Inicialización de historial y disco excluidos.

## producers

Productores preparados en GPU; excluye sensores, H2D, contexto y TTC final.

### `producers/index0_grouped_baseline_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`. Alcance: `PREPARED_GPU_MULTIOBJECT_PRODUCERS_PAIR_PHASE17_GUARDS_NO_SENSOR_H2D_CONTEXT`.

| Configuración | Valor |
| --- | --- |
| query_index | 0 |
| encoder_layout | contiguous |
| transport_variant | original |
| endpoint_reuse | sí |
| cudnn_benchmark | no |
| cudnn_deterministic | no |
| precision | FP32_TF32_OFF |
| input_population | Artificial capacity objects: real histories scaled differently per ID |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | serial | 6 | 15.6035 | 15.82635 | 16.07615 | 16.07723 | 16.0775 | sí |
| 0 | 1 | parallel | 6 | 11.1035 | 11.1611 | 11.3393 | 11.3501 | 11.3528 | sí |
| 1 | 2 | serial | 6 | 22.9450833 | 23.11225 | 23.559075 | 23.595335 | 23.6044 | sí |
| 1 | 2 | parallel | 6 | 18.4729667 | 18.41195 | 19.458975 | 19.640275 | 19.6856 | sí |
| 2 | 20 | serial | 6 | 223.582967 | 223.70095 | 225.3626 | 225.66212 | 225.737 | sí |
| 2 | 20 | parallel | 6 | 206.112217 | 206.06735 | 207.521775 | 207.764835 | 207.8256 | sí |

La columna de paridad corresponde a los TTC comprobados fuera del reloj de productores, frente a la referencia por objeto del probe.

### `producers/index137_event_no_reuse_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`. Alcance: `PREPARED_GPU_MULTIOBJECT_PRODUCERS_PAIR_PHASE17_GUARDS_NO_SENSOR_H2D_CONTEXT`.

| Configuración | Valor |
| --- | --- |
| query_index | 137 |
| encoder_layout | contiguous |
| transport_variant | original |
| endpoint_reuse | sí |
| event_endpoint_reuse | no |
| cudnn_benchmark | no |
| cudnn_deterministic | no |
| precision | FP32_TF32_OFF |
| input_population | Artificial capacity objects: real histories scaled differently per ID |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | serial | 2 | 20.5689 | 20.5689 | 20.88597 | 20.914154 | 20.9212 | sí |
| 0 | 1 | parallel | 2 | 13.8283 | 13.8283 | 14.03638 | 14.054876 | 14.0595 | sí |

La columna de paridad corresponde a los TTC comprobados fuera del reloj de productores, frente a la referencia por objeto del probe.

### `producers/index137_no_reuse_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`. Alcance: `PREPARED_GPU_MULTIOBJECT_PRODUCERS_PAIR_PHASE17_GUARDS_NO_SENSOR_H2D_CONTEXT`.

| Configuración | Valor |
| --- | --- |
| query_index | 137 |
| encoder_layout | contiguous |
| transport_variant | original |
| endpoint_reuse | no |
| cudnn_benchmark | no |
| cudnn_deterministic | no |
| precision | FP32_TF32_OFF |
| input_population | Artificial capacity objects: real histories scaled differently per ID |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | serial | 4 | 21.44485 | 21.42945 | 22.21469 | 22.283018 | 22.3001 | sí |
| 0 | 1 | parallel | 4 | 16.144375 | 16.13585 | 16.518455 | 16.535771 | 16.5401 | sí |

La columna de paridad corresponde a los TTC comprobados fuera del reloj de productores, frente a la referencia por objeto del probe.

### `producers/index137_reuse_v1/summary.json`

Estado: **COMPLETE_WITH_PARITY_FAILURES**. Dispositivo: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`. Alcance: `PREPARED_GPU_MULTIOBJECT_PRODUCERS_PAIR_PHASE17_GUARDS_NO_SENSOR_H2D_CONTEXT`.

| Configuración | Valor |
| --- | --- |
| query_index | 137 |
| encoder_layout | contiguous |
| transport_variant | original |
| endpoint_reuse | sí |
| cudnn_benchmark | no |
| cudnn_deterministic | no |
| precision | FP32_TF32_OFF |
| input_population | Artificial capacity objects: real histories scaled differently per ID |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | serial | 4 | 16.336425 | 16.2995 | 16.96795 | 17.05687 | 17.0791 | no |
| 0 | 1 | parallel | 4 | 12.20575 | 12.25625 | 12.61935 | 12.64251 | 12.6483 | no |

La columna de paridad corresponde a los TTC comprobados fuera del reloj de productores, frente a la referencia por objeto del probe.

### `producers/index488_event_no_reuse_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`. Alcance: `PREPARED_GPU_MULTIOBJECT_PRODUCERS_PAIR_PHASE17_GUARDS_NO_SENSOR_H2D_CONTEXT`.

| Configuración | Valor |
| --- | --- |
| query_index | 488 |
| encoder_layout | contiguous |
| transport_variant | original |
| endpoint_reuse | sí |
| event_endpoint_reuse | no |
| cudnn_benchmark | no |
| cudnn_deterministic | no |
| precision | FP32_TF32_OFF |
| input_population | Artificial capacity objects: real histories scaled differently per ID |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | serial | 2 | 18.12105 | 18.12105 | 18.821835 | 18.884127 | 18.8997 | sí |
| 0 | 1 | parallel | 2 | 13.9405 | 13.9405 | 14.09575 | 14.10955 | 14.113 | sí |

La columna de paridad corresponde a los TTC comprobados fuera del reloj de productores, frente a la referencia por objeto del probe.

### `producers/index488_no_reuse_v1/summary.json`

Estado: **COMPLETE_WITH_PARITY_FAILURES**. Dispositivo: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`. Alcance: `PREPARED_GPU_MULTIOBJECT_PRODUCERS_PAIR_PHASE17_GUARDS_NO_SENSOR_H2D_CONTEXT`.

| Configuración | Valor |
| --- | --- |
| query_index | 488 |
| encoder_layout | contiguous |
| transport_variant | original |
| endpoint_reuse | no |
| cudnn_benchmark | no |
| cudnn_deterministic | no |
| precision | FP32_TF32_OFF |
| input_population | Artificial capacity objects: real histories scaled differently per ID |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | serial | 4 | 20.558225 | 20.3397 | 21.34143 | 21.446766 | 21.4731 | no |
| 0 | 1 | parallel | 4 | 16.7725 | 16.57185 | 17.50493 | 17.619146 | 17.6477 | no |

La columna de paridad corresponde a los TTC comprobados fuera del reloj de productores, frente a la referencia por objeto del probe.

### `producers/index505_event_no_reuse_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`. Alcance: `PREPARED_GPU_MULTIOBJECT_PRODUCERS_PAIR_PHASE17_GUARDS_NO_SENSOR_H2D_CONTEXT`.

| Configuración | Valor |
| --- | --- |
| query_index | 505 |
| encoder_layout | contiguous |
| transport_variant | original |
| endpoint_reuse | sí |
| event_endpoint_reuse | no |
| cudnn_benchmark | no |
| cudnn_deterministic | no |
| precision | FP32_TF32_OFF |
| input_population | Artificial capacity objects: real histories scaled differently per ID |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | serial | 2 | 18.3889 | 18.3889 | 18.60265 | 18.62165 | 18.6264 | sí |
| 0 | 1 | parallel | 2 | 12.9569 | 12.9569 | 13.21106 | 13.233652 | 13.2393 | sí |

La columna de paridad corresponde a los TTC comprobados fuera del reloj de productores, frente a la referencia por objeto del probe.

### `producers/index514_event_no_reuse_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`. Alcance: `PREPARED_GPU_MULTIOBJECT_PRODUCERS_PAIR_PHASE17_GUARDS_NO_SENSOR_H2D_CONTEXT`.

| Configuración | Valor |
| --- | --- |
| query_index | 514 |
| encoder_layout | contiguous |
| transport_variant | original |
| endpoint_reuse | sí |
| event_endpoint_reuse | no |
| cudnn_benchmark | no |
| cudnn_deterministic | no |
| precision | FP32_TF32_OFF |
| input_population | Artificial capacity objects: real histories scaled differently per ID |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | serial | 2 | 18.38615 | 18.38615 | 18.833495 | 18.873259 | 18.8832 | sí |
| 0 | 1 | parallel | 2 | 12.7496 | 12.7496 | 13.07963 | 13.108966 | 13.1163 | sí |

La columna de paridad corresponde a los TTC comprobados fuera del reloj de productores, frente a la referencia por objeto del probe.

### `producers/index533_event_no_reuse_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`. Alcance: `PREPARED_GPU_MULTIOBJECT_PRODUCERS_PAIR_PHASE17_GUARDS_NO_SENSOR_H2D_CONTEXT`.

| Configuración | Valor |
| --- | --- |
| query_index | 533 |
| encoder_layout | contiguous |
| transport_variant | original |
| endpoint_reuse | sí |
| event_endpoint_reuse | no |
| cudnn_benchmark | no |
| cudnn_deterministic | no |
| precision | FP32_TF32_OFF |
| input_population | Artificial capacity objects: real histories scaled differently per ID |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | serial | 2 | 17.81585 | 17.81585 | 17.859905 | 17.863821 | 17.8648 | sí |
| 0 | 1 | parallel | 2 | 12.2451 | 12.2451 | 12.36633 | 12.377106 | 12.3798 | sí |

La columna de paridad corresponde a los TTC comprobados fuera del reloj de productores, frente a la referencia por objeto del probe.

### `producers/index553_chunk1_channels_last_v1/summary.json`

Estado: **COMPLETE_WITH_PARITY_FAILURES**. Dispositivo: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`. Alcance: `PREPARED_GPU_MULTIOBJECT_PRODUCERS_PAIR_PHASE17_GUARDS_NO_SENSOR_H2D_CONTEXT`.

| Configuración | Valor |
| --- | --- |
| query_index | 553 |
| encoder_layout | channels-last |
| transport_variant | original |
| endpoint_reuse | sí |
| event_endpoint_reuse | no |
| cudnn_benchmark | no |
| cudnn_deterministic | no |
| precision | FP32_TF32_OFF |
| input_population | Artificial capacity objects: real histories scaled differently per ID |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 20 | serial | 4 | 548.988125 | 500.72195 | 694.929405 | 721.266201 | 727.8504 | no |
| 0 | 20 | parallel | 4 | 407.08585 | 405.97345 | 445.450465 | 448.765213 | 449.5939 | no |

La columna de paridad corresponde a los TTC comprobados fuera del reloj de productores, frente a la referencia por objeto del probe.

### `producers/index553_chunk1_contiguous_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`. Alcance: `PREPARED_GPU_MULTIOBJECT_PRODUCERS_PAIR_PHASE17_GUARDS_NO_SENSOR_H2D_CONTEXT`.

| Configuración | Valor |
| --- | --- |
| query_index | 553 |
| encoder_layout | contiguous |
| transport_variant | original |
| endpoint_reuse | sí |
| event_endpoint_reuse | no |
| cudnn_benchmark | no |
| cudnn_deterministic | no |
| precision | FP32_TF32_OFF |
| input_population | Artificial capacity objects: real histories scaled differently per ID |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 20 | serial | 4 | 366.97455 | 370.7394 | 387.923 | 388.13324 | 388.1858 | sí |
| 0 | 20 | parallel | 4 | 271.85 | 272.86285 | 287.468805 | 288.385281 | 288.6144 | sí |

La columna de paridad corresponde a los TTC comprobados fuera del reloj de productores, frente a la referencia por objeto del probe.

## prefix

Paquete RAW a representaciones de eventos; excluye modelos y TTC.

### `prefix/compiled_cpu_smoke_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `cpu`. Alcance: `RAW_CPU_PACKET_TO_ALL_EVENT_TENSORS_NO_MODELS_OR_TTC`.

| Configuración | Valor |
| --- | --- |
| normalization | topk |
| data.synthetic | sí |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TemporalPrefixEventHistory | 1 | total | 2 | 1.24515 | 1.24515 | 1.282185 | 1.285477 | 1.2863 | sí |
| TemporalPrefixEventHistory | 20 | total | 2 | 6.92995 | 6.92995 | 6.936205 | 6.936761 | 6.9369 | sí |
| CompiledTemporalPrefixEventHistory | 1 | total | 2 | 2.347 | 2.347 | 2.50243 | 2.516246 | 2.5197 | sí |
| CompiledTemporalPrefixEventHistory | 20 | total | 2 | 15.5869 | 15.5869 | 15.85177 | 15.875314 | 15.8812 | sí |

La paridad canónica indicada corresponde a la ROI primaria; no certifica TTC ni veinte tracks reales.

| Backend | ROI | Exacto frente a buckets | Máx. diferencia | Bytes |
| --- | --- | --- | --- | --- |
| TemporalPrefixEventHistory | 1 | — | — | 2897072 |
| TemporalPrefixEventHistory | 20 | — | — | 4564640 |
| CompiledTemporalPrefixEventHistory | 1 | — | — | 2767024 |
| CompiledTemporalPrefixEventHistory | 20 | — | — | 4044448 |

### `prefix/compiled_cpu_smoke_v2/summary.json`

Estado: **COMPLETE**. Dispositivo: `cpu`. Alcance: `RAW_CPU_PACKET_TO_ALL_EVENT_TENSORS_NO_MODELS_OR_TTC`.

| Configuración | Valor |
| --- | --- |
| normalization | topk |
| data.synthetic | sí |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TemporalPrefixEventHistory | 1 | total | 2 | 1.22465 | 1.22465 | 1.271765 | 1.275953 | 1.277 | sí |
| TemporalPrefixEventHistory | 20 | total | 2 | 6.8828 | 6.8828 | 6.95687 | 6.963454 | 6.9651 | sí |
| CompiledTemporalPrefixEventHistory | 1 | total | 2 | 2.3299 | 2.3299 | 2.43565 | 2.44505 | 2.4474 | sí |
| CompiledTemporalPrefixEventHistory | 20 | total | 2 | 11.6054 | 11.6054 | 12.25304 | 12.310608 | 12.325 | sí |

La paridad canónica indicada corresponde a la ROI primaria; no certifica TTC ni veinte tracks reales.

| Backend | ROI | Exacto frente a buckets | Máx. diferencia | Bytes |
| --- | --- | --- | --- | --- |
| TemporalPrefixEventHistory | 1 | — | — | 2897072 |
| TemporalPrefixEventHistory | 20 | — | — | 4564640 |
| CompiledTemporalPrefixEventHistory | 1 | — | — | 2767024 |
| CompiledTemporalPrefixEventHistory | 20 | — | — | 4044448 |

### `prefix/compiled_gpu_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `cuda`. Alcance: `RAW_CPU_PACKET_TO_ALL_EVENT_TENSORS_NO_MODELS_OR_TTC`.

| Configuración | Valor |
| --- | --- |
| normalization | topk |
| data.synthetic | no |
| data.index | 553 |
| data.query_id | EXTENDED:CPNAO-medium:20200738 |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TemporalPrefixEventHistory | 1 | total | 5 | 6.33748 | 6.3288 | 6.54916 | 6.571352 | 6.5769 | sí |
| TemporalPrefixEventHistory | 20 | total | 5 | 32.68578 | 32.6606 | 33.93308 | 34.153816 | 34.209 | sí |
| CompiledTemporalPrefixEventHistory | 1 | total | 5 | 6.84878 | 6.8749 | 7.4008 | 7.50192 | 7.5272 | sí |
| CompiledTemporalPrefixEventHistory | 20 | total | 5 | 24.65532 | 24.068 | 25.81014 | 25.893548 | 25.9144 | sí |

La paridad canónica indicada corresponde a la ROI primaria; no certifica TTC ni veinte tracks reales.

| Backend | ROI | Exacto frente a buckets | Máx. diferencia | Bytes |
| --- | --- | --- | --- | --- |
| TemporalPrefixEventHistory | 1 | — | — | 2976513264 |
| TemporalPrefixEventHistory | 20 | — | — | 3402602592 |
| CompiledTemporalPrefixEventHistory | 1 | — | — | 2943220976 |
| CompiledTemporalPrefixEventHistory | 20 | — | — | 3269433440 |

### `prefix/cpu_smoke_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `cpu`. Alcance: `RAW_CPU_PACKET_TO_EVENT_REPRESENTATIONS_NO_MODELS_NO_TTC`.

| Configuración | Valor |
| --- | --- |
| roi_size | 8 |
| data.synthetic | sí |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IntegralEventHistory | 1 | total | 3 | 2.30526667 | 2.188 | 2.56492 | 2.598424 | 2.6068 | sí |
| IntegralEventHistory | 20 | total | 3 | 27.7304 | 27.7497 | 30.60396 | 30.857672 | 30.9211 | sí |
| TemporalPrefixEventHistory | 1 | total | 3 | 2.27633333 | 2.2275 | 2.42892 | 2.446824 | 2.4513 | sí |
| TemporalPrefixEventHistory | 20 | total | 3 | 18.8899333 | 19.0319 | 19.58756 | 19.636952 | 19.6493 | sí |

La paridad canónica indicada corresponde a la ROI primaria; no certifica TTC ni veinte tracks reales.

| Backend | ROI | Exacto frente a buckets | Máx. diferencia | Bytes |
| --- | --- | --- | --- | --- |
| IntegralEventHistory | 1 | — | — | 3441800 |
| IntegralEventHistory | 20 | — | — | 6918824 |
| TemporalPrefixEventHistory | 1 | sí | 0 | 2955192 |
| TemporalPrefixEventHistory | 20 | sí | 0 | 4794840 |

### `prefix/cpu_topk_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `cpu`. Alcance: `RAW_CPU_PACKET_TO_EVENT_REPRESENTATIONS_NO_MODELS_NO_TTC`.

| Configuración | Valor |
| --- | --- |
| normalization | topk |
| roi_size | 8 |
| data.synthetic | sí |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IntegralEventHistory | 1 | total | 3 | 3.93956667 | 4.0241 | 4.06244 | 4.065848 | 4.0667 | sí |
| IntegralEventHistory | 20 | total | 3 | 33.1616667 | 31.7478 | 36.88455 | 37.34115 | 37.4553 | sí |
| TemporalPrefixEventHistory | 1 | total | 3 | 3.14356667 | 3.1146 | 3.47028 | 3.501896 | 3.5098 | sí |
| TemporalPrefixEventHistory | 20 | total | 3 | 17.5182333 | 17.491 | 18.53914 | 18.632308 | 18.6556 | sí |

La paridad canónica indicada corresponde a la ROI primaria; no certifica TTC ni veinte tracks reales.

| Backend | ROI | Exacto frente a buckets | Máx. diferencia | Bytes |
| --- | --- | --- | --- | --- |
| IntegralEventHistory | 1 | — | — | 3441800 |
| IntegralEventHistory | 20 | — | — | 6918824 |
| TemporalPrefixEventHistory | 1 | sí | 0 | 2896920 |
| TemporalPrefixEventHistory | 20 | sí | 0 | 4561752 |

### `prefix/gpu_query_graph_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `cuda`. Alcance: `RAW_CPU_PACKET_TO_EVENT_REPRESENTATIONS_NO_MODELS_NO_TTC`.

| Configuración | Valor |
| --- | --- |
| normalization | topk |
| roi_size | 128 |
| data.synthetic | no |
| data.index | 553 |
| data.query_id | EXTENDED:CPNAO-medium:20200738 |
| data.additional_rois | SYNTHETIC_CAPACITY_REGIONS_NOT_ANNOTATED_TRACKS |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IntegralEventHistory | 1 | total | 5 | 7.84078 | 7.9564 | 8.18312 | 8.225024 | 8.2355 | sí |
| IntegralEventHistory | 20 | total | 5 | 73.7359 | 73.9334 | 74.04178 | 74.049636 | 74.0516 | sí |
| TemporalPrefixEventHistory | 1 | total | 5 | 6.46498 | 6.3943 | 6.8028 | 6.84368 | 6.8539 | sí |
| TemporalPrefixEventHistory | 20 | total | 5 | 30.62268 | 30.5775 | 30.8972 | 30.94776 | 30.9604 | sí |

La paridad canónica indicada corresponde a la ROI primaria; no certifica TTC ni veinte tracks reales.

| Backend | ROI | Exacto frente a buckets | Máx. diferencia | Bytes |
| --- | --- | --- | --- | --- |
| IntegralEventHistory | 1 | — | — | 3064677000 |
| IntegralEventHistory | 20 | — | — | 3954721704 |
| TemporalPrefixEventHistory | 1 | sí | 0 | 2976513264 |
| TemporalPrefixEventHistory | 20 | sí | 0 | 3402602592 |

### `prefix/gpu_sort_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `cuda`. Alcance: `RAW_CPU_PACKET_TO_EVENT_REPRESENTATIONS_NO_MODELS_NO_TTC`.

| Configuración | Valor |
| --- | --- |
| normalization | sort |
| roi_size | 128 |
| data.synthetic | no |
| data.index | 553 |
| data.query_id | EXTENDED:CPNAO-medium:20200738 |
| data.additional_rois | SYNTHETIC_CAPACITY_REGIONS_NOT_ANNOTATED_TRACKS |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IntegralEventHistory | 1 | total | 5 | 10.65252 | 10.7783 | 11.06154 | 11.107508 | 11.119 | sí |
| IntegralEventHistory | 20 | total | 5 | 79.41804 | 78.9715 | 80.93416 | 81.097552 | 81.1384 | sí |
| TemporalPrefixEventHistory | 1 | total | 5 | 9.58308 | 9.4137 | 9.9091 | 9.92054 | 9.9234 | sí |
| TemporalPrefixEventHistory | 20 | total | 5 | 59.3747 | 56.2414 | 65.30642 | 65.603204 | 65.6774 | sí |

La paridad canónica indicada corresponde a la ROI primaria; no certifica TTC ni veinte tracks reales.

| Backend | ROI | Exacto frente a buckets | Máx. diferencia | Bytes |
| --- | --- | --- | --- | --- |
| IntegralEventHistory | 1 | — | — | 3064677000 |
| IntegralEventHistory | 20 | — | — | 3954721704 |
| TemporalPrefixEventHistory | 1 | sí | 0 | 2991453304 |
| TemporalPrefixEventHistory | 20 | sí | 0 | 3462329752 |

### `prefix/gpu_topk_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `cuda`. Alcance: `RAW_CPU_PACKET_TO_EVENT_REPRESENTATIONS_NO_MODELS_NO_TTC`.

| Configuración | Valor |
| --- | --- |
| normalization | topk |
| roi_size | 128 |
| data.synthetic | no |
| data.index | 553 |
| data.query_id | EXTENDED:CPNAO-medium:20200738 |
| data.additional_rois | SYNTHETIC_CAPACITY_REGIONS_NOT_ANNOTATED_TRACKS |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IntegralEventHistory | 1 | total | 5 | 9.16228 | 9.3379 | 9.73514 | 9.814388 | 9.8342 | sí |
| IntegralEventHistory | 20 | total | 5 | 81.47858 | 80.9674 | 84.2264 | 84.75256 | 84.8841 | sí |
| TemporalPrefixEventHistory | 1 | total | 5 | 9.83092 | 9.7826 | 10.36524 | 10.430728 | 10.4471 | sí |
| TemporalPrefixEventHistory | 20 | total | 5 | 53.91336 | 53.672 | 59.9284 | 60.49872 | 60.6413 | sí |

La paridad canónica indicada corresponde a la ROI primaria; no certifica TTC ni veinte tracks reales.

| Backend | ROI | Exacto frente a buckets | Máx. diferencia | Bytes |
| --- | --- | --- | --- | --- |
| IntegralEventHistory | 1 | — | — | 3064677000 |
| IntegralEventHistory | 20 | — | — | 3954721704 |
| TemporalPrefixEventHistory | 1 | sí | 0 | 2976511192 |
| TemporalPrefixEventHistory | 20 | sí | 0 | 3402561304 |

## normalization

Normalización de vóxeles preparados; excluye sensores y modelos.

### `normalization/cpu_smoke_v1/summary.json`

Estado: **PASSED**. Dispositivo: `cpu`. Alcance: `SYNTHETIC_PREPARED_VOXEL_NORMALIZATION_WITH_RESET_COPY_NO_SENSOR_NO_MODELS`.

| Configuración | Valor |
| --- | --- |
| cuda_graphs | no |
| precision | FLOAT64_NORMALIZATION_FLOAT32_OUTPUT_TF32_DISABLED |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0; ventanas=6; columnas=163840; ocupación=0.5 | 1 | sort | 3 | 72.0944667 | 72.3261 | 72.93027 | 72.983974 | 72.9974 | sí |
| 0; ventanas=6; columnas=163840; ocupación=0.5 | 1 | topk | 3 | 22.187 | 21.6377 | 23.76134 | 23.950108 | 23.9973 | sí |

### `normalization/gpu_eager_v1/summary.json`

Estado: **PASSED**. Dispositivo: `cuda`. Alcance: `SYNTHETIC_PREPARED_VOXEL_NORMALIZATION_WITH_RESET_COPY_NO_SENSOR_NO_MODELS`.

| Configuración | Valor |
| --- | --- |
| cuda_graphs | no |
| precision | FLOAT64_NORMALIZATION_FLOAT32_OUTPUT_TF32_DISABLED |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0; ventanas=6; columnas=163840; ocupación=0 | 1 | sort | 30 | 1.09667333 | 1.11575 | 1.322765 | 1.377473 | 1.388 | sí |
| 0; ventanas=6; columnas=163840; ocupación=0 | 1 | topk | 30 | 1.52827333 | 1.49475 | 1.818255 | 1.871015 | 1.8874 | sí |
| 1; ventanas=6; columnas=163840; ocupación=0.01 | 1 | sort | 30 | 1.01677 | 1.0295 | 1.19809 | 1.280479 | 1.3138 | sí |
| 1; ventanas=6; columnas=163840; ocupación=0.01 | 1 | topk | 30 | 1.55429333 | 1.57465 | 1.931695 | 2.043205 | 2.068 | sí |
| 2; ventanas=6; columnas=163840; ocupación=0.5 | 1 | sort | 30 | 0.865536667 | 0.81055 | 1.164295 | 1.386813 | 1.4768 | sí |
| 2; ventanas=6; columnas=163840; ocupación=0.5 | 1 | topk | 30 | 1.29519667 | 1.23005 | 1.68309 | 1.779801 | 1.8039 | sí |
| 3; ventanas=6; columnas=163840; ocupación=1 | 1 | sort | 30 | 0.946053333 | 0.94655 | 1.14414 | 1.305663 | 1.3623 | sí |
| 3; ventanas=6; columnas=163840; ocupación=1 | 1 | topk | 30 | 1.37344333 | 1.3729 | 1.655595 | 1.918821 | 2.0247 | sí |
| 4; ventanas=120; columnas=163840; ocupación=0 | 20 | sort | 30 | 17.5257533 | 17.60635 | 18.123185 | 18.185253 | 18.2068 | sí |
| 4; ventanas=120; columnas=163840; ocupación=0 | 20 | topk | 30 | 7.80671 | 7.78915 | 8.658955 | 8.807619 | 8.8334 | sí |
| 5; ventanas=120; columnas=163840; ocupación=0.01 | 20 | sort | 30 | 17.4460933 | 17.32825 | 18.076365 | 18.774699 | 19.029 | sí |
| 5; ventanas=120; columnas=163840; ocupación=0.01 | 20 | topk | 30 | 7.93218667 | 7.88455 | 8.81122 | 8.913226 | 8.9482 | sí |
| 6; ventanas=120; columnas=163840; ocupación=0.5 | 20 | sort | 30 | 17.6656133 | 17.6562 | 18.82473 | 18.954245 | 18.9802 | sí |
| 6; ventanas=120; columnas=163840; ocupación=0.5 | 20 | topk | 30 | 7.948 | 7.8441 | 8.79366 | 9.158495 | 9.2062 | sí |
| 7; ventanas=120; columnas=163840; ocupación=1 | 20 | sort | 30 | 16.5154433 | 16.5124 | 17.18326 | 17.77561 | 18.0166 | sí |
| 7; ventanas=120; columnas=163840; ocupación=1 | 20 | topk | 30 | 6.87999 | 6.8505 | 7.555605 | 7.679695 | 7.7129 | sí |

### `normalization/gpu_graph_v1/summary.json`

Estado: **PASSED**. Dispositivo: `cuda`. Alcance: `SYNTHETIC_PREPARED_VOXEL_NORMALIZATION_WITH_RESET_COPY_NO_SENSOR_NO_MODELS`.

| Configuración | Valor |
| --- | --- |
| cuda_graphs | sí |
| precision | FLOAT64_NORMALIZATION_FLOAT32_OUTPUT_TF32_DISABLED |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0; ventanas=6; columnas=163840; ocupación=0 | 1 | sort | 30 | 0.48805 | 0.4485 | 0.61646 | 0.952263 | 1.0756 | sí |
| 0; ventanas=6; columnas=163840; ocupación=0 | 1 | topk | 30 | 0.39444 | 0.36605 | 0.504605 | 0.843168 | 0.9739 | sí |
| 1; ventanas=6; columnas=163840; ocupación=0.01 | 1 | sort | 30 | 0.51316 | 0.46685 | 0.724495 | 0.968216 | 1.0551 | sí |
| 1; ventanas=6; columnas=163840; ocupación=0.01 | 1 | topk | 30 | 0.44612 | 0.38925 | 0.64736 | 0.889166 | 0.9833 | sí |
| 2; ventanas=6; columnas=163840; ocupación=0.5 | 1 | sort | 30 | 0.54408 | 0.50065 | 0.70238 | 0.910399 | 0.9936 | sí |
| 2; ventanas=6; columnas=163840; ocupación=0.5 | 1 | topk | 30 | 0.443903333 | 0.3961 | 0.66731 | 0.768872 | 0.7901 | sí |
| 3; ventanas=6; columnas=163840; ocupación=1 | 1 | sort | 30 | 0.562693333 | 0.50885 | 0.765935 | 1.187453 | 1.3424 | sí |
| 3; ventanas=6; columnas=163840; ocupación=1 | 1 | topk | 30 | 0.362526667 | 0.3468 | 0.480995 | 0.527888 | 0.5432 | sí |
| 4; ventanas=120; columnas=163840; ocupación=0 | 20 | sort | 30 | 17.2005233 | 17.1499 | 18.07351 | 18.320896 | 18.355 | sí |
| 4; ventanas=120; columnas=163840; ocupación=0 | 20 | topk | 30 | 7.43096333 | 7.30525 | 8.512555 | 9.01632 | 9.0401 | sí |
| 5; ventanas=120; columnas=163840; ocupación=0.01 | 20 | sort | 30 | 17.1193067 | 16.96 | 18.172735 | 18.520216 | 18.61 | sí |
| 5; ventanas=120; columnas=163840; ocupación=0.01 | 20 | topk | 30 | 7.36245 | 7.3083 | 7.95319 | 8.147794 | 8.2207 | sí |
| 6; ventanas=120; columnas=163840; ocupación=0.5 | 20 | sort | 30 | 18.12239 | 17.5002 | 23.34664 | 25.543168 | 26.2597 | sí |
| 6; ventanas=120; columnas=163840; ocupación=0.5 | 20 | topk | 30 | 7.81933333 | 7.50725 | 10.516535 | 10.683339 | 10.6894 | sí |
| 7; ventanas=120; columnas=163840; ocupación=1 | 20 | sort | 30 | 22.7094333 | 21.5005 | 25.564755 | 29.652398 | 31.2253 | sí |
| 7; ventanas=120; columnas=163840; ocupación=1 | 20 | topk | 30 | 8.32513 | 7.91975 | 9.59528 | 9.638819 | 9.653 | sí |

## multiobject

Piloto completo con ROI de capacidad artificiales; no es un stream sostenido.

### `multiobject/pilot_1_2_dedicated_stream_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `—`. Alcance: `ARTIFICIAL_DISTINCT_ROIS_SHARED_REAL_RAW_LAST_PACKET_COMPLETE_UPDATE`.

| Configuración | Valor |
| --- | --- |
| precision | FP32_TF32_OFF |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| piloto completo | 1 | RAW disponible → último TTC | 10 | 29.74366 | 24.82975 | 52.864545 | 70.626909 | 75.0675 | sí |
| piloto completo | 2 | RAW disponible → último TTC | 10 | 33.38843 | 32.99125 | 36.59086 | 38.391292 | 38.8414 | sí |

### `multiobject/pilot_1_2_serial_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `—`. Alcance: `ARTIFICIAL_DISTINCT_ROIS_SHARED_REAL_RAW_LAST_PACKET_COMPLETE_UPDATE`.

| Configuración | Valor |
| --- | --- |
| precision | FP32_TF32_OFF |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| piloto completo | 1 | RAW disponible → último TTC | 3 | 46.5848 | 29.1091 | 76.60417 | 80.825954 | 81.8814 | sí |
| piloto completo | 2 | RAW disponible → último TTC | 3 | 42.0073667 | 40.6862 | 45.30059 | 45.710758 | 45.8133 | sí |

### `multiobject/pilot_1_2_v1/summary.json`

Estado: **FAILED**. Dispositivo: `—`. Alcance: `ARTIFICIAL_DISTINCT_ROIS_SHARED_REAL_RAW_LAST_PACKET_COMPLETE_UPDATE`.

| Configuración | Valor |
| --- | --- |
| precision | FP32_TF32_OFF |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| piloto completo | 1 | RAW disponible → último TTC | 4 | 38.07855 | 29.98695 | 59.77346 | 63.883412 | 64.9109 | no |
| piloto completo | 2 | RAW disponible → último TTC | 4 | 39.7808 | 37.87845 | 45.19519 | 46.193758 | 46.4434 | no |

### `multiobject/pilot_20_chunk1_querygraphs_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `—`. Alcance: `ARTIFICIAL_DISTINCT_ROIS_SHARED_REAL_RAW_LAST_PACKET_COMPLETE_UPDATE`.

| Configuración | Valor |
| --- | --- |
| precision | FP32_TF32_OFF |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| piloto completo | 20 | RAW disponible → último TTC | 3 | 383.714367 | 385.5828 | 386.36472 | 386.434224 | 386.4516 | sí |

### `multiobject/pilot_20_dedicated_stream_v1/summary.json`

Estado: **FAILED**. Dispositivo: `—`. Alcance: `ARTIFICIAL_DISTINCT_ROIS_SHARED_REAL_RAW_LAST_PACKET_COMPLETE_UPDATE`.

| Configuración | Valor |
| --- | --- |
| precision | FP32_TF32_OFF |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| piloto completo | 20 | RAW disponible → último TTC | 3 | 392.5467 | 389.7983 | 410.75525 | 412.61809 | 413.0838 | no |

### `multiobject/pilot_20_exact_cache_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `—`. Alcance: `ARTIFICIAL_DISTINCT_ROIS_SHARED_REAL_RAW_LAST_PACKET_COMPLETE_UPDATE`.

| Configuración | Valor |
| --- | --- |
| precision | FP32_TF32_OFF |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| piloto completo | 20 | RAW disponible → último TTC | 5 | 233.32098 | 194.0419 | 367.51758 | 400.419116 | 408.6445 | sí |

### `multiobject/pilot_2_only_parallel_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `—`. Alcance: `ARTIFICIAL_DISTINCT_ROIS_SHARED_REAL_RAW_LAST_PACKET_COMPLETE_UPDATE`.

| Configuración | Valor |
| --- | --- |
| precision | FP32_TF32_OFF |

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| piloto completo | 2 | RAW disponible → último TTC | 4 | 42.8235 | 33.09075 | 67.17069 | 71.933178 | 73.1238 | sí |

## transport

Transporte sobre features sintéticas preparadas; excluye sensores y TTC.

### `transport/block_gpu_v1/summary.json`

Estado: **PROBE_COMPLETE_NOT_FINAL_TTC_ADMISSION**. Dispositivo: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`. Alcance: `COMPLETE_BIDIRECTIONAL_TRANSPORT_PREPARED_SYNTHETIC_GPU_FEATURES_NO_TTC`.

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle; batch=8; radio=1 | — | graph | 30 | 1.76954 | 1.73445 | 1.943625 | 2.165923 | 2.2388 | sí |
| reuse_samples; batch=8; radio=1 | — | graph | 30 | 1.86817 | 1.8015 | 2.132745 | 2.163474 | 2.172 | sí |
| reuse_reverse; batch=8; radio=1 | — | graph | 30 | 1.29482333 | 1.25975 | 1.477385 | 1.533908 | 1.544 | sí |
| compiled; batch=8; radio=1 | — | graph | 30 | 0.284056667 | 0.2512 | 0.446905 | 0.464233 | 0.4678 | sí |
| oracle; batch=8; radio=4 | — | graph | 30 | 4.75130667 | 4.67155 | 5.15032 | 5.255747 | 5.298 | sí |
| reuse_samples; batch=8; radio=4 | — | graph | 30 | 4.59667667 | 4.4431 | 5.11768 | 5.172935 | 5.1783 | sí |
| reuse_reverse; batch=8; radio=4 | — | graph | 30 | 3.03961667 | 2.93805 | 3.472375 | 3.584316 | 3.6248 | sí |
| compiled; batch=8; radio=4 | — | graph | 30 | 1.04287333 | 1.02785 | 1.1667 | 1.232708 | 1.2457 | sí |

## encoders

Encoders de endpoints preparados; excluye transporte, contexto y TTC.

### `encoder_ea5_b12_v1/summary.json`

Estado: **FAILED_ENCODER_PARITY**. Dispositivo: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`. Alcance: `PREPARED_ENDPOINT_ENCODERS_ONLY_NO_SENSOR_H2D_TRANSPORT_CONTEXT_TTC`.

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E_A5_MATCHED; batch=12 | — | baseline_graph | 20 | 1.823125 | 1.7988 | 1.99659 | 2.000998 | 2.0021 | no |
| E_A5_MATCHED; batch=12 | — | compiled_graph | 20 | 1.39498 | 1.3763 | 1.523365 | 1.664193 | 1.6994 | no |

## compiled_pipeline

Productores con transporte compilado e inputs GPU preparados; la paridad TTC se comprueba fuera del reloj. Excluye RAW, H2D y contexto del tiempo.

### `compiled_pipeline/gpu_idx0_v1/summary.json`

Estado: **FAILED**. Dispositivo: `—`. Alcance: `PREPARED_GPU_PRODUCERS_PAIR_PHASE17_GUARDS_NO_RAW_H2D_CONTEXT`.

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | original_graph | 20 | 20.0803 | 20.45275 | 21.964185 | 22.666197 | 22.8417 | sí |
| 0 | 1 | compiled_transport_graph | 20 | 10.56016 | 10.6217 | 11.61585 | 12.36749 | 12.5554 | sí |

La columna de paridad corresponde a las tres salidas TTC por ruta. El estado del ensayo conserva también fallos de auditoría. La paridad de features intermedias es un diagnóstico separado.

error: `RuntimeError: Compiled producer graph lost its opaque transport operator`

### `compiled_pipeline/gpu_no_fma_exact_div_v4/summary.json`

Estado: **COMPLETE_WITH_PARITY_FAILURES**. Dispositivo: `—`. Alcance: `PREPARED_GPU_PRODUCERS_PAIR_PHASE17_GUARDS_NO_RAW_H2D_CONTEXT`.

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | original_graph | 20 | 18.61502 | 18.42515 | 20.193155 | 21.539191 | 21.8757 | sí |
| 0 | 1 | compiled_transport_graph | 20 | 10.20089 | 10.08835 | 10.96106 | 11.208212 | 11.27 | sí |
| 1 | 1 | original_graph | 20 | 18.95618 | 19.1487 | 20.4676 | 20.65608 | 20.7032 | sí |
| 1 | 1 | compiled_transport_graph | 20 | 10.55057 | 10.6497 | 11.45835 | 11.67647 | 11.731 | sí |
| 2 | 1 | original_graph | 20 | 19.240435 | 19.3197 | 20.571675 | 20.630575 | 20.6453 | sí |
| 2 | 1 | compiled_transport_graph | 20 | 10.88964 | 10.8344 | 12.109645 | 12.236489 | 12.2682 | no |
| 3 | 1 | original_graph | 20 | 18.827965 | 18.89265 | 20.01408 | 20.197696 | 20.2436 | sí |
| 3 | 1 | compiled_transport_graph | 20 | 10.76307 | 10.7428 | 11.99962 | 12.285684 | 12.3572 | no |
| 4 | 1 | original_graph | 20 | 18.404695 | 18.28765 | 19.51635 | 19.52623 | 19.5287 | sí |
| 4 | 1 | compiled_transport_graph | 20 | 10.35518 | 10.45095 | 10.953555 | 10.954391 | 10.9546 | sí |
| 5 | 1 | original_graph | 20 | 20.207165 | 19.98755 | 21.957645 | 22.341369 | 22.4373 | sí |
| 5 | 1 | compiled_transport_graph | 20 | 11.37498 | 11.21235 | 12.53935 | 13.02195 | 13.1426 | sí |
| 6 | 1 | original_graph | 20 | 17.74493 | 17.74665 | 18.465235 | 18.636007 | 18.6787 | sí |
| 6 | 1 | compiled_transport_graph | 20 | 9.997815 | 9.9527 | 10.667405 | 10.836201 | 10.8784 | sí |

La columna de paridad corresponde a las tres salidas TTC por ruta. El estado del ensayo conserva también fallos de auditoría. La paridad de features intermedias es un diagnóstico separado.

### `compiled_pipeline/gpu_no_fma_v3/summary.json`

Estado: **COMPLETE_WITH_PARITY_FAILURES**. Dispositivo: `—`. Alcance: `PREPARED_GPU_PRODUCERS_PAIR_PHASE17_GUARDS_NO_RAW_H2D_CONTEXT`.

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | original_graph | 20 | 20.816355 | 19.46525 | 24.27013 | 24.396746 | 24.4284 | sí |
| 0 | 1 | compiled_transport_graph | 20 | 11.137885 | 10.3036 | 13.5099 | 14.14526 | 14.3041 | sí |
| 1 | 1 | original_graph | 20 | 17.85844 | 17.8892 | 18.489635 | 18.668007 | 18.7126 | sí |
| 1 | 1 | compiled_transport_graph | 20 | 9.88426 | 9.876 | 10.479645 | 10.500089 | 10.5052 | sí |
| 2 | 1 | original_graph | 20 | 17.87667 | 17.763 | 18.600485 | 18.621537 | 18.6268 | sí |
| 2 | 1 | compiled_transport_graph | 20 | 9.749535 | 9.8012 | 10.274215 | 10.365643 | 10.3885 | no |
| 3 | 1 | original_graph | 20 | 17.9546 | 17.9014 | 18.614575 | 18.790515 | 18.8345 | sí |
| 3 | 1 | compiled_transport_graph | 20 | 9.79831 | 9.7568 | 10.45441 | 10.460642 | 10.4622 | no |
| 4 | 1 | original_graph | 20 | 18.098975 | 17.9884 | 18.733815 | 19.015243 | 19.0856 | sí |
| 4 | 1 | compiled_transport_graph | 20 | 9.880895 | 9.8729 | 10.339905 | 10.391661 | 10.4046 | sí |
| 5 | 1 | original_graph | 20 | 17.977495 | 18.05815 | 18.66584 | 18.809328 | 18.8452 | sí |
| 5 | 1 | compiled_transport_graph | 20 | 9.772445 | 9.71465 | 10.33315 | 10.38255 | 10.3949 | no |
| 6 | 1 | original_graph | 20 | 18.154915 | 18.0961 | 19.004355 | 19.335031 | 19.4177 | sí |
| 6 | 1 | compiled_transport_graph | 20 | 10.05057 | 9.9543 | 10.71276 | 11.341432 | 11.4986 | sí |

La columna de paridad corresponde a las tres salidas TTC por ruta. El estado del ensayo conserva también fallos de auditoría. La paridad de features intermedias es un diagnóstico separado.

### `compiled_pipeline/gpu_sensitive_v2/summary.json`

Estado: **COMPLETE_WITH_PARITY_FAILURES**. Dispositivo: `—`. Alcance: `PREPARED_GPU_PRODUCERS_PAIR_PHASE17_GUARDS_NO_RAW_H2D_CONTEXT`.

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | original_graph | 20 | 17.866535 | 17.3967 | 22.587085 | 23.185737 | 23.3354 | sí |
| 0 | 1 | compiled_transport_graph | 20 | 9.60258 | 9.4597 | 10.57088 | 10.573616 | 10.5743 | sí |
| 1 | 1 | original_graph | 20 | 17.075705 | 16.99425 | 17.70918 | 17.760556 | 17.7734 | sí |
| 1 | 1 | compiled_transport_graph | 20 | 9.59396 | 9.40185 | 10.197755 | 11.486031 | 11.8081 | sí |
| 2 | 1 | original_graph | 20 | 18.2248 | 18.20735 | 18.985425 | 19.183405 | 19.2329 | sí |
| 2 | 1 | compiled_transport_graph | 20 | 10.08636 | 10.05485 | 10.849215 | 11.015123 | 11.0566 | no |
| 3 | 1 | original_graph | 20 | 18.97949 | 19.05375 | 20.072145 | 20.457389 | 20.5537 | sí |
| 3 | 1 | compiled_transport_graph | 20 | 10.32863 | 10.21665 | 11.42677 | 12.092074 | 12.2584 | no |
| 4 | 1 | original_graph | 20 | 18.435725 | 18.35965 | 19.395765 | 19.761553 | 19.853 | sí |
| 4 | 1 | compiled_transport_graph | 20 | 10.192405 | 10.1923 | 10.758575 | 10.917795 | 10.9576 | sí |
| 5 | 1 | original_graph | 20 | 19.07976 | 19.0963 | 19.779805 | 20.094521 | 20.1732 | sí |
| 5 | 1 | compiled_transport_graph | 20 | 10.68347 | 10.54045 | 11.815935 | 12.007987 | 12.056 | sí |
| 6 | 1 | original_graph | 20 | 18.391915 | 18.2742 | 19.05532 | 19.142264 | 19.164 | sí |
| 6 | 1 | compiled_transport_graph | 20 | 10.380885 | 10.1561 | 11.82193 | 11.992626 | 12.0353 | sí |

La columna de paridad corresponde a las tres salidas TTC por ruta. El estado del ensayo conserva también fallos de auditoría. La paridad de features intermedias es un diagnóstico separado.

## rgb_uint8

Preparación RGB sintética; incluye el tramo declarado en cada recibo. No incluye modelos ni TTC. Bytes propuestos no equivalen a transferencias ejecutadas.

### `rgb_uint8/cpu_1200x1920_roi_first_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `cpu`. Alcance: `SYNTHETIC_RGB_PREPARATION_ONLY_NOT_TTC_OR_SUSTAINED_SERVICE`.

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | original_preparation | 3 | 49.504 | 49.8234 | 51.35997 | 51.496554 | 51.5307 | sí |
| 0 | 1 | uint8_preparation | 3 | 70.4231333 | 67.0408 | 85.92091 | 87.599142 | 88.0187 | sí |
| 1 | 2 | original_preparation | 3 | 120.5882 | 122.1888 | 122.98638 | 123.057276 | 123.075 | sí |
| 1 | 2 | uint8_preparation | 3 | 119.4465 | 119.5845 | 120.19317 | 120.247274 | 120.2608 | sí |
| 2 | 20 | original_preparation | 3 | 1178.2865 | 1069.6214 | 1371.66401 | 1398.51224 | 1405.2243 | sí |
| 2 | 20 | uint8_preparation | 3 | 3329.66343 | 3457.4945 | 4926.05159 | 5056.59 | 5089.2246 | sí |

| Objetos | Observaciones | Frames/tripleta | H2D ejecutados bytes | H2D propuestos bytes | Píxeles FP32 originales bytes |
| --- | --- | --- | --- | --- | --- |
| 1 | 5 | 3 | 0 | 344224 | 2949120 |
| 2 | 5 | 3 | 0 | 688448 | 5898240 |
| 20 | 5 | 3 | 0 | 6884480 | 58982400 |

### `rgb_uint8/cpu_matrix_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `cpu`. Alcance: `SYNTHETIC_RGB_PREPARATION_ONLY_NOT_TTC_OR_SUSTAINED_SERVICE`.

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | uint8_preparation | 1 | 0.7755 | 0.7755 | 0.7755 | 0.7755 | 0.7755 | sí |
| 1 | 1 | uint8_preparation | 1 | 1.3553 | 1.3553 | 1.3553 | 1.3553 | 1.3553 | sí |
| 2 | 1 | uint8_preparation | 1 | 1.0531 | 1.0531 | 1.0531 | 1.0531 | 1.0531 | sí |
| 3 | 1 | uint8_preparation | 1 | 2.0977 | 2.0977 | 2.0977 | 2.0977 | 2.0977 | sí |
| 4 | 1 | uint8_preparation | 1 | 2.0293 | 2.0293 | 2.0293 | 2.0293 | 2.0293 | sí |
| 5 | 1 | uint8_preparation | 1 | 2.3592 | 2.3592 | 2.3592 | 2.3592 | 2.3592 | sí |
| 6 | 1 | uint8_preparation | 1 | 1.789 | 1.789 | 1.789 | 1.789 | 1.789 | sí |
| 7 | 1 | uint8_preparation | 1 | 2.2096 | 2.2096 | 2.2096 | 2.2096 | 2.2096 | sí |
| 8 | 1 | uint8_preparation | 1 | 2.2018 | 2.2018 | 2.2018 | 2.2018 | 2.2018 | sí |
| 9 | 1 | uint8_preparation | 1 | 2.744 | 2.744 | 2.744 | 2.744 | 2.744 | sí |
| 10 | 1 | uint8_preparation | 1 | 3.7719 | 3.7719 | 3.7719 | 3.7719 | 3.7719 | sí |
| 11 | 1 | uint8_preparation | 1 | 4.6715 | 4.6715 | 4.6715 | 4.6715 | 4.6715 | sí |
| 12 | 1 | uint8_preparation | 1 | 3.9739 | 3.9739 | 3.9739 | 3.9739 | 3.9739 | sí |
| 13 | 1 | uint8_preparation | 1 | 5.0115 | 5.0115 | 5.0115 | 5.0115 | 5.0115 | sí |
| 14 | 1 | uint8_preparation | 1 | 4.011 | 4.011 | 4.011 | 4.011 | 4.011 | sí |
| 15 | 1 | uint8_preparation | 1 | 4.0537 | 4.0537 | 4.0537 | 4.0537 | 4.0537 | sí |
| 16 | 2 | uint8_preparation | 1 | 1.2962 | 1.2962 | 1.2962 | 1.2962 | 1.2962 | sí |
| 17 | 2 | uint8_preparation | 1 | 2.2456 | 2.2456 | 2.2456 | 2.2456 | 2.2456 | sí |
| 18 | 2 | uint8_preparation | 1 | 2.2939 | 2.2939 | 2.2939 | 2.2939 | 2.2939 | sí |
| 19 | 2 | uint8_preparation | 1 | 4.0362 | 4.0362 | 4.0362 | 4.0362 | 4.0362 | sí |
| 20 | 2 | uint8_preparation | 1 | 3.1843 | 3.1843 | 3.1843 | 3.1843 | 3.1843 | sí |
| 21 | 2 | uint8_preparation | 1 | 4.0209 | 4.0209 | 4.0209 | 4.0209 | 4.0209 | sí |
| 22 | 2 | uint8_preparation | 1 | 2.8198 | 2.8198 | 2.8198 | 2.8198 | 2.8198 | sí |
| 23 | 2 | uint8_preparation | 1 | 3.358 | 3.358 | 3.358 | 3.358 | 3.358 | sí |
| 24 | 2 | uint8_preparation | 1 | 3.4714 | 3.4714 | 3.4714 | 3.4714 | 3.4714 | sí |
| 25 | 2 | uint8_preparation | 1 | 4.003 | 4.003 | 4.003 | 4.003 | 4.003 | sí |
| 26 | 2 | uint8_preparation | 1 | 5.9641 | 5.9641 | 5.9641 | 5.9641 | 5.9641 | sí |
| 27 | 2 | uint8_preparation | 1 | 5.5279 | 5.5279 | 5.5279 | 5.5279 | 5.5279 | sí |
| 28 | 2 | uint8_preparation | 1 | 5.7899 | 5.7899 | 5.7899 | 5.7899 | 5.7899 | sí |
| 29 | 2 | uint8_preparation | 1 | 5.6593 | 5.6593 | 5.6593 | 5.6593 | 5.6593 | sí |
| 30 | 2 | uint8_preparation | 1 | 5.4294 | 5.4294 | 5.4294 | 5.4294 | 5.4294 | sí |
| 31 | 2 | uint8_preparation | 1 | 7.6391 | 7.6391 | 7.6391 | 7.6391 | 7.6391 | sí |
| 32 | 20 | uint8_preparation | 1 | 15.6176 | 15.6176 | 15.6176 | 15.6176 | 15.6176 | sí |
| 33 | 20 | uint8_preparation | 1 | 22.4046 | 22.4046 | 22.4046 | 22.4046 | 22.4046 | sí |
| 34 | 20 | uint8_preparation | 1 | 24.3994 | 24.3994 | 24.3994 | 24.3994 | 24.3994 | sí |
| 35 | 20 | uint8_preparation | 1 | 28.3079 | 28.3079 | 28.3079 | 28.3079 | 28.3079 | sí |
| 36 | 20 | uint8_preparation | 1 | 27.5869 | 27.5869 | 27.5869 | 27.5869 | 27.5869 | sí |
| 37 | 20 | uint8_preparation | 1 | 33.9982 | 33.9982 | 33.9982 | 33.9982 | 33.9982 | sí |
| 38 | 20 | uint8_preparation | 1 | 35.0014 | 35.0014 | 35.0014 | 35.0014 | 35.0014 | sí |
| 39 | 20 | uint8_preparation | 1 | 45.0528 | 45.0528 | 45.0528 | 45.0528 | 45.0528 | sí |
| 40 | 20 | uint8_preparation | 1 | 38.0086 | 38.0086 | 38.0086 | 38.0086 | 38.0086 | sí |
| 41 | 20 | uint8_preparation | 1 | 50.7378 | 50.7378 | 50.7378 | 50.7378 | 50.7378 | sí |
| 42 | 20 | uint8_preparation | 1 | 51.4433 | 51.4433 | 51.4433 | 51.4433 | 51.4433 | sí |
| 43 | 20 | uint8_preparation | 1 | 72.777 | 72.777 | 72.777 | 72.777 | 72.777 | sí |
| 44 | 20 | uint8_preparation | 1 | 73.6358 | 73.6358 | 73.6358 | 73.6358 | 73.6358 | sí |
| 45 | 20 | uint8_preparation | 1 | 80.7389 | 80.7389 | 80.7389 | 80.7389 | 80.7389 | sí |
| 46 | 20 | uint8_preparation | 1 | 81.2377 | 81.2377 | 81.2377 | 81.2377 | 81.2377 | sí |
| 47 | 20 | uint8_preparation | 1 | 93.1227 | 93.1227 | 93.1227 | 93.1227 | 93.1227 | sí |

| Objetos | Observaciones | Frames/tripleta | H2D ejecutados bytes | H2D propuestos bytes | Píxeles FP32 originales bytes |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 2 | 0 | 98324 | 393216 |
| 1 | 1 | 3 | 0 | 147488 | 589824 |
| 1 | 2 | 2 | 0 | 147496 | 786432 |
| 1 | 2 | 3 | 0 | 196672 | 1179648 |
| 1 | 3 | 2 | 0 | 196668 | 1179648 |
| 1 | 3 | 3 | 0 | 245856 | 1769472 |
| 1 | 4 | 2 | 0 | 245840 | 1572864 |
| 1 | 4 | 3 | 0 | 295040 | 2359296 |
| 1 | 5 | 2 | 0 | 295012 | 1966080 |
| 1 | 5 | 3 | 0 | 344224 | 2949120 |
| 1 | 6 | 2 | 0 | 344184 | 2359296 |
| 1 | 6 | 3 | 0 | 393408 | 3538944 |
| 1 | 7 | 2 | 0 | 393356 | 2752512 |
| 1 | 7 | 3 | 0 | 442592 | 4128768 |
| 1 | 8 | 2 | 0 | 442528 | 3145728 |
| 1 | 8 | 3 | 0 | 491776 | 4718592 |
| 2 | 1 | 2 | 0 | 196648 | 786432 |
| 2 | 1 | 3 | 0 | 294976 | 1179648 |
| 2 | 2 | 2 | 0 | 294992 | 1572864 |
| 2 | 2 | 3 | 0 | 393344 | 2359296 |
| 2 | 3 | 2 | 0 | 393336 | 2359296 |
| 2 | 3 | 3 | 0 | 491712 | 3538944 |
| 2 | 4 | 2 | 0 | 491680 | 3145728 |
| 2 | 4 | 3 | 0 | 590080 | 4718592 |
| 2 | 5 | 2 | 0 | 590024 | 3932160 |
| 2 | 5 | 3 | 0 | 688448 | 5898240 |
| 2 | 6 | 2 | 0 | 688368 | 4718592 |
| 2 | 6 | 3 | 0 | 786816 | 7077888 |
| 2 | 7 | 2 | 0 | 786712 | 5505024 |
| 2 | 7 | 3 | 0 | 885184 | 8257536 |
| 2 | 8 | 2 | 0 | 885056 | 6291456 |
| 2 | 8 | 3 | 0 | 983552 | 9437184 |
| 20 | 1 | 2 | 0 | 1966480 | 7864320 |
| 20 | 1 | 3 | 0 | 2949760 | 11796480 |
| 20 | 2 | 2 | 0 | 2949920 | 15728640 |
| 20 | 2 | 3 | 0 | 3933440 | 23592960 |
| 20 | 3 | 2 | 0 | 3933360 | 23592960 |
| 20 | 3 | 3 | 0 | 4917120 | 35389440 |
| 20 | 4 | 2 | 0 | 4916800 | 31457280 |
| 20 | 4 | 3 | 0 | 5900800 | 47185920 |
| 20 | 5 | 2 | 0 | 5900240 | 39321600 |
| 20 | 5 | 3 | 0 | 6884480 | 58982400 |
| 20 | 6 | 2 | 0 | 6883680 | 47185920 |
| 20 | 6 | 3 | 0 | 7868160 | 70778880 |
| 20 | 7 | 2 | 0 | 7867120 | 55050240 |
| 20 | 7 | 3 | 0 | 8851840 | 82575360 |
| 20 | 8 | 2 | 0 | 8850560 | 62914560 |
| 20 | 8 | 3 | 0 | 9835520 | 94371840 |

### `rgb_uint8/cpu_roi_first_v1/summary.json`

Estado: **COMPLETE**. Dispositivo: `cpu`. Alcance: `SYNTHETIC_RGB_PREPARATION_ONLY_NOT_TTC_OR_SUSTAINED_SERVICE`.

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | original_preparation | 5 | 3.32668 | 3.2749 | 3.49928 | 3.528016 | 3.5352 | sí |
| 0 | 1 | uint8_preparation | 5 | 3.63772 | 3.6551 | 3.696 | 3.70272 | 3.7044 | sí |
| 1 | 2 | original_preparation | 5 | 8.72866 | 8.7641 | 8.99414 | 9.020828 | 9.0275 | sí |
| 1 | 2 | uint8_preparation | 5 | 8.00892 | 7.9158 | 8.32972 | 8.393544 | 8.4095 | sí |
| 2 | 20 | original_preparation | 5 | 83.3146 | 78.965 | 95.83458 | 96.868676 | 97.1272 | sí |
| 2 | 20 | uint8_preparation | 5 | 58.07622 | 57.7406 | 59.18764 | 59.218888 | 59.2267 | sí |

| Objetos | Observaciones | Frames/tripleta | H2D ejecutados bytes | H2D propuestos bytes | Píxeles FP32 originales bytes |
| --- | --- | --- | --- | --- | --- |
| 1 | 5 | 3 | 0 | 344224 | 2949120 |
| 2 | 5 | 3 | 0 | 688448 | 5898240 |
| 20 | 5 | 3 | 0 | 6884480 | 58982400 |

## scheduler

Control del planificador con RAW y predicciones ficticias. No ejecuta modelos ni GPU y no admite ningún objetivo TTC.

### `compute_council/paced_scheduler_ab_v1/summary.json`

Estado: **COMPLETE_FAKE_CONTROL_ONLY**. Dispositivo: `—`. Alcance: `REAL_PACED_HELPER_THREADS_QUEUE_FAKE_RAW_FAKE_PREDICTIONS_NO_GPU_NO_TTC`.

| Caso | Objetos/ROI | Ruta | n | Media ms | p50 ms | p95 ms | p99 ms | Máximo ms | Paridad del tramo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| event | — | raw_availability_lateness_ms | 3000 | 8.22284703 | 8.247 | 15.225025 | 16.02663 | 16.8984 | — |
| event | — | deadline_response_ms | 3000 | 8.3464227 | 8.3681 | 15.356815 | 16.192146 | 17.1889 | — |
| sleep | — | raw_availability_lateness_ms | 3000 | 0.3510518 | 0.3427 | 0.610805 | 0.825125 | 1.5396 | — |
| sleep | — | deadline_response_ms | 3000 | 0.4668937 | 0.45205 | 0.77045 | 1.043656 | 8.1876 | — |
