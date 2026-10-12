# Continuación multimodal: estado recuperable

Generado 2026-10-12T16:41:58.153231+00:00. Estado: LOCAL_RESULTS_COMPLETE_OFFICIAL_EVALUATION_PENDING; SOTA no demostrado.

Updates científicos: 80000; recuperación superior: 0; límite físico consumido: 80000.

Pasar del 2 % al 4 % cambia la tolerancia de degradación del tramo TTC largo; no cambia las predicciones ni demuestra superioridad sobre otro sistema. PHASE_ONLY ya cumple los criterios originales del 2 % en las semillas 7, 13 y 23, incluido el contraste con el control sin información predictiva event. Sus ventajas externas dependen de la métrica y la cohorte: F_TRUE conserva mejor agregado/MAE FCWD y Dev32 no confirma una mejora consistente de MiD. No se declara una victoria universal ni un resultado oficial de test12.

La comparación pendiente Garl visual/full FCWD ya se ejecutó antes de la nueva autorización de entrenamiento: 630 consultas comprobadas temporalmente, 495 aceptadas por el registro sellado y 472 etiquetas comunes. F_TRUE supera al full registrado tanto en ese soporte común como frente a su fallback event-only sobre 597 etiquetas. No se cambiaron tolerancias. El [informe de esa etapa](../multimodal_validation_20261012/MULTIMODAL_REVIEW_FINAL.md) conserva cobertura, métricas nativas/sensibilidad, 1.229 llamadas instrumentadas, geometría y costes. Sus cero updates describen aquella etapa; esta continuación posterior ejecuta la autorización ampliada del usuario.

## Evidencia interna V

V es desarrollo expuesto. Cada par de correctores conserva 2.500 updates por brazo. Los estados H originales proceden de refinadores supervisados en H y no son OOF. GROUP_CROSSFIT prepara contextos y normalizadores excluyendo el grupo predicho; sus ocho fits de contexto tienen contabilidad propia. Las semillas adicionales de PHASE_ONLY replican las cabezas con productores congelados, no el sistema completo.

| Estudio | Semilla | Brazo | MiD agregado | MAE s | Gates fallidos | IC brazo−control |
| --- | --- | --- | --- | --- | --- | --- |
| BASELINE | 7 | EVENT | 91.300140 | 1.852270 | aggregate_vs_RGB, aggregate_vs_fusion, large | [-3.8621679035432894, 0.3294393313556067] |
| BASELINE | 7 | ZERO | 93.045382 | 1.851931 | aggregate_vs_RGB, aggregate_vs_fusion, large | — |
| PHASE_UNITS | 7 | EVENT | 89.882376 | 1.846064 | large | [-5.389636206247685, -0.5609412822949094] |
| PHASE_UNITS | 7 | ZERO | 92.810052 | 1.852042 | aggregate_vs_RGB, aggregate_vs_fusion, large | — |
| UNIT_REPLICATION | 13 | EVENT | 90.517898 | 1.843921 | large | [-4.100704042569719, 0.2149884042413878] |
| UNIT_REPLICATION | 13 | ZERO | 92.348492 | 1.848880 | aggregate_vs_RGB, aggregate_vs_fusion, large | — |
| UNIT_REPLICATION | 23 | EVENT | 90.257738 | 1.821172 | large | [-4.868064092354796, 0.4001307141483954] |
| UNIT_REPLICATION | 23 | ZERO | 92.309916 | 1.848135 | aggregate_vs_RGB, aggregate_vs_fusion, sign, large, negative | — |
| RELATIVE_PHASE | 7 | EVENT | 90.250266 | 1.884901 | mae_vs_RGB, large | [-5.355518180621468, -0.6255262911840864] |
| RELATIVE_PHASE | 7 | ZERO | 93.131553 | 1.879023 | aggregate_vs_RGB, aggregate_vs_fusion, sign, large | — |
| FUSION_STATE | 7 | FUSED | 93.117998 | 2.005650 | aggregate_vs_RGB, mae_vs_RGB, aggregate_vs_fusion, mae_vs_fusion, sign | [2.423612420526137, 5.283815101955459] |
| FUSION_STATE | 7 | ZERO_FUSED | 89.239073 | 1.837367 | large | — |
| RGB_REGRET | 7 | EVENT | 92.088216 | 1.848408 | aggregate_vs_RGB, aggregate_vs_fusion | [-2.208202289827996, -0.07547567573929281] |
| RGB_REGRET | 7 | ZERO | 93.191735 | 1.881120 | aggregate_vs_RGB, aggregate_vs_fusion, sign, large | — |
| PHASE_ENSEMBLE | 7+13+23 | EVENT | 89.861298 | 1.813344 | large | [-4.938179895791636, -0.34947475877571677] |
| PHASE_ENSEMBLE | 7+13+23 | ZERO | 92.369253 | 1.847306 | aggregate_vs_RGB, aggregate_vs_fusion, large | — |
| FUSION_PRESERVING | 7 | REFINED_RGB | 92.506456 | 2.003201 | aggregate_vs_RGB, mae_vs_RGB, aggregate_vs_fusion, large | [-0.5762574005245096, 1.2557206541216264] |
| FUSION_PRESERVING | 7 | ZERO_RGB | 92.196431 | 2.028992 | aggregate_vs_RGB, mae_vs_RGB, aggregate_vs_fusion, mae_vs_fusion | — |
| PHASE_ONLY | 7 | EVENT | 89.222960 | 1.869604 | ninguno | [-5.737922430198823, -1.3022189428427402] |
| PHASE_ONLY | 7 | ZERO | 92.781912 | 1.785782 | aggregate_vs_RGB, aggregate_vs_fusion | — |
| PHASE_ONLY_REPLICATION | 13 | EVENT | 89.198875 | 1.858718 | ninguno | [-5.734940908789747, -1.3502580619471518] |
| PHASE_ONLY_REPLICATION | 13 | ZERO | 92.781180 | 1.811645 | aggregate_vs_RGB, aggregate_vs_fusion | — |
| PHASE_ONLY_REPLICATION | 23 | EVENT | 89.220028 | 1.862337 | ninguno | [-5.685388400273993, -1.3988404845532494] |
| PHASE_ONLY_REPLICATION | 23 | ZERO | 92.794781 | 1.807807 | aggregate_vs_RGB, aggregate_vs_fusion | — |
| GROUP_CROSSFIT | 7 | EVENT | 90.412912 | 1.893097 | mae_vs_RGB | [-6.766187951416511, -2.2671398803971003] |
| GROUP_CROSSFIT | 7 | ZERO | 94.918069 | 1.795708 | aggregate_vs_RGB, aggregate_vs_fusion, crucial | — |

El gate principal conserva 2 % de tolerancia por bucket. El 4 % en TTC largo es una sensibilidad solicitada después de ver la semilla 7. Su análisis secundario en semillas 13/23 se registró antes de sus nuevas predicciones V; no cambia el principal.

PHASE_ENSEMBLE promedia fases emitidas de todas las semillas 7/13/23 con pesos iguales y cero updates nuevos. FUSION_STATE compara presencia/ausencia de estados F_TRUE sobre el ancla RGB. FUSION_PRESERVING conserva F_TRUE como ancla y compara presencia/ausencia de estados adicionales R_CTX. Esos dos últimos controles permanecen multimodales; sus nombres no significan ausencia de una modalidad en el sistema completo.

PHASE_ONLY conserva el ancla R_CTX y usa un corrector de 1.955 parámetros basado en fases y relojes, sin estados ocultos. Los estudios sucesivos forman una búsqueda adaptativa en V; sus intervalos por contraste no corrigen la selección entre recetas.

## Transferencia congelada

Todas las consultas permanecen. Sin RGB, los nuevos brazos usan exactamente E_CTX de la misma ruta. FCWD tiene registro estimado, no calibración extrínseca oficial. Dev32 conserva 1.024 consultas/946 etiquetas. Los CSV también muestran soporte común, cobertura, todos los comparadores y contrastes descriptivos por secuencia. Dev32 y FCWD ya estaban expuestos durante el desarrollo: esta transferencia es descriptiva y no constituye un nuevo test ciego de la búsqueda completa.

| Estudio | Semilla | Brazo | Cohorte | Ruta padres | Etiquetas | mean MiD | MAE s | MiDc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BASELINE | 7 | EVENT | FCWD | cpu | 597 | 71.436100 | 1.634891 | 125.249380 |
| BASELINE | 7 | ZERO | FCWD | cpu | 597 | 67.083629 | 1.649827 | 101.494163 |
| BASELINE | 7 | EVENT | DEV32 | cpu | 946 | 127.711280 | 1.193430 | 176.544056 |
| BASELINE | 7 | ZERO | DEV32 | cpu | 946 | 131.592029 | 1.220924 | 183.427616 |
| BASELINE | 7 | EVENT | FCWD | native_gpu | 597 | 71.436120 | 1.634766 | 125.250798 |
| BASELINE | 7 | ZERO | FCWD | native_gpu | 597 | 67.082381 | 1.649694 | 101.490048 |
| PHASE_UNITS | 7 | EVENT | FCWD | cpu | 597 | 71.133711 | 1.596072 | 129.938831 |
| PHASE_UNITS | 7 | ZERO | FCWD | cpu | 597 | 67.114889 | 1.647324 | 102.517418 |
| PHASE_UNITS | 7 | EVENT | DEV32 | cpu | 946 | 125.626886 | 1.230327 | 174.709001 |
| PHASE_UNITS | 7 | ZERO | DEV32 | cpu | 946 | 131.881165 | 1.191291 | 183.991447 |
| PHASE_UNITS | 7 | EVENT | FCWD | native_gpu | 597 | 71.131534 | 1.595941 | 129.933912 |
| PHASE_UNITS | 7 | ZERO | FCWD | native_gpu | 597 | 67.113282 | 1.647191 | 102.512174 |
| UNIT_REPLICATION | 13 | EVENT | FCWD | cpu | 597 | 59.263171 | 1.557918 | 98.222755 |
| UNIT_REPLICATION | 13 | ZERO | FCWD | cpu | 597 | 65.300149 | 1.648843 | 96.952126 |
| UNIT_REPLICATION | 13 | EVENT | DEV32 | cpu | 946 | 127.969560 | 1.228086 | 178.835471 |
| UNIT_REPLICATION | 13 | ZERO | DEV32 | cpu | 946 | 131.094626 | 1.188846 | 182.867850 |
| UNIT_REPLICATION | 13 | EVENT | FCWD | native_gpu | 597 | 59.261856 | 1.557775 | 98.221658 |
| UNIT_REPLICATION | 13 | ZERO | FCWD | native_gpu | 597 | 65.298459 | 1.648715 | 96.946157 |
| UNIT_REPLICATION | 23 | EVENT | FCWD | cpu | 597 | 65.521727 | 1.588954 | 119.929689 |
| UNIT_REPLICATION | 23 | ZERO | FCWD | cpu | 597 | 64.693563 | 1.660018 | 93.973200 |
| UNIT_REPLICATION | 23 | EVENT | DEV32 | cpu | 946 | 124.653407 | 1.222678 | 173.857123 |
| UNIT_REPLICATION | 23 | ZERO | DEV32 | cpu | 946 | 129.763376 | 1.196259 | 180.544540 |
| UNIT_REPLICATION | 23 | EVENT | FCWD | native_gpu | 597 | 65.521030 | 1.588821 | 119.930121 |
| UNIT_REPLICATION | 23 | ZERO | FCWD | native_gpu | 597 | 64.692921 | 1.659896 | 93.970303 |
| RELATIVE_PHASE | 7 | EVENT | FCWD | cpu | 597 | 73.525626 | 1.596178 | 142.893317 |
| RELATIVE_PHASE | 7 | ZERO | FCWD | cpu | 597 | 66.582218 | 1.643679 | 100.835718 |
| RELATIVE_PHASE | 7 | EVENT | DEV32 | cpu | 946 | 124.187322 | 1.224172 | 172.240902 |
| RELATIVE_PHASE | 7 | ZERO | DEV32 | cpu | 946 | 132.436979 | 1.232085 | 184.758323 |
| RELATIVE_PHASE | 7 | EVENT | FCWD | native_gpu | 597 | 73.524375 | 1.596037 | 142.892327 |
| RELATIVE_PHASE | 7 | ZERO | FCWD | native_gpu | 597 | 66.580494 | 1.643546 | 100.829801 |
| FUSION_STATE | 7 | FUSED | DEV32 | cpu | 946 | 124.610951 | 1.207419 | 174.443208 |
| FUSION_STATE | 7 | ZERO_FUSED | DEV32 | cpu | 946 | 127.553221 | 1.242159 | 177.308908 |
| FUSION_STATE | 7 | FUSED | FCWD | native_gpu | 597 | 64.289232 | 1.489306 | 127.506602 |
| FUSION_STATE | 7 | ZERO_FUSED | FCWD | native_gpu | 597 | 67.867578 | 1.580760 | 122.329391 |
| RGB_REGRET | 7 | EVENT | FCWD | cpu | 597 | 68.556987 | 1.613227 | 109.674663 |
| RGB_REGRET | 7 | ZERO | FCWD | cpu | 597 | 66.184393 | 1.616652 | 100.343063 |
| RGB_REGRET | 7 | EVENT | DEV32 | cpu | 946 | 130.257146 | 1.232383 | 180.989217 |
| RGB_REGRET | 7 | ZERO | DEV32 | cpu | 946 | 131.529520 | 1.253666 | 182.699692 |
| RGB_REGRET | 7 | EVENT | FCWD | native_gpu | 597 | 68.555920 | 1.613093 | 109.671074 |
| RGB_REGRET | 7 | ZERO | FCWD | native_gpu | 597 | 66.183872 | 1.616559 | 100.340902 |
| PHASE_ENSEMBLE | 7+13+23 | EVENT | FCWD | cpu | 597 | 65.201146 | 1.580199 | 115.751188 |
| PHASE_ENSEMBLE | 7+13+23 | ZERO | FCWD | cpu | 597 | 65.622338 | 1.651822 | 97.577785 |
| PHASE_ENSEMBLE | 7+13+23 | EVENT | DEV32 | cpu | 946 | 125.500632 | 1.258268 | 174.881844 |
| PHASE_ENSEMBLE | 7+13+23 | ZERO | DEV32 | cpu | 946 | 130.865259 | 1.186939 | 182.400482 |
| PHASE_ENSEMBLE | 7+13+23 | EVENT | FCWD | native_gpu | 597 | 65.199578 | 1.580059 | 115.748741 |
| PHASE_ENSEMBLE | 7+13+23 | ZERO | FCWD | native_gpu | 597 | 65.620972 | 1.651696 | 97.572769 |
| FUSION_PRESERVING | 7 | REFINED_RGB | DEV32 | cpu | 946 | 126.677430 | 1.335730 | 176.869610 |
| FUSION_PRESERVING | 7 | ZERO_RGB | DEV32 | cpu | 946 | 129.494481 | 1.363233 | 181.079871 |
| FUSION_PRESERVING | 7 | REFINED_RGB | FCWD | native_gpu | 597 | 67.508500 | 1.524608 | 140.608488 |
| FUSION_PRESERVING | 7 | ZERO_RGB | FCWD | native_gpu | 597 | 54.885678 | 1.461586 | 103.435831 |
| PHASE_ONLY | 7 | EVENT | DEV32 | cpu | 946 | 132.292198 | 1.216539 | 185.793232 |
| PHASE_ONLY | 7 | ZERO | DEV32 | cpu | 946 | 130.757821 | 1.189731 | 181.666152 |
| PHASE_ONLY | 7 | EVENT | FCWD | native_gpu | 597 | 59.504120 | 1.582103 | 93.304249 |
| PHASE_ONLY | 7 | ZERO | FCWD | native_gpu | 597 | 67.684591 | 1.679948 | 101.331169 |
| PHASE_ONLY_REPLICATION | 13 | EVENT | DEV32 | cpu | 946 | 131.701662 | 1.202355 | 184.998891 |
| PHASE_ONLY_REPLICATION | 13 | ZERO | DEV32 | cpu | 946 | 130.654715 | 1.180581 | 181.588850 |
| PHASE_ONLY_REPLICATION | 13 | EVENT | FCWD | native_gpu | 597 | 59.118803 | 1.594180 | 91.540742 |
| PHASE_ONLY_REPLICATION | 13 | ZERO | FCWD | native_gpu | 597 | 67.503937 | 1.698076 | 100.303999 |
| PHASE_ONLY_REPLICATION | 23 | EVENT | DEV32 | cpu | 946 | 131.000175 | 1.211702 | 183.706311 |
| PHASE_ONLY_REPLICATION | 23 | ZERO | DEV32 | cpu | 946 | 130.727344 | 1.196084 | 181.710932 |
| PHASE_ONLY_REPLICATION | 23 | EVENT | FCWD | native_gpu | 597 | 59.879333 | 1.597057 | 92.791049 |
| PHASE_ONLY_REPLICATION | 23 | ZERO | FCWD | native_gpu | 597 | 67.718564 | 1.687749 | 101.069909 |
| GROUP_CROSSFIT | 7 | EVENT | DEV32 | cpu | 946 | 137.004210 | 1.263563 | 194.930046 |
| GROUP_CROSSFIT | 7 | ZERO | DEV32 | cpu | 946 | 133.618868 | 1.207619 | 186.696582 |
| GROUP_CROSSFIT | 7 | EVENT | FCWD | native_gpu | 597 | 65.713557 | 1.576309 | 118.016757 |
| GROUP_CROSSFIT | 7 | ZERO | FCWD | native_gpu | 597 | 73.873205 | 1.710361 | 119.930685 |

La ruta histórica Dev32 era CPU; la de FCWD era GPU. La ruta FCWD conserva matmul TF32 desactivado y cuDNN TF32 activado. El replay global de padres tiene sus recibos separados en parent_replay. Los readouts externos se ejecutan en CPU FP32 y esa precisión se declara en cada sello.

## Diagnóstico del ajuste sobre H

Pérdidas in-sample, sin updates nuevos. No son OOF ni una medida de generalización. La mejora H de FUSED acompañada por su deterioro V es compatible con sobreajuste; no demuestra una causa única.

| Estudio | Brazo | Error absoluto fase H | Pérdida nativa H |
| --- | --- | --- | --- |
| BASELINE | EVENT | 0.00310286 | 0.11014852 |
| BASELINE | ZERO | 0.00347802 | 0.12433334 |
| PHASE_UNITS | EVENT | 0.00305829 | 0.10850258 |
| PHASE_UNITS | ZERO | 0.00348230 | 0.12452161 |
| RELATIVE_PHASE | EVENT | 0.00313172 | 0.11114837 |
| RELATIVE_PHASE | ZERO | 0.00350517 | 0.12538912 |
| RGB_REGRET | EVENT | 0.00329793 | 0.11735145 |
| RGB_REGRET | ZERO | 0.00356509 | 0.12764822 |
| FUSION_STATE | FUSED | 0.00250227 | 0.08768368 |
| FUSION_STATE | ZERO_FUSED | 0.00305469 | 0.10839884 |

## Atribución en los episodios ya publicados

104 trazas de las dos cabezas congeladas sobre los 32 controles H seleccionados originalmente por hash y los 20 episodios externos ya expuestos. Instrumentación con paridad bit a bit; las salidas externas coinciden exactamente con las predicciones selladas. Se reutilizan estados de padres sellados; no es una nueva inferencia sensorial.

| Consulta | Brazo | GT s | R_CTX s | F_TRUE s | Nueva s |
| --- | --- | --- | --- | --- | --- |
| CCRs-1-low-100-overlap-100:24 | EVENT | 0.943878 | 1.726628 | 22.649269 | 1.854228 |
| CCRs-1-low-100-overlap-100:24 | ZERO | 0.943878 | 1.726628 | 22.649269 | 1.722481 |
| CCRs-1-medium-100-overlap-100:01 | EVENT | 6.520134 | -8.321332 | -59.999996 | -10.554130 |
| CCRs-1-medium-100-overlap-100:01 | ZERO | 6.520134 | -8.321332 | -59.999996 | -8.377889 |

La nueva cabeza evita el extremo de F_TRUE en estos dos ejemplos, pero no corrige el signo del caso medium:01 y empeora respecto a R_CTX en ambos. El control ZERO resulta mejor aquí. Son diagnósticos de casos seleccionados por fallos anteriores, no una estimación representativa de la contribución event.

## Desacuerdo de signo entre los padres

Diagnóstico posterior de todas las consultas externas del candidato S7. Los estratos usan signos de E_CTX/R_CTX, sin ajustar umbrales al error TTC. Delta negativo favorece al candidato; la comparación con RGB usa sólo etiquetas con RGB disponible. No es una nueva validación independiente.

| Cohorte | Estrato | Etiquetas comunes | Delta MiD | Delta MAE s |
| --- | --- | --- | --- | --- |
| FCWD | all | 472 | -8.972392 | -0.045375 |
| FCWD | same_sign | 467 | -8.996003 | -0.042019 |
| FCWD | opposite_sign | 5 | -6.767142 | -0.358746 |
| DEV32 | all | 946 | 0.911577 | -0.041325 |
| DEV32 | same_sign | 928 | 0.709197 | -0.086230 |
| DEV32 | opposite_sign | 18 | 11.345398 | 2.273747 |

En Dev32 el desacuerdo concentra un empeoramiento respecto a RGB, aunque la nueva cabeza reduce el error de F_TRUE en ese mismo estrato. Las 13 predicciones negativas de R_CTX entre las 946 etiquetas positivas siguen siendo negativas. En seis de ellas la mediana de los expertos RGB era positiva: el contexto RGB cruza ese signo antes de la corrección nueva. Un signo positivo de los expertos no implica un TTC preciso. Esto localiza una limitación, pero no demuestra causalidad ni autoriza una compuerta ajustada con estas etiquetas.

## Coste y paridad

| Cohorte | Comparaciones | Nativo ms | Reuso ms | Ahorro |
| --- | --- | --- | --- | --- |
| DEV32 | 64 | 1284.593 | 1045.866 | 18.58 % |
| FCWD | 64 | 1116.826 | 823.243 | 26.29 % |

CPU, un hilo, carga concurrente; HDF5 hasta E_CTX con pesos residentes. Incluye reuso exacto de ventanas, mapa y frames del encoder. No incluye adquisición de ROI, carga inicial de pesos, RGB ni registro. Las 128 comparaciones preservan inputs, features, expertos, punto y cuantiles bit a bit. No es una cifra de streaming GPU.

## PHASE_ONLY frente a Garl y F_TRUE

Contrastes descriptivos posteriores a la congelación del candidato semilla 7. Delta negativo favorece PHASE_ONLY. Bootstrap de secuencias completas; FCWD sólo tiene tres secuencias. No corrige la búsqueda adaptativa ni autoriza escoger otra receta según estos resultados externos.

| Cohorte | Comparador | Métrica | Delta | IC95 % |
| --- | --- | --- | --- | --- |
| FCWD | public_Garl_event_lhr | mean_MiD | -2.312403 | [-4.298607, 0.515151] |
| FCWD | public_Garl_event_lhr | MAE | -0.564070 | [-1.083943, -0.259610] |
| FCWD | public_Garl_event_lhr | MiDc | -1.752577 | [-5.759133, 2.279379] |
| FCWD | GARL_FULL_REGISTERED_FALLBACK_EVENT | mean_MiD | -13.219018 | [-18.071542, -8.750096] |
| FCWD | GARL_FULL_REGISTERED_FALLBACK_EVENT | MAE | -0.485242 | [-0.804182, -0.268601] |
| FCWD | GARL_FULL_REGISTERED_FALLBACK_EVENT | MiDc | -34.951468 | [-46.043465, -24.619207] |
| FCWD | F_TRUE_SAME_ROUTE | mean_MiD | 5.574937 | [3.160209, 7.607097] |
| FCWD | F_TRUE_SAME_ROUTE | MAE | 0.152349 | [0.045286, 0.234599] |
| FCWD | F_TRUE_SAME_ROUTE | MiDc | -9.018913 | [-15.451115, -0.570981] |
| DEV32 | public_Garl_event_lhr | mean_MiD | -71.166255 | [-103.313879, -40.292257] |
| DEV32 | public_Garl_event_lhr | MAE | -6.831570 | [-17.379940, -1.028057] |
| DEV32 | public_Garl_event_lhr | MiDc | -101.743130 | [-140.990497, -59.421462] |
| DEV32 | public_Garl_rgb_event_full | mean_MiD | -36.339812 | [-71.925320, -6.512447] |
| DEV32 | public_Garl_rgb_event_full | MAE | -0.651738 | [-1.068316, -0.285212] |
| DEV32 | public_Garl_rgb_event_full | MiDc | -51.448463 | [-103.103499, -1.714675] |
| DEV32 | F_TRUE_SAME_ROUTE | mean_MiD | 2.134754 | [-4.824035, 8.951624] |
| DEV32 | F_TRUE_SAME_ROUTE | MAE | -0.151928 | [-0.446634, 0.097602] |
| DEV32 | F_TRUE_SAME_ROUTE | MiDc | 4.210796 | [-6.785381, 15.949124] |

## Coste sensorial del candidato

CPU FP32 one thread, concurrent system load. Raw sensor read through candidate, resident weights, public/frozen ROI and already sealed FCWD registration. Excludes registration search, detector and initialization; no streaming claim.

| Cohorte | Etapa | p50 ms | p95 ms |
| --- | --- | --- | --- |
| FCWD | event_read_voxels_ms | 438.355 | 2785.877 |
| FCWD | event_producers_pair_ms | 410.482 | 421.557 |
| FCWD | event_context_state_ms | 2.574 | 3.110 |
| FCWD | rgb_read_crop_ms | 219.667 | 283.652 |
| FCWD | rgb_producers_pair_ms | 657.434 | 703.473 |
| FCWD | rgb_context_state_ms | 2.586 | 4.159 |
| FCWD | correction_ms | 0.718 | 0.890 |
| FCWD | sensor_to_candidate_ms | 1630.349 | 4147.542 |
| DEV32 | event_read_voxels_ms | 726.772 | 1935.240 |
| DEV32 | event_producers_pair_ms | 411.811 | 421.138 |
| DEV32 | event_context_state_ms | 2.502 | 2.998 |
| DEV32 | rgb_read_crop_ms | 239.586 | 300.783 |
| DEV32 | rgb_producers_pair_ms | 664.418 | 690.633 |
| DEV32 | rgb_context_state_ms | 2.827 | 3.156 |
| DEV32 | correction_ms | 0.778 | 0.897 |
| DEV32 | sensor_to_candidate_ms | 2022.164 | 3295.278 |

128 ejecuciones con paridad exacta de estados y predicción. El coste de registro FCWD debe añadirse; esta medición no es la latencia completa de un sistema con detector y registro en línea.

## Coste FCWD con padres GPU y registro fresco

FCWD raw sensors, fresh sealed-equivalent registration and frozen GPU parents and CPU readout. One CPU thread; batch4 event observations, batch1 RGB observations. Stage synchronization; no other active Python GPU work. A verified waiting V18 supervisor may retain an idle CUDA context. Normal desktop GPU and concurrent CPU preparation loads allowed and not isolated. Resident weights, public query ROI. Excludes detector, startup and persistent cross-query service/state management. Repeat1 quantiles are warm-access observations, not isolated cold-I/O or streaming latency.

| Etapa | p50 ms | p95 ms |
| --- | --- | --- |
| registration_ms | 148.107 | 195.722 |
| event_read_voxels_ms | 293.177 | 1998.577 |
| event_h2d_ms | 3.324 | 8.107 |
| event_producers_pair_ms | 105.179 | 161.715 |
| event_context_ms | 5.761 | 8.391 |
| rgb_read_crop_ms | 160.774 | 234.697 |
| rgb_h2d_ms | 2.060 | 3.354 |
| rgb_producers_pair_ms | 475.126 | 580.212 |
| rgb_dispatch_and_sync_ms | 0.083 | 0.163 |
| rgb_context_ms | 5.028 | 6.369 |
| states_d2h_ms | 0.158 | 0.230 |
| cpu_correction_ms | 0.523 | 0.722 |
| sensor_to_candidate_ms | 1213.117 | 3060.561 |

64 llamadas con registro, estados y predicciones iguales a sus referencias selladas. No se suman medianas de perfiles separados. Esta ruta procesa historias por consulta y no equivale a un servicio persistente con detector y reutilización de estado entre consultas. Los controles del runner, lectura de referencias de paridad, aserciones y escritura de mediciones quedan fuera del cronómetro.

## Preparación oficial test12

Candidato PHASE_ONLY_EVENT_S7 congelado en 2026-10-12T14:05:52.846980+00:00. Se mantienen los 6.762 tokens oficiales. Las 957 repeticiones por clips solapados se resuelven en 5.805 observaciones físicas tras comprobar igualdad de ventanas, ROI y relojes. Cada observación aporta evidencia una sola vez al contexto; todos los tokens reciben su predicción. No se leen etiquetas privadas.

Los productores/PAIR y refinadores siguen siendo los padres V13 originales. No se presenta esta inferencia como un refit nuevo de TRAIN40. Las fuentes RGB están verificadas contra los hashes de la revisión fijada de eAP. El candidato usa historias causales de observaciones del mismo objeto, con soporte total de hasta 650 ms; Garl full utiliza su pareja temporal nativa.

Durante la recuperación se detectaron 8 predicciones con SHA y CRC discrepantes entre 890 archivos previos. Se conservaron en cuarentena sus bytes y recibos originales, y se recalculan con el modelo congelado. La causa no está establecida; sus fechas coinciden con el entorno temporal de la interrupción. Ningún archivo afectado se acepta cambiando su hash esperado.

Reparación completada: 8 archivos recalculados con CRC válido. Coincidencia con todos los SHA originales esperados: True.

Auditoría de las 6.762 historias: grupos P/H/V/test disjuntos, misma identidad de objeto, disponibilidad causal y aliases con historias idénticas. Soporte sensorial máximo real en microsegundos: {'event': 600006, 'rgb': 600000}. Pesos, normalizadores y código científico original sin cambios desde la congelación. La comprobación de grupos usa las identidades de secuencia publicadas; no establece independencia visual de escenas más allá de esa partición.

La recuperación amplía el alcance del filtro de metadatos de procesos ya validado para cubrir también la admisión compartida. Conserva todos los controles, límites y cálculos científicos; `GUARDED_RESUME_EXECUTION.json` registra esta envoltura adicional. La comparación en vivo de los metadatos Python fue exacta.

La recuperación final carga sólo observaciones ya calculadas, verificando cada hash. Ejecuta el control completo de recursos cada 128 observaciones restauradas o 32 cabezas, y la caducidad en cada elemento. Se mantienen los umbrales y el límite CUDA de 1 GiB. La inicialización y el bloque de cálculo de las cabezas coinciden sintácticamente con el original.

La admisión distingue el supervisor V18 en espera de un trabajador GPU activo: exige estado reciente, código de coordinación verificado y ninguna reserva GPU abierta. Los trabajadores activos o estados ambiguos detienen la ejecución. No se modificó la campaña V18.

Las 16 consultas de recuperación seleccionadas por hash reprodujeron exactamente todos sus arrays guardados, incluidos estados intermedios y comparadores.

Estado: PACKAGED_NOT_SUBMITTED; 6762 predicciones. SHA-256 del ZIP: `60b5a620848659633b875772430cddd7d7532b0fe180ec59904d04b81a9d3ca1`. No enviado, sin score oficial.

El lector público oficial acepta los 6762 tokens y recupera exactamente las predicciones del CSV sellado. Esta prueba verifica el formato y el contenido; no ejecuta scoring ni usa etiquetas.

## Reproducción de las tablas

46778 comprobaciones escalares aprobadas sobre 13 tablas V y 34 tablas externas. Se verifican los endpoints de los 32 fits. Es un recálculo de CSV y una verificación de hashes; no se presenta como una repetición de las inferencias desde los sensores.

## Componentes positivos con los pesos del benchmark

Recálculo descriptivo posterior: `0.5 MiDc + 0.3 MiDs + 0.1 MiDl`. No es un overall_MiD ni un score oficial: falta el componente negativo, que se mantiene desconocido. No se normalizan los pesos ni se cambia la selección del candidato. Los casos TTC >10 s quedan fuera de estos tres buckets, como en las definiciones publicadas.

| Cohorte | Método | MiDc | MiDs | MiDl | Contribución positiva |
| --- | --- | --- | --- | --- | --- |
| FCWD | F_TRUE_SAME_ROUTE | 102.323 | 31.683 | 16.445 | 62.311 |
| FCWD | R_ANCHOR_EVENT | 93.304 | 52.486 | 21.738 | 64.572 |
| FCWD | public_Garl_event_lhr | 95.057 | 36.698 | 34.212 | 61.959 |
| DEV32 | F_TRUE_SAME_ROUTE | 181.582 | 56.217 | 35.636 | 111.220 |
| DEV32 | R_ANCHOR_EVENT | 185.793 | 55.989 | 34.676 | 113.161 |
| DEV32 | public_Garl_event_lhr | 287.536 | 83.127 | 46.893 | 173.396 |

En FCWD se usan 498 etiquetas en (0,10] y se excluyen 99 >10 s. En Dev32 son 926 y 20. La candidata S7 mejora MiDc frente a Garl event-only en FCWD, pero su peor MiDs impide mejorar esta contribución ponderada. Eso explica por qué ganar mean_MiD o un bucket no resuelve el ranking oficial.

## Qué falta para una afirmación de SOTA

El código oficial clasifica por `overall_MiD` ascendente: `0.5 MiDc + 0.3 MiDs + 0.1 MiDl + 0.1 MiDn`. El margen interno del 4 % no modifica esa fórmula. El mean_MiD externo sin negativos tampoco la sustituye. La fórmula se verificó contra el blob público de GitHub y el scorer local, sin abrir referencias privadas; `OFFICIAL_RANKING_PROTOCOL.json` conserva el recibo. No se ha verificado la mejor puntuación actual de la tabla en vivo.

Hay que demostrar superioridad frente a métodos comparables bajo el mismo protocolo y declarar modalidades y métricas. Ganar una métrica oficial no equivale a ganar todos los regímenes. Las réplicas completas de productores/PAIR/cabezas y la transferencia son evidencia pendiente para afirmaciones amplias. El ranking oficial test12 requiere scoring oficial: sus etiquetas TTC no son públicas. Los ZIP de formato no son mediciones de precisión. No se han enviado submissions. [Protocolo del release oficial](https://github.com/NAIL-HNU/Garl-TTC).

El registro geométrico conserva pendientes las correspondencias independientes; NCC no sustituye esa referencia. V15/V17 y los resultados históricos se preservan.
