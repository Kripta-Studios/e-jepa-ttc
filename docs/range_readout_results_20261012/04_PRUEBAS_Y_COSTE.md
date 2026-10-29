# Pruebas, contabilidad y coste


| fit | status | updates | sha256 |
| --- | --- | --- | --- |
| R-C05-CTX-S7 | COMPLETE | 2500 | 7b00523c4c49773f3b6b2710f47f2138df95bf8a1e13130640ea042ae2d9a954 |
| R-C05-S7 | COMPLETE | 6000 | 60caae730b57fa80c3356e2856ac165053748dcee44517d3653197a7a44fd4f1 |
| R-C20-CTX-S7 | COMPLETE | 2500 | 50834270fdccaa9286f1787d88b0cc0acf00f87df8adf0c25a7bef61e5166932 |
| R-C20-S7 | COMPLETE | 6000 | fb30a659a5d67847f45111017b50d27c0c372aec879e50756e35d403089a75b4 |
| S-M0-L0-S7 | COMPLETE | 2500 | 226d8f6c1cba5787041f125e127dae2e1b9401f730acd79fa570d31ca85db104 |
| S-M1-L0-S7 | COMPLETE | 2500 | f5ab85323b1f6276227b9d7cc3d3c3afa7295abae37d9d025c8868a6179ab9e7 |


| fit | seconds | scope |
| --- | --- | --- |
| R-C05-CTX-S7 | 263.283 | H_head_fit |
| R-C05-S7 | 320.057 | P_readout_only |
| R-C20-CTX-S7 | 272.485 | H_head_fit |
| R-C20-S7 | 347.197 | P_readout_only |
| S-M0-L0-S7 | 252.676 | H_head_fit |
| S-M1-L0-S7 | 248.535 | H_head_fit |

## Forward raw de A5


| modelo | mediana_ms | p95_ms |
| --- | --- | --- |
| R-C05-S7 | 59.5693 | 76.7993 |
| R-C20-S7 | 58.8028 | 74.871 |
| historical_A5 | 60.582 | 73.4361 |

CPU FP32, un thread, batch1 T3, misma consulta P, 3 warmups y 30 repeticiones por modelo, orden intercalado. Host compartido. Excluye lectura, preparación de voxels, C2F, PAIR, H8 e historial. No es latencia E2E.

Updates físicos superiores: 22060. Máximos: 56.000 científicos, 500 técnicos, 3.500 recuperación y 60.000 físicos.

Los tests contractuales y logs se conservan en qa/. Las pruebas de continuidad comparan 10 updates sintéticos continuos frente a 5+restore+5. S consume 20 técnicos; R conserva una primera comprobación de 20 y otra de 20 tras añadir los guards explícitos. No se utilizan datos reales para consumir el presupuesto técnico.

Las paridades raw/caché y GPU/CPU verifican outputs, pérdidas y gradientes. La tolerancia 2e-5 se fijó antes de los fits R. Los módulos congelados se comparan exactamente tras el entrenamiento sintético y cada fit R. Los contratos T2/T3 y negativos usan fixtures; no son evidencia de precisión.

TERMINAL_RESUME_PROOF.json acredita la reanudación real de los seis endpoints ya completos: cero updates añadidos y hashes de pesos, curvas, predicciones y freezes conservados. METRICS_REPRODUCTION.json recupera los bits FP32 de las predicciones serializadas y reproduce todas las métricas guardadas. LOSSES_PER_QUERY.csv separa MiD del scorer FP64 y native_aux_MiD, que conserva la aritmética del mean_MiD auxiliar histórico. No se sustituyen predicciones ni se relajan tolerancias para la selección.

Los tiempos de fit proceden de RUN_PROVENANCE e incluyen trabajo vectorial/cabeza y persistencia durante el bucle; excluyen entrenar productores y la preparación de cachés. Los COST.json de V miden sólo forward de cabeza desde caché, batch128. No son latencias E2E ni batch1. PROGRESS.json de extracción registra el tiempo de esa invocación; cuando existe resume, no equivale al total de todos los intentos.

Se preservan el piloto CPU, la pausa controlada de extracción para validar la caché por filas, los fallos de colección iniciales y el ajuste del radio del fixture sintético. No se cambiaron los radios, datos ni parámetros científicos para resolver esos tests. Los artefactos históricos siguen identificados con sus versiones de código.

Ruff/Pyright se ejecutan sobre V15. No se afirma que la suite o CI completa del histórico esté verde. V13/R1 no se modifican; los snapshots de propietarios y recursos conservan el estado observado. No se cambian drivers, paging, energía ni afinidad.

Durante la ejecución cambió la admisión Git de V13, observada y conservada en OWNER_GIT_CHANGE_OBSERVATION.json. V15 no escribió ese archivo. Los padres event y la caché H vinculados por hash se volvieron a validar; la auditoría distingue estos activos científicos de la metadata viva de otra campaña.
