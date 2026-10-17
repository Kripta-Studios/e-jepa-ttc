# V19: latencia desde P2 guardado

Medición V5 completada el 17 de octubre de 2026, con 128 consultas P fijas del primer pack, endpoints L0/LW congelados, CUDA FP32 sin TF32 y 20 calentamientos por endpoint y ruta. Incluye decodificación del P2 comprimido, colación, transferencia a GPU y forward sincronizado hasta TTC. No incluye construir P2 desde eventos crudos ni ejecutar un sistema de dos expertos con router.

| Ruta | Endpoint | Mediana | p95 |
|---|---|---:|---:|
| Batch 4, lote completo | L0 | 36,847 ms | 47,533 ms |
| Batch 4, lote completo | LW | 37,120 ms | 46,519 ms |
| Batch 4, amortizado por consulta | L0 | 9,212 ms | 11,883 ms |
| Batch 4, amortizado por consulta | LW | 9,280 ms | 11,630 ms |
| Batch 1, descriptivo | L0 | 20,066 ms | 26,773 ms |
| Batch 1, descriptivo | LW | 20,076 ms | 27,163 ms |

La amortización divide el coste del lote entre cuatro; no representa la latencia de una petición individual ni incluye espera para formar lotes. Las mediciones comparten los dos modelos residentes. El orden L0/LW por consulta está emparejado y registrado; el orden batch4/batch1 y la carga cold son fijos. La caché del sistema operativo no se controla. No se infiere una ventaja de velocidad de un objetivo de pérdida frente al otro ni una mejora respecto al pipeline sensorial histórico V18.

Batch 4 reproduce exactamente fase raw y TTC guardados en las 128 consultas de ambos endpoints. La fase emitida derivada también es exacta respecto a la misma conversión CUDA del TTC guardado, calculada fuera del tramo medido. Batch 1 conserva signo y estado de saturación; sus máximos errores TTC son 0,000030518 s en L0 y 0,000415802 s en LW. Su paridad es descriptiva, no bit a bit.

Se conservan los intentos anteriores. V3 localizó una deriva determinista por tamaño de batch. V4 conservó 42 filas antes de fallar una comparación entre la conversión de fase CPU y CUDA. El diagnóstico dirigido mostró que raw y TTC eran exactos y la fase derivada LW difería en un ULP. V5 compara esa fase en el mismo backend sin ampliar las tolerancias raw/TTC. Los recibos de fallo y sus hashes permanecen junto a los resultados originales.

La medición no ejecutó backward ni optimizer updates. Los checkpoints son idénticos antes y después. El replay final validó las 128 filas, referencias CUDA, cold, resumen, fuentes y recibos sin ejecutar GPU. Los ajustes científicos, población y valores emitidos no se modificaron para medir latencia.

Resultado: `artifacts/decision_recovery_20261017/cost/batch4_native_and_batch1_descriptive_P2_to_TTC_v5/attempts/8e52692797e83790/RESULT.json`.

SHA256: `56bf767122a81a9b198319bf2655fea38ba94429dc9173bd640ece93628d0fd5`.

Las mejoras previas del tramo de entrenamiento preparado con CUDA graphs están documentadas aparte en [RUNTIME_ADMISSION.md](RUNTIME_ADMISSION.md); no equivalen a una aceleración global desde sensores.
