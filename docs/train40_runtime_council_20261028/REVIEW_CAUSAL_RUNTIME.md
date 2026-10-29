# Revisión independiente del runtime causal

Revisión de solo lectura realizada por `council_audit` el 28 de octubre de 2026.
Alcance: commits `dc7b74b0`, `7bc35069`, `e9a97058`, `8754148c` y `72059cf4`.
No cubre integraciones posteriores de reutilización de transporte o crops.

Veredicto: **ship**, sin hallazgos bloqueantes en ese alcance. El veredicto
evalúa el código y la honestidad de la medición; no admite el objetivo de
50 Hz sostenidos ni menos de 20 ms.

La revisión comprobó que los RAW futuros precargados permanecen privados del
feeder; copia de eventos, sellado de frames, parsing y construcción del payload
ocurren en la adquisición programada. Un hueco RAW invalida la historia antes
de publicar otro TTC. La primación precede al reset del replay, y los costes de
adquisición, preprocesamiento, inferencia y disponibilidad de resultados en
host permanecen dentro del reloj exterior.

Las claves de caché vinculan píxeles inmutables, tiempos, ROI, pesos, runtime y
forma del batch. La caché conserva únicamente features de productores; metadata
y contexto TTC se reconstruyen. El análisis de poblaciones separa respuestas
de error de TTC válidos y mantiene los gates de admisión.

Evidencia examinada:
[memory_cache_overlap1_50_v1](../../artifacts/train40_runtime_council_20261028/paced/memory_cache_overlap1_50_v1/summary.json)
y su [paridad por actualización](../../artifacts/train40_runtime_council_20261028/paced/memory_cache_overlap1_50_v1/parity.json).
El resultado permanece `FAILED`; la cobertura incompleta impide admisión aunque
los TTC efectivamente producidos pasen paridad.

El revisor no ejecutó GPU ni tests. Inspeccionó los recibos de QA existentes;
la evidencia regenerable de tests está en
[measurement_qa_v1](../../artifacts/train40_runtime_council_20261028/measurement_qa_v1/qa.json).
