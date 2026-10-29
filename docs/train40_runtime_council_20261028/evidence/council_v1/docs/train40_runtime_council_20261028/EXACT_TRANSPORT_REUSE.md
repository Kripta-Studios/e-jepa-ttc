La ruta opt-in `reverse-summaries` reutiliza trabajo idéntico del transporte
conservando las expresiones ATen originales. No cambia pesos, normalizadores,
contexto TTC, batch de productores ni fuentes científicas. No usa Inductor,
reentrenamiento ni actualizaciones EMA. El default permanece `original`.

La evidencia previa está en
`artifacts/train40_runtime_council_20261028/producers/exact_transport_sensitive_v1/summary.json`:
462/462 llamadas explícitas de los índices 0, 137, 488, 505, 514, 533 y 553
pasaron la cobertura ordenada, el gate TTC original `atol=rtol=1e-5` y la
igualdad bitexact de las cuatro características frente al core original con
el mismo batch y reutilización de encoders. El recibo confirma pesos,
normalizadores y 599 archivos preservados. Incluye las primeras llamadas,
warmups y repeticiones medidas para original, reverse y reverse-summaries,
cada uno en ejecución serial y paralela.

Ese experimento sólo mide productores sobre entradas GPU ya preparadas.
Preprocesamiento RAW, transferencias y contexto quedan fuera del reloj; la
comparación por llamada se intercala fuera del reloj. Por tanto todavía no
demuestra actualizaciones completas por debajo de 20 ms ni 50 Hz sostenidos,
y no admite dos objetos. Los percentiles y muestras están en el JSON, sin
convertir el inverso de la media en throughput sostenido.

`ProducerRuntime(transport_reuse=...)` admite tres valores:

- `original`: conserva la mapping de modelos y comportamiento anterior.
- `reverse`: aplica `attach_transport_pair_core(reuse_reverse=True)`.
- `reverse-summaries`: aplica
  `attach_transport_reuse(history_group_size=None, summaries=True)`.

Las dos opciones construyen vistas privadas de modelos frozen/eval que
comparten los mismos objetos Parameter. El adaptador se instala antes de
la reutilización de encoders realizada por `ExpertBlockTensorCore`. Se
conserva E OFF/R ON en el servicio y no se deduplican pares temporales del
transporte. Los caches de correlación/resúmenes sólo viven durante un
forward; no reutilizan predicciones o contexto entre consultas. JIT mantiene
`optimized_execution(False)` y las mismas comprobaciones de captura/guards.

`MultiObjectTrain40Service` propaga la opción y exige que un runtime inyectado
declare el mismo modo. Continúa rechazando la combinación de runtime
inyectado y caché RGB sin contrato de backend separado. Una nueva instancia
del servicio sigue vaciando la caché del lifetime anterior.

El siguiente ensayo paced puede añadir
`--transport-reuse reverse-summaries` al comando reproducible de la variante
memory/priming/cache/overlap. Su binding y el hash de runtime de la caché
incluyen explícitamente el modo, el descriptor de reutilización y las fuentes
de ambos adaptadores. `startup.json` conserva además los bindings reales de
las vistas instaladas. No cambia ningún gate de latencia, causalidad, cobertura
o paridad del ensayo sostenido.

Corrección de la auditoría del driver anterior:
`benchmark_train40_producer_variants.py` sólo exigía paridad TTC de la última
repetición; primeras llamadas y warmups no tenían ese gate. No corresponde
atribuirle un defecto confirmado de alias con los buffers CUDA Graph:
`ProducerRuntime._enqueue_block` ya clona sus salidas. El nuevo driver añade
una copia explícita por llamada, consumida inmediatamente fuera del reloj,
para mantener memoria acotada y propiedad independiente del runtime. El
defecto confirmado era cobertura incompleta del gate, no alias demostrado.

QA de la integración: 28/28 casos nuevos pasan en CPU con OMP/MKL=2;
Ruff y Pyright no detectan problemas en los tres archivos de producción ni
en el test nuevo. La regresión de siete módulos ejecutó 220 casos: 217
pasaron, uno omitió GPU y dos resultaron `XPASS(strict)` en las pruebas
antiguas de deduplicación temporal activada. Estos dos casos se reprodujeron
aisladamente, importando `transport.py`, `causal_scale_ttc.py`,
`local_transport.py` y su test, todos idénticos a HEAD. La suite amplia
devuelve fallo por esos XPASS; se conservan sus marcas estrictas y no se
consideran admisión de deduplicación. La nueva ruta siempre usa
`history_group_size=None`. No se ejecutó GPU durante esta QA.
