# Verificación preliminar de la entrega compacta

El ZIP final sigue pendiente de completar B1 y cerrar las decisiones. No se ha creado otra entrega provisional ni un ZIP mayor de 100 MB.

La copia de prueba de los resultados cerrados contiene 40 tablas y 1,065,690 filas. Todos sus escalares contrastados se recalculan con NumPy. Los 18 exports se cargan en CPU y reproducen sus ejemplos con error máximo de 0 s en este entorno. Esto no reproduce sensores crudos ni reentrena los productores.

Recibo local: `verification/COMPACT_CURRENT_WITH_OMISSIONS_PREFLIGHT_20261016.json`, SHA-256 `eace5e7e4de013e022a2cd1902d7eda9ea7f61b5c14c1473f4e06459707daa62`.

El verificador distingue la integridad del paquete, el replay de predicciones/exports y las pruebas que requieren checkpoints de entrenamiento omitidos. Sólo acepta una ausencia cuando el manifiesto la declara; las ausencias no declaradas fallan. Tres pruebas unitarias y el replay de una copia real cubren esa distinción. Ruff y Pyright pasan.

Pruebas no repetidas desde esta copia compacta:

- `PAIR-CORR-FROZEN-S7@100`: 2 payloads de checkpoint omitidos; los recibos originales se conservan.
- `PAIR-CORR-S7@100`: 2 payloads de checkpoint omitidos; los recibos originales se conservan.
- `PAIR-DIRECT-S7@100`: 2 payloads de checkpoint omitidos; los recibos originales se conservan.
- `native_cuda_graph_resume`: 2 payloads de checkpoint omitidos; los recibos originales se conservan.

La prueba del ZIP histórico de 14 de octubre detecta que conserva metadatos de exports pero no sus pesos y ejemplos binarios. Ese archivo se conserva como entrega histórica, sin presentarlo como paquete portable completo. El empaquetador actual incluye pesos y ejemplos de todos los exports; la verificación final se repetirá desde una extracción independiente del ZIP publicado.

El límite de publicación es 100.000.000 bytes. Los estados completos de optimizador, sensores, cachés y padres históricos permanecen locales y se declaran en el inventario de omisiones.

La revisión del empaquetador detectó que la selección de archivos de texto omitía
la extensión `.jsonl`. Se ha corregido para incluir el registro central de
experimentos. La comprobación sobre la campaña real conserva sus 18 entradas y
17.662 bytes, sin modificar el registro, y confirma la inclusión de los registros
originales de fusión y réplica. El [recibo de inclusión](evidence_delivery_20261016/COMPACT_REGISTRY_INCLUSION.json),
también conservado en `verification/COMPACT_REGISTRY_INCLUSION_20261016.json`, registra
`PASS_SELECTION_ONLY`: comprueba qué archivos recogerá el empaquetador; todavía
no prueba un ZIP final ni sustituye su extracción y replay independientes.
El SHA-256 del registro conservado es
`2c552dabb8e91cd21580484055d193e1430b6ef2ab95a15d8bce89b1310a8692`.
