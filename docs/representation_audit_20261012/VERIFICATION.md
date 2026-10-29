# Verificación y límites de reproducción

Las comprobaciones científicas y las comprobaciones del paquete responden a preguntas distintas. Ninguna prueba de software convierte un candidato rechazado en una mejora científica.

## Código y operaciones autorizadas

`TESTS_FINAL_28.log` conserva las 28 pruebas CPU aprobadas de representación, precisión, diferencias temporales, geometría, calibración temporal/espacial, clipping, lectura no lineal y verificación de ZIP. Incluyen la identidad del operador inicial, las rutas sin targets de inferencia, las particiones de parámetros entrenables y paridad frente a operaciones nativas. Las pruebas CPU no realizan updates de optimizador. Ruff y Pyright se conservan en `RUFF_FINAL.log` y `PYRIGHT_FINAL.log`; Pyright comprueba el ámbito V17 indicado en su ejecución, no todas las líneas históricas del repositorio.

`ENDPOINT_AUDIT.json` vuelve a validar los padres admitidos, el hash de cada checkpoint final y su versión durable, los endpoints fijos, la finitud de los tensores, las identidades y hashes de fuentes ejecutadas. Para cada readout compara exactamente los tensores originales A5 fuera del scorer: encoder, foreground, geometría nativa, transporte e incertidumbre permanecen intactos. Los readouts nuevos agregan operadores externos declarados; no se describen como pesos nativos intercambiables sin esos operadores.

## Tres alcances de continuidad

1. Las pruebas técnicas GPU de `qa/*PASS.json` comparan 10 updates continuos con 5 + guardado + restauración + 5. Verifican pesos, optimizador, scheduler, RNG, muestras y curvas conforme al trainer. Esos updates sintéticos están contabilizados; no son experimentos científicos.
2. `SAMPLING_CONTINUITY_AUDIT.json` reproduce cada extracción del sampler de todos los fits desde su semilla, comprueba su RNG final y los relojes de optimizador/scheduler. Los checkpoints de pausa SP0/SP1 retenidos en `resume_audit/` permiten comprobar que sus prefijos de muestras y curvas coinciden con los endpoints posteriores. La pausa antigua de N0 no tiene un checkpoint adicional retenido: el escritor conserva sólo sus últimas dos versiones. No se afirma haber reentrenado cada trayectoria científica completa sin interrupciones para comparar sus pesos.
3. `TERMINAL_RESUME_PROOF.json` ejecuta el comando real `Resume-ResearchV17.ps1 -Queue all` sobre la campaña terminada. Compara todos los hashes finales y la contabilidad antes/después. La comprobación exige cero nuevos updates; las pasadas de QA de caché que no entrenan sí liquidan su tiempo dentro del presupuesto GPU.

La pausa por memoria comprometida y su resolución permanecen en `RESOURCE_DEPENDENCY.json` y `RESOURCE_RESOLVED.json`. No se relajó el guard ni se cambió paging, drivers o configuración global.

## Pesos, predicciones y costes

`PORTABLE_EXPORT_VERIFICATION.json` comprueba la inferencia CPU de cada export con ejemplos preparados incluidos. Un audit hook impide abrir los padres o cachés originales V13/V15. El export NLH es un estado intermedio de geometría, con scorer reiniciado sin entrenar; NL1 es el endpoint final correspondiente.

`BUNDLE_METRIC_REPLAY.json` recalcula exactamente las métricas de las tablas de predicción. No ejecuta los productores ni crea nuevas inferencias. Su cuenta incluye tablas de controles reutilizados y no debe interpretarse como cantidad de escenas independientes. Las filas inválidas se conservan y los agregados no admitidos siguen ausentes.

El perfil CPU excluye encoder, datos crudos y preparación de features; el presupuesto GPU mide tiempo de pared dentro de scopes declarados, incluyendo su I/O. Son mediciones sobre un equipo compartido. Las figuras se regeneraron y revisaron visualmente; sus curvas usan escalas propias porque las pérdidas de objetivos distintos no son comparables como una sola métrica.

## Bundle

`verify_v17_bundle.py` comprueba CRC, SHA-256 exterior, hashes y tamaños de cada miembro, cobertura del manifiesto, duplicados y rutas inseguras. La verificación exterior queda junto al ZIP. La reproducción desde una extracción independiente se documenta también junto al paquete para evitar que una copia de trabajo enmascare una dependencia ausente.

El paquete incluye el entorno observado y todos los resultados positivos y negativos. Excluye datasets crudos y grandes cachés, enumerados con hashes en su inventario de dependencias. No se ha probado una instalación limpia en otra máquina ni se garantiza igualdad bit a bit con otro hardware o versiones. V sigue siendo desarrollo expuesto; este bundle no acredita SOTA ni confirmación externa ciega.
