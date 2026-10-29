# Rendimiento y continuidad del runtime V18

# Runtime optimizado admitido el 13 de octubre

Tras la autorización del usuario para acelerar el entrenamiento activo, NATIVE
continuó desde el update durable 6.131 mediante `fast_run`. Su checkpoint protegido
es `runtime/optimization_20261013/ORIGIN_006131.pt`, SHA-256
`43f8b4fb26516dd5e4292b535a95bc27ed7d1deafedc648cbe14c6a212ac5282`.
Los otros tres endpoints primarios ya estaban completos y no se reentrenaron.

Se activaron CUDA Graphs mediante el backend `cudagraphs`, una caché de entradas
sensoriales de hasta 8 GiB, prefetch de un batch en CPU, memoria host fijada y
transferencia agrupada de escalares de pérdida. Los 211 packs verificados pueden
permanecer abiertos entre scopes; sus fingerprints se revisan incluso en aciertos
de RAM. Las entradas conservan sus valores originales. No hay compresión con
pérdida, AMP, TF32, caché de embeddings detached ni avance anticipado del sampler
autoritativo. Los guards de RAM y disco siguen vigentes.

El código científico registrado, FP32, batch, objetivo, secuencia de muestras y
optimizador se conservan. La implementación de ejecución sí cambia y está fijada
por `RUNTIME_BINDING.json`; no se reescribe la identidad inicial del productor.
`Resume-JointMeasurement.ps1` selecciona esa ruta cuando existe su binding.

La comparación estricta inicial frente a eager **falló**: el probe de backward
observó una diferencia máxima de gradiente de `1.1920928955078125e-7`.
`GRAPH_PROBE.json` y `ADMISSION.json` conservan esos negativos. La política
numérica posterior se registró antes de sus updates de diagnóstico. Pesos y
estado de Adam cumplen `rtol=1e-5, atol=1e-6`; los IDs y RNG son exactos. Las
diferencias de norma de gradiente se preservan y se comprueba la misma decisión
de clipping. No se afirma equivalencia bit a bit entre eager y CUDA Graphs.

La reanudación **dentro del runtime optimizado** sí resultó exacta en los nueve
estados: modelo, Adam, scheduler, sampler, RNG de PyTorch CPU/CUDA, NumPy,
Python y cursor. Se compararon seis updates continuos con un prefijo compartido
de cuatro y dos reanudados: ocho updates físicos, seis lógicos. Este testigo,
junto a los 32 updates del contraste eager/graph, suma 40 updates reales P de
diagnóstico nuevos. No se cuentan como progreso del endpoint primario.

`ACTIVATION_VERIFIED.json` comprueba que los 6.131 registros anteriores y sus
IDs se preservaron exactamente hasta el primer checkpoint optimizado auditado,
6.200, y que los 59 estados de Adam avanzaron 69 pasos sin reinicio ni recuperación.
La verificación definitiva de 12.000 pasó a las 21:00 UTC del 13 de octubre:
`verification/NATIVE_ENDPOINT.json` confirma el mismo prefijo completo, los 59
estados de Adam en 12.000, los 18 tensores del encoder adaptados y cero recuperación
nativa. SHA-256 del endpoint:
`7fee4171269785893bdbdd5e04602d01be8552215a59a1ef7a9a26653253ef3d`.

La observación `verification/HOURLY_20261013T1800.json` midió 1.662 nuevos updates
en 3.339 segundos de pared desde la restauración: aproximadamente 1.792 updates/h.
Incluye los guardados y reinicios de scope intermedios. Es una observación de
producción en ese intervalo, no un benchmark controlado ni coste completo de
la campaña. El diagnóstico frío previo midió un ratio graph/eager de 0,652;
no se mezclan esos dos alcances para atribuir causalmente toda la aceleración.

El replay del bundle ahora recalcula la igualdad de los nueve estados desde
los checkpoints retenidos. Una extracción separada de 23 archivos pasó con cero
updates y CUDA sin inicializar. El ZIP de ese testigo ocupa 9.606.242 bytes,
SHA-256 `9f1692025d083f908b5678815e7a17a5c6b1b1348759a5234b612df3e200dc54`;
está en `verification/OPTIMIZED_RUNTIME_WITNESS_20261013.zip`, junto a sus recibos.
Es un testigo parcial, no la entrega final de V18. Pasaron 32 pruebas CPU de
empaquetado, replay y rechazo de estados alterados.

La primera extracción de ese testigo se creó dentro de la carpeta de campaña;
el contador recursivo sumó otra vez sus journals de seis y dos updates. Se detectó
en la revisión de las 22:00 y se trasladó intacta a
`artifacts/independent_audits/optimized_runtime_20261013`, fuera del árbol contado.
`verification/EXTRACTION_ACCOUNTING_CORRECTION_20261013.json` conserva las rutas,
los hashes y el agregado anterior. Son ocho registros duplicados, no ocho updates
físicos nuevos. No se modificaron los journals originales ni el ledger GPU y se
conservaron las observaciones históricas con ese exceso conservador.

## Histórico: prototipo aislado de lectura anticipada

En el diagnóstico anterior, el trainer ejecutaba la implementación científica de 61ee840.
`UniformPrefetch` no estaba integrado en él. El prototipo no modificaba la arquitectura,
pérdida, optimizador, precisión, batch, sampler ni endpoint de ningún fit activo.

Las curvas durables de los primeros 764 updates nativos muestran medias de
aproximadamente 2,30–4,09 segundos por update en bloques de cien. Incluyen carga,
transferencia, forward/backward y optimización, bajo trabajo concurrente de
preparación sensorial. No se comparan esos tiempos directamente con el preflight
de forward/backward sin optimizador, ni se atribuye toda la diferencia al loader.

El prototipo carga como máximo un batch futuro en un thread propio. Predice sus
índices con una copia del generador y sólo avanza el sampler autoritativo cuando
se consume ese batch. Los índices no consumidos desaparecen al cerrar el loader.
El checkpoint mantiene así el mismo cursor que la ejecución secuencial.

Dos pruebas CPU verifican orden, duplicados, estado del generador y reanudación
5+5 frente a diez consumos continuos. Una comprobación adicional sobre ocho
batches P reales de 32 consultas verifica todos los campos bit a bit, los estados
del sampler y que el RNG global de PyTorch no cambia. Son cero updates de
optimizador. Recibo: `diagnostics/PREFETCH_P_PARITY.json`.

Esto todavía no demuestra una mejora de rendimiento ni reanudación CUDA del
trainer integrado. Una eventual adopción debe medir el beneficio, conservar los
checkpoints existentes y registrar explícitamente la transición de código, con
paridad de entrenamiento y reanudación. No cambiar silenciosamente los hashes
del productor activo ni reescribir su genealogía inicial para instalar el prototipo.

## Diagnóstico real P cerrado a las 17:53 UTC

Se registraron y ejecutaron 80 updates científicos de diagnóstico, separados de
los endpoints primarios: diez seriales frente a cinco con prefetch, checkpoint y
cinco reanudados, en cada uno de los cuatro brazos. Los pesos, optimizador,
scheduler, RNG, sampler, pérdidas y cursor coinciden exactamente en los cuatro.

| Brazo | Serial, s/update | Prefetch, s/update | Ratio |
| --- | ---: | ---: | ---: |
| NATIVE-P2-FT | 1,11815 | 1,10495 | 0,98819 |
| PAIR-DIRECT | 0,41772 | 0,35731 | 0,85538 |
| PAIR-CORR | 0,50314 | 0,41516 | 0,82514 |
| PAIR-CORR-FROZEN | 0,40863 | 0,33079 | 0,80950 |

Son diez pasos medidos por ejecución, después de admisión de páginas sensoriales;
no son una medida de coste end-to-end ni una estimación precisa de toda la campaña.
Se conservan las mediciones individuales, incluidas las fronteras de reanudación.

La regla registrada requería al menos un 10 % de mejora en el modelo nativo.
No se cumple. Decisión: **DO_NOT_INSTALL**. Las mejoras de los otros brazos se
conservan sin cambiar retrospectivamente la regla. La matriz primaria reanuda
desde el update 1.358 con su implementación científica intacta, sin migración de
pesos o identidad. Los 80 updates del diagnóstico cuentan en el techo científico.

Evidencia regenerable: `diagnostics/prefetch_runtime/REGISTRATION.json`,
`COMPLETE.json`, `DECISION.json` y los CSV `TIMING_TO_*.csv` de cada ejecución.
