# Lectura anticipada: prototipo aislado

El trainer primario continúa ejecutando la implementación científica de 61ee840.
`UniformPrefetch` todavía no está integrado en él. No modifica la arquitectura,
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
