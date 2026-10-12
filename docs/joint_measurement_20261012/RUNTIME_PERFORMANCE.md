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
