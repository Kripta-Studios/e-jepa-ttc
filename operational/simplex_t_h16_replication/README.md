# Campaña independiente H16

Este directorio añade una ejecución separada; no modifica código científico,
grafo, freeze, autorización, artefactos ni candidato de T6. No ejecuta Stage70–76.

Orden: `static_audit.py`, `prepare.py`, `register.py`, `train.py` y `analyze.py`.
Todos los entrypoints usan el Python histórico del repositorio vecino ya existente.
No requieren instalar dependencias ni integrar cambios remotos. Sólo `train.py`
puede actualizar optimizador, exclusivamente en los seis IDs autorizados.

`prepare.py` verifica código histórico, identidades TRAIN/OLD_DEV, normalizadores y
controles; no entrena. `register.py` requiere ese preflight completo y publica
el nuevo protocolo con su SHA antes de los updates. No usar sólo el audit estático
como permiso científico. No cambiar código/input/protocolo después del registro.

`train.py` conserva `training.fit` sin editar. El primer fit pausa tras100 y se
reanuda en otra invocación desde101; no hay prueba de entrenamiento adicional.
Su checkpoint completo y prueba sin updates quedan en verification/. Los seis
endpoints2500 se sellan conjuntamente antes de evaluación. Un endpoint válido
ya sellado no se vuelve a entrenar. Cada reserva100 se contabiliza duraderamente.

`analyze.py` aplica inferencia en fragmentos128 y bootstrap en fragmentos256,
con hashes/recibos y recuperación; guarda las pérdidas emparejadas, tablas y
guardrails. Usa los draws históricos en lectura, no vuelve a ejecutar T6.

`deliver.py` empaqueta la entrega y ejecuta `regenerate.py` desde una extracción
independiente, usando sólo los bytes incluidos. La verificación real de ejecución,
análisis y entrega queda registrada conforme se completa cada fase; el código y
los tests operativos por sí solos no son resultados científicos.

Los tests operativos no importan Torch ni ejecutan optimizer. Los límites son
RAM≥2GiB, RSS≤4GiB, disco libre-reserva2GB≥10GB; se añade y declara previamente
Windows commit headroom≥2GiB. No se altera la receta para sortear recursos.
