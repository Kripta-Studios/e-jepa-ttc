# Diseño de recuperación del refit final

## Evidencia aislada

`operational/final_refit/checkpoint_fault_probe.py` ejecuta un optimizador pequeño
en CPU dentro de procesos reales y termina el proceso con `os._exit(91)` en siete
límites controlados: journal de commit antes/después, escritura de versión,
pointer, alias, journal durable y receipt. Tras cada interrupción, reanuda mediante
`ProducerCheckpoint` y compara exactamente pesos, optimizador, scheduler, RNG de
PyTorch/NumPy/Python, generador del sampler, curva, cursor y updates.

El ensayo no representa un corte eléctrico real. No prueba pérdida de la caché del
controlador, sectores rotos, semántica de `fsync` fuera del host ni restauración de
RNG CUDA.

## Cambio propuesto, todavía no aplicado al runtime científico

- Mantener el intervalo autorizado de 100 updates; no añadir un temporizador.
- Guardar también al final de época, pausa solicitada, pausa por recursos y endpoint.
- Mantener el pointer atómico y tres versiones inmutables verificadas por SHA-256.
- Si la versión apuntada no es válida, recorrer versiones anteriores en orden
  descendente y admitir la primera cuyo payload, identidad y hash sean válidos.
- Registrar versión elegida, motivo del fallback, updates rehechos y
  `recovery_upper`; nunca contar trabajo rehecho como progreso nuevo.
- Corregir el guard global para sumar
  `max(0, completed_updates - durable_updates)` además de `recovery_upper` y el
  update pendiente. Con checkpoints cada 100 updates, ese hueco es parte del
  máximo trabajo incierto reejecutable.
- Vincular cada archivo de curva a la versión de checkpoint que contiene el mismo
  cursor. Hoy el cursor conserva `pending_curve`, pero los JSON externos de curva
  no tienen un hash incluido en el checkpoint.
- Medir en el checkpoint real tiempo de serialización, `fsync`, hash, alias y receipt
  sólo como diagnóstico. El intervalo de 100 updates permanece fijo. El
  microbenchmark CPU no estima el coste del checkpoint RGB real.

## Migración necesaria

Modificar `operational/rgb_port/train_producers.py` cambia un source hash congelado
por `PRODUCER_PLAN.json`, la admisión eager y la identidad del checkpoint. Un run ya
iniciado no puede adoptar ese código mediante un simple reinicio: la restauración
exige igualdad de identidad completa.

La migración debe conservar el checkpoint actual, registrar los hashes de código
anterior y nuevo, demostrar que arquitectura, receta, parámetros y estado del
optimizador no cambian, y autorizar explícitamente una transición de identidad. No
debe reinicializar el modelo ni reetiquetar un checkpoint técnico como científico.

El snapshot previo al cambio debe conservar checkpoint, pointer, receipt, journal,
plan e identidad. La ruta de fallback sólo puede considerar versiones con un
sidecar creado al guardarlas que vincule nombre, SHA-256, identidad y contador de
updates; no debe aceptar archivos encontrados por glob sin esa autenticación.
