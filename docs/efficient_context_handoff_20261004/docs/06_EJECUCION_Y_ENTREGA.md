# 6. Ejecución, recuperación, pruebas y entrega

## Aislamiento y recursos
Usar rama, worktree y artifacts nuevos. Leer los roles y estados locales; no asumir
que Stage70 terminó por ausencia de un PID. Fuentes históricas C:/E: en sólo lectura;
TRAIN manifestado y OLD_DEV ya expuesto son los ámbitos permitidos. No extraer ZIP
sobre un worktree activo ni actualizar su entorno global.

Límites: 2 GiB de RAM disponible, 4 GiB de RSS del árbol, 10 GB decimales libres tras
reservas y 10 GB de artefactos nuevos como máximo. Cuatro threads numéricos, dos
interop. Registrar también memoria comprometida en Windows. GPU de 12 GB es el hardware,
no una reserva garantizada: perfilar picos y respetar su propietario. No matar otros
procesos, reiniciar el equipo, modificar pagefile ni desactivar seguridad para pasar
una admisión. Nuevos permisos de esta campaña no modifican la historia de permisos.

## Separar raíces de lectura y escritura
Un worktree nuevo no contiene automáticamente los artifacts locales de T0/T6 ni
los productores originales. Registrar explícitamente `code_root`,
`historical_artifact_root`, `raw_train_root` y `new_output_root` como rutas absolutas.
Resolverlas desde los manifiestos existentes. No duplicar todos los artifacts ni
cambiar sus bindings para simular que se generaron en el worktree nuevo.

El runner histórico contiene constantes ROOT/OUT/NIGHT. No ejecutarlo sin adaptar
su interfaz, porque podría escribir intentos sobre la campaña cerrada. Crear una
interfaz de referencia que reciba raíces de sólo lectura y una raíz de resultados
nueva, conservando sus operaciones numéricas. Cualquier extracción de funciones
comunes necesita tests de paridad. No parchear archivos ni locks del worktree ajeno
para hacer funcionar la comparación. Reutilizar el intérprete compatible o un
entorno aislado; registrar su identidad, sin actualizar globalmente dependencias.

## Orquestación
Un líder gestiona contratos y cola; un implementador puede trabajar en el runtime;
un revisor independiente comprueba paridad y tests. Como máximo dos workers ligeros
a la vez dentro del presupuesto total, un único trainer pesado y un escritor por
output. No es necesario construir otro MCP o sistema de orquestación completo.
Utilizar Git, pytest y los journals ya disponibles. No encargar al revisor una
búsqueda abierta de arquitecturas.

## Estados y reanudación
PENDING, IMPLEMENTING, TESTED, FROZEN, RUNNING, PAUSED_RESOURCE, COMPLETE,
BLOCKED_DEPENDENCY, FAILED_INTEGRITY, NEGATIVE e INCONCLUSIVE. Un fallo de admisión
no es un negativo científico. Registrar IDs y presupuesto antes del primer update.
Checkpoint completo: pesos, optimizador, scheduler, RNG, sampler, contador y bindings.
Persistir bootstrap y publicación por fragmentos, no sólo un heartbeat.

No volver a cerrar T6 por un recibo incidental. Ante un problema transitorio,
registrar el path y los recursos, esperar con backoff ligero y avanzar una rama
independiente viable. Ante un hash cambiado, bloquear lo afectado sin reinterpretarlo.

Un nombre de métrica o metadata no equivale automáticamente a fuga de targets.
Registrar contenido visible y alcance del acceso; tampoco reclasificar una exposición
ambigua como metadata sin pruebas. Si información excluida pudo guiar decisiones,
separar desarrollo de confirmación. Un checkout nuevo no deshace lo que se aprendió.

## Pruebas y alcance
TDD para límites, máscaras, exclusión de productores, invalidación de caché, anchors,
cold starts, gaps, count/rate, padding, timestamps y recuperación. Ejecutar pytest
focalizado, Ruff, formato y tipado sobre cambios; ampliar la suite cuando haya
recursos. Registrar los fallos legacy sin ocultarlos ni usarlos para cancelar tareas
independientes. No ejecutar optimizadores en tests reales fuera de la cola autorizada;
las pruebas sintéticas tienen su contabilidad separada.

Validación fuerte de fuentes al admitir, en boundaries y al publicar, con detección
barata de cambios entre medias. No rehashear streams gigantes por cada draw. Tampoco
eliminar procedencia para acelerar. Un hash no sustituye a comprobar query mappings.

## CLI que debe implementar el agente
```text
python -m operational.efficient_context.run status --protocol <p>
python -m operational.efficient_context.run all --protocol <p> --resume
python -m operational.efficient_context.run package --protocol <p>
```
Este módulo NO está incluido como trainer integrado. Implementarlo, probar su help
y registrar los comandos reales antes de usarlo. No invocar launchers antiguos que
reabran T6. Después de QA y freeze, ejecutar las tareas viables; no regresar únicamente
con código o instrucciones. Si el cliente termina la sesión, dejar recuperación
concreta, sin prometer trabajo futuro en background.

## Entregables
`FINAL_REPORT.md`, `NEXT_DECISION.json`, `SOURCE_ADMISSION.json`,
`EXPERIMENT_MANIFEST.json`, `ACCOUNTING.json`, `INPUT_OUTPUT_PARITY.json`,
`PROFILE_COMPONENTS.csv`, `RUNTIME_COMPARISON.csv`, `H8_WIDE_RESULTS.json`, `EWMA_TRANSPORT_CV_RESULTS.json`,
`GARL_COMPARISON.json`, `COMPARATOR_CONTRACTS.json`, `TTC_TARGET_SEMANTICS.md`,
`TEST_RESULTS/`, `COMMAND_LOG.jsonl`, `STAGE70_INTERFACE_STATUS.md`,
`RESUME_PROOF.json` e `INFERENCE_INTERFACE.md`.

Separar ejecución, utilidad, mecanismo, rendimiento y confirmación. Un ámbito no
ejecutado se entrega con estado y causa, no con cifras estimadas. Bundle esencial
con código/configs/diffs, pesos nuevos, predicciones, curvas, normalizadores, inputs
incluidos o bindings externos, draws y recibos. No copiar 350 MB históricos a cada
entrega: referenciarlos por hash. Regenerar desde una extracción independiente dentro
del alcance incluido. No prometer wallclock idéntico ni raw autónomo si no está incluido.

Verificar CRC, manifiesto y SHA externo. No push, releases o submissions automáticos.
Commits selectivos de cambios propios; nunca borrar trabajo ajeno para presentar
un git status limpio.
