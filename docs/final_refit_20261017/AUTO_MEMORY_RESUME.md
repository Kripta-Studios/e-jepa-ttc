# Reanudación automática por recursos — 18 de octubre de 2026

El usuario autorizó reanudar automáticamente los productores cuando vuelvan
a estar disponibles los márgenes de RAM física y memoria comprometida. El
lanzador anterior declaraba `automatic_restart: false`; no reanudaba por sí solo.

Implementación: `operational/final_refit/auto_resume.py`. Es un supervisor externo
al runtime V6. Conserva sus 31 fuentes vinculadas, las recetas, checkpoints,
optimizador, scheduler y RNG. No añade updates ni altera el endpoint de 49.932.

## Política activa

- Un único propietario usa el mismo mutex global que el lanzador anterior.
- Mantiene un publisher compartido y supervisa cada productor independientemente.
- Comprueba recursos cada 10 segundos y exige 30 segundos de observaciones válidas
  con RAM física disponible y margen comprometido de Windows de al menos 2 GiB.
- El inventario firmado, TTL, identidad del publisher y límites de disco/RSS
  siguen siendo obligatorios; memoria suficiente por sí sola no anula otros guardas.
- Escalona la carga de procesos: arranca uno y espera a que complete un update
  antes de permitir el arranque del otro. Después entrenan en paralelo.
- Sólo reintenta una salida 3 con checkpoint completo: journal y receipt deben
  coincidir, sin updates pendientes, y la pausa debe corresponder al mismo update.
  Admite memoria insuficiente con inventario válido, o inventario caducado/publisher
  ausente con vinculación correcta al runtime. Antes de arrancar exige de nuevo
  inventario firmado válido y 30 segundos de recursos estables.
- Una indisponibilidad del publisher ya no convierte una pausa guardada en error
  definitivo. Si el publisher muere, o permanece caducado durante 60 segundos,
  sólo se reemplaza cuando todos los entrenadores han salido y guardado. Se
  reinicia el contador de secuencia del lector. Máximo tres reemplazos sin recuperar
  30 segundos de recursos válidos; no se amplía el TTL ni se omiten firmas.
- Tras una nueva pausa espera 60, 120, 240, 480 y después hasta 600 segundos entre
  intentos del mismo productor. Se exige también estabilidad de los recursos.
- No reintenta errores de código, pérdida no finita, integridad, I/O o fallos de
  guardado. En particular, un `MemoryError` durante un checkpoint requiere revisión.
- Un `STOP_REQUEST` manual se conserva e impide reanudar ese productor.
- Una nueva autorización explícita puede ejecutarse con `--resume-manual`.
  Esta opción exige para cada fit pausado estado previo `MANUAL_STOP`, salida 3, evidencia
  coherente y completa, hashes de checkpoints y de las fuentes ejecutadas válidos.
  Archiva las marcas únicamente tras validar ambos fits y registra un receipt
  `MANUAL_RESUME_*.json`. Después vuelve a exigir 30 segundos de recursos válidos;
  no habilita reanudar automáticamente futuras pausas manuales.
  Si el otro fit ya está `COMPLETE`, valida su endpoint y checkpoint y lo conserva
  completado, sin volver a lanzarlo. Esta transición se verificó para reanudar
  R_C2F desde 39.969 updates con R_A5 terminado en 49.932 el 19 de octubre.
- Para detener toda la supervisión, crear `AUTO_RESUME_STOP` en el directorio del
  runtime. El supervisor pide guardado a los productores, espera su salida y
  después cierra únicamente su publisher. No termina a la fuerza los entrenadores.
- Los checkpoints siguen guardándose cada 100 updates y al pausar limpiamente.

## Estado y alcance

`artifacts/final_refit_20261017/ACTIVE_RUNTIME.json` apunta al receipt
`LAUNCHER_AUTO_*.json` actual. Este contiene heartbeat, PID, estados por productor,
logs de cada intento, contadores de reintentos y recursos observados. Los eventos
de salida y arranque quedan como `STATUS_AUTO_*.json` sin sobrescribir los anteriores.

La automatización depende de que el supervisor y Windows sigan ejecutándose.
No se ha instalado ninguna tarea de inicio, ni cambiado el pagefile, límites
globales de Windows o procesos ajenos. Un desplome abrupto de memoria aún puede
impedir guardar; el supervisor no considera confirmado un checkpoint incompleto.

## Validación

- 71 pruebas de política de reanudación y recursos compartidos aprobadas.
- Incluyen pausa manual, errores de checkpoint, evidencia antigua, falta de
  inventario válido, ambos límites, estabilidad, backoff y rechazo de duplicados.
- Ruff y Pyright: sin errores.
- Los 31 hashes de fuentes ejecutadas coinciden con `PRODUCER_PLAN.json` V6.
- Incluyen caducidad recuperable, rechazo de firmas/identidades incorrectas,
  reemplazo acotado del publisher y prohibición de reemplazarlo con entrenadores vivos.
- Evidencia técnica actual: `artifacts/final_refit_20261019_inventory_recovery/TESTS.xml`.
- La autorización posterior a la pausa manual restaura R_A5 desde 42.517 updates
  y R_C2F desde 25.436. Ambos hashes de checkpoint se verificaron antes de arrancar.
- Preflight: las pausas de R_A5 (42.507) y R_C2F (25.364) tenían esos mismos
  updates guardados, sin pendientes; ambas se debían a inventario caducado.

La activación y avance reales se comprueban aparte en los receipts y journals;
las pruebas de política no son entrenamientos científicos adicionales.

## Recuperación ante inventario incompleto por AccessDenied

La pausa del 19 de octubre a las 18:42 locales conservó los 46.093 updates de
R_C2F. Un proceso no inspeccionable (`AccessDenied`, RSS sin cota) invalidó la
cobertura del inventario, aunque había memoria disponible. El supervisor anterior
clasificó esa pausa limpia como error definitivo.

Ahora una pausa con checkpoint completo y coincidente, snapshot autenticado y
vigente, binding correcto, sin error global de scan y únicamente `AccessDenied`
en los procesos sin cota pasa a `WAITING_MONITOR`. El publisher sano sigue
escaneando; no se consume su presupuesto de reinicios por esa falta de cobertura.
El entrenamiento solo vuelve cuando el inventario recupera cobertura y todos los
controles permanecen válidos durante 30 segundos. Si persiste la falta de acceso,
continúa esperando sin entrenar. Los errores de integridad siguen bloqueando.

No se modifican la vigencia del snapshot, las firmas, los mínimos de 2 GiB ni los
checkpoints cada 100 updates. El preflight verificó ambos hashes de checkpoint y
los 31 hashes del runtime científico, conservando R_A5 completo en 49.932 updates.
Evidencia: `artifacts/final_refit_20261019_inventory_recovery/PREFLIGHT.json`.

## Verificación de la recuperación del monitor

A las 16:52:59 UTC el supervisor corregido seguía vivo. Ambos productores habían
ejecutado nuevos updates: R_A5 pasó de 42.507 a 42.508 y R_C2F de 25.364 a 25.379.
R_A5 volvió a pausar limpiamente por caducidad y fue relanzado automáticamente;
su segundo intento estaba en `STARTING`. Esta evidencia verifica el reintento,
no demuestra entrenamiento continuo sin pausas. Se conserva en
`artifacts/final_refit_20261018_monitor_recovery/ACTIVATION_VERIFIED.json`.

Persiste un coste del inventario: un perfil aislado midió 9,708 s por scan,
de los que 9,224 s correspondieron a 147 llamadas nativas `proc_info` de psutil.
La vigencia sigue siendo 15 s desde el inicio del scan. La reanudación automática
corrige el bloqueo terminal, pero todavía pueden ocurrir pausas por caducidad.
Un prototipo de consultas de identidad con cuatro procesos midió 3,151 s en frío
y 2,165 s en caliente. No es un inventario completo equivalente ni está aplicado
al publisher: faltan comprobar cobertura, RSS, reutilización de PID y errores.
