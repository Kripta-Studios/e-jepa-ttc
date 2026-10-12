# Orden de ejecución de la matriz primaria

El 12 de octubre, después del checkpoint nativo de 3.010 updates, la cadencia
reciente proyectaba más tiempo para terminar sólo ese control que el presupuesto
GPU restante. La proyección es orientativa: el rendimiento del equipo ha variado.
La decisión utiliza tiempos y contadores, sin resultados H/V ni selección por loss.

Se registró en `PRIMARY_EXECUTION_ORDER.json` el orden:

1. PAIR-DIRECT-S7
2. PAIR-CORR-S7
3. PAIR-CORR-FROZEN-S7
4. Reanudar NATIVE-P2-FT-S7 hasta su endpoint original.

Todos mantienen 12.000 updates, seed7, inicialización, sampler, objetivo,
optimizador y fuentes científicas. El config original no cambia. La decisión
enlaza `scheduling/PAIRS_FIRST/EVIDENCE.json`, con los tiempos observados y el
presupuesto en el momento de registrarla. No se instala el prefetch rechazado.

`operational.joint_measurement.run` usa la permutación registrada después de
validar que contiene todos los brazos, sin duplicados, y que sus hashes de config
y fuentes científicas coinciden. Una configuración o fuente distinta se rechaza.
La primera evaluación H sigue exigiendo los cuatro endpoints completos y congelados.
El comando de reanudación continúa siendo:

```text
python -u -B -m operational.joint_measurement.run
```

La transición del proceso anterior se realiza con un controlador finito de
frontera durable. Comprueba PID, fecha de creación, comando y worktree; adquiere
el lock GPU propio entre scopes; exige presupuesto liquidado, journal durable,
checkpoint íntegro y estado del runner posterior al cierre. Sólo entonces
termina ese proceso. El recibo queda en `scheduling/PAIRS_FIRST/STOP_RESULT.json`.
No se debe iniciar otro runner hasta confirmar ese recibo y la ausencia del dueño
anterior. `STOP_REQUEST.json` por sí solo no acredita una parada.

Once pruebas CPU verifican la permutación, conservación de los controles,
bloqueo de H si el nativo sigue incompleto, rechazo de estados obsoletos o trabajo
sin checkpoint, y rechazo de procesos ajenos. La transición y el primer update
del nuevo orden deben comprobarse además en el estado real de ejecución.

La transición se ejecutó a las 19:51:43 UTC. El nativo quedó en 3.108 updates,
todos durables, sin pendientes ni recuperación. Su checkpoint tiene SHA-256
`86a0aeb1d4873c0575d9af5fab1a21973ff066da284d2841c170051ed25af543`.
El proceso anterior salió antes de iniciar el nuevo runner. Su log se conserva
en `PRIMARY_RESUME_AFTER_PREFETCH.log`; el nuevo utiliza `PRIMARY_PAIRS_FIRST.log`.

A las 19:54:34 UTC DIRECT había ejecutado sus primeros siete updates. El recibo
`verification/PRIMARY_ORDER_RESUME.json` comprueba identidad de las fuentes
científicas/configuración entre ambos brazos, endpoint y semilla originales,
gradientes finitos y no nulos en los 18 tensores del encoder y conservación del
checkpoint nativo. Todavía no acredita un endpoint completo ni precisión TTC.
