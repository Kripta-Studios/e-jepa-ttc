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
