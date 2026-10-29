# 06 — Ejecución, recursos y entrega

## Dos sesiones, no dos escritores en la misma campaña

V13 mantiene su propietario y contrato. Enviar `COORDINATE_V13.md` a esa sesión cuando corresponda. La nueva trabaja en `scientific-recovery-v14-range-transfer`, desde el commit V13 auditado. V12 es evidencia de sólo lectura. No limpiar, reserializar ni corregir recibos del entrenamiento ajeno.

Antes de actuar, inventariar procesos y endpoints locales actuales. Si el HEAD V13 ya avanzó, registrar nuevo commit y leer su admisión; no resetearlo a la instantánea. Los consumidores V14 fijan sólo fuentes/padres completos. No construir un clone sustitutivo de main si falta el objeto de un commit.

## Recursos

Máximo conjunto23 GB decimales de RSS del proyecto, no por sesión. Reserva dura del host2 GiB e histéresis3 GiB. V14≤4 GiB de RSS, hasta 4 threads numéricos y CPU FP32; reducir preparación/profiling antes de presionar el entrenamiento V13.10 GB libres después de reservas para checkpoints/resultados. Controlar también memoria comprometida Windows.

No fijar afinidades ni tocar energía, drivers, paging o ControlCenter. No servicios de cómputo externos. La ausencia instantánea de GPU no concede permiso para competir con R1/V13. La implementación y recálculo de CSV pueden avanzar con CPU acotada.

## Presupuesto

25.000 updates científicos máximos;500 sintéticos técnicos;4.500 de recuperación;30.000 físicos. Contar optimizaciones guardadas y trabajo ejecutado/repetido/incierto separadamente. No usar las reservas como permiso para otra receta. Descubrir y reutilizar endpoints equivalentes en vez de relanzarlos.

No deadline artificial de horas. Un plazo posterior del usuario exige guardar la frontera y entregar parcial, no cambiar endpoint científico ni declarar completo un fit incompleto.

## Continuidad

Checkpoint cada 100 updates y pausa: pesos, optimizer, scheduler, cursor, sampler, RNGPython/NumPy/Torch y contador. Escritura atómica y referencia a último durable. Validar fuentes al admitir y en fronteras; no ejecutar hash de GB o git rev-parse por cada minibatch. Monitorizar con estado reciente y coste bajo.

Cada análisis se guarda por fragmentos con identificación de draws/grupos. Reanudar bootstrap y publicación desde fragmentos válidos, sin cambiar semillas hasta obtener intervalos favorables. La ausencia de un archivo debe identificar su path concreto; no ser un bucle opaco de 500 intentos.

Estado N0,N1,N2,SCREEN,REPLICATION,TRANSFER,REPORT. Si faltan padres, preparar código y entregar `WAITING_FINAL_PARENTS` con sus IDs; no rellenar resultados con V12. Si hay tareas independientes listas, ejecutarlas. Si ninguna lo está, entregar comando real de reanudación y dependencia exacta, sin heartbeat interminable.

## Entrega mínima

- `RANGE_TRANSFER_FINAL_REPORT.md`: comparación con fuentes y nuevas hipótesis, factores aislados, precisión/coste y límites.
- `NEXT_DECISION.json`: estados, candidato o ausencia, no promoción retroactiva V13, trabajos futuros no autorizados.
- `SOURCE_BINDINGS.json`, `ACTIVE_STATE.json`, `ROLE_AUDIT.json`, `MASS_AUDIT.csv`, `GRADIENT_AUDIT.csv`.
- Endpoints/pesos nuevos, fuentes de pares, normalizadores referenciados, predicciones y pérdidas por query/grupo/banda, curvas y draws.
- Pruebas de paridad L0 y resume, contabilidad física/técnica/científica y logs de fallos.
- `COMMANDS_EXECUTED.md` y comando exacto probado para continuar.
- ZIP esencial con hash externo y manifiesto interno; regeneración independiente que declare si parte de caché o raw.

La entrega distingue desarrollo H/V, transferencia expuesta FCWD/Dev32 y test oficial no puntuado. No proclamar seguridad por FR=0 ni calibración por producir cuantiles.
