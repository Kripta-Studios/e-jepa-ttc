# Cola posterior a los productores RGB

**Registro histórico de la primera cola.** La continuación científica posterior
está implementada en [SCIENTIFIC_OOF_QUEUE.md](SCIENTIFIC_OOF_QUEUE.md). Los límites
que siguen describen este observador inicial, conservado para trazabilidad.

La petición del 19 de octubre autoriza continuar hacia OOF, PAIR, E_CTX/R_CTX,
PHASE_ONLY, exportación y predicciones test12. No autoriza enviar a CodaBench.

## Estado real

El grafo está registrado en
`artifacts/final_refit_20261019_queue/QUEUE_PLAN.json`. El controlador ejecuta
automáticamente el prefijo disponible: esperar los endpoints RGB completos y
verificar los cinco splits OOF. **No existe aún ejecución científica completa de
OOF en este repositorio.** Después de verificar los splits, la cola termina con
`BLOCKED_IMPLEMENTATION`; no comienza entrenamiento ni inferencia test12.

Este límite es técnico, no una nueva solicitud de permiso. `fold_producers.py`
rechaza `scientific=True`; `fold_training.py` rechaza `technical=False`. Sus
pruebas de hasta diez updates no producen endpoints científicos. No se eliminan
esas protecciones ni se usa el stacking in-sample como sustituto de OOF.

## Dependencias registradas

1. R_A5 y R_C2F: 49.932 updates, endpoint científico, checkpoint completo y hash
   verificado. Se espera también la salida del supervisor anterior.
2. Cinco rotaciones de grupos 24/8/8. Productores event/RGB por rotación, PAIR
   sobre su propio encoder, PHASE17, normalizadores/refinadores por rol y generación
   sobre grupos excluidos de toda su genealogía. La rotación histórica no se da
   por adoptada sin sus comprobaciones.
3. Unión OOF de PHASE17 y PHASE20, identidad exacta y cobertura de TRAIN40,
   normalizadores OOF. No se mezclan tokens latentes de encoders independientes.
4. PAIR finales con productores TRAIN40; E_CTX/R_CTX finales sobre PHASE17 OOF;
   PHASE_ONLY final sobre salidas OOF de los refinadores.
5. Sellado de la cadena de despliegue, export y paridad, admisión de inputs test12,
   predicciones y ZIP local. Ningún nodo realiza un envío externo.

Los nodos posteriores al prefijo son dependencias pendientes, identificadas como
`executor=unimplemented`, no comandos ficticios ni trabajos terminados.

## Recuperación y observabilidad

El plan vincula sus fuentes y contratos por SHA-256. El estado se escribe de
forma atómica en `QUEUE_STATUS.json` cada 30 segundos mientras espera. Un lock
del sistema operativo evita dos controladores y se libera tras una caída. Al
reanudar se vuelven a comprobar las evidencias; no se confía en una etiqueta
`COMPLETE` antigua. Los pasos disponibles son idempotentes y no hacen updates.

Se respetan `STOP_REQUEST` de la cola y de los fits, y `AUTO_RESUME_STOP` del
runtime. No se borran marcas manuales. El controlador no cambia fuentes del
entrenamiento activo, límites de 2 GiB, guardado cada 100 updates, RNG ni pesos.

Comando de reanudación del prefijo actualmente ejecutable:

```powershell
.\scripts\resume_final_refit_queue.ps1
```

El comando no instala un servicio ni una tarea de inicio de Windows. Para habilitar
los nodos científicos hace falta implementar y verificar los adaptadores,
genealogía transitiva, generación, adaptación de caches OOF a la receta final y
sellado de despliegue, además de registrar sus recursos. Una nueva implementación
requiere un nuevo plan vinculado: cambiar un JSON a `COMPLETE` no habilita nada.

## Verificación

Pruebas específicas: `tests/unit/test_final_refit_continuation_queue.py`.
Evidencia de los grupos reales: `OOF_ROLE_PREFLIGHT.json` en el directorio de la
cola. Verifica 88.744 consultas, 40 grupos y cinco particiones; no lee targets ni
demuestra aún la genealogía de modelos que todavía no se han entrenado.
