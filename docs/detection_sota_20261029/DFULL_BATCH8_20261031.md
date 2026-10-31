# Dfull: continuación autorizada con batch 8

El usuario autorizó aplicar batch 8 tras el perfil de throughput. La continuación
se lanzó desde el checkpoint 7.168 del run batch 4, con 28.672 exposiciones ya
consumidas. El proceso anterior se detuvo en el reemplazo atómico del checkpoint;
el ledger y el estado del sampler se comprobaron antes de cerrar el productor.

El [recibo de transición](evidence/dfull_batch8_20261031/BATCH_TRANSITION.json)
identifica checkpoint, configuración, ledger y plan. El nuevo run es
`/workspace/phase_release/dfull_campaign_v1/r2_dfull_seed7_batch8_v1`, con
[configuración explícita](evidence/dfull_batch8_20261031/config_batch8.yaml).
El controlador es 12328 y el trainer 12409 en este lanzamiento.

## Presupuesto y estado

Se mantienen 281.116 exposiciones sobre los mismos 70.279 frames train legibles:
cuatro pasadas, sin cambiar selección ni calibración. No se mantiene el antiguo
número de pasos multiplicándolo por ocho. El contador final pasa a 38.726 pasos
AdamW acumulados, incluidos los 7.168 pasos batch 4 previos; quedan 31.558 pasos
en el segmento nuevo. La mayoría usa ocho imágenes. Se usan lotes de cuatro
en las fronteras necesarias para conservar exactamente los puntos de evaluación
y el presupuesto final originales.

Se restauran estrictamente modelo, momentos y contadores AdamW, sampler y RNG
desde el productor histórico verificado. El nuevo scheduler envuelve la función
original: consulta el LR en el reloj de exposiciones de cuatro imágenes, con
el mismo warmup, cosine, multiplicador del backbone y LR base. No aplica escalado
lineal del LR al aumentar batch. Sus checkpoints incluyen ambos relojes y el hash
del plan, para poder reanudar sin reiniciar el schedule.

Es una receta de entrenamiento nueva: restaurar estado y RNG no hace equivalente
un paso de ocho imágenes a dos pasos de cuatro. Cambian la frecuencia de AdamW,
las operaciones de batch y la evolución estocástica. La mejora de throughput no
demuestra por sí misma que se conserve o mejore la calidad del detector.

## Verificación y automatización

Cuatro pruebas cubren conservación total de exposiciones y fronteras, continuidad
del LR, preservación del estado AdamW, rechazo de relojes inconsistentes y el
mismo orden del sampler a través de cambios de época y lotes parciales.
`audit_detection_batch_transition.py` verifica además el prefijo real ejecutado:
tokens exactos, tamaño efectivo del lote, exposiciones acumuladas y LR aplicado.

Los módulos y el recibo forman parte del snapshot y manifiesto del productor
nuevo. El productor batch 4, sus checkpoints y sus evidencias permanecen
conservados. Se mantienen FP32 highest, arquitectura, pérdidas y decoder.

El controlador verifica el ledger conjunto: prefijo comprometido batch 4 y
continuación batch 8. Evaluación, selección, comprobación del checkpoint elegido
y `FINAL_MODEL.json` siguen automatizados. La selección se realiza después de
haber consumido al menos una pasada completa; los puntos se conservan por número
de imágenes, aunque ahora tienen otros números de update.

La copia local incorpora ambos runs. El checkpoint de transición y el código
congelado se conservan en `D:/PHASE_TTC_DFULL_20261031`; el mapa del productor nuevo
es `FROZEN_CODE_RESTORE_MAP_BATCH8.json` y su materializador es
`materialize_backup_batch8.py`. La periodicidad de backup sigue siendo 30 minutos,
con el PC despierto y conectado. `LATEST_VERIFIED.json` delimita cada copia real.

El coste inicial de volver a abrir TAR y llenar caché se mide aparte del throughput
estable. El porcentaje de progreso debe calcularse con exposiciones acumuladas
divididas por 281.116, no comparando directamente el contador nuevo con 70.279.
