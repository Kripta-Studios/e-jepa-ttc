# R2-rare-v1: continuación del 2026-10-31

La receta de referencia sigue siendo R2 semilla 7, ya terminada. Esta intervención
es un experimento nuevo con la misma inicialización, datos, optimización, arquitectura
y presupuesto. No es fine-tuning del checkpoint final R2: eso cambiaría el presupuesto
y confundiría la comparación. Su único factor es el calendario de muestreo.

El pod nuevo conserva el volumen `pjgl7mi03u`, montado en `/workspace`. CUDA funciona
con Torch 2.11.0+cu128, Python 3.11.18 y una A40. Se verificaron los cuatro hashes de
checkpoints finales/seleccionados de R0 y R2 antes de continuar. Los datos no se han
retransferido y esos entrenamientos no se repiten.

## Calendario preregistrado

El calendario usa únicamente los tokens train de D4096 v5 y sus identidades admitidas
en la auditoría de población. Mantiene los 4.096 frames únicos y 32.768 exposiciones
(8.192 updates, batch 4). Se parte del calendario uniforme del sampler original con
semilla 7 y se sustituyen exposiciones de frames sin clases objetivo por exposiciones
de bicicleta/other_vehicle. El RNG del calendario es privado y no consume el RNG del
modelo.

Hay 75 frames objetivo y 75 pares únicos (secuencia, clase, track_id). Sus exposiciones
de frames pasan de 600 a 2.400. Se desplazan 1.800 exposiciones comunes, preservando
al menos una de cada frame. Cada frame y cada identidad objetivo quedan limitados
a 4× su exposición uniforme. Es un límite relativo a la exposición original, no una
equiparación entre identidades. Una de las 8.192 tandas contiene un frame repetido;
se conserva y registra como consecuencia del muestreo con repetición.

Las exposiciones de identidades anotadas de bicicleta pasan de 1.296 a 5.184 y las de
other_vehicle de 56 a 224. Esto no añade diversidad ni objetos físicos independientes.

Calendario exacto y denominadores: [sampling_plan.json.gz](evidence/rare_sampling_v1/sampling_plan.json.gz)
y [sampling_audit.json](evidence/rare_sampling_v1/sampling_audit.json).
SHA-256 del JSON sin comprimir:
`034871ec7dab86aa87f8c75a719103b202408e63e5bb105c1ef49f57cd20fb89`.

## Ejecución y reanudación

Se materializó un productor separado del R2 histórico. Conserva preparación,
modelo, optimizador, caché CUDA, prefetch y scorer del productor admitido; cambia
la construcción del sampler y añade su plan/módulo a la procedencia congelada.
El cargador verifica el hash del calendario, tokens en orden, presupuesto y límites
efectivos. El checkpoint guarda la identidad del calendario y el cursor exacto.

La primera admisión abortó antes de entrenar por comparar una lista del JSON con la
tupla normalizada del manifest. Se corrigió la normalización, se añadió una prueba
para ambos contenedores y se creó `producer_v2`; el primer productor permanece como
evidencia. El calendario no cambió.

La prueba GPU de ocho updates terminó con `BUDGET_STOPPED`, guardó checkpoint y
evaluación diagnóstica de parada. Sus ocho batches coinciden exactamente con el
calendario. El controlador admite ese checkpoint y continúa desde update 9 hasta
8.192; las ocho actualizaciones están incluidas en el presupuesto total. La evaluación
diagnóstica extra de la prueba corta no se utiliza para seleccionar checkpoint.

- Productor: `/workspace/phase_release/workspace/phase-ttc-s0-evaluation/output/rare_sampling_v1/producer_v2`.
- Campaña: `/workspace/phase_release/rare_sampling_campaign_v1`.
- Run: `r2_rare_seed7`.
- Estado autoritativo: `STATUS.json`; puntero global: `/workspace/phase_release/ACTIVE_CAMPAIGN.json`.
- Log de entrenamiento: `training.log`; log del controlador: `controller.log`.
- Admisión: [SMOKE_ADMISSION.json](evidence/rare_sampling_v1/SMOKE_ADMISSION.json).
- Receta: [config.yaml](evidence/rare_sampling_v1/config.yaml).
- Código congelado: [RARE_SAMPLING_RECEIPT.json](evidence/rare_sampling_v1/RARE_SAMPLING_RECEIPT.json).

El checkout reducido del pod conserva como base Git `fa0a26d0`; los cambios nuevos se
identifican por sus snapshots y hashes de productor, módulo y plan, además del commit
local de esta entrega. El campo Git del runner no se presenta como si contuviera por
sí solo toda la intervención.

## Comparación automática al terminar

El controlador posee un único hijo, no reintenta fallos y no lanza semillas adicionales.
Al terminar, verifica summary, checkpoints, linaje y ambas evaluaciones completas.
Comprueba identidad/hash del R2 previo y población idéntica de evaluación antes de
escribir `COMPARISON.json`.

Regla de screening fijada antes del nuevo resultado:

1. AP macro completa superior a R2.
2. Promedio AP de bicicleta y other_vehicle superior a R2.
3. Ninguna otra clase pierde más de 0,01 AP absoluto.

Se conserva el resultado aunque falle estos criterios. La regla es operacional,
no significación estadística. Se compara el final a update 8.192 sobre selección
completa de 23.958 frames; no se elige una variante usando test. Una mejora con
semilla 7 no confirma SOTA ni precisión TTC. Las semillas 13 y 23 siguen aplazadas.

## Comprobaciones

Pasan 20 tests de entrenamiento/muestreo: límites materializados, diversidad,
exclusión de roles no train, rechazo de calendarios inviables, reanudación exacta,
agotamiento del presupuesto, corrupción de plan/identidades y normalización de
tokens lista/tupla. Ruff y Pyright pasan para los archivos nuevos. La admisión GPU
añade ocho actualizaciones reales y comprobación de su calendario; la continuación
usa el cargador estricto de modelo, optimizador, scheduler, sampler y RNG.

Generación del calendario:

```text
python scripts/prepare_detection_rare_sampling.py --manifest output/campaign_data/d4096_v5_readable/d4096_manifest.json --population output/campaign_data/d4096_v4_final/frame_population.jsonl --output output/rare_sampling_v1/sampling_plan.json
```

El comando exige una salida nueva. La campaña nueva queda en ejecución; este
documento no acredita aún una ganancia de AP ni la finalización del entrenamiento.
