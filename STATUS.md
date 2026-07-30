# Estado del repositorio

Actualizado: 2026-07-30.

Compatibilidad histórica: CPLA-high is diagnostic only; no se utiliza como
test final ni como sustituto del Benchmark-10 sellado.

Validación local del 30 de julio de 2026: `224 passed` en el árbol versionado,
Ruff sin errores,
validación del orquestador superada y export ONNX de OGE verificado
numéricamente.

## Conclusión ejecutiva

La confirmación histórica demuestra que Dense Patch puede superar a A0 en un
split cuando converge, pero grouped CV multisemilla selecciona A0. No existe
todavía SOTA oficial. Lo cerrado es:

1. reproducción exacta del checkpoint histórico;
2. separación entre el ancla histórica y el control matched;
3. implementación corregida de Dense Patch, AttnRes, Object-KDA y geometría;
4. réplica Garl-TTC alineada con el código público;
5. navegación transformada al frame de la cámara de eventos;
6. orquestación Core/Garl sin colisión de resúmenes;
7. early stopping y checkpointing acotado;
8. grouped CV de 5 folds × 3 seeds para A0/A1;
9. freeze de A0 antes del diagnóstico familiar OOD;
10. evaluación separada validation/family-OOD con bootstrap por secuencia.

El screen de ocho épocas era insuficiente para Dense: A1 alcanzó su mejor
checkpoint en la época 20. AttnRes y KDA no mejoraron con el presupuesto largo.

## BASE histórico exacto

Artefacto:
`artifacts/audit/oge_sota/historical_base_reproduction.json`.

```text
arquitectura     EventTubeletTransformerEncoder
canales          10 eventos + 11 auxiliares
dimensión        192
profundidad      6
heads            6
patch            16
pooling          media de tokens finales
salida           log-TTC
train/validation 7.835 / 2.040 ventanas
checkpoint       epoch 26/30
SSL              epoch 6/30
```

| Métrica validation | Valor |
|---|---:|
| MAE | 0,3228917687 s |
| RMSE | 0,5844324448 s |
| error relativo medio | 8,1553575311 % |

Las predicciones son idénticas byte a byte al artefacto histórico. Este modelo
se denomina `B0_HISTORICAL_BASE_EXACT`; no es el control entrenado en la matriz
object-cache.

## Resultado negativo conservado

La matriz FlowMimic multisemilla histórica obtuvo:

| Variante | MAE medio | Cambio frente a BASE |
|---|---:|---:|
| BASE | 0,332715 s | - |
| alignment global | 0,369604 s | +11,1 % |
| inverse-TTC sintético | 0,432909 s | +30,1 % |
| ambas pérdidas | 0,435869 s | +31,0 % |

El código activo FlowMimic se retiró, pero el resumen negativo se conserva para
evitar sesgo de publicación.

## Matriz matched

Comparación Core:

- `A0_MATCHED_GLOBAL`;
- `A1_MATCHED_DENSE_BLOCK`;
- `A2_MATCHED_DENSE_ATTNRES`;
- `K1_OBJECT_KDA`;
- `A4_GT_GEOMETRY`.

A0, A1, A2 y K1 comparten:

- selección exacta de muestras;
- backbone y cabeza iniciales con hashes iguales;
- batch, épocas, optimizador, LR y weight decay;
- loss log-TTC;
- regla de early stopping;
- validación macro por secuencia.

La comparación Garl incluye G0–G7: direct, LHR, early/late fusion y foreground.
El backbone del screen es ResNet-50. El backbone compacto queda limitado a
smoke.

## Confirmación Core comparable

Todos los brazos usaron 1.208 ventanas train y 314 validation, checkpoint BASE
idéntico, máximo 40 épocas, batch 16 x acumulación 2, LR `3e-5`, weight decay
`1e-3` y early stopping con mínimo 10 épocas y paciencia 6.

| Variante | Mejor época | Completadas | Error rel. macro | Score | MAE macro | ms |
|---|---:|---:|---:|---:|---:|---:|
| A0 global | 17 | 23 | 16,129 % | 0,32523 | 0,701 s | 8,98 |
| **A1 Dense** | **20** | 26 | **15,210 %** | **0,30543** | **0,628 s** | 17,15 |
| A2 AttnRes | 10 | 16 | 16,136 % | 0,32503 | 0,653 s | 16,70 |
| K1 Object-KDA | 7 | 13 | 16,960 % | 0,34139 | 0,731 s | 16,99 |

Decisión de ese gate histórico: promover A1 a grouped CV; no promover A2 ni
K1. A1 mejora frente a A0 un 5,70 %
en error relativo macro, 6,09 % en score y 10,45 % en MAE, pero cuesta 1,91
veces la latencia.

Esta decisión histórica queda supersedida para la arquitectura final por el
grouped CV, no borrada como resultado positivo de un split.

## Grouped CV A0/A1 cerrado

Los 30 runs esperados están completos. Cada pareja fold/seed pasó la auditoría
de cache, samples, backbone, cabeza y trainer.

| Variante | Score ± sd seeds | Error rel. ± sd | MAE ± sd | ms/ventana |
|---|---:|---:|---:|---:|
| **A0 global** | **0,58452 ± 0,00853** | **30,25 % ± 0,52** | 1,011 ± 0,039 s | 4,54 |
| A1 Dense | 0,59312 ± 0,00349 | 30,55 % ± 0,06 | **1,007 ± 0,013 s** | 9,82 |

A1 empeora 1,47 % el score y 0,99 % el error relativo; solo mejora 0,41 % el
MAE. A0 gana 10/15 pares en score/error relativo y 8/15 en MAE. A1 cuesta
1,58× entrenamiento y 2,16× latencia. Ningún bootstrap pareado por secuencia
demuestra una diferencia distinta de cero. Decisión final de arquitectura:
`A0_MATCHED_GLOBAL`.

## Ablación R1 bbox-ROI

`R1_MATCHED_BBOX_ROI` usa la bbox GT únicamente para seleccionar los tokens
densos que alimentan la misma cabeza TTC. Los cinco folds de seed 7 terminaron
con la misma configuración matched:

| Variante | Score | Error rel. | MAE | Tiempo total |
|---|---:|---:|---:|---:|
| A0 seed 7 | 0,58125 | 30,16 % | 0,966 s | 1.443 s |
| R1 seed 7 | 0,59814 | 30,99 % | 1,010 s | 2.410 s |

R1 empeora 2,90 % el score, 2,74 % el error relativo y 4,55 % el MAE, con
1,67× tiempo. Se rechaza sin gastar seeds 13/21. La bbox como pooling no
reemplaza una estimación explícita de expansión/FoE.

## Ajuste final y OOD

El perfil matched gana al throughput un 10,95 % en score medio de tres seeds,
a cambio de 4,02× tiempo. Validation seleccionó A0 matched seed 13 antes de
abrir family-OOD.

| Split | Secuencias / ventanas | Score | Error rel. macro | MAE macro |
|---|---:|---:|---:|---:|
| validation | 5 / 314 | 0,28992 | 14,46 % | 0,541 s |
| family-OOD reutilizado | 8 / 481 | 0,53784 | 30,56 % | 0,805 s |

El OOD degrada 85,5 % el score, 111,4 % el error relativo y 48,8 % el MAE. Su
bootstrap 95 % de MAE es 0,593–1,128 s. El holdout es disjunto del ajuste pero
no virgen para el proyecto. Benchmark-10 no se abrió.

## Screen Garl local

El screen ResNet-50 usa el mismo batch efectivo y las mismas actualizaciones
por época que Core. El mejor brazo local fue G5 RGBE-LHR early con 36,52 % de
error relativo macro. Ningún brazo se promueve aún: el protocolo público Garl
entrena 50 épocas y construye late fusion desde ramas LHR preentrenadas; el
screen corto solo sirve para descarte, no para afirmar paridad con el paper.

## Geometría y ego-motion

Corregido:

- inverse-TTC en el endpoint temporal actual;
- último par válido para height/area/affine;
- event contrast object-centric;
- heading en grados convertido y desenvuelto;
- velocidad GNSS norte/este/arriba transformada a la cámara de eventos;
- brazo rígido navegación–cámara incluido en la velocidad;
- warp causal de rotación y traslación.

La traslación necesita profundidad. La evaluación con distancia oficial EvTTC
se etiqueta `translation_compensated_box_mixture_oracle`; no puede alimentar el
modelo final. La variante desplegable necesitará profundidad predicha.

La geometría causal de escala bbox, calibrada solo con train, cubre 311/314
ventanas. El híbrido determinista geometría + fallback A1 obtiene 14,790 % de
error relativo macro frente a 15,210 % de A1, pero su score empeora de 0,30543
a 0,31144 por RMSE. No pasa el gate predeclarado del 5 %.

El port causal trazable a STRTTC implementa NLTS, contornos, normal flow local,
RANSAC y solver de tres parámetros. En el screen 200 ms resuelve 27/40 muestras
y falla 13; las métricas de las exitosas son 112,96 % de error relativo macro.
La cobertura incompleta se registra y el brazo queda rechazado.

## Datos

- `datasets/evttc`: 32 secuencias públicas etiquetadas.
- `datasets/evttc_official_benchmark_sealed`: no inspeccionado.
- `E:\eAP_dataset\data\train`: train-40 completo, 216 archivos y 536,64 GiB.
- `datasets/CARLA_DVS_Looming_Dataset/random_spawn`: 1.406 secuencias y
  71,64 GiB extraídos; 1.395 válidas con contexto de 100 ms.
- eAP no contiene TTC oficial en el release local.
- pseudo-TTC eAP: 195.024/804.510 filas válidas (24,24 %), no oficial y fuera
  de la selección de arquitectura.
- CARLA: 412 colisiones con coche, 347 con peatón y 636 negativos; manifest y
  split bloqueado firmados, lectura mmap con `allow_pickle=False`.

## Almacenamiento y hardware

- cache EvTTC v6 comprimido y separado por etapa;
- no se materializa un cache global de voxels;
- `best`, `last` y `weights_only` por run;
- BF16, `pin_memory`, prefetch y workers persistentes;
- batch 24 para Garl ResNet-50 en Screen;
- microbatch 4 x acumulación 6 para G7 foreground, batch efectivo 24;
- teachers no cargados durante los screens.

## Próximos gates

1. Mantener A0 como final y A1 como hipótesis, no como ganador.
2. Ejecutar un piloto eAP SSL de 2–4 secuencias sin pseudo-TTC y repetir el
   mismo fine-tuning EvTTC; escalar a 40 solo si mejora.
3. Repetir Garl con el presupuesto y pretraining por ramas del código oficial.
4. Mantener módulos bbox-free bloqueados: la geometría no pasó su gate.
5. Abrir Benchmark-10 solo bajo una decisión explícita posterior; este trabajo
   no lo consumió ni afirma SOTA.

## Estado Git

La implementación y documentación v6 deben quedar asociadas a un commit limpio
antes de considerar cualquier nuevo resultado como promocionable. Los
checkpoints y caches permanecen fuera de Git; solo se versionan código,
configuración, manifests pequeños, métricas resumidas y documentación.
