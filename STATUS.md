# Estado del repositorio

Actualizado: 2026-07-30.

Compatibilidad histórica: CPLA-high is diagnostic only; no se utiliza como
test final ni como sustituto del Benchmark-10 sellado.

Validación local del 30 de julio de 2026: `202 passed`, Ruff sin errores,
validación del orquestador superada y export ONNX de OGE verificado
numéricamente.

## Conclusión ejecutiva

La implementación v6 ya permite una comparación justa, pero todavía no existe
una mejora científicamente demostrada sobre `BASE`. Lo cerrado es:

1. reproducción exacta del checkpoint histórico;
2. separación entre el ancla histórica y el control matched;
3. implementación corregida de Dense Patch, AttnRes, Object-KDA y geometría;
4. réplica Garl-TTC alineada con el código público;
5. navegación transformada al frame de la cámara de eventos;
6. orquestación Core/Garl sin colisión de resúmenes;
7. early stopping y checkpointing acotado.

Los resultados smoke son diagnósticos de integración. Los screens de 304/80
ventanas y hasta ocho épocas son la próxima evidencia de promoción.

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

## Datos

- `datasets/evttc`: 32 secuencias públicas etiquetadas.
- `datasets/evttc_official_benchmark_sealed`: no inspeccionado.
- `E:\eAP_dataset\data\train`: descarga train-40 ya iniciada.
- eAP no contiene TTC oficial en el release local.
- pseudo-TTC eAP: no oficial y fuera de la selección de arquitectura.

## Almacenamiento y hardware

- cache EvTTC v6 comprimido y separado por etapa;
- no se materializa un cache global de voxels;
- `best`, `last` y `weights_only` por run;
- BF16, `pin_memory`, prefetch y workers persistentes;
- microbatch 8 x acumulación 16 para Garl ResNet-50;
- microbatch 1 x acumulación 128 para G7 foreground;
- teachers no cargados durante los screens.

## Próximos gates

1. Screen Core histórico.
2. Screen Garl ResNet-50.
3. Promoción solo de brazos que superen el gate.
4. Grouped CV de cinco folds, seed 7.
5. Seeds 7, 13 y 21 para BASE y máximo dos finalistas.
6. Módulos bbox-free solo si geometría bbox-GT supera BASE.
7. Freeze y una inferencia final sobre Benchmark-10.

## Estado Git

La implementación y documentación v6 deben quedar asociadas a un commit limpio
antes de considerar cualquier nuevo resultado como promocionable. Los
checkpoints y caches permanecen fuera de Git; solo se versionan código,
configuración, manifests pequeños, métricas resumidas y documentación.
