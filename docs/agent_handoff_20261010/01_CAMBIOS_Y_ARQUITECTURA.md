# 01 — Cambios recientes y arquitectura que se está comparando

## Sistema de referencia

El sistema H8 de esta campaña recibe representaciones de eventos alrededor de una ROI y combina características temporales de productores A5, C2F y PAIR. Las cabezas consumen ocho observaciones y producen estimaciones TTC; se evalúan tanto semillas individuales como la mediana fija de tres cabezas. El historial V12 se adapta causalmente con un máximo de 650 ms. La representación H8 documentada en la revisión tiene forma `[8,3,12,128,128]` por consulta.

La preparación incluye lectura HDF5, selección temporal, recorte/remapeo espacial y voxels. La ejecución de la red incluye productores, construcción de características y cabezas. El coste de lectura y representación puede superar al de las cabezas. Por ello no basta con comparar el número de parámetros o el tiempo de un forward con entradas preparadas.

La carga y normalización de modelos de transferencia se siguen desde [`FrozenModels`](../../operational/evttc_transfer/models.py); la conversión TTC/fase, desde [`simplex_t/phase.py`](../../src/e_jepa_ttc/simplex_t/phase.py). La geometría de productores se encuentra en [`causal_scale_ttc.py`](../../src/e_jepa_ttc/models/causal_scale_ttc.py). Para reproducir una consulta hay que fijar productores, cabezas, normalizador y representación, no solo un archivo de pesos de la cabeza. El [RESULT del replay FCWD](../sota_evidence_20261010/evidence/streaming/fcwd_stream_cpu/RESULT.json) registra identidades SHA y updates de cada componente.

El esquema PHASE17 que construye `device_features` es:

| Columnas | Señal |
|---|---|
| 0–1 | Estadísticas escalares de conteo/tasa ROI |
| 2–4 | Transporte, confianza y log-varianza A5 |
| 5–7 | Transporte, confianza y log-varianza C2F |
| 8–10 | Fases A5, C2F y PAIR |
| 11–13 | PAIR−A5, PAIR−C2F, C2F−A5 |
| 14–16 | Valores absolutos de esas diferencias |

El estudiante consume exclusivamente columnas `(0,1,2,3,4,8)`, predice `(5,6,7,9,10)` y reconstruye algebraicamente las seis diferencias. Su red es 6→64→64→5 con SiLU y normalización fijada. Esta especificación explica qué información puede conservar y qué señales imita; no implica que la imitación produzca siempre el mismo TTC.

La evaluación utiliza cajas anotadas. «Solo eventos» describe la modalidad consumida por la estimación en inferencia; no implica ausencia de supervisión RGB durante el entrenamiento ni detección autónoma de objetos. El entrenamiento de los productores empleó teacher RGB y supervisión geométrica. Tampoco existe aquí una ablation que atribuya la ventaja exclusivamente al objetivo JEPA.

## Cronología verificable por commits

| Commit V12 | Trabajo incorporado | Dónde estudiarlo |
|---|---|---|
| `0a06ce6` | Corrección de prueba R1 para independizarla del deadline | Diff del commit y pruebas correspondientes; no es una ganancia TTC |
| `8c02888` | Revisión de entrada, ejecución compacta, cabeza directa y latencia emparejada | [`operational/ttc_revision`](../../operational/ttc_revision), [protocolo](../ttc_revision_20261009/PROTOCOL.md) |
| `46603a4` | Comparación con checkpoints Garl, MiD y test12 | [`operational/garl_comparison`](../../operational/garl_comparison), [evidencia Garl](../sota_evidence_20261010/evidence/garl) |
| `dcb1f8b` | Estado incremental, reproyección, estudiante, productores, graphs y paquetes compactos | [`operational/streaming_revision`](../../operational/streaming_revision) |
| `df46b15` | Publicación de evidencias de los experimentos | [`docs/sota_evidence_20261010`](../sota_evidence_20261010) |
| `6834735` | Paquetes de handoff preservados byte a byte | Historial Git y artefactos documentados en esa revisión |

En V13, `3be588d`, `4ecc218` y `4cb5f26` trasladan las herramientas V12; `b5effd4` añade el adaptador nativo y la continuidad; `da068eb` añade la admisión Git y evidencia final. El informe 06 detalla por qué trasladar herramientas no equivale a trasladar pesos ni resultados.

## Revisión del 9 de octubre: entrada y cabeza TTC

[`inputs.py`](../../operational/ttc_revision/inputs.py), clase `EventPreparer`, conserva el lector y un buffer crudo por secuencia. Invalida el buffer cuando cambian archivo, identidad o intervalo incompatible. La primera revisión usa 128 MiB; existe una ablation posterior de 512 MiB, publicada por separado. Se comparten doce ventanas distintas que antes se voxelizaban veinticuatro veces. Recalcular ROI y normalización sigue siendo necesario para la ruta exacta.

La preparación de Garl event-only se separó de H8: Garl necesita sus dos ventanas de 100 ms, no la construcción del historial H8. El coste histórico compartido inflaba artificialmente el baseline event-only. Las tablas nuevas preservan un baseline legacy para explicar el ahorro, pero la comparación principal debe usar la entrada nativa de cada sistema.

[`runtime.py`](../../operational/ttc_revision/runtime.py), `device_features` y `H8Runtime`, compacta la ejecución, elimina observaciones de relleno que los productores calculaban y conserva características en GPU. Se verifica paridad numérica: cambiar el orden de operaciones flotantes no permite prometer identidad bit a bit.

[`head.py`](../../operational/ttc_revision/head.py) implementa `DirectTTCHead` y `symmetric_ttc_loss`. La salida asinh/sinh cruza cero continuamente y limita TTC a ±60 s. Combina residuos en segundos, relativos con denominador mínimo 0,5 s y en coordenada asinh. No vuelve simétricos en segundos todos los gradientes alrededor de cualquier target.

Se entrenaron tres cabezas nuevas, 2.500 updates por semilla, sobre características TRAIN40 de productores congelados. Cambian además batch 128→256, dimensión oculta 160→64 y scheduler de H8 por LR constante 3e-4. Por tanto, **esta comparación no aísla el efecto de la pérdida**. No se seleccionó el checkpoint usando FCWD: el endpoint estaba fijado. El resultado negativo FCWD debe conservarse.

## Revisión del 10 de octubre: streaming

La arquitectura incremental separa tres capas de reutilización:

1. **Eventos crudos:** `PacketRing` conserva paquetes con representación compacta y una retención acotada. La selección de eventos debe ser exacta.
2. **Voxels:** `IncrementalPreparer` reutiliza ventanas exactas o, si se activa explícitamente, reproyecta voxels históricos a una ROI nueva. Esta segunda ruta es aproximada.
3. **Características:** `FeatureState` retiene resultados de productores por identidad temporal, secuencia y objeto; `StreamRuntime` computa misses y construye el contexto de la cabeza.

No se debe denominar «caché exacta» a cualquier hit. Un hit de eventos exacto, un hit con ROI aproximada y una característica conservada bajo tolerancia tienen distintos contratos. Los resultados registran reuse, nuevos cálculos, hits aproximados y geometría de ROI.

[`kernels.py`](../../operational/streaming_revision/kernels.py) añade correlación vectorizada, soporte de CUDA Graphs, padding de cabezas y precisión selectiva del encoder. Las salidas de graphs necesitan memoria propia antes de reutilizar el grafo. El estudiante de [`distill.py`](../../operational/streaming_revision/distill.py) imita características de C2F/PAIR a partir de A5; no es un estudiante end-to-end entrenado directamente para TTC.

## Estado de las variantes

| Variante | Qué conserva/cambia | Evidencia actual |
|---|---|---|
| H8 original | Productores y tres cabezas de referencia | Dev32, FCWD, TRAIN40 y predicciones test12 |
| Direct | Nuevas cabezas; productores congelados | Mejora MAE Dev32, empeora FCWD; no se promueve como solución general |
| H4/H2 | Truncación del contexto de la cabeza existente | Ablation de características guardadas; no entrenamiento dedicado |
| H8 warp | Ocho observaciones con voxels históricos reproyectados | Pilotos GPU y FCWD completo CPU |
| H8 warp+student | Sustituye parte del cálculo de productores | Mejora FCWD frente a H8, empeora Dev32 y no alcanza Garl en MiD FCWD |
| Student INT8 | Cuantización dinámica Linear en CPU | Error y latencia medidos; no ventaja suficiente para adopción |
| V13 streaming | Adaptador a productores y contratos nativos V13 | Pruebas y probe sintético; falta evaluación final de modelos completos |

## Qué no cambió

Los resultados históricos permanecen archivados; no se reemplazan los fallos por un candidato favorable. Los datos siguen en E:. Las mejoras de inferencia no introducen retrospectivamente una nueva loss en los entrenamientos matched V13 ni convierten sus checkpoints BF16 en FP32. Este handoff no modifica código de entrenamiento, pesos, splits o política de selección.
