# CUDA, precisión numérica y transporte de tensores

## 1. Alcance y criterio de evidencia

Este documento reconstruye la evolución de la ejecución CUDA, la precisión numérica y el transporte CPU/GPU durante las campañas de medición conjunta, refit científico y migración a Runpod. Separa deliberadamente cuatro estados que no deben confundirse:

1. **Ejecutado en entrenamiento científico**: apareció en la ruta real que produjo actualizaciones de optimizador.
2. **Admitido por una prueba técnica**: superó paridad o rendimiento, pero no necesariamente se activó en la campaña científica.
3. **Explorado parcialmente**: se probó un kernel o subgrafo aislado; no valida el modelo completo.
4. **Rechazado**: cambió el orden numérico, no alcanzó el umbral o careció de cobertura suficiente.

Esta taxonomía es importante porque el repositorio contiene trabajo real con CUDA Graphs, pero no existe evidencia para afirmar que todos los entrenamientos posteriores los usaran. En particular, el refit final BF16 mantuvo `cudagraph_local_transport=false` y su prueba de grafo fue sólo parcial.

## 2. Cronología resumida

| Etapa | Precisión y dispositivo | Transporte o grafo | Resultado verificable |
|---|---|---|---|
| Medición conjunta A1 | FP32 en CUDA | Grafo del modelo activo; pérdida y Adam fuera del grafo | GPU media 9,8%; el siguiente batch dominaba el muestreo del hilo principal |
| B1, inferencia preparada | FP32 en CUDA | CUDA Graph admitido para `forward` preparado | Medianas 158,46→9,43 ms y 158,29→9,10 ms; no incluye lectura del sensor |
| B1, entrenamiento | FP32 en CUDA | Captura del modelo; buffer afín tuvo que preasignarse | Una captura inicial falló por una copia CPU→GPU dentro del grafo; la corrección fue operativa, no un cambio científico |
| C1/D1 | CUDA | Probes y gates registrados | No hay recibo suficiente para afirmar activación científica |
| Refit concurrente inicial | BF16 en CUDA | Particiones dinámicas, múltiples transferencias escalares | Se localizaron sincronizaciones host y un hotspot de indexado; no se modificó durante el diagnóstico |
| Runtime v4 del refit | BF16 en CUDA | Metadatos de pérdida en host y transferencia escalar agrupada; sin CUDA Graph de producción | 77 pruebas y matriz BF16 aprobadas; reutilización de transporte y fusión de particiones rechazadas |
| Runpod RTX 4090 | BF16 nativo disponible | Varios procesos independientes, no una prueba de CUDA Graph | Escalado agregado medido de uno a seis workers; sin atribución causal a grafos |

## 3. Contrato de precisión: FP32 histórico frente a BF16 científico

Las campañas no usaron una única precisión. La medición conjunta A1/B1 documentó ejecución FP32 y dejó explícito que BF16, FP16 y TF32 no estaban activos en aquella ruta. Por tanto, sus cifras de latencia y sus resultados de CUDA Graph no deben extrapolarse como validación BF16.

El refit final sí fijó BF16 como parte de la identidad científica, junto con semilla 7, AdamW, scheduler, orden de muestras, checkpoint cada 100 actualizaciones y la misma arquitectura y pérdida. El runtime v4 no cambió estos elementos. Su recibo de admisión, [`RUNTIME_OPTIMIZATION_ADMISSION.json`](../../artifacts/final_refit_20261017_runtime_v4/RUNTIME_OPTIMIZATION_ADMISSION.json), registra:

- 77 pruebas focalizadas aprobadas;
- matriz de 16 casos BF16 para `R_A5` y `R_C2F`, epochs 2/4, longitudes T2/T3/mixtas y batch 32;
- comparación de salidas, pérdida, gradientes, optimizador, estado, RNG y reanudación;
- tolerancias fijas `atol=2e-5` y `rtol=2e-4`;
- cero actualizaciones científicas o técnicas del optimizador durante la admisión;
- SHA de admisión `e59c5017…`.

También se comprobó que la máscara física CPU/CUDA era idéntica en las 88.744 consultas TRAIN; el artefacto queda ligado por SHA `69b918…`. Esta cobertura es más fuerte que una simple prueba de que la GPU soporta BF16: protege los límites discretos que determinan qué ejemplos entran en cada rama.

En Runpod se verificó además la capacidad del hardware, no un nuevo resultado científico: [`ENVIRONMENT_V3.json`](../../artifacts/runpod_migration_20261026/ENVIRONMENT_V3.json) registra Python 3.11.15, PyTorch 2.11.0+cu128, CUDA 12.8, RTX 4090 y `bf16=true`, con cero actualizaciones durante el inventario. El preflight de red preservó 7.093 entradas del bundle, una población de 54.615 consultas y el endpoint retenido de 30.726; cuatro consultas CPU de referencia fueron bit-exactas. Véase [`NETWORK_TECHNICAL_OVERLAY_V3.md`](../runpod_migration_20261026/NETWORK_TECHNICAL_OVERLAY_V3.md).

## 4. Sincronizaciones CPU/GPU encontradas

El diagnóstico concurrente del refit no se limitó a la utilización global. Reconstruyó la ruta de actualización y encontró que un batch de 32 se dividía por `(historical_role, T)` en entre una y seis particiones físicas. Cada partición implicaba ramas Python y materialización de escalares:

- tres conversiones `bool(...)` posteriores a CUDA en la validación del batch;
- cinco `.item()` en T2, una adicional en T3 y un `bool` de finitud relacional;
- nueve puntos de sincronización por partición T2 y diez por partición T3, además de la comprobación final del total;
- dieciséis conversiones separadas `float(tensor.cpu())` después del paso del optimizador.

El perfil CUDA de tres actualizaciones señaló `aten::_index_put_impl_` con 135.624,454 µs en 258 llamadas y el backward de indexado con 132.111,549 µs en 240 llamadas. Ese dato identifica una familia de operaciones costosa, pero no atribuye el tiempo a una única línea ni demuestra que reescribir el indexado mantenga la misma aritmética BF16. La evidencia completa está en [`CONCURRENT_RUNTIME_BOTTLENECK_AUDIT.md`](../final_refit_20261017/CONCURRENT_RUNTIME_BOTTLENECK_AUDIT.md).

Runtime v4 admitió dos cambios acotados:

- `host_loss_metadata=true`: preparar en CPU metadatos que no necesitaban nacer como tensores CUDA durante el prefetch;
- `scalar_transfer=true`: transferir métricas escalares conjuntamente después de la actualización.

No se afirmó que estas medidas eliminaran todas las sincronizaciones. El recibo deja `transport_reuse=false`, `merge_parts_by_t=false` y `cudagraph_local_transport=false`. La justificación y el resultado posterior se documentan en [`RUNTIME_V4_APPLIED.md`](../final_refit_20261017/RUNTIME_V4_APPLIED.md).

## 5. Reutilización de transporte y fusión de particiones: resultados negativos

La optimización denominada *transport reuse* pretendía reutilizar representaciones transportadas para reducir trabajo y sincronización. El forward aislado coincidía, pero el cambio alteraba el orden de acumulación de gradientes BF16. Los cuatro casos completos `R_A5` de epoch 4 superaron la tolerancia fijada. El rechazo y su recibo histórico `RGB_OPTIMIZATION_ADMISSION_TRANSPORT_REJECTED.json` están inventariados en [`RUNTIME_V4_APPLIED.md`](../final_refit_20261017/RUNTIME_V4_APPLIED.md); la copia actual del repositorio no conserva ese JSON como archivo navegable independiente.

La fusión de particiones con el mismo T tampoco se admitió. Falló en los cuatro casos T2/mixtos de A5/C2F con clasificación `REJECT_RNG_OR_NUMERICAL_DRIFT`. La razón científica es suficiente aunque el throughput potencial fuera atractivo: cambiar el agrupamiento puede cambiar el consumo de RNG y el orden de reducción, y con BF16 esas diferencias no son meramente cosméticas.

Estos resultados negativos fijaron un límite operativo útil: se optimizaron metadatos y transferencia de escalares, pero se conservó el orden de forwards y acumulaciones que define la receta.

## 6. CUDA Graphs: qué estuvo activo y qué no

### 6.1 A1: grafo activo en entrenamiento FP32

En A1 los grafos de forward/backward del modelo sí estaban activos y aparecieron en las pilas de ejecución. El grafo no cubría la pérdida ni Adam. En una captura de 40 segundos, el hilo principal pasó aproximadamente:

- 54,43% esperando el siguiente batch;
- 34,58% en forward/pérdida/backward;
- 3,49% en optimizador;
- 3,07% en journal;
- 1,77% en transferencia;
- 0,99% en clip.

Treinta actualizaciones en 44,15 s equivalieron a 1,472 s/update, unas 2.446 updates/h. La GPU media fue 9,8%, por lo que el grafo del modelo no resolvía el cuello de botella de entrada. El hijo se reiniciaba cada 900 s, perdiendo tanto caché como capturas. Fuente: [`A1_TRAINING_BOTTLENECK_20261016.md`](../joint_measurement_20261012/A1_TRAINING_BOTTLENECK_20261016.md).

### 6.2 B1: inferencia preparada realmente graficada

El evaluador original B1 no usaba grafos. Una variante v2 sí fue admitida para la parte preparada del forward, con 132 predicciones exactas, pesos y RNG sin cambios y cero actualizaciones. Las medianas medidas fueron:

| Rama | Eager | CUDA Graph | Cociente aproximado |
|---|---:|---:|---:|
| CORR | 158,46 ms | 9,43 ms | 0,0595 |
| TTC control | 158,29 ms | 9,10 ms | 0,0575 |

El alcance excluye carga del sensor, decodificación y preparación de entrada. Es evidencia de aceleración del forward preparado, no de una mejora end-to-end equivalente. Fuente: [`B1_INFERENCE_RUNTIME.md`](../joint_measurement_20261012/B1_INFERENCE_RUNTIME.md).

En el entrenamiento B1 hubo además una incidencia reproducible: la primera captura intentaba copiar una matriz afín constante de CPU a GPU dentro del grafo. Se corrigió preasignando el buffer. Una transición de etapa invalidó la captura y el coordinador reconstruyó el proceso. Esto muestra que la captura era real, pero también que su validez dependía de buffers, shapes y fase.

### 6.3 C1 y D1: estudio y gates, no activación demostrada

C1 registró un probe de grafo geométrico, y D1 registró QA de paridad con un máximo de 30 actualizaciones técnicas. Los documentos de ejecución no aportan un recibo GPU completo que permita afirmar que el entrenamiento científico final se ejecutó con esos grafos. Deben describirse como trabajo preparado o estudiado. Véanse [`C1_EXECUTION.md`](../joint_measurement_20261012/C1_EXECUTION.md) y [`D1_EXECUTION.md`](../joint_measurement_20261012/D1_EXECUTION.md).

### 6.4 Refit BF16: prueba parcial, producción desactivada

El refit final tenía longitudes T2/T3/mixtas, entre uno y seis forwards por batch, ramas Python, I/O, buffers dinámicos y requisitos estrictos de RNG/reanudación. Por eso no se admitió una captura completa.

El recibo histórico `PARTIAL_GRAPH_PROBE.json`, resumido y ligado desde [`RUNTIME_V4_APPLIED.md`](../final_refit_20261017/RUNTIME_V4_APPLIED.md), prueba únicamente un muestreador bilineal determinista, sin parámetros, en cuatro firmas FP32 de batch 1/4 y lado 16/32. Dio:

- salida máxima diferente: 0;
- gradiente de grid máximo diferente: 0;
- gradiente del campo: entre `1.19e-7` y `4.77e-7`;
- RNG sin cambios;
- restauración de buffers exacta;
- cero actualizaciones del optimizador.

El propio recibo declara `PASS_PARTIAL_GRAPH_VIABLE` y `production_graphs_enabled=false`. Apareció además un warning de stream de `AccumulateGrad`. La conclusión correcta es que un subgrafo era viable, no que el refit BF16 estuviera graficado.

## 7. Batching y concurrencia en RTX 4090

Las rotaciones cortas de Runpod permiten separar escalado por procesos de las optimizaciones de un solo update. El baseline de un worker, [`SINGLE_6MIN_RESULT.json`](../../artifacts/runpod_migration_20261026/SINGLE_6MIN_RESULT.json), registró 1.142 updates en 360,471 s:

- 190,084 updates/min;
- GPU media 18,55%, p95 59%;
- VRAM 3.150 MiB;
- CPU de proceso 47,09% de un core;
- máximo cgroup 25,56 GB.

Las rotaciones posteriores midieron:

| Workers | Ritmo por worker | Agregado | Relación agregada frente a uno |
|---:|---:|---:|---:|
| 1 | 190,084 upd/min | 190,084 | 1,000 |
| 2 | 183,001 y 185,168 | 368,169 | 1,937 |
| 3 | 176,668; 178,501; 179,001 | 534,169 | 2,810 |
| 6 | A5 126,67–127,83; C2F 112,50–116,00 | 726,505 | 3,822 |

En dos workers la GPU media fue 40,47%, p95 83%, VRAM 5.847 MiB. El escalado agregado fue útil, mientras el ritmo individual cayó de forma gradual. Estas cifras describen esa ventana, hardware, datos y overlay. No demuestran que CUDA Graphs fueran la causa, ni que seis workers sean universalmente óptimos; de hecho, fases posteriores necesitaron control de capacidad por RAM y page cache.

## 8. Rendimiento posterior al runtime v4

La primera ventana de 180 s antes del cambio había observado aproximadamente 30,17 upd/min en A5 y 18,51 en C2F, GPU media 50,9%, VRAM 6.657 MiB y temperatura 86,6 °C. Tras runtime v4, una ventana comparable observó:

- C2F: 771→817, 15,77 upd/min, 3,805 s/update;
- A5: 6.190→6.258, 23,31 upd/min, 2,574 s/update;
- guard: 20,42 y 18,92 ms;
- GPU media 39,6%, VRAM 7.338 MiB, 62,6 W, 86,5 °C.

La ventana posterior fue más lenta. Por ello el informe no atribuyó un speedup global a los cambios. Una prueba A5 en solitario de cinco minutos dio 27,6 upd/min frente a 27,0 en una referencia concurrente anterior, una diferencia observada de +2,2%, insuficiente como estimación causal controlada. Los datos están en [`POST_CHANGE_180S`](../../artifacts/final_refit_20261017_runtime_v4/POST_CHANGE_180S) y [`RUNTIME_V4_APPLIED.md`](../final_refit_20261017/RUNTIME_V4_APPLIED.md).

## 9. Límites y reglas que quedaron establecidos

- No activar CUDA Graphs completos en una receta BF16 dinámica basándose en un probe de bilinear FP32.
- No reutilizar transportes ni fusionar particiones si cambia el orden de acumulación, RNG o estado reanudable.
- No inferir speedup causal a partir de dos ventanas con distinta concurrencia, temperatura, cache o presión de recursos.
- Medir por separado preparación, transferencia, forward capturado, pérdida, backward y optimizador.
- Mantener el recibo de precisión ligado a salidas, gradientes, estado, RNG y reanudación, no sólo a la pérdida final.
- Tratar las sincronizaciones `.item()`, `bool(tensor)` y transferencias escalares como deuda medible; reducirlas sólo con una matriz BF16 equivalente a la de runtime v4.

## 10. Índice de evidencia principal

- [`CONCURRENT_RUNTIME_BOTTLENECK_AUDIT.md`](../final_refit_20261017/CONCURRENT_RUNTIME_BOTTLENECK_AUDIT.md): perfil de hilo, CUDA, sincronizaciones y tiempos de prefetch.
- [`RUNTIME_V4_APPLIED.md`](../final_refit_20261017/RUNTIME_V4_APPLIED.md): decisiones admitidas, rechazadas y medición posterior.
- [`RUNTIME_OPTIMIZATION_ADMISSION.json`](../../artifacts/final_refit_20261017_runtime_v4/RUNTIME_OPTIMIZATION_ADMISSION.json): contrato BF16, opciones y pruebas.
- `PARTIAL_GRAPH_PROBE.json`, inventariado en [`RUNTIME_V4_APPLIED.md`](../final_refit_20261017/RUNTIME_V4_APPLIED.md): alcance exacto del probe parcial; el JSON independiente no está en la copia actual.
- `COMPUTATION_REUSE_PROOF.json`, inventariado en el mismo informe: prueba histórica de que el cambio admitido no alteró el AST de cómputo; SHA `ed9c…`. El JSON independiente tampoco está en la copia actual.
- [`A1_TRAINING_BOTTLENECK_20261016.md`](../joint_measurement_20261012/A1_TRAINING_BOTTLENECK_20261016.md): grafos activos y cuello de entrada A1.
- [`B1_INFERENCE_RUNTIME.md`](../joint_measurement_20261012/B1_INFERENCE_RUNTIME.md): grafo de inferencia admitido y límites de la comparación.
- [`SINGLE_6MIN_RESULT.json`](../../artifacts/runpod_migration_20261026/SINGLE_6MIN_RESULT.json), [`DUAL_6MIN/RESULT.json`](../../artifacts/runpod_migration_20261026/DUAL_6MIN/RESULT.json) y [`DUAL_ANALYSIS.json`](../../artifacts/runpod_migration_20261026/DUAL_ANALYSIS.json): escalado inicial de Runpod.
