# MonoDGP portable: ejecución completa en la GPU local

La variante `monodgp_pytorch_grid_sample_v1` ejecuta el detector completo con
ocho clases, pérdidas, backward, actualización y decode. Usa operaciones PyTorch
diferenciables en GPU. **No se compiló `MultiScaleDeformableAttention`: el G1
original sigue bloqueado; la variante portable supera una comprobación funcional
sintética con alcance explícito.** Su precisión eAP se mide por separado.

Se conserva ResNet-50, hidden dimension 256, cuatro niveles, tres capas de encoder
y decoder según la configuración upstream, cabezas de clase/caja/dimensión/
profundidad/ángulo, predictor de profundidad y segmentación regional. Los primeros
smokes y los dos runs de overfit usaron inicialización aleatoria, sin descargar
pesos preentrenados. La configuración inicial con 11 grupos contiene 42.168.788
parámetros. El piloto posterior incorpora un backbone ImageNet descargado
explícitamente y una cabeza de velocidad, descritos más abajo.

## Backend y comprobaciones independientes

`src/e_jepa_ttc/detection/portable_attention.py` sustituye la interfaz de la
extensión por una función ordinaria de PyTorch, preservando autograd. Combina
bilinear sampling por nivel y sumas ponderadas. Fija `align_corners=False` y
`padding_mode=zeros`; rechaza coordenadas no finitas antes del muestreo.

Las seis pruebas iniciales incluyen valores conocidos en centros de píxel,
muestreo fuera de la imagen, rechazo de índices de nivel incorrectos, rechazo de
NaN, gradcheck CPU en float64 y comparación CPU/GPU de outputs y primeros
gradientes en float32 y float64. No se compara la función consigo misma para
proclamar paridad. Estas pruebas no equivalen a comparar un binario externo CUDA
que no existe en el entorno.

La documentación de PyTorch indica que el backward CUDA de `grid_sample` puede
ser no determinista; las semillas registradas no garantizan identidad bit a bit.
[API oficial, PyTorch 2.11](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.grid_sample.html).

## Fuentes y cambios declarados

La fuente original permanece en `output/detector_sources/MonoDGP`, fijada a
`aa059a18214aebf644510e7f0793971b403f9d14`. El runner genera una copia separada en
`MonoDGP-portable`, un patch y hashes de cada archivo modificado. No modifica
los recibos importados ni el código histórico de PHASE.

Cambios de la variante:

- Atención deformable mediante PyTorch; ninguna importación de la extensión.
- Imports compatibles con Torch 2 para las condiciones históricas de versión.
- `weights=None` explícito en ResNet: evita descargas implícitas. [Torchvision 0.26](https://docs.pytorch.org/vision/0.26/models/generated/torchvision.models.resnet50.html).
- Propagación explícita de `group_num` a los constructores 2D y 3D; upstream
  omite este argumento y deja 1 aunque modelo y matcher reciban 11.
- Tensor de one-hot en el dispositivo de los targets; escala de la pérdida del
  mapa de profundidad derivada del shape real.
- Uso de `fy` en el prior basado en altura vertical, en lugar de `fx`.
- Rechazo de costes Hungarian no finitos antes de cualquier saneamiento.
- Pérdida de dimensiones cero y diferenciable cuando no hay objetos emparejados.

Son cambios de implementación y de semántica declarados, no una reproducción
idéntica de los resultados publicados de MonoDGP.

## Mediciones sintéticas obtenidas

RTX 5070 Ti Laptop, Torch 2.11.0+cu128, float32, batch 1. Las duraciones incluyen
comprobaciones de finitud; las lecturas breves no constituyen un benchmark de
rendimiento sostenido.

| Perfil | Resultado | Pasos calientes observados | Pico asignado por Torch |
|---|---|---|---|
| 384×1280, 11 grupos | Dos updates y decode completos | 0,472 s en el segundo paso | 2.502.063.104 bytes |
| 384×640, 11 grupos | Tres updates y decode completos | 0,299 / 0,321 s | 1.524.552.192 bytes |
| 384×640, 1 grupo | Tres updates y decode completos | 0,352 / 0,391 s | 1.286.721.536 bytes |

Los recibos son [nativo](evidence/portable_smoke_native_v2.json),
[640, 11 grupos](evidence/portable_smoke_640_group11.json) y
[640, 1 grupo](evidence/portable_smoke_640_group1.json). Se mantienen 11 grupos
para el primer experimento real: la reducción no es necesaria por memoria y
estas observaciones no acreditan una ventaja de velocidad al usar un grupo.

El primer smoke falló en el comprobador de actualización, que miraba una cabeza
extra creada pero no usada por upstream. El recibo
[fallido](evidence/portable_smoke_native.json) se conserva. Se corrigió el
comprobador para inspeccionar `class_embed[dec_layers-1]` y se repitió la ejecución.

El decode nativo produce 50 registros finitos de 14 valores. En la primera
prueba aleatoria las dimensiones físicas todavía eran inválidas: finitud y
ejecución no demuestran detecciones útiles. Esa primera variante no incluía
cabeza de velocidad; la variante posterior del piloto sí la incorpora.

## Entorno y reproducción

Se creó `.venv-detector`, separado de PHASE. La primera instalación desde la
caché de uv reveló archivos ausentes en Torch; se reinstaló exclusivamente ese
paquete desde el wheel oficial con `--no-cache`. El entorno sano de PHASE se
conservó. Las versiones instaladas están en
[portable_environment_freeze.txt](evidence/portable_environment_freeze.txt).
El [lock con hashes](evidence/portable_dependency_lock.txt) fija las dependencias
resueltas del entorno propio; no modifica `uv.lock` de PHASE.

```powershell
& .venv-detector/Scripts/python.exe -m pytest tests/unit/test_portable_attention.py

& .venv-detector/Scripts/python.exe scripts/smoke_monodgp_portable.py `
  --width 640 --height 384 --group-num 11 --steps 3 `
  --output output/portable_smoke_reproduction.json

& .venv-detector/Scripts/python.exe scripts/train_monodgp_portable.py `
  --data-root E:/eAP_dataset --manifest output/eap_rgb_pilot/manifest.json `
  --candidate-split data/splits/eap_detection_candidate_v1.json `
  --output output/portable_experiments/overfit16_reproduction `
  --role overfit16 --steps 400 --max-seconds 600 `
  --width 640 --height 384 --group-num 11 --lr 0.0002 --seed 7
```

Cada ejecución real guarda preregistration antes del primer paso: tokens,
manifest/hash, configuración, hashes de código, semilla, presupuesto y criterio
de aceptación. El criterio del primer overfit es media de `loss_center` en los
16 frames al finalizar ≤50% de la media inicial, parámetros finitos y cabeza
activa actualizada. AP y errores de detección se reportan aunque sean cero o
desfavorables; superar la pérdida no permite declarar precisión útil.

Las inferencias pasan `targets=None`, usan el RGB completo y calibración, y
decodifican sin consultar GT. La supervisión se usa después para las pérdidas
y métricas. La evaluación final recarga el checkpoint con `strict=True`.
Los pesos, imágenes y fuentes clonadas permanecen en `output/` local.

## Primer experimento real: resultado negativo conservado

`overfit16_seed7_v1` terminó por el límite preregistrado de 600 segundos de
entrenamiento: alcanzó 344 de los 400 pasos máximos. Conserva 16 frames reales,
47 objetos admitidos de seis clases; otros 33 objetos quedaron fuera del perfil
por profundidad o visibilidad, con razones registradas por frame.

| Medida sobre los mismos 16 frames train | Antes | Después |
|---|---:|---:|
| Media de `loss_center` | 0,0798634 | 0,0441130 |
| Pérdida de profundidad | 30,3357 | 2,85423 |
| Pérdida total ponderada | 211,401 | 32,4108 |
| AP101 de desarrollo, macro de seis clases presentes | 0 | 0,103415 |

**El criterio de overfit falla:** razón 0,552355, superior a 0,5. La mediana por
paso sostenido fue 1,697 s, frente a las mediciones sintéticas breves más rápidas.
El checkpoint se recargó estrictamente y los parámetros permanecieron finitos.
Los errores de los verdaderos positivos a 2 m fueron ATE 1,416 m, ASE 0,5330 y
AOE 0,08858 rad. Son medidas sobre train, no evidencia de generalización. Las
predicciones todavía contienen duplicados y geometría imprecisa.

Los artefactos completos están bajo
`output/portable_experiments/overfit16_seed7_v1`: preregistration, snapshot de
código, curva JSONL, checkpoint con optimizer, predicciones iniciales/finales y
`report.json`. Este run no se reescribe ni se reetiqueta como aprobado.

## Rendimiento sostenido y precisión numérica

El [perfil sobre un frame train real](evidence/portable_precision_profile.json)
midió tres pasos de calentamiento y diez pasos completos por configuración:

| Configuración | Mediana por paso | Resultado funcional |
|---|---:|---|
| FP32, matmul `highest` | 2,852 s | PASS |
| FP32, matmul `high` | 2,806 s | PASS |
| BF16 autocast, matmul `high` | 3,025 s | PASS |

No se obtuvo aceleración útil con BF16 en estas condiciones; se conserva FP32
`highest` para comparar con v1. BF16 elevó además la memoria privada del proceso.
La diferencia máxima de profundidad en el forward inicial de precisión mixta
fue 0,2074, registrada sin afirmar equivalencia numérica exacta.

La memoria privada del proceso se estabilizó desde el cuarto paso, alrededor de
7,0 GB para FP32. La RAM física disponible del host fue aproximadamente
0,4–1,5 GiB durante el perfil; el espacio de commit disponible llegó a menos de
0,6 GiB. La memoria GPU asignada por Torch se mantuvo cerca de 1,5 GB. Estas
observaciones apuntan a presión de memoria del host, no a saturación de VRAM;
no se cerraron aplicaciones del usuario ni se cambiaron políticas del sistema.
Las duraciones CUDA registradas son el intervalo del stream e incluyen posibles
huecos entre kernels; no equivalen a tiempo de GPU ocupada.

## Continuación del overfit y reproducibilidad

La segunda etapa `overfit16_seed7_v2_resume` recuperó pesos y estado AdamW del
checkpoint anterior, reinició explícitamente el RNG y el sampler con semilla 7
y realizó 456 updates adicionales: 800 acumulados. Su evaluación inicial fue
idéntica a la final de v1. El umbral conservado fue el 50 % de la pérdida inicial
original, 0,03993169, fijado por hash del resultado v1 antes de entrenar.

El resultado final fue `loss_center=0,02757039`, razón 0,345219, y supera ese
subgate de ingeniería. AP101 de desarrollo sobre las seis clases presentes fue
0,201421; ATE 1,08117 m, ASE 0,190580 y AOE 0,083536 rad. La etapa duró
787,156 s, con mediana 1,71877 s/update y 1,525 GB de memoria GPU asignada.
Todavía son medidas de sobreajuste sobre 16 frames, con duplicados y clases
escasas; no prueban generalización. V1 conserva su resultado FAIL.

Un proceso separado de inferencia sin cargar anotaciones reprodujo exactamente
las predicciones v2 (`max_abs_difference=0`), con mediana de forward de 44,26 ms.
Los snapshots versionables
[v1](evidence/portable_v1_code_snapshot.zip) y
[v2](evidence/portable_v2_code_snapshot.zip) conservan código, patch y hashes
exactos, porque el HEAD de aquellos runs precedía a su incorporación a Git.

## Perfil del piloto con velocidad aprendida

La variante `monodgp_pytorch_grid_sample_velocity_v1` añade una MLP de dos
componentes por decoder. La pérdida L1 suma ambos errores por objeto válido,
con coeficiente 0,1. Los targets corresponden a los campos XY publicados; no
se interpretan como velocidad relativa radial ni se usan para derivar TTC.

Se verificó el modelo completo con ocho clases y cabeza de velocidad:
42.367.706 parámetros, forward, pérdidas, backward, actualización real de las
tres cabezas utilizadas y decode finito. El
[recibo del smoke](evidence/portable_velocity_pretrained_smoke.json) conserva
estos resultados. Los pesos oficiales ResNet50 ImageNet1K V1 se descargaron de
forma explícita: 102.530.333 bytes, SHA256
`0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a`.
Se cargaron estrictamente 265 tensores del backbone, incluidos los 106 buffers
de estadísticas de 53 FrozenBatchNorm; solo se omitió el clasificador `fc`.
Las cabezas detectoras se inicializan al azar.

El [perfil con cuatro imágenes train reales](evidence/portable_velocity_batch_profile.json)
midió tres pasos de calentamiento y cinco completos por batch, a 640×384,
FP32 `highest`, group11 y tensores contiguos NCHW:

| Batch | Mediana por update | Imágenes/s | Pico GPU asignado |
|---|---:|---:|---:|
| 1 | 0,19635 s | 5,09 | 1,53 GB |
| 2 | 0,21567 s | 9,27 | 2,54 GB |
| 4 | 0,28341 s | 14,11 | 4,53 GB |

El piloto se preregistró antes de ejecutarse: 256 imágenes de dos secuencias
train, 256 imágenes de dos secuencias de selección separadas, batch 4,
512 updates máximos (2.048 exposiciones, ocho pasadas si se completa), semilla
7, AdamW con LR global 2e-4, decay 1e-4, clip 1,0 y límite de entrenamiento
1.200 s. El perfil usa el mismo LR para backbone y cabezas, como el helper
original; nuestro decay incluye también biases, diferencia registrada.
No se fija un umbral de precisión para declarar éxito científico del piloto:
se conservarán todos los resultados train y selección, incluso AP cero.
El [preflight de admisión real](evidence/pilot_target_admission.json) encuentra
829 objetos train de siete clases: `other_vehicle` tiene tres anotaciones raw
pero ninguna supera el filtro, y `bicycle` conserva solo un objeto. Selección
contiene 938 objetos de cinco clases. Hay siete frames train y uno de selección
sin targets admitidos; se conservan como fondo. Todos los objetos admitidos
disponen de dos componentes de velocidad finitas. La cabeza tiene ocho salidas,
pero este piloto no demuestra entrenamiento positivo real de las ocho clases.
La proyección breve de 145 s de cómputo no garantiza velocidad sostenida;
los perfiles previos mostraron variación de latencia y presión de memoria host.

## Resultado del piloto y coste observado

`pilot256_seed7_velocity_v1` terminó los **512 updates y 2.048 exposiciones**
previstos dentro del límite. Cada uno de los 256 tokens train apareció
exactamente ocho veces; ningún token de selección apareció en la curva de
entrenamiento. Hubo 56 exposiciones sin targets admitidos y 6.632 exposiciones
de objetos con targets de velocidad válidos. Los parámetros permanecieron
finitos, cambiaron la cabeza de clase y las tres cabezas de velocidad utilizadas,
y el checkpoint completo se recargó con `strict=True`.

El [informe íntegro](evidence/portable_pilot_report.json), la
[preregistración](evidence/portable_pilot_preregistration.json) y el
[resumen verificable de ejecución](evidence/portable_pilot_execution_summary.json)
conservan las cifras y sus hashes. Los scores siguientes son de desarrollo,
en escala 0–1; **no son resultados oficiales eAP**.

| Medida | Train, 256 frames | Selección, 256 frames |
|---|---:|---:|
| Clases presentes tras filtros | 7 | 5 |
| AP101, macro de clases presentes | 0,058375 | 0,020641 |
| AP101, ocho clases con ausentes a cero | 0,051078 | 0,012900 |
| ATE de desarrollo, m | 1,26568 | 1,22148 |
| ASE de desarrollo | 0,480624 | 0,529690 |
| AOE de desarrollo, rad | 0,174325 | 0,179422 |
| Matches a 2 m con velocidad | 492 | 401 |
| MAE de velocidad X/Y publicada, m/s | 0,33568 / 0,02634 | 0,59209 / 0,03088 |

La media de pérdida de centro train pasó de 0,0794023 a 0,0281789, y la de
profundidad de 27,3725 a 2,23247. El piloto no tenía un umbral preregistrado de
precisión para declararlo un detector útil. El AP sigue siendo bajo: en selección
solo car, truck y motorcycle tienen matches a 2 m (342, 58 y 1), mientras bus
y pedestrian no tienen ninguno. Se guardaron las 50 propuestas por imagen,
sin umbral de score para AP; sus 12.800 propuestas incluyen 12.399 no emparejadas
a 2 m. Esto requiere mejorar ranking, supresión de duplicados y geometría antes
de afirmar robustez. Las macros train/selección tienen distintos conjuntos de
clases y no deben interpretarse como una diferencia de generalización aislada.
La comparación de velocidad con cero y media train se conserva por separado en
los controles offline; el MAE por sí solo no prueba aprovechamiento de imagen.

Los [controles de imagen prefijados](evidence/pilot256_image_signal_controls.json)
obtuvieron AP101 de selección 0,004126 para el prior constante ajustado solo en
train y 0,003728 al permutar las predicciones entre frames sin puntos fijos,
frente a 0,020641 del modelo. Esto aporta evidencia descriptiva de correspondencia
con la imagen en este pequeño conjunto, sin eliminar su baja precisión absoluta.
El [control de velocidad sobre los mismos 401 matches](evidence/pilot256_velocity_signal_controls.json)
compara la cabeza con cero y la media global train: MAE X de 0,59209 / 1,38074 /
0,75762 m/s, respectivamente. En Y la cabeza es peor: 0,03088 frente a 0,02086 /
0,02062 m/s. El beneficio observado está limitado a la primera componente y a
los objetos emparejados de tres clases; no demuestra velocidad física completa
ni precisión TTC.

El entrenamiento consumió 1.096,590 s (18,28 min). La suma de los tramos de step
cronometrados fue 564,524 s, con mediana 1,02226 s; otros 532,065 s ocurrieron
fuera de ese tramo, incluyendo preparación, transferencias y escritura. Las
evaluaciones inicial train, final train y selección tardaron 101,016 / 59,600 /
53,449 s. Desde preregistración hasta cierre transcurrieron 1.316,055 s
(21,93 min), sin incluir construcción de modelo/fuentes ni carga del backbone
anteriores a la preregistración. El pico Torch asignado fue 4.544.696.320 bytes,
y el reservado 4.773.117.952 bytes. El perfil breve de 145 s no predijo el coste
integral. No se utilizó cloud ni se cargó gasto de entrenamiento externo.

Una inferencia posterior en proceso separado, sin leer anotaciones, reprodujo
exactamente las predicciones guardadas de 16 frames train y 256 de selección,
incluida velocidad (`max_abs_difference=0`). Las medianas de forward sincronizado
fueron 43,26 y 47,89 ms respectivamente; excluyen lectura, preparación, decode,
asociación y PHASE. Los siete frames temporales del mismo checkpoint se
entregaron al puente de PHASE por separado.

## Reproducción del piloto registrado

El run utilizó el commit `3ebac039e56e3e5af98e0e3302084f00923235a1` y el
[snapshot de diez fuentes y patch](evidence/portable_pilot_code_snapshot.zip).
La reproducción histórica debe usar esa fuente y el lock del detector, con el
manifest SHA `048636cd79e8f5160b97a8056b0e2ed6f78846f4bbb365b77efa5884766be159`
y el backbone oficial ya verificado. El comando ejecutado fue:

```powershell
& .venv-detector/Scripts/python.exe scripts/train_monodgp_portable.py `
  --data-root E:/eAP_dataset --manifest output/eap_rgb_pilot/manifest.json `
  --output output/portable_experiments/pilot256_seed7_velocity_v1 `
  --role train --steps 512 --max-seconds 1200 --group-num 11 --batch-size 4 `
  --width 640 --height 384 --lr 0.0002 --seed 7 --predict-velocity `
  --velocity-loss-coef 0.1 `
  --backbone-weights output/external_weights/resnet50-0676ba61.pth `
  --precision fp32 --matmul-precision highest
```

El runner rechaza sobrescribir un directorio existente: usar otro directorio
para una nueva réplica. El guard de admisión añadido después de este run exige
además `--candidate-split data/splits/eap_detection_candidate_v1.json`; esa
validación del run histórico es retrospectiva y no debe atribuirse al runner
congelado. El checkpoint local pesa 498.531.791 bytes y tiene SHA256
`dc0ce304ad1c8a791ef66395be85b09f22526bc11a834472dcb18ce16384e338`.

## Límites científicos y licencias

Este experimento usa el modelo de proyección pinhole del visualizador original
y su aproximación RGB remapeada; no resuelve por sí mismo todos los subgates de
rectificación, velocidad o scorer. El adaptador eAP cuenta cajas descartadas por
profundidad/visibilidad, conserva las dimensiones explícitas y exporta score de
clase sigmoid. No produce una submission oficial ni inventa velocidad.

El operador deriva de la referencia Deformable DETR, con copyright SenseTime
2020 y licencia Apache 2.0; se conserva la
[licencia completa](evidence/deformable_detr_LICENSE.txt), obtenida del
[repositorio original](https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE).
MonoDGP distribuye su repositorio bajo MIT, copyright Fanqi Pu 2025; la copia
local preserva su `LICENSE` y las cabeceras específicas de los archivos.
