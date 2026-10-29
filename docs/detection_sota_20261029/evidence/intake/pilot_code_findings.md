# Auditoría de código del piloto para el plan posterior

## Alcance y evidencia

- Auditoría de solo lectura del repositorio `phase-ttc-sota-review`, commit `24b8f36f6e4aa17c965f9526d8cf486b512a8046`, incluyendo `AGENTS.md`, los tres informes nuevos y código real. No se modificó el repositorio, no se entrenó, no se accedió a los datasets ni a checkpoints pesados y no se reprodujeron las 232 pruebas publicadas.
- Referencia MonoDGP inspeccionada localmente en `external/MonoDGP`, revisión `aa059a18214aebf644510e7f0793971b403f9d14`.
- Las curvas archivadas se analizaron con Python/stdlib; no requieren GPU. El snapshot ZIP de código del piloto confirma el optimizador y ausencia de scheduler del entrenamiento real. Los guards de admisión actuales se incorporaron después: no atribuirlos retroactivamente a la ejecución original.

## Conclusión principal

Hay una implementación funcional y evidencia de aprendizaje en un protocolo pequeño. El cuello todavía no está identificado con resolución suficiente para atribuirlo a la arquitectura. La prioridad es completar cobertura/protocolo, diagnosticar el checkpoint fijo y convertir el runner mínimo en un entrenador reproducible, antes de cambiar backbone, introducir eventos o entrenamiento compartido. Una campaña más larga sin estas comprobaciones podría optimizar una población incompleta y objetivos auxiliares que todavía no representan bien eAP.

## Hechos del código y consecuencias

| Componente | Hecho comprobado | Consecuencia / decisión propuesta |
|---|---|---|
| Entrenador | `scripts/train_monodgp_portable.py:375–376`: un AdamW con LR 2e-4 y weight decay 1e-4 para todos los parámetros entrenables. No scheduler, warmup, EMA ni LR diferencial por backbone. | Añadir configuración explícita y comparar recetas controladas; no afirmar que se reprodujo la receta completa de MonoDGP. |
| Backbone | MonoDGP `backbone.py:71–73`: stem/layer1 congelados; layer2–4 entrenables con `train_backbone=True`; FrozenBatchNorm. | No describirlo como encoder totalmente congelado ni totalmente entrenable. Registrar nombres/grupos/contadores y hacer ablation de LR por grupo si el diagnóstico lo justifica. |
| Inicialización | `backbone_initialization.py:12–15,72–153`: ResNet50 IMAGENET1K_V1, hash `0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a`, carga estricta y comparación exacta de buffers y parámetros. Cabezas nuevas aleatorias. | Mantener esta base verificable. Cambiar preentrenamiento es experimento posterior; no vender una actualización de pesos como corrección de pesos ausentes. |
| Referencia upstream | `configs/monodgp.yaml`: batch8; máximo250 épocas; LR2e-4; escalones85/125/165/205 a factor0.5; warmup=False. `optimizer_helper.py:15–23` excluye biases del decay, pero NO usa LR distinto para backbone. | Si se propone warmup o LR del backbone 0.1×, marcarlo como propuesta, no restauración exacta del paper. No copiar 250 épocas sin definir exposición y población eAP. |
| Datos | `monodgp_eap_adapter.py:103–109`: resize bilinear a640×384 + normalización ImageNet; sin augmentations. Upstream usa1280×384, photometric distortions, crop, flip y mixup3D. | Medir escala de objetos; comenzar por photometric augmentation, y añadir transforms geométricos solo con actualización exacta de K y fixtures. No trasladar flip/crop KITTI sin revisar geometría eAP. |
| Targets | `monodgp_eap_adapter.py:119–151`: profundidad[0.1,60), todas las esquinas delante, centro proyectado dentro, rectángulo recortado con tamaño>1px y distancias l/r/t/b no negativas. `native_ground_truth` se crea solo tras estos filtros. | La evaluación publicada es de esa población admitida, no del universo anotado. Mantener métrica histórica y crear un perfil nuevo, con auditoría de denominadores y exclusiones antes de entrenar. |
| Cajas2D | `monodgp_eap_adapter.py:140–164`: rectángulo de cuboide3D proyectado y recortado; región foreground es unión de esos rectángulos. | La región NO está a cero para positivos. Es amodal proyectada; no es una máscara de visibilidad ni caja2D anotada. Comprobar degradación por oclusión, truncación y filtros. |
| Región upstream | `kitti_dataset.py:296–309`: caja2D anotada; región se rellena antes de rechazar centros fuera de imagen323–359. | La diferencia es real; el efecto no está medido. Considerar regiones ignore para objetos visibles rechazados o ablation de loss_region; nunca llamar a todos los ignorados objetos visibles. |
| Profundidad | Upstream `monodgp.py:247–263`, parche `smoke_monodgp_portable.py:164`: z=fy·H_pred/h2D_pred + error aprendido. No es un promedio de tres profundidades; ese código está comentado. | Medir con GT la validez del prior según clase, yaw, distancia y truncación, y el tamaño de la corrección. La pinhole reproduce una aproximación oficial sobre el plano distorsionado, no una rectificación perfecta. |
| Dimensiones | Targets absolutos h,w,l, `adapter.py:161`; upstream `meanshape=False` también produce mean_size cero. Cabeza unconstrained y decoder descarta tamaños≤0. | No hay evidencia de un bug por faltar class means. Probar log-dimensiones/softplus o residual sobre medias obtenidas SOLO en train si la auditoría muestra ese cuello. Registrar descartes e inicialización. |
| Yaw | Adapter154–156 usa rayo del centro3D para alpha; decoder304 invierte con el mismo rayo y rotación completa. Upstream usa centro2D de bbox para su alpha. | No restaurar ciegamente alpha KITTI. Mantener fixtures independientes y resolver por separado la convención del scorer oficial. |
| Queries | Config:50 queries,11 grupos; adapter184–186 falla si hay más de50 targets, no trunca silenciosamente. | Auditar densidad de escenas nuevas. El coste del entrenamiento crece con grupos; no cambiar queries/grupos por intuición ni silenciosamente. |
| Score | `decode_eap:283–284`: una clase por query, score=max sigmoid, top50 queries. Upstream toma topK sobre Q×C y multiplica por exp(−logvar_depth). | Exportar score de clase y calidad por separado; comparar ranking con checkpoint fijo, sin asumir que la incertidumbre está calibrada. |
| Datos de diagnóstico | Decoder lee pred_depth[:,1] pero no lo exporta; tampoco query_id, logits completos, caja2D, prior y residual. | Añadir sidecar diagnóstico sin cambiar predicciones del baseline y repetir inferencia sin labels. Indispensable para atribuir errores. |
| NMS | No NMS en evaluación detectora. Bridge sí tiene NMS class-aware de ROIs. | No explicar AP por falta de NMS sin análisis. DETR es set predictor; probar NMS solo como ablation si hay duplicación demostrada. |
| Velocidad | Target copia dos componentes publicados con máscara missing, loss L10.1 y cabeza por decoder. Semántica física completa no validada. | Su mejora frente a cero no valida velocidad física ni TTC. Separar contrato publicado, estimación física y score de asociación. |
| Entrada/salida | `PreparedCache`32, CPU sin DataLoader/prefetch; GPU transfer síncrono. `EAPRGBDataset`:2 shards/2 secuencias cache, tar.getnames por apertura y SHA256 por lectura. | Perfilar RAM, I/O, preparación y transferencia antes de cambiar arquitectura. Ampliar secuencias puede causar thrashing. Usar workers limitados, índice/prefetch acotados y recibos de contenido. |
| Atención | `portable_attention.py:33,45,88`: conversiones de shape y verificaciones que pueden sincronizar CPU/GPU; `grid_sample` por nivel. | Optimización posterior guiada por profiler. No claim de equivalencia del kernel CUDA original; CUDA backward puede no ser bitwise determinista. |
| Resume | Runner293 reinicia RNG/sampler declaradamente; checkpoint479–495 guarda modelo/optimizador, no RNG ni cursor de muestreo. Solo checkpoint final y validación final. | Para campañas largas añadir checkpoints periódicos, RNG/sampler/scheduler/AMP states y protocolo de selección. No confundir resume documentado con replay bitwise. |

## Curvas recalculadas

Archivo: `docs/detection_local_20261029/evidence/training_curves/pilot256_seed7_velocity_v1.jsonl`.

Cada una de las256 imágenes aparece8 veces,2048 exposiciones y512 updates. Al agrupar64 updates por pasada:

| Pasada | Pérdida media ponderada | Loss centro media | Gradnorm preclip medio |
|---:|---:|---:|---:|
|1|62.361467|0.081357|93.313341|
|2|41.365748|0.061217|65.469535|
|3|36.098981|0.046995|73.867912|
|4|33.587438|0.038861|82.377130|
|5|30.088573|0.032805|76.955841|
|6|28.226374|0.028978|92.964993|
|7|27.333711|0.025331|91.029482|
|8|25.704930|0.025395|103.987560|

Los512 gradnorms antes de clip son>1: mínimo25.430437, mediana70.487160, máximo1775.952026. Esto confirma clipping frecuente, pero NO demuestra explosión ni que quitarlo ayude. Hay que instrumentar gradientes por cabeza y ratios actualización/peso, pérdidas individuales y sus escalas. La pérdida de entrenamiento baja; no hay curva AP de validación intermedia para afirmar convergencia, saturación ni duración óptima. El gap train0.05837 vs selección0.02064 tampoco usa la misma macro:7 vs5 clases.

ASE selección: car0.16515, truck0.74103 y motorcycle0.68289 (solo1match). Es motivo para inspeccionar dimensiones por clase, no prueba de orden de dimensiones incorrecto. ATE/AOE/ASE macro solo cubren clases con matches y un único match motorcycle aporta1/3 de la macro de error; no describen todas las clases.

## Experimentos prioritarios y decisiones

### P0: contrato de evaluación y población

Mantener intacto `rgb_box_v1` y las cifras históricas. Versionar un perfil amplio que especifique denominadores raw/admitted, rango, visibilidad, truncación, max_queries, clases y coste. Auditar por secuencia/sesión los positivos de las ocho clases y la frecuencia de filtros. No puede resolverse `other_vehicle` con una nueva cabeza: ya hay ocho salidas, faltaron positivos admitidos. La política de muestreo debe asegurar soporte por grupos independientes y reportar las clases imposibles; una sola bicicleta no es evidencia de generalización. Hacer varios splits/semillas solo después de fijar el protocolo y reservar un conjunto de confirmación no usado para escoger variantes. Resolver el scorer antes de presentar cifras oficiales; mientras tanto, métricas de desarrollo etiquetadas.

### P1: diagnóstico congelado, sin nuevo entrenamiento

1. Añadir export opcional de `query_id`, ocho logits, UV, lrtb, depth, logvar, dimensiones, alpha, prior/corrección, velocity. Sidecar versionado; comprobar que activarlo conserva exactamente la salida normal.
2. Evaluar recall class-agnostic por distancia/rango/tamaño y clase verdadera, manteniendo matching determinista y denominadores. Si recall de propuestas es bajo, la calibración de score no lo recuperará.
3. Oráculos exclusivamente diagnósticos: GT class, centro/profundidad, dimensiones/yaw, ranking perfecto sobre el MISMO conjunto de propuestas y matching predefinido. Separar upper bounds y no presentar outputs con GT como predicciones del sistema.
4. Distribuciones de errores en profundidad vs UV, tamaño por componente y cola, yaw circular/simétrico según contrato; precisión-recall y contribuciones por secuencia.
5. Correlación score/TP y logvar_depth/error; duplicados por instancia vs fondos vs clase incorrecta. Comparar one-class/query con topK(Q×C) con presupuesto constante. Solo ensayar NMS si la duplicación es sustancial. Temperatura monótona global no mejora AP al conservar ranking; útil para calibrar umbrales de asociación, no como solución automática a AP.
6. Auditoría de prior con GT: `z - fy*H/h2D` usando caja completa y recortada, mediana/colas por clase, rango, yaw y borde de imagen. Reproyecciones visuales estratificadas revisadas por humano/agente. Comprobar cuánto de `loss_region` y `loss_depth_map` etiqueta como fondo a objetos visibles excluidos.

Archivos: `monodgp_eap_adapter.py`, `infer_monodgp_eap.py`, un nuevo analizador de errores y fixtures/tests dedicados. No modificar scorer oficial para igualar implementación local.

### P2: entrenador escalable y baseline RGB más amplio

Cambios: configuración declarativa; ledger de parámetros; optimizer groups configurables; scheduler por updates/exposiciones; evaluación periódica en selección; best/last checkpoint con manifest de selección; estados RNG/sampler/scheduler; sampler y reader con memoria acotada; profiler por etapa. Preservar el runner piloto o compatibilidad de receta y snapshots.

Comenzar con el R50/ImageNet verificado. Comparar a igual población/exposición el baseline receta actual con scheduler y luego augmentación fotométrica. Resolver cobertura de ocho clases antes de amplificar raras: muestreo por clase a nivel de secuencia/frame y reponderación son ablations diferentes. Registrar su distribución efectiva y limitar repetición de casos únicos. Probar mayor resolución con K/centros actualizados y presupuesto medido en GPU local; batch/grad accumulation puede cambiar dinámica, por lo que no es equivalencia automática.

No fijar una campaña de250 épocas porque eso sea el upstream KITTI. Preregistrar primero presupuesto y puntos de evaluación; ampliar cuando las curvas de validación y recursos indiquen beneficio y el procedimiento de selección lo permita. En prueba final usar semillas múltiples y dispersión entre grupos, sin inventar significación a partir de frames correlacionados.

Pruebas/gates: arrays de targets/decoding equivalentes con lectura optimizada; geometría de resize/flip/crop con K; no acceso a selection/test dentro del sampler; cobertura de clases; checkpoint/resume conserva estados y orden programado; LR efectivo esperado en milestones; parámetros congelados y actualizados comprobados; pérdidas/gradientes finitos; limites RAM/VRAM y rendimiento medido tras calentamiento. Comparar tolerancias numéricas explícitas en CPU/FP32; no prometer replay CUDA bitwise con grid_sample.

### P3: modificaciones guiadas por P1

- Si domina tamaño/profundidad: positivo por softplus/log-dim, residual sobre medias solo train, prior más robusto para truncación/pitch/roll, o ablation de dependencia geométrica; una variable por experimento y volver a validar decoding/geometría.
- Si domina clasificación/cola: soporte de escenas, sampler balanceado vs pesos de pérdida/focal; inspeccionar false positives de clases mayoritarias y coste en recall común.
- Si domina ranking: rama de calidad geométrica entrenada con labels train/OOF o score combinado validado; conservar `detection_score`, `quality_score`, `ttc_confidence` distintos. La fórmula original exp(−logvar) es un control, no certeza de mejora.
- Si domina poca resolución: resolución mayor o multiescala con prueba de K y stride; comparar coste total y colas de latencia, no solo FPS del detector.
- Si faltan propuestas en escenas densas: queries/grupos parametrizados, pero mantener comparable la métrica y estudiar coste del matching.
- Comparar MonoDETR control con el MISMO adaptador/protocolo/initialization/budget; cambiar a ImVoxelNet solo si el port/inductiva está justificado por resultados. No inferir superioridad en eAP de liderazgo en KITTI.

### P4: asociación, temporalidad, eventos y PHASE

El bridge actual toma el objeto de mayor score del último frame (`infer_detected_roi_phase.py:287`), asociación greedy con IoU2D/clase solo al frame anterior y resetea por gap150ms; demuestra una trayectoria, no servicio multiobjeto robusto. Registra causalidad por cutoff del sensor, no latencias de llegada medidas (`physical_arrival_timestamps_measured=False`). TTC no tiene verdad alineada en el ejemplo y TRAIN40 ya expuso PHASE a esas secuencias.

Tras baseline RGB: servicio por track con estado explícito, covarianza/quality/madurez, gates3D+2D, oclusiones y predictor causal, split/merge/ID-switch con reinicio de historia cuando corresponda, outputs abstain por falta de historia. Emitir cobertura y riesgo-condicional TTC; evaluar primero detector fijo/tracker fijo/PHASE fijo y separar mejoras de detección, identidad y TTC. Velocidad single-frame actual no valida física: comparar track finitedifference con dt/extrínseca/egomotion correctamente definidos. No integrar este vector como TTC cinemático mientras semántica esté bloqueada.

Eventos después: rama tardía/cross-attention con causalidad de ventanas, controles RGB solo, eventos vacíos/permutados y presupuesto comparable; medir ganancias en fallos RGB observados y coste/calibración. Backbone compartido solo si se demuestra beneficio frente a módulos independientes y se controla interferencia entre losses, exposición TRAIN40 y latencia. No es primera tarea para elevar AP0.02064.

## Riesgos concretos y alternativas

- No dataset/checkpoints en este entorno: análisis de artefactos y código es reproducible, pero oráculos, nuevo training y GPU deben ejecutarse donde están los medios con hashes fijados.
- Cualquier ambigüedad oficial yaw/velocity/rectificación mantiene gate correspondiente bloqueado; la prueba del bundle permite un baseline local, no disuelve esos bloqueos.
- Cuello RAM/IO demostrado en perfiles; FP16/BF16 y mayores batches requieren evaluación nueva en condiciones controladas, no asumir velocidad por soporteCUDA.
- Aumentar dataset sin revisar prepared cache32 y tar LRU2 puede empeorar throughput. Priorizar indexado y prefetch acotados sobre cargar todo.
- Narrow G0_RGB_BOX_V1/G1_PORTABLE_V1/G2 no significan G0/G1 universales; preservar el nombre de variante y evidencia histórica.

