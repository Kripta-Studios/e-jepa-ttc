# Informe de implementación: detector 3D eAP y puente PHASE-TTC

Fecha: 2026-10-29. Estado documentado: commit `5c9fc73062f109f808be732a4a10d6f26a91afee` de la rama `dev/eap-detection-local-20261029`.

## Objetivo y resultado de ingeniería

El trabajo añadió a PHASE-TTC una ruta experimental que parte de una imagen RGB eAP completa y produce detecciones con clase, centro 3D, dimensiones, yaw, velocidad planar y confianza. Una asociación causal convierte las detecciones temporales en una historia de ROIs. El runtime PHASE existente consume esa historia junto con RGB y eventos para devolver TTC firmado, fase y límites de fase.

La cadena ejecutada es:

```text
RGB eAP + K_event + T_event_ego
  -> MonoDGP portable
  -> cajas eAP y velocidad XY aprendida
  -> NMS y asociación causal
  -> ROI común + historia RGB/eventos
  -> pesos finales PHASE
  -> TTC firmado + fase + q10/q90 de fase
```

Esta implementación resuelve la viabilidad del flujo y permite entrenar, guardar, recargar e inferir en la GPU local. Su alcance científico sigue acotado: no equivale a una puntuación oficial eAP, no valida las ocho clases con soporte suficiente y no mide precisión TTC nueva. Los resultados cuantitativos están separados en [RGB_TRAINING_RESULTS.md](RGB_TRAINING_RESULTS.md) y [TRAINING_TABLES.md](TRAINING_TABLES.md).

## Componentes incorporados

| Archivo | Responsabilidad |
|---|---|
| `src/e_jepa_ttc/detection/eap_rgb_data.py` | Acceso acotado a Parquet y TAR RGB, rutas seguras, cachés limitadas, calibración y etiquetas raw |
| `src/e_jepa_ttc/detection/geometry_contract.py` | Construcción de corners según el bundle original, validación de calibración y disponibilidad causal |
| `src/e_jepa_ttc/detection/monodgp_eap_adapter.py` | Conversión eAP↔targets MonoDGP, resize de intrínseca, encoding de yaw y decode a cajas nativas |
| `src/e_jepa_ttc/detection/portable_attention.py` | Sustitución diferenciable en PyTorch de la atención deformable externa |
| `src/e_jepa_ttc/detection/backbone_initialization.py` | Carga explícita y verificada del backbone ResNet-50 ImageNet1K V1 |
| `src/e_jepa_ttc/detection/velocity_extension.py` | Cabezas de velocidad, target con máscara y pérdida L1 para los dos componentes publicados |
| `src/e_jepa_ttc/detection/development_metrics.py` | AP101 y errores ATE/ASE/AOE de desarrollo con contrato explícito |
| `src/e_jepa_ttc/detection/velocity_diagnostics.py` | Diagnóstico de velocidad sobre matches de clase y distancia XY |
| `src/e_jepa_ttc/detection/pilot_admission.py` | Validación fail-closed del manifest, split, hashes, roles y disjunción antes de usar el modelo |
| `scripts/train_monodgp_portable.py` | Preregistración, entrenamiento, evaluación, checkpoint y recibos reproducibles |
| `scripts/infer_monodgp_eap.py` | Inferencia sin etiquetas, carga de fuentes congeladas, hashes de medios/K/T y comprobación de paridad |
| `scripts/infer_detected_roi_phase.py` | NMS, asociación causal, preparación sensorial y llamada al runtime PHASE |
| `scripts/check_detection_image_signal.py` | Controles de prior constante y permutación de predicciones |
| `scripts/check_detection_velocity_signal.py` | Comparación de la cabeza de velocidad con cero y media train sobre los mismos matches |

La implementación conserva `e_jepa_ttc` como namespace y no modifica pesos, resultados o helpers históricos de PHASE. La comparación contra la base `f2903f0bfd47d7bf56358a94ad5fac0b8a8906ef` no encontró archivos históricos modificados, eliminados o renombrados. La [revisión independiente](INDEPENDENT_FINAL_REVIEW.md) comprobó esa preservación.

## Contrato de datos y selección del piloto

`EAPRGBDataset` recibe una lista explícita de `sample_token`. Lee solo las filas necesarias de `data/train.parquet`, abre los shards RGB correspondientes y recupera las etiquetas de la secuencia. Cada `RawRGBSample` transporta imagen RGB `uint8` HWC, anotaciones sin reinterpretar, intrínseca, transformación ego→evento, timestamps de exposición, procedencia del medio y SHA-256 del miembro RGB.

La selección fijó dos secuencias train y dos de selección, con 128 frames por secuencia. Los 256 tokens de cada rol son únicos y no comparten secuencias. Los 16 frames del overfit pertenecen a train. El selector toma posiciones equiespaciadas sin usar etiquetas ni resultados. El [recibo de admisión](evidence/pilot_manifest_admission.json) liga manifest, split y `data/train.parquet` mediante hashes.

La protección actual se ejecuta antes de construir el detector. Comprueba esquema, revisión del dataset, roles train/selection/calibration, duplicados, prefijos, rutas públicas train, disjunción y pertenencia de `overfit16`. Se añadió después del piloto; el recibo retrospectivo confirma los inputs usados sin atribuir esa validación al código antiguo.

## Resolución geométrica

La geometría se resolvió contra el bundle original y el visualizador oficial fijado. Se recuperó `annotations.pkl` de una secuencia train mediante una descarga Range acotada y se comparó con la release HF. Las 17.611 filas emparejadas por secuencia, nombre de frame e instancia coinciden en cajas, clase, velocidad, calibración y exposición.

El adaptador usa el contrato documentado en [GEOMETRY_RESOLUTION.md](GEOMETRY_RESOLUTION.md):

- `bbox_3d_ego = [x, y, z, length, height, width, yaw]`;
- centro geométrico y ejes ego X hacia delante, Y a la izquierda, Z hacia arriba;
- rotación `Rz(-yaw)` y transformación `T_event_ego` de ego a cámara;
- salida semántica `size_lwh_m = bbox[[3, 5, 4]]`.

Tres imágenes RGB reales y 19 cuboides produjeron corners compatibles con la función oficial y un residual máximo de 0,9575 px frente a las cajas 2D publicadas después de clipping. El adaptador aplica el resize a la intrínseca con la transformación de centros de píxel, conserva el centro geométrico y convierte el rumbo entre los marcos ego/cámara de forma invertible. Rechaza profundidad, tamaño o valores no finitos; cuenta cada descarte en vez de sanearlo en silencio.

La prueba no certifica rectificación RGB/eventos perfecta ni la convención que emplea el scorer privado para yaw. El modelo pinhole reproduce la aproximación del visualizador sobre RGB remapeada al plano distorsionado del sensor de eventos.

## Port de MonoDGP

MonoDGP quedó fijado a la revisión `aa059a18214aebf644510e7f0793971b403f9d14`. No se obtuvo una compilación validada de su extensión CUDA externa. La ruta ejecutada usa RTX 5070 Ti Laptop, el runtime CUDA 12.8 de PyTorch 2.11 y un backend PyTorch con `grid_sample`, sumas ponderadas por nivel y autograd. Los tests incluyen casos analíticos, rechazo de NaN, `gradcheck` CPU float64 y comparación CPU/GPU de salidas y gradientes.

El port corrigió la propagación de `group_num`, el escalado del mapa de profundidad, el uso de `fy` en el prior vertical, la creación de tensores en el dispositivo correcto y el rechazo de costes Hungarian no finitos. El score exportado es la probabilidad sigmoid de clase.

El piloto conserva ResNet-50, cuatro niveles de features, tres capas de encoder/decoder, 50 queries y 11 grupos. El backbone ImageNet se cargó desde un archivo con SHA-256 conocido: coincidieron 265 tensores, incluidos parámetros y estadísticas FrozenBatchNorm; las cabezas detectoras empezaron con inicialización aleatoria. [PORTABLE_DETECTOR.md](PORTABLE_DETECTOR.md) registra el entorno, los patches y los perfiles.

## Cabeza de velocidad

La variante de velocidad añade una MLP de dos salidas a cada capa activa del decoder. El target contiene `velocity[N,2]` y `velocity_valid[N]`. La pérdida suma el error absoluto de ambos componentes por objeto válido, con coeficiente 0,1, y conserva una salida diferenciable cero cuando no existen targets válidos. La velocidad no participa en el coste de matching.

El decode solo emite `velocity_mps` cuando el checkpoint contiene la cabeza aprendida y los valores son finitos. No fabrica `[0,0]` para modelos anteriores. Los dos componentes mantienen el significado nativo de la release pública; el código no los declara velocidad radial ni compensada por ego-motion y nunca deriva TTC a partir de ellos. [NATIVE_VELOCITY_HEAD.md](NATIVE_VELOCITY_HEAD.md) documenta el contrato y las pruebas matemáticas.

## Asociación y puente hacia PHASE

El detector trabaja sobre imágenes completas. Después del decode, el puente aplica NMS por clase y asociación causal por IoU, clase, margen de ambigüedad y gap temporal. Si no obtiene una historia suficiente o inequívoca, se abstiene. Para la ejecución registrada construyó una ROI común a siete frames y recortó cuatro observaciones de eventos y cinco RGB según sus timestamps disponibles.

El runtime carga PHASE mediante `CompetitionRuntime.from_candidate`. Un hook observa las salidas de la cabeza final y comprueba la paridad del TTC. El puente mantiene separados clase/confianza, caja 3D, velocidad, TTC, fase y q10/q90. Los límites pertenecen al espacio de fase; no forman un intervalo TTC continuo cuando cruzan cero.

La [salida integrada](evidence/integrated_prediction_pilot_v1.json) liga checkpoint, predicción temporal, índice de caja, pesos PHASE y fuentes mediante hashes. El ID del track solo vale dentro de esa ejecución. La ROI elegida por el piloto no representa el mismo objeto que la del overfit anterior, aunque ambos ejemplos usen el mismo token de consulta.

## Fallos detectados y correcciones

| Problema | Corrección aplicada |
|---|---|
| El smoke comprobaba una cabeza de clase creada pero no usada | Se inspecciona `class_embed[dec_layers-1]`, la cabeza activa |
| El lector cargaba Parquet completo antes de filtrar | Se filtran tokens mediante Arrow antes de materializar pandas |
| El runner aceptaba manifests sin comprobar separación | `pilot_admission.py` valida hashes, roles y filas canónicas antes del modelo |
| `--tokens-json` no ligaba siempre la calibración | Inferencia registra el SHA del manifest y hashes de K/T por frame |
| Comparación exacta de floats fallaba entre intérpretes | Tolerancia recursiva `1e-12` solo para floats; tipos y estructura siguen exactos |
| Git podía normalizar bytes de recibos | `evidence/.gitattributes` desactiva normalización para archivos con hashes |
| Repetir ingestión para procesar más objetos duplicaría buffers PHASE | El puente offline prepara una historia única; la refactorización del servicio streaming multiobjeto sigue pendiente |

Las correcciones no reescriben recibos históricos. Los runs anteriores al commit que incorporó su fuente incluyen ZIPs con los bytes exactos del código ejecutado.

## Gates resueltos y límites pendientes

Los gates amplios del handoff original continúan registrados como bloqueados. Se abrieron dos admisiones más estrechas:

| Admisión | Evidencia disponible | Lo que no demuestra |
|---|---|---|
| `G0_RGB_BOX_V1` | Correspondencia original/HF, reproyección real, contrato de targets y medios train seleccionados | Scorer privado, rectificación perfecta, semántica física completa de velocidad, medios Test12 completos |
| `G1_PORTABLE_V1` | Detector completo, forward/backward, actualización, decode, recarga e inferencia independiente | Compilación o paridad con la extensión CUDA externa, determinismo bit a bit, precisión robusta |
| Puente PHASE de desarrollo | Asociación causal, ROIs detectadas y salidas nativas con hashes | Precisión TTC, calibración q10/q90, identidad multiobjeto general o latencia extremo a extremo |

No se usaron etiquetas Test12 ni se realizó una submission. El piloto contiene siete clases train admitidas y cinco en selección; `other_vehicle` carece de positivos admitidos y `bicycle` conserva un solo objeto train. PHASE ya estuvo expuesto a TRAIN40, incluidas estas secuencias, por lo que la separación del detector no crea una evaluación independiente de PHASE.

## Decisiones siguientes

La siguiente versión debe ampliar el muestreo bajo un protocolo fijado antes de observar resultados para conseguir soporte de las ocho clases. También debe agrupar sesiones relacionadas, ejecutar varias semillas y evaluar secuencias sin exposición de PHASE. Antes de una submission debe reproducir el contrato del scorer para yaw, velocidad y errores TP.

La comparación RGB frente a RGB+eventos necesita el mismo detector, presupuesto y split, más controles con eventos anulados o desordenados. La asociación requiere pruebas con cruces, oclusiones y múltiples objetos, junto con una medida de la degradación TTC causada por errores de ROI e identidad. Compartir backbone o entrenar ambas tareas conjuntamente queda después de esas mediciones.

La entrega versiona código, documentación, licencias, hashes, manifiestos de tokens públicos, curvas escalares y recibos agregados. El dataset eAP permanece en `E:/eAP_dataset`; los pesos PHASE, en su candidato local de D:. El checkpoint del detector de 498 MB, los renders y las predicciones fila a fila permanecen en `output/`. [final_rgb_verification.json](evidence/final_rgb_verification.json) registra 232 tests correctos, Ruff/formato sobre 49 archivos y Pyright sin errores en ocho módulos nuevos; esa suite cubre la extensión de detección y el puente, no todo PHASE ni una evaluación oficial.
