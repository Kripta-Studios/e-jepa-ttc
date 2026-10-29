# G1 — Viabilidad de MonoDGP y control MonoDETR

Fecha de inspección: 29 de octubre de 2026. PHASE-TTC base inicial:
`e58d0106e1f114f00f796987f70cd622cc6d186e`; base final inspeccionada:
`f2903f0bfd47d7bf56358a94ad5fac0b8a8906ef`. El nuevo commit solo añade los dos
informes previos, idénticos a los adjuntos; código y entorno histórico no cambian.

**Estado: G1 bloqueado; G2 no ejecutado.** Se han inspeccionado fuentes,
creado un entorno aislado para las auditorías y probado las comprobaciones
nuevas. No se ha ejecutado un detector, compilado su extensión ni medido su
memoria en GPU. El equipo accesible es un entorno Linux sin PyTorch instalado,
sin `nvidia-smi`, sin `nvcc` y sin dispositivo CUDA accesible. Esto no describe ni
certifica el estado actual del portátil del usuario.

## 1. Evidencia y alcance

| Comprobación real | Resultado | Evidencia |
|---|---|---|
| Clones públicos fijados y archivos inspeccionados | Completada | [fuentes y hashes](../../configs/detection/detector_sources_v1.json) |
| Entorno de auditoría separado, CPython 3.12.14 | Creado | [paquetes exactos](evidence/g1_audit_environment_freeze.txt) |
| Pruebas de bloqueo y separación entre prerequisitos/modelo | 11 pruebas pasan | [salida pytest](evidence/g1_tests.txt) |
| Ruff de los dos scripts y sus pruebas | Pasa | Comando reproducible en §7 |
| Inventario de GPU, Torch y compilador | Bloqueado | [entorno](evidence/g1_environment.json), salida 2 |
| Paridad CUDA MonoDGP | No ejecutada: falta Torch/CUDA | [resultado](evidence/g1_monodgp_operator.json), salida 2 |
| Paridad CUDA MonoDETR | No ejecutada: falta Torch/CUDA | [resultado](evidence/g1_monodetr_operator.json), salida 2 |
| Forward, backward, pérdidas y decode de detector completo | No ejecutados | Campos explícitos `NOT_RUN` en el inventario |
| VRAM pico del detector | No medida | `null`, nunca cero |

La sintaxis Python de los clones se analiza en
[g1_upstream_static_audit.json](evidence/g1_upstream_static_audit.json).
Poder analizar un archivo con `ast.parse` no demuestra que sus imports,
operadores, entrenamiento o inferencia sean compatibles. Los avisos de escapes
en docstrings tampoco equivalen a fallos CUDA.

La receta GPU queda como
[environment_candidate_v1.json](../../configs/detection/environment_candidate_v1.json):
**propuesta, no entorno resuelto ni probado**. Mantiene Torch `2.11.0+cu128` y
Torchvision `0.26.0+cu128`, versiones ya fijadas por el `uv.lock` histórico de
PHASE, en otro entorno. No se ha tocado `pyproject.toml`, `uv.lock` ni el entorno
de PHASE. El lock completo del port queda pendiente: fijar dos paquetes no es
una demostración de reproducibilidad del detector.

## 2. Candidatos y selección

| Candidato | Revisión inspeccionada | Decisión |
|---|---|---|
| [MonoDGP][dgp] | `aa059a18214aebf644510e7f0793971b403f9d14` | Primer candidato para port y medición |
| [MonoDETR][detr] | `6994b9f512400b258c6edb75f77423beb9c126f2` | Control de familia; no fallback de compatibilidad independiente |
| [ImVoxelNet en MMDetection3D][imvoxel] | No se ha fijado ni instalado una versión local | Alternativa si la ruta anterior no resulta viable; necesita su propio gate |

MonoDGP aporta queries visuales desacopladas, corrección del error geométrico de
profundidad y una cabeza de regiones; estos mecanismos se describen en el
[artículo original CVPR 2025][dgp-paper] y aparecen en el código inspeccionado.
Es una base monocular sin necesidad de poses temporales para comenzar. MonoDETR
sirve como control cercano porque MonoDGP deriva de esa familia. **No se ha
comparado ninguno en eAP ni se afirma que MonoDGP vaya a ganar.** Los resultados
KITTI publicados no estiman AP eAP por distancia de centros.

Las licencias raíz son MIT (`LICENSE` en MonoDGP y `LICENSE.txt` en MonoDETR).
Los archivos heredados de atención deformable contienen avisos Apache-2.0:
hay que conservar sus avisos al incorporar código. La licencia del repositorio
no acredita automáticamente los términos de pesos o datos. En esta iteración
no se han copiado fuentes de terceros a PHASE, descargado checkpoints ni
redistribuido datos.

## 3. Compatibilidad: problemas comprobados en fuentes

Los dos README proponen Python 3.8 y una instalación histórica Torch 1.9/CUDA
11.1. Esa receta no debe sustituir el entorno actual. [PyTorch documenta][torch]
soporte Blackwell y wheels CUDA 12.8 desde 2.7; eso acredita el framework,
no estas extensiones ni el port a Windows.

| Hallazgo leído | Archivo upstream | Consecuencia para el port |
|---|---|---|
| Compilación exige `torch.cuda.is_available()` y `CUDA_HOME` | `lib/models/<familia>/ops/setup.py` | Sin GPU/toolkit el propio build rechaza la operación |
| Flags explícitos `-arch=sm_60` y `sm_60/61/70/75` | Mismo `setup.py` en ambos | Sustituir en un patch revisado por arquitectura obtenida del dispositivo; no asumir que `TORCH_CUDA_ARCH_LIST` anula flags explícitos |
| `THC/THCAtomics.cuh`, `AT_ASSERTM`, `value.type()` y dispatch histórico | `ops/src/cuda/*`, `ops/src/ms_deform_attn.h` | Compatibilidad con Torch moderno pendiente de compilación real; revisar API/headers manteniendo semántica |
| Import de extensión `MultiScaleDeformableAttention` | `ops/functions/ms_deform_attn_func.py` | Una instalación de Torch que funciona no basta |
| Forward comparado mediante `print(fwdok)` | `ops/test.py` | Un exit code cero upstream puede acompañar una comparación falsa; usar assertions |
| `pretrained=is_main_process()` en ResNet | `backbone.py` | Evitar descarga implícita durante smoke: hacer explícito `weights=None`/pretrained desactivado en el port |
| `.cuda()` y `device='cuda'` en pérdidas | `monodgp.py` | Hacer consistente el dispositivo; no prueba CPU del modelo original |
| `nan_to_num` sobre coste de matching | `matcher.py` | Verificar finitud antes del matching; no ocultar NaN bajo un coste válido |

Estos son riesgos localizados y comportamientos observados en código. **No se
ha observado un error de compilación en una RTX 5070 Ti** porque no hay acceso a
ella. Ni el fracaso ni el éxito de compilación se pueden inventar a partir de
esta lectura.

MonoDETR comparte la misma familia de extensión y flags antiguos. Cambiar a
MonoDETR puede simplificar el modelo, pero no elimina por sí solo el bloqueo del
operador. ImVoxelNet evita esa arquitectura, pero [MMDetection3D][mm-install]
tiene su propio stack MMEngine/MMCV/MMDetection y operadores compilados; no es
un fallback garantizado para Blackwell. Su volumen 3D añade una dimensión de
memoria que también habrá que perfilar.

## 4. Adaptación a eAP: no basta `num_classes = 8`

El código original es KITTI. La siguiente tabla identifica las zonas concretas
que el adaptador futuro debe reemplazar o parametrizar. Es **trabajo pendiente**.

| Zona upstream | Comportamiento real | Trabajo eAP propuesto |
|---|---|---|
| `lib/datasets/kitti/kitti_dataset.py` | Tres clases; `Pedestrian/Car/Cyclist`; máximo 50; `writelist` Car en YAML | Nuevo loader de escena eAP con las ocho clases del contrato G0; nunca usar el loader KITTI con etiquetas simplemente renombradas |
| Mismo loader | Resolución 1280×384; filtros de dificultad, profundidad 2–65 m, centro fuera de imagen y truncación | Registrar cada exclusión y definir políticas eAP; no heredar silenciosamente dificultad o rango KITTI |
| Mismo loader | Tamaños `[h,w,l]`; centro anotado inferior corregido con `-h/2` | Transformar desde geometría eAP verificada; contratos de centro y orden explícitos |
| `lib/helpers/decode_helper.py` | Decode añade `h/2` para volver a base; `alpha → ry`; score nativo `p_clase * exp(-depth_log_variance)` | Conservar score bruto, probabilidad sigmoide de clase y parámetro de profundidad; definir por separado el score exportable y su rango |
| `monodgp.py`, líneas 247–249 | Prior de profundidad `altura_3D / altura_píxeles * calibs[:,0,0]`; utiliza `fx` para una altura vertical | Fixture independiente con `fx ≠ fy`, resize anisótropo y rectificación; actualizar K por sí solo no verifica este prior |
| `monodgp.py::loss_depth_map` | Multiplicador fijo `[80,24,80,24]` | Derivarlo de la forma real del mapa de profundidad; comprobar resize/padding e intrínsecos |
| `depth_predictor/ddn_loss/ddn_loss.py` | `forward` usa defaults `bin_depths`: 80 bins, máximo 60 m | Un solo contrato de bins/rango para predictor y pérdida; rango basado en train y definición de evaluación |
| `det2d_transformer.py`, `det3d_transformer.py` | En entrenamiento usan `self.group_num * 50` | Mantener 50 en primer smoke; parametrizar y testear si se cambia número de queries |
| `monodgp.py::build`, matcher y criterio | Cabezas configurables, asignación por grupos | Cambiar clasificación a ocho clases y demostrar targets/negativos/casos vacíos correctos |
| `monodgp.py` y salida nativa | Clase, 2D/centro proyectado, dimensión, profundidad, ángulo; sin velocidad | Velocidad futura separada y con semántica G0 validada; no rellenar ausentes con cero |
| `tester_helper.py`, evaluator KITTI | Selección por resultado KITTI Car/AP3D | Sustituir por evaluación local eAP declarada de desarrollo mientras falte scorer oficial |

La clase `QuerySelection` definida en `matcher.py` tiene valores por defecto
11×50, pero no está instanciada en la ruta `build` inspeccionada; no se presenta
como bloqueo activo. Los literales 50 en los dos transformers sí están en la
ruta de entrenamiento. Esta distinción impide convertir código sin uso en una
conclusión incorrecta.

Las 2D boxes del loader KITTI provienen de anotaciones. En eAP debe comprobarse
si existe un target 2D compatible; proyectar un cuboide crea una región amodal,
no una caja de superficie visible. En MonoDGP esa diferencia afecta tanto a la
pérdida de caja como al prior de profundidad y al target de la cabeza de regiones.
Desactivar mixup/crop/flip geométrico en el primer overfit hasta verificar cada
transformación. La normalización ImageNet no autoriza mantener medias de
dimensiones KITTI; cualquier estadístico eAP se calcula solo sobre train.

**El score nativo no está acotado por uno.** El segundo canal de `pred_depth`
es el parámetro llamado `depth_log_variance` en `loss_depths`; el decoder lo
transforma con `exp(-valor)` y multiplica por la salida sigmoide de clase. Un
valor negativo puede producir score mayor que uno. Por ejemplo, `p_clase=0,8`
y parámetro `-1` producen aproximadamente `2,17`; es una comprobación algebraica,
no una predicción observada. El adaptador debe conservar `native_score_raw`,
`class_probability_raw` y ese parámetro de profundidad para que el ranking sea
auditable. La política de `score_export` queda pendiente del contrato de
submission verificado: fijar y versionar una transformación explícita o el uso
de la probabilidad de clase, comprobando sus efectos en desarrollo. No aplicar
un clamp silencioso, llamar calibrada a una salida sigmoide ni reutilizar
ninguna de estas cantidades como confianza TTC. El umbral upstream se aplica
antes de la multiplicación; también debe registrarse esa diferencia respecto
de umbralizar el score final.

**El prior geométrico requiere un fixture adicional.** En un pinhole sintético
rectificado, un segmento vertical de altura métrica H a profundidad Z produce
altura de imagen `h = fy * H / Z`; el prior de ese fixture debe recuperar Z
usando fy. Probar `fx = fy` no detecta que el código usa `calibs[:,0,0]`.
Incluir casos con `fx ≠ fy`, con escalas distintas por eje, padding y cámara
rectificada; transformar juntos proyección, caja, intrínsecos y `img_sizes`,
manteniendo explícito si cada campo corresponde a la imagen original o a la
entrada del backbone. Comparar el prior antes de la corrección aprendida para
que esta no oculte una inconsistencia. Resolver la convención de la ruta eAP y,
si procede, portar la fórmula con una prueba que falle sobre el supuesto
original. Esta observación del código no demuestra por sí sola un error en los
resultados KITTI ni autoriza modificar resultados o artículos históricos.

## 5. Contrato para completar G1 en la GPU accesible

Secuencia propuesta, sin entrenar un dataset y posible independientemente de G0:

1. Obtener nombre, capacidad CUDA, VRAM total/libre, driver, OS, `nvcc`, compilador
   y versiones exactas en el entorno **separado**. No asumir 12 GB o una arquitectura
   a partir de una conversación anterior. Fijar source SHA y guardar patch del port.
2. Resolver el entorno candidato, guardar lock con hashes y log de instalación.
   No instalar el `requirements.txt` antiguo sin revisar y resolver sus rangos.
   Compilar la extensión en ese entorno y guardar comando/log/binario SHA-256.
3. Ejecutar [check_deformable_operator.py](../../scripts/check_deformable_operator.py).
   Compara forward y gradientes de value/sampling locations/attention weights con
   la referencia `grid_sample` del mismo upstream en FP32 y FP64. Exige valores
   finitos y `assert_close`; conserva tolerancias explícitas en el JSON.
   Un PASS aquí es **solo del operador**, nunca del detector.
4. Implementar el runner de detector completo con `MonoDGP.build(cfg)`, primero
   pesos aleatorios y sin descarga implícita. Batch 1, imagen `[1,3,384,1280]`,
   `num_queries=50`, `group_num=11`, `num_classes=8`, seed 7, FP32. Los 11 grupos
   de entrenamiento explican por qué batch 1 no equivale a 50 queries de entrenamiento.
5. Crear fixture sintético autocontenido de ocho objetos con etiquetas 0…7,
   cajas positivas/no cuadradas, distintos centros y profundidades, calibración
   pinhole `[B,3,4]`, `img_sizes=[width,height]` y targets con los tipos del loader:
   `labels`, `boxes` normalizadas cxcywh, `boxes_3d` normalizadas cxcy/l/r/t/b,
   `depth`, `size_3d` hwl, `heading_bin`, `heading_res`, `calibs`, `obj_region`.
   Estas convenciones son del fixture KITTI-like; no resuelven G0 eAP.
6. Comprobar **todos** los tensores de salida y cada componente de pérdida antes
   de sumar. Backward; registrar gradientes no nulos/finito de backbone y cabezas
   activas, y una actualización de optimizador reproducible. Probar por separado
   frame vacío para las pérdidas donde esté definido. No aceptar solo descenso
   de una pérdida total ni que el matcher convierta NaN en cero.
7. Pasar a `eval`, decodificar, verificar forma, clases, finitud, centros,
   dimensiones y yaw del fixture. Probar encode/decode de targets conocidos
   independientemente del modelo aleatorio. Un modelo aleatorio no tiene por qué
   producir dimensiones físicas válidas: registrar invalidaciones y hacer que el
   futuro adaptador mantenga positividad con semántica explícita, no clamps ocultos.
   Añadir el fixture del prior vertical con `fx ≠ fy` y resize anisótropo antes
   de aceptar cambios de resolución; conservar score nativo y comprobar por
   separado el contrato del score exportado, incluyendo valores nativos mayores
   que uno y la posición del threshold respecto a la reponderación de profundidad.
8. Medir por separado arranque/steady state, forward, entrenamiento y decode.
   Sincronizar CUDA, registrar `max_memory_allocated/reserved`, proceso vía
   `nvidia-smi`, RAM y forma/batch/precisión; no incluir PHASE residente en este
   entrenamiento por etapas. No inferir memoria del modelo desde el operador.
9. Repetir con AMP solo después de FP32 y con comparación de salidas/gradientes.
   Si hay OOM de un ejemplo, acumulación de gradiente no lo arregla. Probar forma
   reducida únicamente después de corregir el factor fijo del depth-map y
   conservar una prueba de geometría de resize/padding. Registrar el cambio.

El runner de detector completo de los puntos 4–9 **no está implementado en esta
entrega**: no se ha portado el detector ni se puede validarlo en esta máquina.
G1 permanecerá bloqueado hasta guardar esos resultados reales en la GPU objetivo.
El futuro runner puede llamarse `scripts/smoke_eap_detector.py`; ese nombre es
una tarea del plan, no un comando disponible ahora.

## 6. Criterio de elección/fallback

MonoDGP avanza a G2 solo con G0 resuelto y G1 completo. Si la extensión no compila,
separa causa de toolchain, API C++ y arquitectura; un patch de compatibilidad debe
pasar la comparación de operador y no cambiar silenciosamente el algoritmo.
Como diagnóstico puede usarse el core PyTorch de atención deformable, pero ese
backend tendrá nombre/config propios y sus resultados no acreditarán el kernel.

MonoDETR debe superar el mismo gate si se usa como control. Si el port o memoria
requiere cambiar sustancialmente ambos modelos, evaluar una única configuración
ImVoxelNet de una cámara, volumen y rango acotados. No instalar simultáneamente
tres stacks en PHASE ni abrir una campaña de comparaciones antes de tener un
baseline funcional. No se fija un plazo, coste monetario, FPS ni rendimiento
esperado para decidir qué es «desproporcionado»; registrar cambios y riesgos
técnicos necesarios y escoger la ruta que realmente pueda comprobarse.

## 7. Reproducción de lo entregado

Desde la raíz del worktree. `DETECTION_AUDIT_ENV` y `DETECTOR_SOURCES` son rutas
configurables fuera del checkout; los comandos siguientes no modifican PHASE:

```bash
DETECTION_AUDIT_ENV=../venv-detection-audit
DETECTOR_SOURCES=../external
python3 -m venv "$DETECTION_AUDIT_ENV"
"$DETECTION_AUDIT_ENV/bin/python" -m pip install -r configs/detection/requirements-audit.txt
"$DETECTION_AUDIT_ENV/bin/python" -m pytest tests/unit/test_detection_environment.py -ra
"$DETECTION_AUDIT_ENV/bin/ruff" check scripts/probe_detection_environment.py scripts/check_deformable_operator.py tests/unit/test_detection_environment.py
"$DETECTION_AUDIT_ENV/bin/python" scripts/probe_detection_environment.py --monodgp-root "$DETECTOR_SOURCES/MonoDGP" --monodetr-root "$DETECTOR_SOURCES/MonoDETR" --output /tmp/g1_environment.json
```

El último comando devuelve **2 por diseño**: el inventario no puede certificar
G1 aunque todos los prerequisitos estén presentes. `--run-cuda-primitives`
habilita únicamente un producto matricial con backward; esa memoria queda
etiquetada como primitiva y nunca rellena `detector_peak_memory_bytes`.

Clones reproducibles si todavía no existen:

```bash
git clone https://github.com/PuFanqi23/MonoDGP.git "$DETECTOR_SOURCES/MonoDGP"
git -C "$DETECTOR_SOURCES/MonoDGP" checkout --detach aa059a18214aebf644510e7f0793971b403f9d14
git clone https://github.com/ZrrSkywalker/MonoDETR.git "$DETECTOR_SOURCES/MonoDETR"
git -C "$DETECTOR_SOURCES/MonoDETR" checkout --detach 6994b9f512400b258c6edb75f77423beb9c126f2
```

Tras crear y validar el entorno CUDA separado y compilar un port revisado,
ejecutar con **su** intérprete (no con el entorno de auditoría sin Torch):

```bash
"$DETECTOR_ENV/bin/python" scripts/check_deformable_operator.py --family monodgp --source-root "$DETECTOR_SOURCES/MonoDGP" --seed 7 --output /tmp/monodgp_operator.json
```

`DETECTOR_ENV` identifica un entorno futuro todavía no creado en esta entrega.
El wrapper es ejecutable ahora, pero aquí produce el bloqueo registrado por
ausencia de Torch. Un PASS futuro exige ejecución real y conserva
`g1_passed=false`: aún faltará el detector completo. En Windows adaptar la ruta
del intérprete a `Scripts/python.exe`; no se ha probado compilación nativa Windows
ni WSL. No se ha contratado GPU ni servicio de pago.

## Fuentes primarias

[dgp]: https://github.com/PuFanqi23/MonoDGP/tree/aa059a18214aebf644510e7f0793971b403f9d14
[detr]: https://github.com/ZrrSkywalker/MonoDETR/tree/6994b9f512400b258c6edb75f77423beb9c126f2
[dgp-paper]: https://openaccess.thecvf.com/content/CVPR2025/html/Pu_MonoDGP_Monocular_3D_Object_Detection_with_Decoupled-Query_and_Geometry-Error_Priors_CVPR_2025_paper.html
[torch]: https://pytorch.org/blog/pytorch-2-7/
[imvoxel]: https://github.com/open-mmlab/mmdetection3d/tree/main/configs/imvoxelnet
[mm-install]: https://mmdetection3d.readthedocs.io/en/latest/get_started.html
