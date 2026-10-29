# Extensión de PHASE-TTC a percepción 3D: investigación del swarm

Fecha de corte: **29 de octubre de 2026**. Base inspeccionada: `main`, commit
`e58d0106e1f114f00f796987f70cd622cc6d186e`.

Estado: **propuesta de investigación e ingeniería; detector nuevo no implementado**.
Complemento: [dictamen del council](council_review.md).

## 1. Objetivo y conclusión

La extensión debe recibir RGB, eventos, calibración y timestamps públicos, detectar
objetos en 3D, asociarlos de forma causal y estimar TTC firmado por objeto. La salida
interna reunirá clase, centro métrico, dimensiones, orientación, velocidad, score de
detección, identidad temporal, TTC y límites nativos de fase. Cada campo conservará
su significado y estado de disponibilidad.

PHASE-TTC aporta una cadena TTC entrenada, lectores eAP, preparación de regiones,
contexto temporal y un servicio multiobjeto. **No aporta hoy un detector de escena
ni un asociador que produzca sus propias ROIs e identidades.** El nuevo trabajo
requiere un detector completo, con representación espacial, pérdidas, decodificación
y evaluación; añadir salidas a la cabeza final de veinte variables no resuelve eso.

Recomendamos construir primero una ruta modular:

```text
RGB + eventos + calibración + reloj de disponibilidad
    -> detector 3D de escena
    -> asociación causal y ciclo de vida de objetos
    -> ROIs por sensor y preparación temporal compatible
    -> servicio PHASE-TTC por objeto
    -> resultados conjuntos con validez y procedencia

Exportación eAP: solo detecciones en el esquema oficial.
Análisis del sistema: detecciones + identidad + TTC + fase + diagnóstico.
```

La segunda ruta, con backbone compartido y aprendizaje multitarea, queda como
experimento posterior. Puede reducir cómputo y aprovechar supervisión geométrica,
pero cambiará la distribución de features y requerirá volver a validar el estimador.

Esta revisión no demuestra robustez general del sistema actual, mejora de AP por
usar TTC, transferencia a ocho clases ni funcionamiento en tiempo real de escena
completa. Tampoco establece un ganador SOTA en eAP. Identifica referencias
comparables, candidatos reproducibles y pruebas que permitirán elegir.

## 2. Cómo se realizó la investigación

Cinco agentes trabajaron en paralelo. El agente principal contrastó sus hallazgos
con archivos y fuentes; tres agentes distintos hicieron la revisión del council.
Son agentes de IA de esta sesión, no revisores humanos ni validación externa.

| Agente del swarm | Responsabilidad |
|---|---|
| `swarm_event_detection` | Detección RGB–eventos, sensores requeridos y literatura 2024–2026 |
| `swarm_camera_3d` | Detectores monoculares/temporales, código, pesos y coste de adaptación |
| `swarm_repo_architecture` | Interfaces reales de PHASE-TTC y diseños de integración |
| `swarm_data_benchmark` | Manifests, medios, geometría, submission y protocolo eAP |
| `swarm_tracking_uncertainty` | Asociación, incertidumbre, evaluación y exposición de secuencias |

Buscamos expresamente publicaciones de 2026, incluidas agosto y octubre. Priorizamos
artículos originales, proceedings, código de autores y releases oficiales. Abrimos
repositorios cuando un resultado indexado decía «code soon»: MAGneT-3D ya tenía
código publicado antes del corte. Una búsqueda bibliográfica no certifica
exhaustividad; los releases vivos pueden cambiar después de esta fecha.

Convenciones de evidencia:

- **Observado:** leído en código, manifest, archivo o implementación pública.
- **Publicado:** afirmación de los autores, sin reproducción nuestra.
- **Propuesto:** diseño o experimento que todavía no hemos ejecutado.
- **Pendiente:** información insuficiente o contrato sin verificar.

No entrenamos modelos, instalamos stacks de detección, modificamos pesos, enviamos
submissions ni descargamos medios. Leímos manifests y muestras pequeñas; una
etiqueta 3D remota se inspeccionó en memoria. Los checkpoints externos se comprobaron
como enlaces documentados, no como pesos descargados y ejecutados. La presencia de
un archivo local no equivale a validar su checksum o todo su contenido.

## 3. Frontera entre los dos benchmarks

| Propiedad | GarlTTC / PHASE-TTC actual | eAP 3D Detection |
|---|---|---|
| Unidad | Consulta de un objeto con región suministrada | Fotograma con conjunto desconocido de objetos |
| Supervisión | TTC y geometría por consulta | Clases y cajas 3D de los objetos anotados |
| Localización al inferir | Externa | Predicha por el participante |
| Salida | TTC por token | Lista de cajas, clases, yaw, velocidad y score por token |
| Métrica principal | Error de TTC/fase | AP por distancia de centros |
| Test local inspeccionado | Entregas TTC de 6.762 tokens | 35.940 tokens de fotograma |

El [README oficial eAP][eap-code] enlaza detección a **CodaBench 16717**. Los correos
aportados por el usuario identifican **17289** como la competición TTC. No son el
mismo evaluador. En esta sesión, el acceso web a CodaBench devolvió contenido ajeno
y las consultas directas a detección devolvieron HTTP 404. **No verificamos el
programa oficial de scoring de detección ni su disponibilidad operativa.**

El contrato aportado por el usuario exige ocho clases (`car`, `bicycle`, `truck`,
`pedestrian`, `motorcycle`, `bus`, `tricycle`, `other_vehicle`), AP con matching a
0,5/1/2/4 m y errores ATE/ASE/AOE. Faltan por contrastar en el scorer: interpolación
de AP, umbral TP, filtros de rango, límite de cajas, promedios, clases sin GT,
simetrías de yaw y tratamiento de predicciones inválidas. Hasta comprobarlo, una
implementación propia será una **métrica de desarrollo**, no una réplica oficial.

La velocidad es un campo exigido por el formato; el contrato recibido no lista AVE
como métrica de ranking. Mejorar velocidad o TTC no implica mejorar AP.

Los correos de los autores aclaran que GarlTTC referencia medios eAP y que las
etiquetas TTC Test12 son privadas. Para EvTTC, describen ROIs anotadas, tres
escenarios, ventanas y selección de muestras. No certifican una lista exhaustiva de
queries ni el archivo de checkpoint usado históricamente para Tabla VI. Es
correspondencia suministrada por el usuario, no un nuevo resultado reproducido ni
permiso para usar ROIs de otra tarea como entrada del detector de test.

## 4. Auditoría de datos locales

Rutas inspeccionadas: `E:/eAP_dataset`, `E:/GarlTTC_dataset`, `E:/Garl-TTC` y el ZIP
`sample_submission.zip` facilitado en Downloads. Son ubicaciones de esta auditoría;
el código futuro debe aceptar rutas configurables.

| Elemento | Resultado observado |
|---|---|
| `data/train.parquet` eAP | 118.247 fotogramas, 40 secuencias, tokens únicos |
| Medios train referenciados | 40/40 HDF5 y 136/136 TAR presentes |
| Etiquetas de detección train | 0/40 `labels.parquet` presentes en las rutas del manifest |
| `data/test.parquet` eAP | 35.940 fotogramas, 12 secuencias |
| Medios test referenciados | 12/12 HDF5 y 36/45 TAR presentes |
| Efecto de los nueve TAR ausentes | 4.132 fotogramas, siete secuencias |
| ZIP del usuario | Un único `submission.json`; claves exactamente iguales a test; todas las listas vacías |
| GarlTTC train | 88.744 consultas; las mismas 40 secuencias que eAP train |

La etiqueta 3D remota inspeccionada contiene clases, cajas, identificadores y
velocidades: existe supervisión pública para desarrollar detección. Las anotaciones
TTC de Garl no sustituyen el conjunto completo de objetos de cada fotograma.

Revisiones HF comprobadas:

- eAP: [`a814a946e60d37f7242358af5f344542177e50c5`][eap-hf].
- GarlTTC: [`9f477b5c4d5f9cdc32e7cc4bb3b186301979d6b5`][garl-hf].

La API indica 128.416.244 bytes para las cuarenta etiquetas faltantes y
4.793.784.320 bytes para los nueve TAR: **4.922.200.564 bytes**, unos 4,92 GB
decimales. No es necesario volver a descargar todos los medios.

### 4.1 Archivos RGB ausentes

Todos llevan el prefijo `data/test/`:

```text
3RUZHwnCwK/rgb_shards/rgb-00003.tar
71wUSQYMpk/rgb_shards/rgb-00001.tar
71wUSQYMpk/rgb_shards/rgb-00003.tar
89fvtHGHx2/rgb_shards/rgb-00001.tar
89fvtHGHx2/rgb_shards/rgb-00003.tar
HHJe79eY2J/rgb_shards/rgb-00003.tar
Mu6kaUjIz3/rgb_shards/rgb-00001.tar
kafYzX6kJ4/rgb_shards/rgb-00003.tar
vhjB5OViA7/rgb_shards/rgb-00003.tar
```

### 4.2 Recuperación reproducible propuesta

El swarm ejecutó el primer `--dry-run` sin descargar etiquetas. Estos comandos
documentan una fase posterior; no indican que hayamos completado los datos:

```powershell
$eapRevision = 'a814a946e60d37f7242358af5f344542177e50c5'
$eapRoot = 'E:/eAP_dataset' # Sustituir por la raíz local configurada.
hf download NAIL-HNU/eAP-dataset --type dataset --revision $eapRevision --include 'data/train/*/labels.parquet' --local-dir $eapRoot --dry-run

$missingRgb = @(
  'data/test/3RUZHwnCwK/rgb_shards/rgb-00003.tar'
  'data/test/71wUSQYMpk/rgb_shards/rgb-00001.tar'
  'data/test/71wUSQYMpk/rgb_shards/rgb-00003.tar'
  'data/test/89fvtHGHx2/rgb_shards/rgb-00001.tar'
  'data/test/89fvtHGHx2/rgb_shards/rgb-00003.tar'
  'data/test/HHJe79eY2J/rgb_shards/rgb-00003.tar'
  'data/test/Mu6kaUjIz3/rgb_shards/rgb-00001.tar'
  'data/test/kafYzX6kJ4/rgb_shards/rgb-00003.tar'
  'data/test/vhjB5OViA7/rgb_shards/rgb-00003.tar'
)
hf download NAIL-HNU/eAP-dataset @missingRgb --type dataset --revision $eapRevision --local-dir $eapRoot --dry-run
```

En la implementación se retiraría `--dry-run` y se verificarían hashes, miembros
TAR y referencias. `hf cache verify` puede comprobar archivos contra esa revisión;
no sustituye la validación geométrica ni temporal.

### 4.3 Bloqueo geométrico: no asumir que los formatos coinciden

Hay tres contratos: bundle original con `annotations.pkl`, Parquet de HF y JSON de
submission. No hemos demostrado su conversión completa.

En una fila pública de HF observamos:

```text
bbox_3d_ego[3:6] = [5.9296875, 2.6953125, 2.076171875]
size             = [2.076171875, 5.9296875, 2.6953125]
```

Esto demuestra una permutación, **no identifica por sí solo cada eje físico**.
El [README original fijado][eap-code] documenta `[x,y,z,l,h,w,yaw]`, Z-up y
`T_event_ego` de ego a cámara evento. El [visualizador][eap-visualizer] coloca `w`
sobre X local, `l` sobre Y y aplica una rotación con otra convención de signo que
nuestro helper. El [helper local](../../src/e_jepa_ttc/data/eap.py#L190) interpreta
`[x,y,z,l,w,h,yaw]`. El empaquetado HF puede haber convertido ejes y dimensiones:
la diferencia **no demuestra un bug en el código existente**.

Antes de entrenar: resolver el puente con código oficial de conversión o fixtures
públicos independientes; probar cuboides no cúbicos, yaw 0 y ±π/2, centro geométrico
frente a base, ida/vuelta de transforms y reproyección sobre RGB/eventos. Una
prueba que convierte y desconvierte usando dos veces la misma suposición errónea
no aporta evidencia suficiente.

`velocity` expresada en marco ego no aclara si es velocidad absoluta expresada en
ese marco o relativa al vehículo. Hay que fijar esa semántica antes de relacionarla
con TTC. Los manifests aportan `K_event` y `T_event_ego`, pero no acreditan poses ego
temporales ni `K_rgb`/distorsión. Una extrínseca fija no reemplaza ego-motion.

Las muestras PNG inspeccionadas tienen 1280×720. El visualizador proyecta cajas
sobre ambas modalidades, pero no basta para demostrar cómo rectificaron los PNG de
HF. El tamaño idéntico tampoco prueba alineación pixel a pixel.

Su ventana de eventos de visualización está centrada alrededor de exposición;
copiarla como preprocesado causal puede consumir eventos futuros respecto al
instante declarado. El pipeline deberá fijar final de exposición, query y máximo
timestamp consumido.

## 5. Inventario del repositorio: reutilización y fronteras

| Implementación existente | Uso posible y límite |
|---|---|
| [Lectores eAP](../../src/e_jepa_ttc/data/eap.py#L154) | Manifests, etiquetas, HDF5 indexado y RGB TAR; falta el contrato de muestras de escena para entrenar detección |
| [Proyección de cajas conocidas](../../src/e_jepa_ttc/data/eap.py#L225) | Herramienta de geometría; no predice cajas ni resuelve las convenciones HF por sí sola |
| [Recorte temporal común](../../src/e_jepa_ttc/data/event_v4_geometry.py#L68) | Preserva expansión entre endpoints; mantener su contrato al usar ROIs predichas |
| [Productores TTC](../../src/e_jepa_ttc/models/causal_scale_ttc.py#L195) | Geometría 2D normalizada y features densas candidatas a transferencia; no detecciones métricas |
| [Cabeza final](../../operational/final_refit/training.py#L85) | `[B,340]` y logits RGB → veinte features seleccionadas → fase/TTC/q10/q90 |
| [Predictor sellado](../../operational/final_refit/test12.py#L623) | Nueve componentes y hashes; su wrapper `predict` entrega un escalar TTC |
| [Servicio multiobjeto](../../operational/train40_runtime/multiobject.py#L214) | Buffers compartidos y regiones por objeto; exige ROIs e IDs estables suministrados |
| [Contexto batched](../../operational/train40_runtime/batched_context.py#L92) | Salidas TTC final/evento/RGB y validez; esta ruta todavía no expone límites de fase |
| [Export de cabeza](../../operational/final_refit/export.py#L358) | Ya declara TTC, fase y q10/q90; no es un detector sensor→caja ni garantiza esas salidas en todos los servicios |
| [Export GarlTTC](../../operational/sota_evidence/submission.py#L16) | Valida tokens y valores TTC; requiere un exportador distinto para detección eAP |

El council corrigió dos simplificaciones del primer inventario: **ya hay servicio
multiobjeto** y **ya hay export de cuantiles de la cabeza**. El trabajo pendiente
es producir ROIs/IDs y transportar los límites por la ruta elegida, conservando
paridad con el cálculo canónico.

El servicio admite entre uno y veinte objetos por llamada/configuración actual.
Eso no establece un límite del benchmark. Una escena con cero detecciones debe
exportar `[]` sin invocar un servicio que exige al menos una región; una escena
con más de veinte debe conservar todas las detecciones. El procesamiento TTC por
chunks o su política de indisponibilidad requieren medición propia.

`reconstruct_eap_object_states` calcula targets a partir de anotaciones y pendientes
locales que pueden incorporar observaciones posteriores. No se reutilizará como
estimador causal de velocidad de test.

## 6. Literatura: qué puede transferirse

Las cifras y disponibilidades de esta sección proceden de los autores. No
reprodujimos esos experimentos. AP 2D, AP3D KITTI por IoU y AP eAP por distancia no
son intercambiables. Un trabajo más reciente no es necesariamente la mejor base
para nuestro hardware, modalidades o protocolo.

### 6.1 Eventos y fusión RGB

| Trabajo | Señales y resultado | Evidencia de implementación | Decisión para este proyecto |
|---|---|---|---|
| [eAP, marzo 2026][eap-paper] | RGB+eventos → BEV/cajas 3D; TTC separado | Repo de datos/visualización; detector completo y pesos no verificados | Referencia más cercana al problema y supervisión |
| [Ev-3DOD, CVPR 2025][ev3dod-paper] | RGB+LiDAR+eventos; actualiza propuestas 3D | [Código MIT][ev3dod-code] | Idea de actualización entre frames; no importar como camera-only |
| [Ev-Stereo3D, ICCV 2025][evstereo-paper] | Dos cámaras evento → cajas 3D | [Código][evstereo-code], stack antiguo/spconv; licencia raíz no identificada | Fuera del MVP por falta de estéreo evento |
| [DAGr, Nature 2024][dagr-paper] | RGB+grafo asíncrono de eventos → cajas 2D | [Código y ejemplo de pesos GPL-3.0][dagr-code], operaciones CUDA | Referencia temporal; coste de licencia/port y sin escala 3D |
| [FlexEvent, NeurIPS 2025][flex-paper] | Fusión 2D a frecuencias variables | [Código MIT y pesos enlazados][flex-code] | Referencia reproducible para fusión, no detector 3D listo |
| [ERA-Det, 19-10-2026][era-paper] | Routing de fiabilidad y resolución asimétrica, 2D | [Repo][era-code] sin código/pesos de entrenamiento en la inspección | Hipótesis de fusión posterior, no dependencia disponible |
| [AERODet, 02-08-2026][aero-paper] | Localización evento/clasificación RGB, 2D | Código/pesos oficiales no verificados | Inspiración para separar funciones; no prueba universal de superioridad de eventos |
| [PEOD, noviembre 2025][peod-paper] | Dataset y evaluación RGB–eventos 2D en condiciones difíciles | [Release][peod-code] | Diseñar evaluación por condición y controles unimodales |
| [RVT, CVPR 2023][rvt-paper] | Backbone recurrente de eventos, detección 2D | [Código MIT y pesos][rvt-code] | Candidato de encoder evento, sujeto a perfil de memoria |

Dos referencias de coste evitan extrapolaciones: eAP publica unos **125,07 ms**
para su detector 3D en A100-40G; sus aproximadamente 200 FPS pertenecen al estimador
TTC sobre ROI. ERA-Det publica **34,3 GB** de entrenamiento y 22 ms batch uno FP16
en RTX 6000D. Ninguna cifra demuestra viabilidad de esa configuración en 12 GB.
Frecuencia nominal de decisión, latencia de sensor y throughput de cómputo son
medidas diferentes.

### 6.2 Detección monocular y temporal

| Trabajo | Ventaja relevante | Requisitos y disponibilidad | Prioridad propuesta |
|---|---|---|---|
| [MonoDETR, ICCV 2023][monodetr-paper] | Baseline monocular guiado por profundidad | [MIT, configs y pesos][monodetr-code]; atención deformable y entorno antiguo | Control de referencia |
| [MonoDGP, CVPR 2025][monodgp-paper] | Queries 2D y corrección de geometría/profundidad | [MIT, pesos/logs][monodgp-code]; R50, 38,9 M parámetros según paper | Primer candidato, condicionado a geometría y smoke de operaciones |
| [MonoSAOD, CVPR 2026][monosaod-paper] | Teacher–student con anotaciones parciales | [Código][monosaod-code]; pesos finales no comprobados; preparación extra | Ablation de pocas etiquetas después del baseline |
| [MAGneT-3D, agosto 2026][magnet-paper] | Propuestas adaptativas y memoria monocular temporal | [Código/pesos enlazados][magnet-code], ego-pose; tres categorías agrupadas | Referencia temporal principal, con barreras de señales y entorno |
| [StreamPETR, ICCV 2023][streampetr-paper] | Memoria centrada en objetos | [Código y pesos][streampetr-code]; configuración multivista y ego-motion | Patrón arquitectónico, no port directo |
| [Sparse4D v3, 2023/2024][sparse4d-paper] | Propagación y refinamiento temporal de instancias | [MIT, pesos][sparse4d-code], operación CUDA y entrenamiento publicado multi-GPU | Alternativa posterior; no asumir coste de laptop |
| [BEVDet4D, 2022][bevdet-paper] | Acumulación temporal BEV | [Apache-2.0][bevdet-code]; variantes Depth/Stereo con otros requisitos | Referencia BEV; elegir explícitamente señales y supervisión |
| [ImVoxelNet, WACV 2022][imvoxel-paper] | Features de imagen proyectadas a vóxeles | [Implementación MMDetection3D][imvoxel-code] | Alternativa cercana al detector eAP; controlar volumen/memoria |
| [MonoUNI, NeurIPS 2023][monouni-paper] | Normalización de profundidad e intrínsecos | [Repo][monouni-code] centrado en Rope3D, stack antiguo; licencia no verificada | Fuente de ideas, no primera dependencia |
| [Map-Det3D, agosto 2026][mapdet-paper] | Reconstrucción métrica y detección temporal | [Apache-2.0, pesos][mapdet-code]; interiores, class-agnostic | Exploratorio; no evidencia de adecuación a conducción eAP |

La elección provisional **MonoDGP + control MonoDETR** reduce diferencias de familia
y permite comenzar sin ego-pose temporal. MonoDGP publica entrenamiento con una
RTX 3090 y 42 ms de inferencia batch uno en ella; no garantiza memoria ni latencia
en la RTX 5070 Ti Laptop. Si el port de operaciones resulta desproporcionado,
ImVoxelNet es una alternativa que debe superar el mismo gate de entorno y memoria.
No proponemos portar todos estos modelos.

MAGneT-3D merece atención por su alineación monocular y fecha reciente. Verificamos
el commit `d93ba645e07a9be5240fd6337406d8c76a067d51`, del **18-10-2026**: contiene
implementación, configuraciones y enlaces a checkpoints. Su configuración necesita
`ego_pose`/inversa y agrupa `pedestrian`, `vehicle`, `bicycle`. Su setup documenta
Torch 1.9/CUDA 11.1 y MMCV 1.6. Adaptarlo exige ocho clases, señales temporales válidas
y port de operaciones; el hecho de tener código no elimina esas tareas.

### 6.3 Asociación e incertidumbre

| Referencia | Uso acotado | Límite |
|---|---|---|
| [AB3DMOT, 2020][ab3dmot] | Kalman 3D y asociación húngara como baseline interpretable | Reajustar ruido de cajas monoculares; resultados originales no son eAP |
| [ByteTrack, ECCV 2022][bytetrack] | Asociación 2D en dos etapas cuando profundidad sea inestable | No produce velocidad métrica; excluir refinamientos offline con futuro |
| [MOTIP, CVPR 2025][motip-paper] | Identidad aprendida desde historia de objetos | [Código Apache-2.0][motip-code]; opción posterior, no empezar con otro módulo complejo |
| [HOTA][hota] | Separar calidad de detección y asociación | Añadir métricas locales de historia TTC; no sustituye AP ni error TTC |
| [CAST, WACV 2026][cast] | Asociación medida en ventanas de contexto | Código enlazado no accesible en revisión; usar idea y declarar implementación propia |
| [Guo et al., ICML 2017][calibration] | Diferencia entre score y probabilidad calibrada | Calibración de clase no calibra geometría ni TTC |
| [Adaptive Conformal Inference, 2021][aci] y [ECI, ICLR 2025][eci] | Investigación de cobertura bajo dependencia/cambio temporal | Requieren feedback de errores; test privado no ofrece etiquetas online |

## 7. Diseño propuesto de la nueva parte

### 7.1 Arquitectura A: detector independiente y puente causal

El detector conserva mapas espaciales hasta decodificar centro, dimensiones,
orientación, clase y score. Una cabeza de velocidad supervisada o un módulo temporal
aportará velocidad con convención explícita. Una imagen aislada no determina de
forma suficiente la velocidad de cada objeto; una estimación aprendida necesita
validación propia, aunque el benchmark solo la exija como campo de entrega.

Para ROIs, comparar la caja visible 2D predicha con la proyección del cuboide 3D.
No son equivalentes bajo truncamiento u oclusión. El puente debe reproducir primero
el preprocesado entrenado: el historial retrospectivo dentro de una región actual
no es idéntico a construir recortes con cajas de tracking de cada instante.

La asociación comienza con una opción simple. Comparar posteriormente, usando las
mismas detecciones, compatibilidad 2D tipo ByteTrack y compatibilidad 3D tipo
AB3DMOT. La incertidumbre de profundidad debe influir en el gate; un centro
monocular ruidoso no es una posición exacta. Los IDs son internos, no los `track_id`
de anotaciones de test.

El gestor mantendrá estado por `(sequence_id, track_id)` y un ciclo de vida:
tentativo, confirmado, temporalmente perdido, cerrado. Ante cambio de identidad,
rollback o discontinuidad, reiniciará o invalidará la historia. Un objeto nuevo
puede producir detección antes de disponer de TTC. No debe desaparecer del JSON
porque PHASE todavía no tenga endpoints válidos.

### 7.2 Contrato interno propuesto

Es un esquema de diseño, no una API implementada:

| Grupo | Campos y semántica requerida |
|---|---|
| Identidad | `sequence_id`, `sample_token`, `detection_id`, `track_id` opcional |
| Tiempo | `measurement_us`, `exposure_end_us`, `available_us`, `delivered_us`, `max_input_us` |
| Geometría | Marco/versionado, centro m, dimensiones lwh m, yaw rad, velocidad y definición del marco |
| Localización por sensor | ROI RGB/evento, transformación del recorte, procedencia `predicted`, clipping/visibilidad |
| Detección | `class_name`, `detection_score`; incertidumbre geométrica solo si se estima |
| Asociación | Estado, edad, ambigüedad; score de asociación separado del de detección |
| TTC | `ttc_signed_s` o ausencia, `point_phase`, `phase_q10`, `phase_q90`, historia efectiva, estado de saturación |
| Validez | Motivo de ausencia: inicio, historia insuficiente, identidad ambigua, modalidad ausente, geometría inválida, presupuesto excedido |

La cabeza actual calcula límites de fase, pero la ruta batched TTC necesita una
ampliación comprobada contra el export canónico. Un fallback event-only tampoco
hereda automáticamente los límites de la cabeza de fusión RGB; deberá declarar
qué estimador produjo cada campo y qué límites están disponibles.

El export eAP contendrá solo `meta` y `results` con sus seis campos por detección:
`class_name`, `translation_m`, `size_lwh_m`, `yaw_rad`, `velocity_mps` y `score`.
Los extras de TTC/identidad se guardarán en un artefacto interno separado hasta
que un contrato oficial permita incluirlos. No rellenar scores de detección con
confianza TTC ni usar TTC como score AP sin un experimento independiente.

### 7.3 Escala, causalidad y disponibilidad

Todos los endpoints de un ejemplo PHASE deben compartir un sistema de recorte.
Redimensionar cada bbox ajustada de forma independiente puede eliminar la
expansión física. Un filtro de cajas puede reducir jitter, pero también retrasar o
suprimir esa expansión: su beneficio debe medirse, no suponerse.

La cadena temporal existente admite hasta 650 ms de historia. La caja del frame t
solo está disponible después de exposición y ejecución del detector. Procesar
historia con esa caja es causal a la entrega si todo input ya está disponible;
no demuestra que la predicción estuviera disponible en t. Los registros deben
separar tiempo físico, disponibilidad y respuesta.

Reutilizar features solo cuando coincidan las condiciones de ROI, ventana, delta
y disponibilidad del runtime existente. El cambio de recorte puede invalidar el
cache aunque el track conserve su ID. Los eventos fuera de ventana y frames
futuros deben rechazarse antes de consultar el sensor.

### 7.4 TTC y velocidad 3D

Para profundidad óptica Z y derivada relativa, una convención habitual es
`tau = -Z / dZ_dt`; con aproximación positiva, `tau = Z / v_approach`. El centro
del cuboide y la superficie más próxima producen valores distintos. El vector
planar del submission requiere transformación y semántica ego antes de utilizarse
en esta ecuación. Giro del vehículo, oclusión y cambio del punto visible añaden
problemas que una división escalar no resuelve.

Usar `ttc_geometry` como baseline/diagnóstico con máscara de validez. Una pérdida
de consistencia TTC–velocidad será opcional y posterior; no forzar al estimador a
copiar una velocidad geométrica no validada. Un TTC óptico de aproximación no
certifica intersección futura de trayectorias con movimiento lateral.

### 7.5 Arquitectura B: compartir backbone

Después de A, probar features espaciales compartidas con cabezas de detección y
features por ROI/tiempo para TTC. El orden experimental será: capas tempranas
compartidas con adaptadores, ajuste parcial y, solo si procede, entrenamiento
conjunto. Mantener un control de capacidad y presupuesto equivalente.

Es un modelo nuevo. Un embedding arbitrario no cumple la interfaz de las veinte
variables de fase/timing. Las pérdidas de clasificación pueden favorecer
invariancia a escala mientras TTC necesita sensibilidad al cambio de escala;
evaluar esa interferencia. No atribuir resultados al objetivo JEPA sin controles
específicos; el [ADR-0042](../decisions/ADR-0042-phase-ttc-name.md) conserva esa
limitación científica y el namespace `e_jepa_ttc`.

### 7.6 Supervisión y cabezas aprendidas

El primer detector debe mantener la asignación y las pérdidas de su implementación
de referencia siempre que el contrato eAP lo permita. La adaptación propuesta
añade o cambia targets explícitos, no aprende cajas a partir del escalar TTC.

| Salida | Supervisión pública y candidato de objetivo | Precaución |
|---|---|---|
| Clase y presencia | Categoría, objetos y fondo; clasificación/focal según baseline | Fijar mapeo a ocho clases y tratamiento de ignorados antes del muestreo |
| Centro/profundidad | Caja métrica transformada al marco del detector; regresión robusta | Escala absoluta depende de supervisión y calibración, no de TTC |
| Dimensiones | Dimensiones positivas verificadas; parametrización logarítmica o la nativa | No reutilizar medias KITTI como si fueran priors eAP; estimarlas solo en train |
| Yaw | Orientación con convención comprobada; objetivo periódico/sin-cos o nativo | Evitar discontinuidad ±π y definir simetrías por clase conforme al scorer |
| Velocidad | Campo público con semántica validada; regresión y máscara de validez | No convertir missing en velocidad cero ni confundir absoluto con relativo |
| ROI visible 2D opcional | Anotación válida o proyección declarada como target amodal | Proyección 3D no demuestra contorno visible bajo oclusión |
| TTC/fase auxiliar posterior | Solo consultas con verdad TTC verificada; pérdida actual o versionada | Máscaras separadas de detección; no generar verdad usando predicciones del propio detector |

La correspondencia entre predicciones y GT, los negativos y la decodificación
deben seguir el detector elegido. Pesos de pérdidas, balance de clases y thresholds
se seleccionan en desarrollo, nunca en test. Si se añade TTC multitarea, comparar
gradientes, capacidad y presupuesto contra la versión sin esa pérdida.

Para eventos, comenzar con una representación densa causal de escena y un encoder
compatible. Comparar fusión simple con RGB antes de routing complejo. Los tensores
por ROI de PHASE y sus normalizaciones no se sustituyen por features de escena
sin adaptación y comprobación. Cualquier resize, crop o flip de entrenamiento
debe actualizar intrínsecos, ROIs y targets geométricos de forma coherente.

## 8. Incertidumbre y dominio numérico

Separar cuatro cantidades: score de detección, incertidumbre de localización,
confianza/ambigüedad de asociación y límites de fase. Ninguna calibración de una
de ellas demuestra calibración de las restantes.

La [transformación actual](../../src/e_jepa_ttc/simplex_t/phase.py#L16) limita la
magnitud de TTC a 60 s y asigna cero fase a +60 s. `ttc_to_phase` rechaza TTC
positivos menores o iguales a 0,1 s. Son condiciones numéricas del contrato,
no garantías físicas ni cobertura probabilística. Una modificación exige nueva
versión y comparación con la ruta congelada.

Si `phase_q10 < 0 < phase_q90`, convertir y ordenar los dos extremos no produce
un intervalo continuo ordinario de TTC firmado: la inversa cruza singularidad y
salto de signo, con saturación adicional en la implementación. Mantener límites
nativos y marcar ambigüedad de signo. Tocar cero requiere un caso distinto: el
intervalo `[0, fase_positiva]` solo emite TTC positivo bajo la convención saturada,
y `[0,0]` emite +60 s. Un intervalo de fase negativa que termina en cero incluye
el salto numérico hacia +60 s en ese extremo. Probar esos bordes explícitamente,
sin confundir saturación con distancia física a una colisión.
La etiqueta «intervalo al 80 %» requiere
cobertura empírica y una definición explícita, no solo nombres q10/q90.

Calibrar sobre secuencias reservadas y mostrar cobertura, anchura y riesgo frente
a cobertura por clase, signo, distancia, visibilidad y condición sensorial. Los
métodos conformales con feedback no autorizan recalibración online en test privado.

## 9. Protocolo experimental y exposición de datos

### 9.1 Dos carriles válidos, con afirmaciones distintas

**Piloto operativo:** usar los pesos TRAIN40 para probar contratos, causalidad,
paridad, formato y coste. Cualquier TTC interno sobre esas cuarenta secuencias se
etiquetará `development-exposed`; dividirlas ahora no borra su exposición anterior.

**Experimento de generalización:** separar secuencias para entrenamiento, selección
y calibración; entrenar productores sin los grupos evaluados y generar ROIs/features
fuera de fold para las etapas posteriores. Registrar exposición por módulo, incluidos
pesos pretrained, teachers y decisiones de selección. Tres semillas para los
resultados principales y bootstrap por secuencia, no por ventanas independientes.

Un detector entrenado y seleccionado sin las secuencias reservadas puede tener
evaluación AP disjunta aunque un PHASE downstream ya las haya visto, **si PHASE no
interviene en entrenamiento, cajas, scores, filtrado ni selección del detector**.
La independencia de esa evaluación no se extiende al TTC conjunto. Si PHASE
retroalimenta el detector, también debe auditarse esa exposición para AP.

Las ROIs/identidades de GarlTTC test no se usarán como detector eAP. No son una
alternativa a detectar objetos aunque sean públicas para otra tarea. Tampoco se
reservarán etiquetas privadas para debugging, ajuste o selección.

### 9.2 Matriz mínima, ejecutada de forma secuencial

| Experimento propuesto | Pregunta |
|---|---|
| D0: detector RGB | ¿Existe un baseline geométricamente correcto y reproducible? |
| D1: mismo detector + eventos, fusión simple | ¿Aportan eventos con presupuesto y protocolo controlados? |
| D2: fusión adaptativa frente a D1 | ¿Mejora frente a concatenación sin ocultar coste/capacidad extra? |
| O: ROI e identidad anotadas + PHASE | Referencia condicionada a localización; no sistema desplegable |
| R: ROI predicha + asociación GT solo de diagnóstico | Daño atribuible a localización/recorte |
| T: ROI predicha + asociación causal real | Rendimiento completo y fallos de identidad |
| P: perturbaciones de ROI/identidad/historia | Sensibilidad y políticas de recuperación |
| A: adaptación TTC con ROIs fuera de fold | ¿Recupera el cambio de distribución sin exposición al fold? |
| M: backbone compartido/multitarea | ¿Mejora el compromiso entre detección, TTC y coste frente a A modular? |

No ejecutar un producto cartesiano de todos los detectores, trackers y fusiones.
Seleccionar un baseline, aislar una hipótesis y ampliar solo con evidencia.

### 9.3 Métricas y denominadores

- **Detección:** AP y ATE/ASE/AOE según contrato verificado; desgloses por clase y
  condiciones con etiquetas disponibles. No derivar etiquetas meteorológicas del
  nombre de secuencia sin documentarlo.
- **TTC:** error de fase, MAE, mediana, p95, error de signo y saturación. Comparar
  soporte común O/R/T y además cobertura sobre **todas las consultas TTC elegibles
  con verdad verificable**; no asumir que toda caja 3D tiene TTC fiable.
- **Asociación:** ID switches, fragmentación, HOTA/IDF1 con matching declarado y
  porcentaje de historias de 650 ms contaminadas por otra identidad.
- **Disponibilidad:** objetos no detectados, falsos positivos, historias cortas,
  fallos sensoriales y causas de abstención. Un TTC menor sobre objetos fáciles no
  compensa esconder misses del denominador.
- **Incertidumbre:** cobertura/anchura en fase, curvas riesgo–cobertura y análisis
  de cruces de cero. Verificar soporte por clase antes de afirmar ocho clases TTC.
- **Eficiencia:** latencia de escena p50/p95/p99, memoria pico, colas, drops y edad de
  medida. Medir el camino completo; sumar p95 de módulos no da el p95 del sistema.

Fijar márgenes de aceptación y protocolo antes de observar test. El test oficial
se reservará para evaluación final del sistema congelado y admitido por las reglas.

## 10. Hardware, dependencias y entrega

La máquina inspeccionada dispone de RTX 5070 Ti Laptop, 12.227 MiB de VRAM total.
El proyecto declara Python >=3.11,<3.14 y Torch >=2.5,<2.12. Una receta externa con
Torch 1.9/CUDA 11.1 no debe sustituir el entorno actual. Probar operaciones
deformables/sparse en un entorno separado, con versión de CUDA y soporte Blackwell
comprobados. La compatibilidad del núcleo PyTorch no demuestra la de extensiones
de terceros ni Windows. WSL/Linux es una alternativa a medir, no una solución ya
instalada o verificada.

Para dimensionar, comenzar con batch uno y una configuración compacta; registrar
forward/backward y memoria del detector. Medir aparte la inferencia conjunta con
los productores TTC residentes: el entrenamiento por etapas no exige mantenerlos
en GPU. Reducir imagen
o volumen puede ayudar, pero modifica alcance/small objects y exige comparación.
Acumulación de gradiente no elimina la memoria de activaciones de un solo ejemplo.

El [runtime publicado](../../paper/sections/runtime.tex) midió un objeto durante
cinco segundos, RAW precargado y disco excluido: p95 29,14 ms. Existe software
multiobjeto, pero esa cifra no demuestra su coste. Medir cero, uno, varios y más
de veinte objetos, densidad de eventos, cambios de ROI y pérdida de cache. No se
fija aquí un presupuesto de nube ni una duración de entrenamiento sin perfil.

Licencias verificadas de código orientan la selección, no cubren automáticamente
pesos/datos. El árbol HF eAP inspeccionado no contiene `LICENSE` ni campo `license`
en la tarjeta; documentar términos pendientes antes de redistribución. No inferir
licencia del dataset a partir de la licencia del paper. DAGr GPL-3.0 añade un
aspecto de integración que no tienen candidatos MIT; revisar la composición antes
de publicar derivados.

## 11. Hoja de ruta con criterios de paso

Todos los entregables de esta tabla son **propuestos**, no archivos ya creados.

| Fase | Entregable | Criterio para avanzar | Si falla |
|---|---|---|---|
| G0: contrato y datos | Manifest fijado, matriz de exposición, fixtures geométricos y temporales | Resolver orden/ejes/yaw/centro/RGB y velocidad; medios necesarios íntegros | Continuar análisis, no entrenar geometría ni publicar métricas 3D |
| G1: viabilidad del detector | Entorno aislado, smoke forward/backward, memoria y overfit pequeño | Operaciones correctas, pérdidas finitas, gradientes y decodificación coherentes | Reducir configuración o escoger otra base antes de entrenar completo |
| G2: baseline RGB | Pesos, config, logs, predicciones y métricas por secuencia | Aprendizaje verificable en validación disjunta; evaluador local probado y sus límites declarados | Corregir datos/targets/decode; no esconderlo tras fusión |
| G3: eventos | Comparación D0/D1 y después D2 si procede | Beneficio o trade-off defendible bajo protocolo/coste común | Mantener RGB como baseline; registrar resultado negativo |
| G4: puente TTC | Diagnósticos O/R/T, paridad y perfil de escena | IDs/ROIs causales, cobertura y fallos explícitos, sin truncar detecciones por límite TTC | Ajustar asociación/preparación o abstener TTC conservando detecciones |
| G5: adaptación/calibración | OOF, pesos fold-locales, análisis de intervalos | Mejora validada sin contaminación y con alcance de clases declarado | Mantener sistema modular congelado y límites de fase sin promesas probabilísticas |
| G6: evaluación oficial | Export exacto, validación de tokens/cajas y registro de revisión | Scorer/reglas accesibles y sistema congelado, entradas permitidas | Conservar evaluación local; no llamarla score oficial |
| G7: backbone compartido | Ablation multitarea frente a referencia modular | Mejora medida de calidad/coste sin degradación no aceptada | Conservar módulos separados |

G0–G2 se pueden desarrollar con datos públicos de entrenamiento; completar todos
los RGB de test es necesario antes de inferencia test completa, no antes de un
piloto train. Si no se consigue el scorer, se puede investigar con métricas
etiquetadas de desarrollo, pero no afirmar paridad ni resultado oficial.

### 11.1 Módulos candidatos

Preservar namespace, checkpoints, hashes y experimentos históricos. Posibles rutas
nuevas, sujetas a diseño al implementar:

```text
src/e_jepa_ttc/data/eap_detection.py
src/e_jepa_ttc/detection/{types,model,decode,training}.py
src/e_jepa_ttc/tracking/{association,state}.py
src/e_jepa_ttc/inference/detection_ttc.py
src/e_jepa_ttc/inference/eap_detection_export.py
src/e_jepa_ttc/evaluation/{detection,detection_ttc}.py
configs/detection/ y configs/experiment/ para protocolos versionados
```

El puente reutilizará `MultiObjectTrain40Service` cuando cumpla el contrato; no
duplicará buffers ni supondrá que soporte IDs/ROIs predichos sin pruebas.

### 11.2 Pruebas necesarias

Reutilizar [tests de geometría](../../tests/unit/test_event_v4_geometry.py),
[lectura eAP](../../tests/unit/test_eap.py),
[restricciones de inferencia](../../tests/unit/test_final_refit_test12.py) y
[streaming](../../tests/test_rgb_port_streaming.py). Añadir pruebas que detecten:

1. Permutación dimensional, yaw invertido o proyección incorrecta con fixture externo.
2. Consumo de frames/eventos futuros o etiquetas en inferencia.
3. Borrado de expansión por resizing independiente de endpoints.
4. Cruce de dos objetos, ID swap, oclusión, nueva identidad y reset causal.
5. Falta de historia: conservar detección y expresar TTC ausente.
6. Cero objetos y más de veinte: conservar el contrato completo de detección.
7. Transporte de q10/q90 y TTC equivalente al cálculo canónico, con cruces de cero.
8. Exportación con todas las claves exactas, clases válidas, valores finitos y dimensiones positivas.
9. Contabilización de misses/abstenciones y métricas conocidas en fixtures sintéticos.
10. Medición completa de escena y recursos con régimen de llegada declarado.

## 12. Preguntas abiertas que cambian decisiones

- ¿Cuál es la conversión oficial exacta entre bundle, HF y JSON eAP?
- ¿Qué rectificación RGB/evento y semántica de velocidad aplica el release HF?
- ¿Qué reglas y scorer están vigentes en detección 16717 y qué fuentes externas admite?
- ¿Qué soporte real tienen las ocho clases en supervisión TTC y sus variantes de movimiento?
- ¿Qué detector compacto compila y aprende en el entorno compatible con Blackwell?
- ¿Cuánto degrada PHASE con ROIs predichas y cuánto viene de identidad frente a recorte?
- ¿Cuánto cuesta compartir sensores y procesar TTC de todos los objetos de una escena?
- ¿Aporta el backbone compartido algo más que aumentar capacidad o presupuesto?

Estas preguntas definen el trabajo siguiente. Ninguna se cierra mediante una tabla
SOTA de otro dataset o mediante el resultado TTC existente.

## Referencias primarias consultadas

Los enlaces de las tablas identifican artículos y código por separado. Las revisiones
fijadas son preferibles a ramas vivas para reproducir esta revisión.

[eap-paper]: https://arxiv.org/html/2603.16303v1
[eap-code]: https://github.com/NAIL-HNU/eAP_dataset/blob/2c5f3a74ba5b693b746820a0a355684e3597dc2d/README.md
[eap-visualizer]: https://github.com/NAIL-HNU/eAP_dataset/blob/2c5f3a74ba5b693b746820a0a355684e3597dc2d/release_visualizer/visualize_release_sequence.py#L459
[eap-hf]: https://huggingface.co/datasets/NAIL-HNU/eAP-dataset/tree/a814a946e60d37f7242358af5f344542177e50c5
[garl-hf]: https://huggingface.co/datasets/NAIL-HNU/GarlTTC-dataset/tree/9f477b5c4d5f9cdc32e7cc4bb3b186301979d6b5
[ev3dod-paper]: https://arxiv.org/abs/2502.19630
[ev3dod-code]: https://github.com/mickeykang16/Ev3DOD
[evstereo-paper]: https://arxiv.org/abs/2508.02288
[evstereo-code]: https://github.com/mickeykang16/Ev-Stereo3D
[dagr-paper]: https://doi.org/10.1038/s41586-024-07409-w
[dagr-code]: https://github.com/uzh-rpg/dagr
[flex-paper]: https://arxiv.org/abs/2412.06708
[flex-code]: https://github.com/DylanOrange/flexevent
[era-paper]: https://arxiv.org/html/2610.26090v1
[era-code]: https://github.com/Peijun1122/ERA-Det
[aero-paper]: https://arxiv.org/html/2608.01302v1
[peod-paper]: https://arxiv.org/abs/2511.08140
[peod-code]: https://github.com/bupt-ai-cz/PEOD
[rvt-paper]: https://openaccess.thecvf.com/content/CVPR2023/html/Gehrig_Recurrent_Vision_Transformers_for_Object_Detection_With_Event_Cameras_CVPR_2023_paper.html
[rvt-code]: https://github.com/uzh-rpg/RVT
[monodetr-paper]: https://arxiv.org/abs/2203.13310
[monodetr-code]: https://github.com/ZrrSkywalker/MonoDETR
[monodgp-paper]: https://openaccess.thecvf.com/content/CVPR2025/html/Pu_MonoDGP_Monocular_3D_Object_Detection_with_Decoupled-Query_and_Geometry-Error_Priors_CVPR_2025_paper.html
[monodgp-code]: https://github.com/PuFanqi23/MonoDGP
[monosaod-paper]: https://arxiv.org/abs/2604.01646
[monosaod-code]: https://github.com/VisualAIKHU/MonoSAOD
[magnet-paper]: https://arxiv.org/abs/2608.14282
[magnet-code]: https://github.com/mo-sameh/MAGneT-3D/tree/d93ba645e07a9be5240fd6337406d8c76a067d51
[streampetr-paper]: https://arxiv.org/abs/2303.11926
[streampetr-code]: https://github.com/exiawsh/StreamPETR
[sparse4d-paper]: https://arxiv.org/abs/2311.11722
[sparse4d-code]: https://github.com/HorizonRobotics/Sparse4D
[bevdet-paper]: https://arxiv.org/abs/2203.17054
[bevdet-code]: https://github.com/HuangJunJie2017/BEVDet
[imvoxel-paper]: https://arxiv.org/abs/2106.01178
[imvoxel-code]: https://github.com/open-mmlab/mmdetection3d/tree/main/configs/imvoxelnet
[monouni-paper]: https://openreview.net/forum?id=v2oGdhbKxi
[monouni-code]: https://github.com/Traffic-X/MonoUNI
[mapdet-paper]: https://arxiv.org/abs/2608.12179
[mapdet-code]: https://github.com/cvg/Map-Det3D
[ab3dmot]: https://github.com/xinshuoweng/AB3DMOT
[bytetrack]: https://github.com/FoundationVision/ByteTrack
[motip-paper]: https://openaccess.thecvf.com/content/CVPR2025/html/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.html
[motip-code]: https://github.com/MCG-NJU/MOTIP
[hota]: https://arxiv.org/abs/2009.07736
[cast]: https://openaccess.thecvf.com/content/WACV2026/html/Bai_CAST_Evaluating_Multi-Object_Trackers_with_Context-Aware_Switch_and_Transfer_Scores_WACV_2026_paper.html
[calibration]: https://proceedings.mlr.press/v70/guo17a.html
[aci]: https://papers.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html
[eci]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/cab5ae2704d3e01f06a92512a5376b87-Abstract-Conference.html
