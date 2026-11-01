# G0 — auditoría de datos y geometría eAP

**Estado: G0 BLOQUEADO.** La inspección de metadatos, etiquetas públicas y geometría
del visualizador original está terminada dentro de este alcance. La conversión
entre releases y la reproyección sobre sensores reales no están verificadas. No
se ha entrenado detección, no se han leído etiquetas test ni se han descargado
eventos o RGB. Fecha de inspección: 29 de octubre de 2026.

Base inicial PHASE-TTC: `e58d0106e1f114f00f796987f70cd622cc6d186e`.
Base final de `main`, revalidada antes del cierre:
`f2903f0bfd47d7bf56358a94ad5fac0b8a8906ef`. El avance solo incorpora los dos
informes previos, idénticos a los adjuntos; no cambia runtime, artículo ni manifests.
La auditoría final registra esta última base y los hashes de sus fuentes ejecutadas;
`g0_sources.json` conserva el recibo histórico de la inspección inicial.
Se preservan el namespace
`e_jepa_ttc`, los lectores y productores históricos, pesos, resultados y artículo.
Este documento acompaña a [DEVELOPMENT_PLAN.md](../../DEVELOPMENT_PLAN.md).

## 1. Qué se ha comprobado

Se han leído `AGENTS.md`, ambos documentos recibidos, los lectores reales eAP,
el inventario histórico, los splits y las fuentes oficiales siguientes:

- [Código eAP, revisión 2c5f3a74](https://github.com/NAIL-HNU/eAP_dataset/tree/2c5f3a74ba5b693b746820a0a355684e3597dc2d).
- [Dataset HF, revisión a814a946](https://huggingface.co/datasets/NAIL-HNU/eAP-dataset/tree/a814a946e60d37f7242358af5f344542177e50c5).
- [Tarjeta HF fijada](https://huggingface.co/datasets/NAIL-HNU/eAP-dataset/blob/a814a946e60d37f7242358af5f344542177e50c5/README.md).
- [Página de los autores](https://nail-hnu.github.io/eAP_dataset/), guardando hash de
  la respuesta inspeccionada; es una fuente descriptiva viva, no la revisión HF.

Se ha descargado exclusivamente metadata y las **40 etiquetas train faltantes
identificadas en los documentos**, con revisión fija y SHA-256: 128.416.244 bytes
de etiquetas. Los dos manifests ocupan otros 5.228.535 bytes. No se ha descargado
el dataset completo ni se ha reparado el disco `E:` del usuario: ese disco y su
GPU no están montados en esta sesión. Las copias para esta auditoría permanecen
fuera del repositorio; Git solo contiene hashes, contratos, resultados agregados,
código de auditoría y un fixture numérico sintético.

La salida regenerable es [g0_audit.json](evidence/g0_audit.json), y
[g0_sources.json](evidence/g0_sources.json) conserva revisión, URLs, hashes y
recibos de las cuarenta etiquetas. El inventario fijado completo está en
[eap_detection_public_files_v1.json](../../data/manifests/eap_detection_public_files_v1.json).

| Comprobación real | Resultado |
|---|---|
| API HF fijada | 278 archivos; revisión exacta `a814a946e60d37f7242358af5f344542177e50c5` |
| Train | 118.247 frames, 40 secuencias, 118.247 tokens únicos |
| Referencias train | 40 HDF5, 136 TAR RGB, 40 Parquet de etiquetas |
| Test, solo metadatos estructurales | 35.940 frames, 12 secuencias, tokens únicos |
| Referencias test | 12 HDF5, 45 TAR; `labels_path` nulo en todos los frames |
| Intersección train/test | Sin intersección de tokens ni secuencias |
| Exposición RGB train | Duraciones 886–19.992 µs; inicio creciente por secuencia |
| Exposición RGB test | Duraciones 805–19.992 µs; inicio creciente por secuencia |
| Matrices | Una calibración distinta en train y dos en test; estructura finita y rígida válida |
| Etiquetas descargadas | 40/40, 804.510 filas; hashes exactos |
| Claves etiqueta | Sin duplicados `(sample_token, track_id)` ni tokens/IDs de secuencia ajenos |
| Cajas y velocidades | Forma, finitud y dimensiones positivas en las 804.510 filas |

Estos recuentos proceden del release HF fijado. La página del proyecto describe
un alcance diferente: 58 secuencias, 46 train y 12 test, y otra cifra de cajas.
No se sustituye TRAIN40 por TRAIN46 ni se interpreta esa diferencia como error
demostrado. Falta documentar cómo se relacionan las versiones, el muestreo de
frames y las unidades de recuento. Los manifiestos SHA-256 determinan la población
de este piloto.

### 1.1 Soporte real de clases en todas las etiquetas train

| Clase | Filas anotadas |
|---|---:|
| car | 496.284 |
| truck | 179.441 |
| motorcycle | 80.997 |
| pedestrian | 29.377 |
| bus | 10.210 |
| tricycle | 5.559 |
| bicycle | 2.069 |
| other_vehicle | 573 |
| **Total** | **804.510** |

Son anotaciones por frame, no objetos independientes. El desbalance y la
correlación temporal importan al definir un piloto. `g0_audit.json.labels` incluye
recuentos por secuencia para revisar los splits sin abrir test. La existencia de
ocho clases de detección no prueba soporte TTC válido para todas ellas.

### 1.2 Revalidación del informe previo sobre faltantes

Las nueve rutas TAR test citadas en `swarm_research.md` existen en el inventario
remoto fijado. La unión de sus referencias afecta a **4.132 frames de siete
secuencias** y suma **4.793.784.320 bytes**, confirmando el alcance numérico del
informe. **No se ha revalidado que sigan ausentes en el ordenador del usuario.**
El informe actual registra `NOT_INSPECTED` para esos paths locales cuando la raíz
no es accesible. Una raíz inexistente tampoco cuenta como inspección del disco.

Las cuarenta etiquetas suman 128.416.244 bytes; etiquetas más los nueve TAR suman
4.922.200.564 bytes. Solo se han recuperado las etiquetas para esta auditoría.
El entrenamiento piloto no necesita recuperar RGB test. Un frame sin datos no
se convierte en una detección vacía para ocultar el error.

### 1.3 ZIP de submission

El ZIP aportado contiene un único `submission.json`, 35.940 tokens exactamente
iguales al manifest test y listas vacías. El contenido tiene SHA-256
`c307b01e150753254ce15776eb02b08a66f53ac893323d21c2ddd5a58a589106` y Git blob
`49868f7af188dc3733459097430dfc77e2a8d8db`, idéntico al blob anunciado por HF para
el template oficial. Esto verifica **identidad de archivo y contrato de claves**.
Las listas vacías no verifican convenciones geométricas por objeto.

La tarjeta HF y el `meta` del template documentan ocho clases y seis campos:
`class_name`, `translation_m`, `size_lwh_m`, `yaw_rad`, `velocity_mps`, `score`.
La tarjeta especifica `[length,width,height]` y velocidad planar `[vx,vy]`. No se
ha verificado el scorer ni sus filtros, límites, matching o tratamiento de casos
inválidos; este documento no atribuye carácter oficial a una métrica local.

## 2. Geometría: partes resueltas y partes abiertas

### 2.1 Bundle original: convención ejecutada, no deducida del nombre

Se ha ejecutado la función oficial `bbox_to_corner_ego` extraída sin modificar del
[visualizador fijado](https://github.com/NAIL-HNU/eAP_dataset/blob/2c5f3a74ba5b693b746820a0a355684e3597dc2d/release_visualizer/visualize_release_sequence.py).
El fixture utiliza un cuboide no cúbico de dimensiones `l=6,h=2,w=4`, centro
`(10,3,2)` y cuatro yaw: `0`, `+π/2`, `−π/2`, `0,37` rad. Sus ocho esquinas
numéricas quedaron guardadas en
[eap_original_visualizer_corners.json](../../tests/fixtures/detection/eap_original_visualizer_corners.json).

La implementación oficial establece, para **ese bundle original**:

- Caja `[x,y,z,l,h,w,yaw]`.
- Ancho sobre X local, largo sobre Y local y altura sobre Z local.
- Rotación positiva horaria en el plano XY.
- Centro geométrico: se suman desplazamientos simétricos `±dimensión/2`.
- Proyección `p_camera = T_event_ego · p_ego`; no se invierte esa matriz.

El nuevo helper `original_bundle_corners` reproduce esos fixtures con tolerancia
absoluta de `1e-6 m`; no cambia `data/eap.py` ni autoriza aplicarlo a HF.
La prueba de dirección de transform usa además un caso calculado a mano:
X cámara = −Y ego + 1; Y cámara = −Z ego + 2; Z cámara = X ego + 3.
El punto `(10,2,1)` se transforma en `(-1,1,13)`, con proyección conocida.

![Fixture sintético de la geometría original](evidence/g0_original_geometry.svg)

**Esta figura es una reproyección sintética, no una validación de calibración con
sensores reales.** [g0_original_geometry.json](evidence/g0_original_geometry.json)
registra matrices, esquinas proyectadas, errores y el hash de la fuente externa.

### 2.2 Parquet HF: relaciones verificadas en 804.510 filas

En todas las filas de las cuarenta secuencias se cumple:

```text
translation == ego_translation == bbox_3d_ego[:3]
size == bbox_3d_ego[[5,3,4]]
yaw == bbox_3d_ego[6]
rotation == [cos(yaw/2), 0, 0, sin(yaw/2)]
velocity == velocity_3d[:2]
velocity_3d[2] == 0
```

Las igualdades de campos son exactas, salvo el chequeo trigonométrico con
tolerancia `1e-12`. `ego_translation` copia el centro del objeto en estas filas;
**no es una pose ego temporal** que pueda alimentar un detector como MAGneT-3D.
La relación trigonométrica por sí sola tampoco documenta el orden semántico de
cuaternión ni corrige la diferencia de signo con el visualizador original.

Ejemplo real del Parquet: dimensiones en `bbox_3d_ego` =
`[4.8046875,1.6025390625,1.8271484375]`, mientras `size` =
`[1.8271484375,4.8046875,1.6025390625]`. La permutación se conoce; su puente físico
con `[l,w,h]` del submission y con el bundle original sigue **sin verificarse**.
Falta un conversor oficial o una pareja del mismo objeto/frame antes y después
del empaquetado HF. La consistencia de 804.510 filas no sustituye esa prueba.

El helper histórico de PHASE interpreta `[x,y,z,l,w,h,yaw]`, largo sobre X local
y yaw antihorario. La diferencia está comprobada entre funciones. **No demuestra
un bug histórico**, porque no se ha demostrado que ambos reciban el mismo formato
físico. No se altera el helper ni se recomputan sus resultados publicados.

### 2.3 Rectificación y correspondencia RGB–eventos

Los manifests contienen `K_event` y `T_event_ego`; no `K_rgb`, distorsión, mapas de
rectificación ni poses ego entre instantes. La página de los autores describe una
correspondencia de base estrecha, con disparidad menor de cinco píxeles. Eso no
prueba identidad píxel a píxel ni el preprocesado exacto aplicado a los PNG HF.

No hay RGB ni HDF5 accesibles aquí. Falta ejecutar reproyecciones sobre frames
train reales, guardar overlays sin selección por resultados y comprobar el
tratamiento de bordes, puntos detrás de cámara, truncamiento y ventanas evento.
No se cambia `K_event` por `K_rgb` implícitamente. La reutilización de ROIs
multimodales queda bloqueada hasta disponer del contrato preciso.

### 2.4 Velocidad y causalidad

El README original indica velocidad en marco ego; la página descriptiva añade
velocidad relativa al ego para la construcción TTC. Ninguna de estas lecturas
demuestra el proceso de conversión del campo HF, su signo, si está compensado
por rotación ego ni el suavizado temporal de las etiquetas. Se conservan ambos
niveles de evidencia, sin fijar una conversión basándose en un nombre.

Antes de usar velocidad como target o en `tau=-Z/dZ_dt`, hace falta contrastar un
fixture independiente o exportador, definir si se refiere al centro o superficie,
comprobar movimientos conocidos y tratar valores ausentes mediante máscara.
No se rellena missing con cero. Las velocidades anotadas se usan aquí solo para
auditar etiquetas train, nunca como input del modelo ni como estado de test.

Los timestamps de exposición son enteros y monótonos, pero **no prueban
disponibilidad de llegada**. El visualizador oficial emplea 50 ms centrados en el
medio de exposición; esa ventana puede incluir eventos posteriores al final RGB.
El nuevo chequeo separa `exposure_end_us`, `max_input_us`, `input_available_us` y
`decision_us`, rechaza input futuro y llegada tardía y exige que una entrada no
esté disponible antes de medirse. Son pruebas de contrato; la ausencia de HDF5
impide certificar todavía el stream real. El reconstructor histórico con pendientes
locales no se reutiliza como estimador causal en inferencia.

## 3. Subgates y criterio de desbloqueo

| Subgate | Estado obtenido | Evidencia o trabajo pendiente |
|---|---|---|
| G0.M: revisión, schema y tokens | PASS | Ambos manifests completos, hashes, clases y template |
| G0.L: etiquetas train | PASS | 40 archivos, 804.510 filas, SHA-256 y consistencia estructural |
| G0.B: geometría del bundle original | PASS, alcance original | Función oficial ejecutada + fixture no cúbico + reproyección sintética |
| G0.H: puente original ↔ HF ↔ submission | BLOCKED | Conversor oficial o pareja independiente + fixture con yaw y centro |
| G0.R: rectificación RGB/evento | BLOCKED | Conversor/calibración y reproyecciones sobre train real |
| G0.V: semántica de velocidad | BLOCKED | Marco, relativo/absoluto, referencia física, signo y construcción temporal |
| G0.C: causalidad del stream real | BLOCKED | Eventos, exposiciones, llegadas y pruebas de prefijos reales |
| G0.I: integridad de medios locales | BLOCKED | Disco no accesible; no se certifican TAR, miembros ni HDF5 |
| **G0 total** | **BLOCKED** | No habilita entrenamiento geométrico ni piloto G2 |

`configs/detection/eap_contract_v1.json` registra estos límites. El gate no se abre
con `training_geometry_approved=true`: las afirmaciones necesarias deben ser
`VERIFIED` y tener recibos JSON específicos, legibles, con revisión fija y SHA-256
que coincida. Un recibo declara claim, método, fuentes, resultado de comprobaciones
y limitaciones. La integridad de bytes no demuestra física correcta: los nuevos
recibos necesitan revisión independiente antes de aprobarlos.
Los paths de recibos son relativos al directorio del contrato pasado mediante
`--contract`, por ejemplo `evidence/hf_to_bundle_equivalence.json` bajo
`configs/detection/`. Se rechazan rutas absolutas, `..`, separadores no portables
y enlaces simbólicos cuyo destino salga de esa raíz; un hash correcto de un
archivo externo no basta para admitirlo. Las fuentes primarias pueden citarse
desde el contenido del recibo sin convertirlas en paths de lectura externos.

## 4. Código nuevo y pruebas

- `src/e_jepa_ttc/detection/data_audit.py`: manifests, checks de filas, relaciones
  de etiquetas, template y recuperación acotada.
- `src/e_jepa_ttc/detection/geometry_contract.py`: geometría **solo del bundle
  original**, estructura de calibración, relojes causales y gate de evidencia.
- `scripts/audit_eap_detection.py`: auditoría y recuperación con dry-run por defecto.
- `scripts/check_eap_geometry_reference.py`: reejecución de fuente oficial fijada,
  comparación con fixture y SVG sintético regenerable.
- `tests/unit/test_eap_detection_audit.py`: fixtures externos, permutaciones,
  timestamps futuros/tardíos, duplicados, test-label rejection, rutas inseguras,
  revisión no fijada, descargas corruptas y preservación de archivos preexistentes.

Las pruebas no descargan datos ni entrenan. Las comprobaciones finales dieron
39 casos correctos; la revisión independiente añadió regresiones para evidencia
no vinculada, tokens de otra secuencia y enlaces simbólicos fuera de la raíz.
El resultado final y la revisión están
en el informe general de esta iteración. Un `PASS` sintético no cambia el estado
de G0 real.

## 5. Reproducción

Ejecutar desde la raíz del worktree. Sustituir las rutas de ejemplo por rutas
locales existentes; no tratar `E:` como montado si no está accesible. El entorno
de auditoría es independiente del entorno de PHASE y del candidato detector.

```bash
uv venv --python 3.12 ../.venv-g0
uv pip install --python ../.venv-g0/bin/python -r configs/detection/g0_requirements.lock.txt
```

Windows puede usar `../.venv-g0/Scripts/python.exe`; fijar `PYTHONPATH=src` mediante
la sintaxis del shell correspondiente. En Linux, con un directorio vacío elegido
para metadata/etiquetas:

```bash
PYTHONPATH=src ../.venv-g0/bin/python scripts/audit_eap_detection.py recover \
  --metadata-root ../external/eap_metadata --output ../external/eap_recovery_plan.json

PYTHONPATH=src ../.venv-g0/bin/python scripts/audit_eap_detection.py recover \
  --metadata-root ../external/eap_metadata --download \
  --output ../external/eap_recovery_receipt.json
```

La recuperación solo permite `data/train.parquet`, `data/test.parquet` y los
cuarenta `data/train/<sequence>/labels.parquet`, con techo conjunto de 140 MB.
No descarga RGB, HDF5 ni etiquetas test. Verifica tamaño y SHA-256 antes de admitir
cada archivo; un archivo preexistente distinto se conserva y produce error.
Se rechazan rutas con traversal y padres simbólicos que resuelvan fuera de la
raíz elegida. En la auditoría, las etiquetas se leen desde `metadata_root` y
RGB/HDF5 desde `dataset_root`: no es necesario duplicar las etiquetas recuperadas.

```bash
PYTHONPATH=src ../.venv-g0/bin/python scripts/audit_eap_detection.py audit \
  --metadata-root ../external/eap_metadata \
  --sample-submission ../upload/sample_submission.zip \
  --output docs/detection_g0_g2_20261029/evidence/g0_audit.json
```

**Resultado esperado en esta sesión: exit code 2, `BLOCKED`**, con informe escrito.
El código de salida no debe ocultarse usando `|| true` en un pipeline que vaya a
autorizar entrenamiento. Cuando exista una raíz real de medios, añadir
`--dataset-root /ruta/eAP --verify-present --verify-tar-members`; el hashing
puede leer centenares de GB locales, por lo que se solicita explícitamente.
La auditoría distingue existencia, tamaño, checksum, miembros TAR y geometría.

Para reproducir el fixture externo:

```bash
git clone https://github.com/NAIL-HNU/eAP_dataset.git ../external/eAP_dataset
git -C ../external/eAP_dataset checkout 2c5f3a74ba5b693b746820a0a355684e3597dc2d
PYTHONPATH=src ../.venv-g0/bin/python scripts/check_eap_geometry_reference.py \
  --official-visualizer ../external/eAP_dataset/release_visualizer/visualize_release_sequence.py \
  --output-dir docs/detection_g0_g2_20261029/evidence
PYTHONPATH=src ../.venv-g0/bin/python -m pytest tests/unit/test_eap_detection_audit.py -q
```

El siguiente paso necesario es obtener el contrato/conversor exacto del release y
ejecutar las comprobaciones de sensor sobre **train**, con acceso al disco local.
Mientras tanto, G1 puede analizar dependencias y preparar pruebas en paralelo,
pero ni ese análisis ni los fixtures sintéticos autorizan G2.
