# Revisión independiente de código y conclusiones

Fecha: 29 de octubre de 2026. Base inicial inspeccionada de PHASE-TTC:
`e58d0106e1f114f00f796987f70cd622cc6d186e`.
**Base final de `main` inspeccionada:
`f2903f0bfd47d7bf56358a94ad5fac0b8a8906ef`.**
Rama de trabajo: `dev/eap-detection-g0-g1-20261029`.

**Dictamen de código: las correcciones solicitadas están incorporadas; las
comprobaciones reproducidas pasan dentro de su alcance. G0 y G1 siguen bloqueados
y G2 no se ha ejecutado.** La revisión no certifica geometría HF, calibración de
sensores reales ni compatibilidad del detector con una GPU.

**Revisión de `DEVELOPMENT_PLAN.md` y `EXECUTION_REPORT.md`: completada.** Tras las
correcciones descritas abajo, no quedan hallazgos materiales abiertos en los
cambios revisados. La conclusión admisible es la entrega de plan y herramientas
de auditoría; no el cierre de los gates de datos/GPU ni un baseline entrenado.

La revisión la realizó un agente distinto de los autores del código G0/G1 y del
plan. El revisor no modificó las implementaciones. Los problemas se comunicaron
a sus autores, que incorporaron las correcciones; después se volvió a leer el
código y ejecutar sus pruebas.

## 1. Alcance inspeccionado

Se leyeron `AGENTS.md`, los dos documentos aportados, el artículo y sus evidencias,
los lectores eAP, el runtime TRAIN40 y el head final existentes. La revisión del
cambio incluye:

- `src/e_jepa_ttc/detection/{data_audit,geometry_contract,protocol}.py`.
- `scripts/audit_eap_detection.py`, `scripts/check_eap_geometry_reference.py`,
  `scripts/prepare_detection_protocol.py`, `scripts/probe_detection_environment.py`
  y `scripts/check_deformable_operator.py`.
- Sus tres archivos de pruebas en `tests/unit/` y el fixture externo en
  `tests/fixtures/detection/`.
- Configuraciones en `configs/detection/`, inventario público fijado y candidato
  de partición por secuencias.
- Informes G0/G1, sus JSON de evidencia y las fuentes primarias clonadas a las
  revisiones declaradas.
- `DEVELOPMENT_PLAN.md` completo y `EXECUTION_REPORT.md`.

Se comprobó independientemente el avance final de `main`: el diff entre `e58d010`
y `f2903f0` contiene únicamente los dos MD de
`docs/phase_3d_ttc_20261029/`. Sus SHA-256 coinciden exactamente con los adjuntos
ya leídos. Runtime, AGENTS, artículo, manifests y resultados históricos son
idénticos entre ambas revisiones. La procedencia anterior de los recibos y del
split se conserva; la auditoría G0 regenerada registra `f2903f0`.

La comparación de archivos previamente seguidos por Git con la base no mostró
cambios históricos. Los archivos nuevos se revisaron directamente en el worktree;
los informes de auditoría registran hashes de sus productores, además del commit
base. Esto no convierte el commit base en una revisión que ya contenga el código
nuevo.

## 2. Hallazgos de código y correcciones verificadas

| Hallazgo inicial | Consecuencia | Corrección comprobada |
|---|---|---|
| El gate de geometría aceptaba cualquier cadena no vacía como path y hash de evidencia | Se podían eliminar bloqueos con un path inexistente y un hash arbitrario | `geometry_blockers` exige raíz explícita, SHA-256 válido, bytes existentes y coincidentes, schema de recibo, claim y revisión correctos, método, fuentes, resultado y limitaciones; regresor cubre ausencia, hash distinto y schema inválido |
| Un recibo con SHA correcto podía residir fuera de `evidence_root` mediante `..` o un symlink | El gate podía depender de un archivo ajeno al árbol portable declarado | Se exigen paths relativos canónicos, sin traversal ni separadores no portables, y resolución dentro de la raíz; se prueban rechazo exterior y admisión del mismo recibo dentro |
| El conjunto global de tokens train permitía asociar a una secuencia un token de otra | Una fila con `sequence_id` local y token extranjero podía pasar el chequeo estructural | Validación explícita del prefijo `sequence_id:` además de pertenencia al conjunto; regresor con ambos tokens en train |
| La presencia de un argumento `dataset_root` se confundía con revalidar los nueve TAR ausentes históricos | El informe podía atribuir una ausencia local sin inspeccionarla | Estado de directorio real y estados por cada path; `NOT_INSPECTED` cuando no es accesible, y ausencia confirmada solo cuando los nueve paths fueron comprobados como ausentes |
| Una ruta textual segura podía atravesar un directorio symlink hacia fuera del root de recuperación | La descarga podía escribir fuera de la raíz elegida | Resolución del destino y temporal dentro de `root.resolve()`; regresor preserva intacto un directorio externo |
| Los Parquet recuperados en `metadata_root` también se exigían en `dataset_root` | Un uso legítimo de raíces separadas podía bloquear disponibilidad por pedir duplicados | Labels se resuelven desde `metadata_root`; RGB y eventos desde `dataset_root`; se registra la raíz de verificación |
| La bandera de verificación podía representar una opción solicitada, en vez del resultado obtenido | Ambigüedad al interpretar disponibilidad e integridad | Campos separados para comprobación solicitada y verificación real; los resultados requieren medios train completos y miembros TAR encontrados |

La integridad de un recibo de geometría no demuestra que su interpretación física
sea correcta. El código y el informe lo dicen expresamente. Antes de cambiar una
afirmación a `VERIFIED` hará falta revisar sus fuentes, fixtures y reproyecciones;
no basta fabricar un JSON que cumpla el schema.

La recuperación mantiene un repositorio HF y una revisión exactos, limita la lista
a manifests y etiquetas train, establece un techo de 140 MB, verifica tamaño y
SHA-256 y preserva archivos existentes distintos y temporales preexistentes. No
hay una ruta autorizada de descarga de RGB test, eventos o etiquetas test en esta
herramienta. Las pruebas de recuperación usan respuestas simuladas y no descargan
datos.

## 3. Comprobaciones y reproducción independiente

Desde la raíz del worktree, con el entorno de auditoría separado:

```bash
../.venv-g0/bin/python -m pytest \
  tests/unit/test_eap_detection_audit.py \
  tests/unit/test_detection_protocol.py \
  tests/unit/test_detection_environment.py -ra

../.venv-g0/bin/python -m ruff check \
  src/e_jepa_ttc/detection \
  scripts/audit_eap_detection.py scripts/check_eap_geometry_reference.py \
  scripts/prepare_detection_protocol.py scripts/probe_detection_environment.py \
  scripts/check_deformable_operator.py \
  tests/unit/test_eap_detection_audit.py tests/unit/test_detection_protocol.py \
  tests/unit/test_detection_environment.py

../.venv-g0/bin/python -m ruff format --check \
  src/e_jepa_ttc/detection \
  scripts/audit_eap_detection.py scripts/check_eap_geometry_reference.py \
  scripts/prepare_detection_protocol.py scripts/probe_detection_environment.py \
  scripts/check_deformable_operator.py \
  tests/unit/test_eap_detection_audit.py tests/unit/test_detection_protocol.py \
  tests/unit/test_detection_environment.py
```

La verificación final registra **63 pruebas correctas, cero fallos, cero errores
y cero skips**: 39 de G0, 13 de protocolo y 11 de entorno. Se comprobó el JUnit
[final_tests.xml](evidence/final_tests.xml). Ruff pasa y **12 archivos** pasan la
comprobación de formato. El revisor ejecutó independientemente el pase completo
de 57 casos previo a la última protección de paths y volvió a ejecutar los nueve
casos de geometría que incluyen las seis regresiones nuevas después de ese
cambio. También confirmó la ejecución de `--help` con el `PYTHONPATH` corregido.
Se leyó [final_verification.json](evidence/final_verification.json), que conserva
los comandos, exit codes, hashes de las doce fuentes Python y alcance de la
verificación final; coincide con estos resultados y con los gates bloqueados.

No se ejecutó una suite histórica de entrenamiento o de runtime porque estos
módulos no se modificaron, no hay pesos disponibles y este entorno no contiene
Torch. Las pruebas unitarias de una guardia CUDA no son pruebas CUDA.

Comprobaciones independientes adicionales:

| Evidencia | Reproducción del revisor | Alcance |
|---|---|---|
| Fixture de esquinas del bundle original | Se extrajo y ejecutó exclusivamente la función `bbox_to_corner_ego` de la fuente oficial fijada; los cuatro casos coinciden exactamente con el fixture numérico archivado | Demuestra la convención del visualizador original, no equivalencia HF |
| Fuente del visualizador | SHA-256 y commit coinciden con el fixture | Revisión `2c5f3a74ba5b693b746820a0a355684e3597dc2d` |
| MonoDGP y MonoDETR | Se comprobaron ambos commits y los 20 hashes de archivos declarados | La lectura corresponde a las fuentes declaradas; no demuestra compatibilidad de ejecución |
| Tres informes G1 | Los hashes `probe_source_sha256` coinciden con los scripts revisados | Evidencia vinculada a los productores actuales |
| Auditoría G0 completa | El revisor volvió a leer las cuarenta etiquetas y ambos manifests mediante el CLI; exit 2, siete bloqueos y 804.510 filas, sin diferencias en resultados respecto a la auditoría archivada | La última regeneración tiene hashes de productor, contrato y manifest coincidentes y commit `f2903f0`; sus resultados son idénticos al run independiente, descontando procedencia y escritura de rutas |
| Preflight G1 | Ejecución independiente nueva: exit 2, estado `BLOCKED`, detector `NOT_RUN` y memoria `null` | No se infiere compatibilidad del equipo del usuario |
| Partición de secuencias | Regenerada en archivo temporal: asignaciones y procedencia idénticas a la versión incluida al ejecutarse sobre su base inicial | 24/8/8; candidato de desarrollo, sin datos test |
| Evidencias públicas del artículo | Se recomputaron los 13 hashes del registro público y coincidieron | Preservación y trazabilidad de resultados históricos archivados |
| Estado de Git | No hay modificación de archivos históricos seguidos frente a la base inspeccionada | El trabajo se añade en archivos nuevos |

La regeneración del fixture se puede repetir con el nuevo script, dirigiendo los
resultados a un directorio temporal para preservar la evidencia archivada:

```bash
PYTHONPATH=src ../.venv-g0/bin/python scripts/check_eap_geometry_reference.py \
  --official-visualizer ../external/eAP_dataset/release_visualizer/visualize_release_sequence.py \
  --output-dir /tmp/phase-independent-geometry-review
```

## 4. Validez de las conclusiones de G0

Los recuentos de manifests y filas describen el release HF fijado. La revisión
acepta distinguirlos de las cifras de otra versión descrita en la página del
proyecto. No hay justificación para sustituir TRAIN40 por un supuesto TRAIN46.

El fixture oficial establece para el bundle original orden `[x,y,z,l,h,w,yaw]`,
dimensiones sobre X/Y/Z locales como ancho/largo/altura, yaw horario, centro
geométrico y multiplicación de puntos ego por `T_event_ego`. En HF se comprueba una
permutación numérica de campos; ese hecho no prueba el puente físico al original
ni al submission. Las dos conclusiones se mantienen separadas.

El template de submission tiene listas vacías: puede verificar tokens, clases y
campos, pero no la semántica de una caja, el orden real de dimensiones o el signo
de yaw. Tampoco acredita un scorer. No se ha producido una métrica de detección
que pueda denominarse oficial.

La exposición RGB, los timestamps máximos de entrada, la llegada y la decisión
se tratan como relojes distintos. La selección de eventos centrada del
visualizador oficial es una herramienta de visualización y no valida una ventana
causal online. Falta inspeccionar HDF5 y frames reales.

Por tanto, **G0 total continúa bloqueado** por el puente entre formatos,
rectificación/correspondencia multimodal, semántica de velocidad, causalidad del
stream real e integridad de los medios locales. Los PASS parciales de metadata y
del bundle original no abren G2.

## 5. Validez de las conclusiones de G1 y protocolo

El inventario diferencia un import de Torch, una multiplicación CUDA de prueba,
la paridad de una extensión y un detector completo. `probe_detection_environment`
no puede devolver un PASS de G1; `check_deformable_operator` conserva
`g1_passed=false` incluso cuando la paridad de operador pasase en otro equipo.
Forward, backward, decode y VRAM del detector permanecen `NOT_RUN`/`null` en la
evidencia de esta sesión.

Las incompatibilidades potenciales de la extensión proceden de lectura de código
antiguo y flags observados. El informe no las presenta como un fallo de
compilación ya medido en la GPU del usuario. MonoDETR comparte esa extensión y no
es un fallback de compatibilidad independiente. La receta de Torch/Torchvision
está etiquetada como propuesta; no existe todavía un entorno GPU de detector
resuelto y comprobado.

El candidato de partición toma únicamente el manifest train y separa secuencias
completas. Su método determinista no consulta predicciones ni resultados. La
validación detecta solapamientos, omisiones, secuencias prohibidas y grupos de
sesión compartidos cuando se proporciona ese mapa. La ausencia del mapa no se
presenta como independencia entre sesiones.

La independencia de ese split solo puede aplicarse a futuros ajustes del detector
que respeten los roles. Los pesos TRAIN40 de PHASE ya estuvieron expuestos a sus
cuarenta secuencias; no pasan a ser independientes por cambiar un JSON de split.
Un experimento posterior con esos pesos congelados sobre esas secuencias debe
etiquetarse como desarrollo de integración.

**G1 continúa bloqueado y G2 no se ha ejecutado.** Esta conclusión es consistente
con los datos disponibles y con la condición solicitada por el usuario.

## 6. Revisión del plan y del informe de ejecución

Se contrastó el plan completo con el código real y con los informes G0/G1. Las
correcciones materiales solicitadas al autor quedaron incorporadas:

| Punto | Corrección verificada en el plan |
|---|---|
| Comandos en un venv limpio | Se declara activación del entorno y `PYTHONPATH`; el CLI G0 ya no se presenta como importable sin instalar el paquete o añadir `src` |
| Alcance real del auditor | NEXT-01 distingue la auditoría global implementada de una admisión futura por tokens/subset; no existe ni se anuncia un flag de subset |
| Exposición de productores | El carril de generalización incluye pretraining supervisado y SSL; un protocolo transductivo requiere etiqueta propia |
| Calibración TRAIN40 | Reservar secuencias para el calibrador no convierte en independiente un productor que ya las vio |
| Score de MonoDGP | Se documenta `p_clase * exp(-log_variance)` sin cota superior; el score nativo se conserva y `class_sigmoid_v1` es una política exportable propuesta y explícita, sin clamp oculto ni claim de calibración |
| Prior vertical de profundidad | Se identifica el uso upstream de `fx` y se exige un fixture con `fx != fy`, resize anisótropo y rectificación; cambiar K por sí solo no acredita el prior |

El informe de ejecución distingue lo implementado de los runners futuros y no
convierte propuestas en mediciones. La receta GPU sigue sin resolver; no hay
forward/backward/decode ni memoria de detector. Los resultados TTC históricos se
atribuyen a evidencia archivada. Se declara que el scorer original no está
distribuido y que su archivo de resultados no vincula un hash de las predicciones
enviadas; no se usan como resultados eAP de detección.

### 6.1 Cobertura de los once requisitos

| Requisito solicitado | Secciones del plan | Evaluación de cobertura |
|---|---|---|
| Estado real: existente, reutilizable y ausente | §1 | Inventario contrastado con código; no inventa detector, tracker ni pesos locales |
| Crítica de informes previos | §2 | Conserva acuerdos y corrige geometría, chunks, identidad, GPU y scorer |
| Arquitectura, contratos y salidas | §3 y §8 | Marco interno propuesto, times y nulabilidad explícitos; export eAP separado de TTC |
| Detector y alternativas | §5 | MonoDGP condicionado, MonoDETR control de familia, ImVoxelNet con gate propio |
| Datos, geometría, velocidad y causalidad | §4 | Subgates, fuentes/fixtures, overlays y bloqueos reales |
| Multiobjeto, ROIs, identidad e historia | §8 | Ingestión única, cero objetos, chunks, buffers prestados, generaciones, caches y abstenciones |
| Entrenamiento, validación, calibración y evaluación | §6 y §7 | Secuencias completas, exposición de toda la cadena y métricas de desarrollo |
| Fases, archivos, tareas, pruebas y aceptación | §9 | Dependencias y nombres futuros diferenciados de código existente |
| Experimentos decisivos | §10 | Eventos, efecto de ROI/asociación, adaptación OOF y backbone compartido |
| Recursos y riesgos | §11 | GPU no accesible, dependencias, memoria sin medir, datos/pesos y fallbacks |
| Ruta crítica y primera iteración ejecutable | §12 | Comandos reales y tickets NEXT-01–09; no simula comandos que resolverían evidencia inexistente |

### 6.2 Contraste de la integración con PHASE

Se verificó que la propuesta exige ingestión global incluso con cero detecciones,
evita repetir `update()` sobre el mismo paquete al dividir objetos y respeta que
`query_many()` presta buffers. Distingue IDs aportados de pertenencia histórica
demostrada; los GRU se reconstruyen por llamada, por lo que no inventa un estado
recurrente persistente por track. La retención actual de cache por llamada y el
problema de expulsión entre chunks quedan expuestos.

La ampliación de cuantiles se plantea como ruta nueva con paridad frente al export
canónico, manteniendo los productores y hashes históricos. No reconstruye
anchuras a partir del TTC escalar ni ordena extremos que atraviesan fase cero
como si formasen un intervalo TTC ordinario. Backbone compartido y entrenamiento
multitarea siguen en fases posteriores con comparación modular explícita.

## 7. Dictamen final y límites

**Revisión completada del código nuevo, el plan, los informes de ejecución y sus
conclusiones. No quedan correcciones materiales abiertas de esta revisión.** Los
archivos revisados son adecuados para auditar datos, registrar bloqueos y guiar
la continuación condicionada. Las pruebas no permiten declarar el sistema 3D
funcional, una GPU compatible ni un rendimiento de detección.

Permanecen los bloqueos científicos y de acceso documentados: correspondencia
original/HF/submission, rectificación real, semántica de velocidad, causalidad y
medios del workstation, además del entorno y smoke completo de detector en GPU.
La revisión no los exime ni los transforma en aprobación. **G0: BLOCKED. G1:
BLOCKED. G2: NOT_RUN.**
