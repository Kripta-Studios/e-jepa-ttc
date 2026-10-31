# Informe de ejecución — auditoría y preparación de detección 3D

Fecha: 29 de octubre de 2026. Documento principal: [DEVELOPMENT_PLAN.md](../../DEVELOPMENT_PLAN.md).

## 1. Resultado y alcance real

Se ha preparado el plan de desarrollo completo y se han implementado herramientas de auditoría de datos/geometría, preflight del detector y partición por secuencias. **G0 sigue bloqueado, G1 sigue bloqueado y G2 no se ha ejecutado.** No hay detector eAP entrenado ni resultados de detección 3D obtenidos en esta iteración.

El encargo contiene una petición inicial de planificación y después un alcance operativo explícito G0/G1, con G2 condicionado. Se ha aplicado este último alcance para las comprobaciones y herramientas reversibles, sin superar los gates ni iniciar entrenamiento. El documento principal distingue **H** (hecho comprobado), **P** (afirmación publicada), **D** (decisión propuesta), **HIP** (hipótesis) y **B** (bloqueo).

| Gate | Resultado observado | Qué falta para cerrarlo |
|---|---|---|
| G0: metadatos y etiquetas | Verificados dentro del release HF fijado | No equivale a verificar sensores y geometría entre formatos |
| G0: geometría del bundle original | Fixture no cúbico contra función oficial y proyección sintética correctos | Pareja original/HF o conversor, contrato de submission, rectificación real, velocidad y causalidad |
| G0 total | **BLOCKED** | Además, medios RGB/eventos locales no accesibles |
| G1: preparación | Fuentes fijadas, riesgos localizados, entorno de auditoría separado y pruebas de bloqueo | Resolver y compilar entorno GPU, ejecutar modelo completo |
| G1 total | **BLOCKED** | GPU local no accesible; forward, backward, decode y VRAM no medidos |
| G2: overfit y piloto RGB | **NOT_RUN** | G0 y G1 completos; después loader/targets, overfit y evaluación por secuencias |

Los blockers se mantienen aunque todas las pruebas unitarias de las herramientas pasen. No se interpreta ausencia de GPU como un fallo demostrado de MonoDGP en el portátil del usuario.

## 2. Git, fuentes y preservación

Se clonó el repositorio público y se creó el worktree aislado `phase-ttc-work`, rama `dev/eap-detection-g0-g1-20261029`. En el espacio de trabajo no había un checkout previo ni cambios del usuario que incorporar. La inspección comenzó sobre `e58d0106e1f114f00f796987f70cd622cc6d186e`.

Una comprobación posterior de `origin/main` detectó **`f2903f0bfd47d7bf56358a94ad5fac0b8a8906ef`**, publicado el 29-10-2026 a las 11:56:49 +0100. Se inspeccionó y se avanzó la rama mediante fast-forward. Este commit solo añade `docs/phase_3d_ttc_20261029/swarm_research.md` y `council_review.md`; sus hashes son idénticos a los adjuntos que ya se habían leído. Código, `AGENTS.md`, artículo, resultados y manifests históricos no cambiaron. Los recibos anteriores conservan la revisión en que se generaron; no se reescribe su procedencia.

La base final inspeccionada y del parche entregado es `f2903f0bfd47d7bf56358a94ad5fac0b8a8906ef`. Se conserva el namespace `e_jepa_ttc`. Todos los cambios de esta iteración son archivos nuevos; no se modifica ningún archivo distribuido en esa base. Tampoco se cambian `pyproject.toml`, `uv.lock`, lectores/productores legacy, artículo, tablas, hashes ni evidencias históricas. No había checkpoints locales accesibles en el checkout para ejecutar PHASE. No se han descargado, sustituido o entrenado pesos ni se han enviado submissions o cambios remotos.

| Fuente | Revisión inspeccionada |
|---|---|
| PHASE-TTC, base final | `f2903f0bfd47d7bf56358a94ad5fac0b8a8906ef` |
| eAP original | `2c5f3a74ba5b693b746820a0a355684e3597dc2d` |
| eAP Hugging Face | `a814a946e60d37f7242358af5f344542177e50c5` |
| MonoDGP | `aa059a18214aebf644510e7f0793971b403f9d14` |
| MonoDETR | `6994b9f512400b258c6edb75f77423beb9c126f2` |

[inspection.json](evidence/inspection.json) registra los hashes de los tres adjuntos y la evolución de la base. Las fuentes externas, URLs y hashes individuales están en [g0_sources.json](evidence/g0_sources.json) y [detector_sources_v1.json](../../configs/detection/detector_sources_v1.json). Los clones externos y los datos descargados permanecen fuera del repositorio.

## 3. Cambios implementados

| Área | Archivos principales | Funcionalidad entregada |
|---|---|---|
| Auditoría de datos | `src/e_jepa_ttc/detection/data_audit.py`, `scripts/audit_eap_detection.py` | Validación de manifests/etiquetas, tokens, hashes, faltantes y template; recuperación pública acotada |
| Geometría y evidencia | `detection/geometry_contract.py`, `scripts/check_eap_geometry_reference.py` | Fixture del bundle original, estructura de extrínsecas, clocks y gate que requiere recibos vinculados a bytes |
| Contrato y fuentes | `configs/detection/eap_contract_v1.json`, `data/manifests/eap_detection_public_files_v1.json` | Estado bloqueado explícito y manifest público de 278 archivos fijados |
| Entorno/operadores | `scripts/probe_detection_environment.py`, `scripts/check_deformable_operator.py` | Inventario sin certificar el modelo; comparación CUDA de operador con referencia y assertions cuando exista el entorno |
| Reproducibilidad G1 | `configs/detection/requirements-audit.txt`, `detector_sources_v1.json`, `environment_candidate_v1.json` | Lock del entorno de auditoría, fuentes fijadas y receta GPU propuesta no validada |
| Protocolo | `detection/protocol.py`, `scripts/prepare_detection_protocol.py`, `data/splits/eap_detection_candidate_v1.json` | Split determinista de secuencias train, detección de exposición y rechazo de sobrescritura |
| Pruebas | Tres nuevos archivos `tests/unit/test_*detection*.py` y fixture en `tests/fixtures/detection/` | Casos positivos/negativos, procedencia, finitud estructural, rutas y gates cerrados |
| Documentación | Plan, informes G0/G1, este informe y revisión independiente | Arquitectura, tareas futuras, contratos, experimentos y aceptación por fase |

No se han implementado todavía `smoke_eap_detector.py`, `train_eap_detector.py`, `evaluate_eap_detection.py`, un detector eAP, un tracker ni el puente detector–PHASE. Esos nombres son tickets futuros claramente identificados en el plan.

## 4. Resultados obtenidos

### 4.1 Datos y geometría

La auditoría leyó **118.247 frames de 40 secuencias train** y **35.940 frames de 12 secuencias test**, estos últimos únicamente como metadatos estructurales. No encontró intersección de tokens ni secuencias. Se recuperaron solo las cuarenta etiquetas train faltantes identificadas, 128.416.244 bytes, junto a los dos manifests, 5.228.535 bytes, y metadata pequeña. No se descargaron imágenes RGB, HDF5, TAR test ni el dataset completo.

Las cuarenta etiquetas contienen **804.510 filas**. Se comprobaron hashes, formas, finitud, positividad de dimensiones, pertenencia token/secuencia, ausencia de claves duplicadas y relaciones algebraicas de los campos. En todas las filas:

```text
size = bbox_3d_ego[[5, 3, 4]]
translation = ego_translation = bbox_3d_ego[:3]
yaw = bbox_3d_ego[6]
rotation = [cos(yaw/2), 0, 0, sin(yaw/2)]
velocity = velocity_3d[:2]; velocity_3d[2] = 0
```

Esto **no prueba** que un nombre HF identifique una convención física ni que `ego_translation` sea una pose del vehículo. En este release ese campo coincide numéricamente con el centro del objeto.

El fixture ejecuta la función del visualizador original con dimensiones distintas y yaw 0, ±π/2 y 0,37. Verifica su orden `[x,y,z,l,h,w,yaw]`, extensión local ancho/largo/alto sobre X/Y/Z, yaw horario y centro geométrico. Comprueba por separado que el visualizador transforma puntos mediante `T_event_ego` hacia cámara. El SVG conservado es **una proyección sintética**, no una reproyección validada de sensores eAP. Falta demostrar original→HF→submission con referencias correspondientes.

Las nueve rutas TAR test del informe previo suman 4.793.784.320 bytes y afectan a 4.132 frames de siete secuencias, según los manifests. **No se puede confirmar que sigan ausentes en el ordenador del usuario**: su disco no está montado. Se registra `NOT_INSPECTED`, no una observación falsa de ausencia. No se descargan porque no son necesarios para G2.

El ZIP aportado contiene un único JSON, con los 35.940 tokens test y listas vacías. Su SHA-256 y Git blob coinciden con el template HF. Sirve para identidad y claves; las listas vacías no prueban yaw, dimensiones o reglas del scorer. La auditoría detallada y los conteos de ocho clases están en [G0_DATA_GEOMETRY.md](G0_DATA_GEOMETRY.md).

### 4.2 Detector y GPU

La máquina de esta sesión tiene CPython 3.12.14 y no expone dispositivos NVIDIA, `nvidia-smi`, `nvcc` ni Torch instalado. Se crearon entornos aislados para auditoría y se ejecutaron los scripts de preflight. El resultado de cada uno es **exit code 2 / BLOCKED**; el script no confunde una comprobación de prerequisitos con un detector válido.

| Comprobación requerida G1 | Medición real en esta sesión |
|---|---|
| Compilación de extensión | No ejecutada |
| Paridad CUDA MonoDGP/MonoDETR | Bloqueada por entorno |
| Forward de detector completo | No ejecutado |
| Backward, pérdidas y gradientes finitos | No ejecutados |
| Decodificación eAP | No implementada ni ejecutada |
| VRAM máxima, latencia y throughput | No medidos; valores `null`/`NOT_RUN` |

La revisión de fuentes identificó flags sm60/61/70/75, APIs CUDA/ATen históricas, tests upstream que solo imprimen comparaciones, supuestos KITTI de clases y filtros, constantes de resolución/profundidad, score nativo sin cota superior y un prior vertical que usa `fx`. Son tareas concretas de port, **no fallos medidos en la GPU objetivo**. MonoDETR comparte el operador; ImVoxelNet introduce su propio stack. La receta Torch 2.11.0+cu128 / torchvision 0.26.0+cu128 está marcada `PROPOSED_NOT_INSTALLED_OR_VALIDATED`, sin lock GPU completo. Véase [G1_DETECTOR_FEASIBILITY.md](G1_DETECTOR_FEASIBILITY.md).

### 4.3 Protocolo y evidencias PHASE

El candidato de split contiene 24 secuencias train, ocho de selección y ocho de calibración, semilla 7, orden por SHA-256 de `seed:sequence_id`. Su estado es `CANDIDATE_METADATA_ONLY`. No se usaron métricas de modelos, test ni píxeles para elegirlo. Aún deben verificarse grupos de adquisición relacionados y soporte de clases en los tokens concretos del piloto. El commit inicial de su receipt se conserva y la reproducción final verifica las mismas asignaciones.

La independencia del futuro detector **no vuelve independientes los pesos TRAIN40 de PHASE**. Las cuarenta secuencias ya estuvieron expuestas a esos productores. Selección, SSL/pretraining, normalizadores, teachers y calibración tienen reglas explícitas en el plan; una calibración reservada no borra exposición del productor.

Se ejecutó `paper/scripts/verify_public_data.py`: verificó 13 hashes públicos, ocho recomputaciones de transferencia y 12 campos de scores archivados. La cifra archivada weighted MiD 35,80145158082389 pertenece al resultado TTC conservado en el artículo; **no es una evaluación de detección 3D, una nueva ejecución del scorer ni una comprobación actual del ranking**. El scorer de detección 3D no se obtuvo de forma verificable y no se produjo ninguna métrica de detección en esta iteración.

## 5. Reproducción y resultados de verificación

Desde el worktree, ajustar solo las raíces si la disposición local es distinta. El entorno de auditoría se crea separado; no instalar dependencias en PHASE:

```bash
uv venv --python 3.12 ../.venv-g0
uv pip install --python ../.venv-g0/bin/python -r configs/detection/g0_requirements.lock.txt
source ../.venv-g0/bin/activate
export PYTHONPATH="$PWD/src"
```

Recuperación permitida, primero dry-run. El `--download` siguiente descarga únicamente las rutas allowlisted de manifests y etiquetas train, a revisión fija, con techo total de 140 MB. Un archivo preexistente distinto genera error y se conserva:

```bash
python scripts/audit_eap_detection.py recover \
  --metadata-root ../external/eap_metadata --output /tmp/eap_recovery_plan.json
python scripts/audit_eap_detection.py recover \
  --metadata-root ../external/eap_metadata --download --output /tmp/eap_recovery_receipt.json
python scripts/audit_eap_detection.py audit \
  --metadata-root ../external/eap_metadata \
  --sample-submission ../upload/sample_submission.zip \
  --output /tmp/eap_g0_reproduced.json
```

El último comando termina con **2/BLOCKED** en las condiciones de esta sesión. Cuando los medios estén accesibles, usar `--dataset-root`, `--verify-present` y `--verify-tar-members`. El auditor implementado exige todos los medios referenciados de train; **no tiene flag de subset del piloto**. NEXT-01 especifica cómo añadir una admisión por tokens y hash si se necesita, preservando el estado global.

Con los clones externos fijados a las revisiones de §2, ejecutar:

```bash
python scripts/check_eap_geometry_reference.py \
  --official-visualizer ../external/eAP_dataset/release_visualizer/visualize_release_sequence.py \
  --output-dir /tmp/eap_original_geometry_reproduced
python scripts/probe_detection_environment.py \
  --monodgp-root ../external/MonoDGP --monodetr-root ../external/MonoDETR \
  --output /tmp/eap_detector_environment.json
python scripts/check_deformable_operator.py \
  --family monodgp --source-root ../external/MonoDGP \
  --output /tmp/eap_monodgp_operator.json
python scripts/check_deformable_operator.py \
  --family monodetr --source-root ../external/MonoDETR \
  --output /tmp/eap_monodetr_operator.json
```

Los tres últimos comandos están bloqueados aquí. No usar `|| true` para convertirlos en una aprobación. Los informes G0/G1 contienen instalación, flags y alcance exactos; no presentan un runner futuro como comando ya disponible.

Las pruebas focalizadas y la preservación de evidencias se reproducen con:

```bash
python -m pytest \
  tests/unit/test_eap_detection_audit.py \
  tests/unit/test_detection_protocol.py \
  tests/unit/test_detection_environment.py -ra
python paper/scripts/verify_public_data.py
```

**Resultado final: 63 pruebas pasan** (39 G0, 13 protocolo y 11 entorno G1), sin fallos ni skips en estos tres archivos. Ruff pasa y los 12 archivos Python nuevos cumplen formato. Los resultados finales, comandos, códigos de salida y hashes de los scripts están en [final_verification.json](evidence/final_verification.json), con logs adyacentes. También se comprueba el patch. Estas pruebas validan herramientas nuevas y preservación documental; **no representan la suite completa de PHASE ni una ejecución CUDA**. Las pruebas históricas que requieren Torch/checkpoints/datos no se han ejecutado en este entorno.

## 6. Revisión independiente y correcciones

Otro agente revisó código, fuentes, conclusiones y el plan completo; su resultado está en [INDEPENDENT_REVIEW.md](INDEPENDENT_REVIEW.md). Se corrigieron problemas reales de admisión de evidencia, pertenencia token→secuencia y seguridad de rutas. Las regresiones rechazan evidencia inexistente/no vinculada y descargas que modificarían archivos preexistentes o escaparían de la raíz.

La revisión también hizo explícitos el scope global del auditor, `PYTHONPATH` en reproducción, exposición SSL y de calibración TRAIN40, política de score nativo/exportado y prueba geométrica del prior `fx`/`fy`. Ninguna corrección permite aprobar un gate por declaración manual. La integridad de un receipt prueba sus bytes, no su validez física; el cierre de cada afirmación geométrica necesita evidencia y revisión sustantiva.

## 7. Siguiente paso y límites

La ruta crítica tiene dos ramas independientes. La primera necesita un conversor oficial o correspondencia original/HF, calibración/rectificación, semántica de velocidad y medios train accesibles; debe producir fixtures y reproyecciones reales. La segunda necesita acceso a la GPU local para resolver el entorno aislado, compilar y validar el operador y ejecutar el smoke de detector completo. **No se requieren nube de pago, medios test ni backbone compartido para avanzar.**

Cuando ambas ramas pasen, NEXT-06–08 describen loader eAP de ocho clases, overfit funcional de 16 frames y piloto limitado de dos secuencias train más dos de selección. Los presupuestos son propuestas de prueba, no resultados ni estimaciones de rendimiento. Un fallo conserva logs/checkpoint y bloquea la siguiente etapa; no se cambia el umbral después de observarlo para aprobar.

El puente posterior se desarrollará sobre `detector → asociación causal → ROIs/historia → PHASE`. Requiere ingestión única por paquete, tratamiento explícito de buffers prestados/chunks, generaciones de track y abstención TTC cuando falte historia. Eventos, adaptación de PHASE y backbone compartido quedan sujetos a experimentos posteriores. Se entrega preparación reproducible y un plan ejecutable; **no se declara terminada ninguna de las fases que permanece bloqueada**.
