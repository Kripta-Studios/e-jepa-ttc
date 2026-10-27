# 08. Exportación, preservación y reproducción

La entrega conserva un sistema final identificado, sus dependencias de entrenamiento e inferencia, predicciones test12, diagnósticos y copias locales verificadas. El cierre de preservación ocurrió el **27 de octubre de 2026 a las 17:40:51 UTC**. Esta hora es la del recibo local, no el final de todos los entrenamientos ni la duración activa de GPU. La posterior comparación EvTTC se conserva en una carpeta separada, sin reescribir el cierre original.

El recibo principal es [LOCAL_FINAL_PRESERVATION_COMPLETE.json](D:/E_JEPA_POD2_RETURN_20261027/LOCAL_FINAL_PRESERVATION_COMPLETE.json); el mapa de restauración es [RESTORE_INDEX.json](D:/E_JEPA_POD2_RETURN_20261027/RESTORE_INDEX.json). Ambos describen lo efectivamente preservado, no una lista de tareas pendientes.

## 1. Identidad de la cadena final

El candidato se llama `PHASE_ONLY_EVENT_S7_TRAIN40_SCIENTIFIC_OOF`. Es multimodal RGB+eventos. `EVENT` en el nombre de la cabeza no convierte a todo el sistema en event-only. Su dependencia desplegada mínima comprende cuatro productores, dos PAIR, dos refinadores de contexto y una cabeza de fase: **nueve checkpoints**, más dos normalizadores y las configuraciones de entrada.

| Componente | Endpoint registrado | Función en la ruta |
|---|---:|---|
| E_A5 | 49.932 updates | Productor de eventos |
| E_C2F | 49.932 updates | Productor de eventos de la segunda receta |
| R_A5 | 49.932 updates | Productor RGB |
| R_C2F | 49.932 updates | Productor RGB de la segunda receta |
| PAIR_E | 11.264 updates | Etapa PAIR de eventos |
| PAIR_R | 11.264 updates | Etapa PAIR RGB |
| E_CTX_MATCHED | 12.044 updates | Refinador temporal de eventos |
| R_CTX | 12.044 updates | Refinador temporal RGB |
| PHASE_ONLY_EVENT_S7_TRAIN40 | 12.044 updates | Fusión/corrección final de fase |

La suma de estos endpoints es 258.388 updates, pero **no es el coste completo de la campaña**: no incluye ajustes auxiliares OOF, recuperaciones, perfiles ni etapas anteriores. Los cuatro productores finales están ligados a TRAIN40, 88.744 consultas y 40 grupos. Los productores event finales reutilizan padres completos de octubre; no se presentan como entrenados de nuevo al ejecutar las últimas cinco etapas. Fuentes: [auditoría de alcance TRAIN40](../../artifacts/final_diagnostics_20261027/DEPLOYMENT_TRAIN40_SCOPE.json), [extracción de estados y pérdidas guardadas](D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE/diagnostics/TRAINING_LOSS_EXTRACTION.json).

La identidad científica inmutable se conserva en [MODEL_SEAL.json](D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE/MODEL_SEAL.json), SHA-256:

```text
cf5f4eac22e572679c77cd0f6a16425f978600597d4be77f0b1e576c226165a4
```

[PORTABLE_PATHS.json](D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE/PORTABLE_PATHS.json) relaciona las rutas originales con los archivos locales, SHA y miembros de archivo correspondientes. No se reescribió el sello para sustituir rutas Linux por rutas Windows. Esa separación permite restaurar el sistema sin cambiar su identidad.

## 2. Qué está exportado realmente

`PHASE_ONLY_HEAD.ts` es un **TorchScript de la cabeza sobre el estado físico de 340 coordenadas**. No incluye la lectura RAW, reconstrucción de eventos, imágenes, cuatro productores, PAIR e historiales. Por tanto, no es un único fichero ejecutable sensor→TTC ni una exportación ONNX de todo el sistema.

El [recibo de exportación](D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE/PHASE_ONLY_HEAD.json) registra igualdad exacta del `state_dict`, batches de paridad 1 y 7, diferencia absoluta máxima 0,0 y tolerancia 1e−6. Su campo `end_to_end_sensor_runtime_exported` es `false`; `EXPORTED_NOT_EVALUATED` describe el ámbito de ese artefacto. La paridad valida el comportamiento de la cabeza exportada bajo las entradas probadas, no precisión de benchmark ni latencia del sistema completo.

| Artefacto | SHA-256 |
|---|---|
| Checkpoint de la cabeza | `a24bfc72ca7f53a93d53db806b6866c6a5a4b6abde074b5dfec50436b7005408` |
| TorchScript de cabeza | `e28b95cdc9cfcebf4597356d04ec1f2dccca7b9bc17a2f6e6c1d431e1a7e4e37` |

El hash exacto del TorchScript, sin abreviaturas, es:

```text
e28b95cdc9cfcebf4597356d04ec1f2dccca7b9bc17a2f6e6c1d431e1a7e4e37
```

## 3. ZIP de test12: cobertura y trazabilidad

El archivo final es [PHASE_ONLY_S7_TRAIN40_SCIENTIFIC_OOF.zip](D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE/PHASE_ONLY_S7_TRAIN40_SCIENTIFIC_OOF.zip), de **88.968 bytes**, SHA-256:

```text
5eb69749de7aff682eea640d843ee635379702337d929c86f50472264f1d88bc
```

Contiene exactamente `submission.json` en la raíz, esquema `garlttc_prediction_v1`, con 6.762 tokens oficiales y valores finitos. Los tokens corresponden a **5.805 consultas físicas más 957 aliases**. Cada consulta física se computó una vez y su valor se reutilizó para los tokens equivalentes; los aliases no añaden muestras independientes. Hay recibos completos para las doce secuencias, cero fallbacks, y las banderas de lectura de targets privados y envío externo son falsas. [Recibo de predicciones](D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE/PREDICTION_RECEIPT.json).

La [revisión local del paquete](../../artifacts/final_diagnostics_20261027/FINAL_PACKAGE_REVIEW.json) verificó el conjunto exacto de tokens, ausencia de claves JSON duplicadas, hashes de nueve checkpoints y dos normalizadores, e historia causal del mismo objeto. El máximo span observado fue 600.006 μs. La admisibilidad de ese soporte temporal en el benchmark sigue siendo una cuestión distinta de la aceptación del parser.

La revisión documentó 589 diferencias de representación float64 entre la serialización histórica CSV→JSON y las cifras guardadas, con máximo de 7,1054e−15 segundos; **todos los valores coinciden a la precisión float32 original**. El ZIP no se cambió para eliminar esa diferencia de representación. No se han leído labels privados, no se ha enviado el ZIP y no se conserva en esta entrega una puntuación oficial obtenida de CodaBench.

## 4. Qué se recuperó del Pod y cuánto ocupa

El cierre local referencia **48 archivos de conservación**, que suman **22.788.329.378 bytes** en sus rutas actuales. Su inventario contiene **263.923 miembros únicos**. La carpeta D: contiene además seis snapshots anteriores preservados: el inventario físico observado al redactar este capítulo tiene 54 archivos `.tar`/`.tar.gz`, con 22.932.723.618 bytes. Estas cuentas se obtienen del mapa `groups` del cierre y del tamaño local de los archivos; no incluyen todas las copias materializadas o la comparación EvTTC posterior.

| Grupo de contenido | Preservación |
|---|---|
| Nueve pesos finales, normalizadores y configs | Materializados en `FINAL_CANDIDATE`, con SHA y archivo de origen |
| Estados de entrenamiento y recuperación | Archivos de fits, checkpoints, cursores, optimizadores, scheduler y registros asociados |
| Rotaciones y entradas OOF | Features, generaciones, soportes y metadatos de eventos/RGB en archivos específicos |
| Test12 preparado | 5.805 tensores event, features, fragmentos de predicción y recibos de doce secuencias |
| Código y entorno | Código congelado, metadatos del proyecto, herramientas operativas y versiones de dependencias |
| Diagnósticos | Telemetría, tiempos, cachés, fases, pérdidas ya guardadas y manifiestos de fuentes |
| RAW público | Revisiones y hashes fijados; se permite volver a descargar medios públicos sin reentrenar ni regenerar features ya preservadas |

La política RAW no equivale a afirmar que se han retenido todos los HDF5 y TAR públicos en D:. Los medios públicos grandes pueden haberse desalojado tras verificar y preservar sus productos. Su recuperación requiere red y acceso a la revisión pública indicada. Los datos generados que evitan recomputación —tensores, OOF, features, pesos y predicciones— sí forman parte de la conservación declarada. El cierre registra `all_archive_members_previously_sha_verified=true` y `scientific_tests_or_inference_reexecuted_by_preservation=false`.

Para RGB test12, el [plan de descarga](../../artifacts/runpod_event_migration_20261026/final_20261027/TEST12_RGB_DOWNLOAD_PLAN.json) fija la revisión pública `a814a946e60d37f7242358af5f344542177e50c5`, los SHA y tamaños de cada TAR, y el binding de inputs `b78730abe68de7fbe06c6c98b25d5fa025c04d9303720b680ce0c0f575d2e526`. La restauración debe usar esas identidades, no reemplazarlas por la revisión `main` que exista en el futuro. Los recibos de eventos y RGB están en los archivos `test12_event_metadata` y `test12_rgb_metadata` ligados desde el índice de restauración.

## 5. Evidencia suficiente para analizar sin repetir el trabajo científico

El [directorio de diagnósticos final](D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE/diagnostics) contiene:

- `QUERY_DIAGNOSTICS.csv`: 5.805 llamadas físicas con predicciones, fases, intervalos nativos y datos de caché/tamaño.
- `INSTRUMENTED_WALL_DISTRIBUTIONS.csv`: distribuciones por secuencia y estrato de caché; no latencias de kernels.
- `SEQUENCE_FOOTPRINT.csv` y `EVENT_PREPARATION.csv`: tamaños de medios/tensores/features y tiempos registrados de preparación.
- `PASSIVE_TELEMETRY.csv`: **529 muestras** de recursos, con ventanas UTC en `RESOURCE_WINDOWS_5MIN.csv`; no se rellenan huecos inventando actividad.
- `HISTORICAL_TRAINING_LEDGER.csv`: endpoints y snapshots; un intervalo de calendario puede incluir pausas o reapertura de un fit completo.
- `TRAINING_LOSS_BY_UPDATE.csv`: **58.660 pérdidas de entrenamiento** extraídas de cursores existentes, sin forward ni optimizer updates.
- `SOURCE_MANIFEST.json`, `METHODOLOGY.json` y código del informe: trazabilidad de archivos y definiciones reproducibles.

Las 58.660 pérdidas corresponden a PAIR_E/PAIR_R (11.264 cada uno) y E_CTX/R_CTX/PHASE_ONLY (12.044 cada uno). Los checkpoints completos de los cuatro productores no conservan un `cursor.losses` con toda su curva: la extracción lo declara ausente y no interpola una curva ficticia a partir del endpoint. Las pérdidas de objetivos diferentes tampoco se comparan entre etapas como si midieran la misma cantidad. [Recibo de extracción](D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE/diagnostics/TRAINING_LOSS_EXTRACTION.json).

## 6. Entornos reales, no solo el nombre de la plantilla cloud

La captura del Pod del 27 de octubre a las 16:33:57 UTC registra Python 3.11.15, Torch 2.11.0+cu128, CUDA compilada 12.8, cuDNN runtime 91900 y driver NVIDIA 595.91.07, con RTX 4090 de 24.564 MiB visibles. El contenedor era Ubuntu 24.04.5 LTS. La plantilla inicial elegida en el panel no basta para reconstruir estas versiones finales. [Inventario completo](../../artifacts/final_diagnostics_20261027/ENVIRONMENT_FULL_20261027.json).

Ese inventario contiene 87 distribuciones con nombre y versión, sin URLs de instalación con credenciales. También conserva afinidad, cgroups y código observado en disco frente a procesos activos. Se veían 32 CPU lógicas y 16 físicas, pero la cuota era `1360000/100000` = 13,6 CPU equivalentes; el límite cgroup de memoria era 60.999.999.488 bytes. La memoria de page cache no debe confundirse con memoria residente de modelos ni con la RAM anunciada originalmente por el proveedor.

La transferencia EvTTC posterior se ejecutó en Windows, con una RTX 5070 Ti Laptop y sus propios bindings. Sus tiempos no se mezclan con la RTX 4090. Para reproducir un resultado se deben tomar las versiones y modos numéricos del binding correspondiente, no asumir equivalencia entre los dos hosts.

## 7. Reproducción por niveles, sin trabajo redundante

**Nivel 1: inspección y comprobación de identidad.** No requiere GPU. Se conservan los archivos del cierre y sus SHA; `RESTORE_INDEX.json` permite localizar cualquier dependencia original por archivo TAR, miembro y hash. Ejemplo de comprobación del ZIP final en PowerShell:

```powershell
Get-FileHash D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE/PHASE_ONLY_S7_TRAIN40_SCIENTIFIC_OOF.zip -Algorithm SHA256
```

El resultado esperado es el hash exacto de la sección 3. Restaurar un miembro exige comprobar el SHA del archivo de conservación y del miembro. La ruta histórica incluida en un recibo describe su procedencia; `PORTABLE_PATHS.json` suministra la ubicación actual de los componentes desplegados.

**Nivel 2: regenerar las tablas de transferencia desde predicciones selladas.** Este comando lee los TTC públicos únicamente después de validar ambos sellos. No vuelve a inferir ni modifica pesos:

```powershell
../e-jepa-ttc/.venv/Scripts/python.exe scripts/final_train40_transfer_report.py --candidate D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE --output docs/final_train40_transfer_20261027 --copy-to D:/E_JEPA_POD2_RETURN_20261027/FINAL_TRAIN40_TRANSFER_20261027
```

Requiere los manifiestos, freezes y fragmentos de `artifacts/final_train40_transfer_20261027`, además de los `ttc.csv` públicos en las rutas del manifiesto. El código usa NumPy/Pandas y Matplotlib para la figura; no importa los modelos ni crea un contexto CUDA. La entrega conserva 11 tests nuevos de los gates y agregaciones, Ruff satisfactorio y Pyright sin errores. [Tests registrados](../../artifacts/final_train40_transfer_20261027/REPORT_TESTS_FINAL.log), [tipado](../../artifacts/final_train40_transfer_20261027/REPORT_PYRIGHT_FINAL.log).

**Nivel 3: regenerar diagnósticos de test12 desde archivos locales.** Se recomienda una carpeta de salida nueva para no alterar el cierre preservado:

```powershell
../e-jepa-ttc/.venv/Scripts/python.exe scripts/runpod_final_diagnostics_report.py --archive-root D:/E_JEPA_POD2_RETURN_20261027 --candidate-root D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE --mode final --output D:/E_JEPA_POD2_RETURN_20261027/DIAGNOSTICS_REGENERATED
```

El comando consume metadatos y diagnósticos ya guardados. Para extraer las pérdidas carga los checkpoints en CPU; no hace forward ni actualiza optimizadores. Su `SOURCE_MANIFEST.json` identifica los archivos elegidos. Los snapshots antiguos que siguen en el directorio se distinguen por sus recibos y contenidos; no deben sumarse como inferencias nuevas.

**Nivel 4: inferencia o entrenamiento de nuevo.** No es necesario para revisar las cifras entregadas. Si se decide hacerlo, hay que restaurar los inputs fijados, código, configs, normalizadores y checkpoints correspondientes; validar sus bindings y declarar hardware, precisión y estado de cachés. Reanudar entrenamiento exige además el estado completo de optimizador, scheduler, sampler y RNG: cargar únicamente pesos no equivale a reanudar el mismo experimento. Las rotaciones OOF tampoco son nueve redes desplegadas en ensemble; son dependencias de generación de datos y genealogía de entrenamiento.

## 8. Estado de publicación y límites conservados

La evidencia soporta identidad técnica, trazabilidad, reproducción de tablas y preparación de submission. El cierre local es independiente de que el Pod siga existiendo. Su conservación no se basa en la sesión de SSH ni en que permanezca abierta una terminal.

El [capítulo de evaluación](07_EVALUATION_METRICS_AND_LATENCY.md) distingue resultados internos, transferencia EvTTC y latencias. El ZIP test12 está empaquetado, no enviado; la cabeza está exportada, no toda la aplicación sensor→TTC. Un envío oficial, la comprobación de sus reglas y el recibo de evaluación son pasos externos distintos. No se presenta documentación, paridad numérica o cobertura de tokens como precisión oficial medida.
