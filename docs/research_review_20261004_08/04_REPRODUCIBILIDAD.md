# Reproducibilidad, contabilidad y conservación

## Qué se guarda y qué publica GitHub

Las raíces originales permanecen separadas: `artifacts/efficient_context_20261004`, `artifacts/train40_system_20261005`, `artifacts/evttc_transfer_20261008` y `artifacts/evttc_rgb_transfer_20261008`. Los informes de esta carpeta son una entrega nueva; no reescriben recibos históricos ni cambian hashes de pesos. La carpeta handoff que ya estaba sin seguimiento en el workspace se conserva fuera de este commit.

Este push publica la rama `scientific-recovery-v12-efficient-context`, con sus commits previos y estos informes. No hace merge a main ni crea una submission. El repositorio recibe evidencia compacta: resultados JSON, predicciones CSV, contratos, QA, contabilidad y generador. Los checkpoints grandes y datasets se conservan localmente y se identifican por hash; **no se promete reproducir inferencia desde un checkout sin conseguir también esas dependencias**.

## Identidad de los pesos

Los [bindings del sistema](evidence/event_only/DEPENDENCIES.json) fijan A5, C2F, PAIR, las tres cabezas y normalización. Las cargas son estrictas. Para evitar duplicar manualmente una larga lista de hashes, el JSON es el inventario normativo de esos pesos.

El checkpoint Garl event-only tiene SHA-256 `fcaf9be47e2dafc6f73c6c3ebd102595ae06119dcae78aea698a42627b2b4fef`. El full RGB+eventos tiene `e96a613a4fb877a1969d57ab562cadba89961fb202f5f2f2f0658f333a0d443e`; su configuración, `c7dedc9b93d32a416b8e92a6b474cc5a17ffd96b2cdaae3a4554e3cc0ee28454`. Repo y revisión están en [DEPENDENCIES full](evidence/rgb_event/DEPENDENCIES.json). El hash prueba identidad del archivo, no certifica por sí solo ausencia de contaminación en el entrenamiento del proveedor.

El manifiesto EvTTC compartido tiene SHA-256 `d9fe973c37f9e97ac7cb8a46de01bfe461ccc2672328bfa3e51b853763589bd5`. La integridad de 32 HDF5 se comprobó mediante hashes locales y estabilidad de tamaño/mtime: 197.781.014.225 bytes. Eso identifica la copia local, no constituye autenticación criptográfica por el publicador. Se conserva el preflight rechazado y las razones de corrección sin usarlo como población puntuada.

## Bundles conservados

- TRAIN40, `essential_bundle.zip`: SHA-256 `6cb350449c7f27076159086ac021033fa5f9f031c134a6e373c308449c405292`; 413.185.835 bytes, 4.300 miembros verificados.
- EvTTC event-only, `EVTTC_ESSENTIAL.zip`: `68afaa9a8e5b30603d93c524e15be73c9eea6528b480412e9ecd2c613e6721b9`; 3.474.997 bytes, 2.163 miembros verificados.
- EvTTC RGB+eventos, `FULL_RGB_ESSENTIAL.zip`: `5f38258e49d144ddc586f23a67d6afbfcbf8a4557b9ee8184caeb4eb2d6c7208`; 6.662.038 bytes, 2.094 miembros verificados.
- Addendum de análisis del 8 de octubre v2: `f5f6aa620ff0574919dd3d8d14a5f7493ddd0fa7bc48d45b59f96c554695e3fa`; 49 miembros verificados.

Los contadores son entradas de payload verificadas; cada ZIP tiene además su manifiesto interno, por lo que su número total de entradas es una unidad mayor.

Los dos bundles EvTTC tienen copias verificadas en `E:/EJEPA_results/comparisons_20261008/`. Full contiene preservación del bundle event-only. Sus pesos grandes permanecen fuera del ZIP esencial, con dependencia por hash. Los recibos de [preservación](evidence/rgb_event/BASELINE_PRESERVATION.json), [verificación de bundle](evidence/rgb_event/BUNDLE_VERIFICATION.json) y [backup](evidence/rgb_event/BACKUP_VERIFICATION.json) acompañan al informe. Un backup en otro volumen conectado no sustituye una política de copia externa permanente.

## Contabilidad de ejecución

La [tabla regenerada](tables/ACCOUNTING.md) descompone la cota física de **155.078 sobre un techo de 240.000**. Contiene 40.748 de campaña original, 114.204 científicos TRAIN40, 91 de recuperación TRAIN40 y 35 de técnica sintética. Los 40.748 ya incluyen 22.500 WIDE, 17.948 productores Garl y una cota de recuperación de 300. No se suman otra vez las 22.600 operaciones de un recibo parcial WIDE ni los 40.783 de un snapshot QA antiguo.

La campaña original distingue recuperación confirmada inferior de 50 y cota superior de 300. Por ello 155.078 es una cota conservadora y no un contador exacto de todas las actualizaciones físicas. Las 1.728 mediciones históricas, R1 y ambas inferencias EvTTC son mediciones, no entrenamiento: añaden cero updates de optimizador. El saldo hasta el techo no autoriza por sí solo nuevos experimentos.

## Pruebas y ambiente

La evaluación event-only registra 21 tests, Ruff correcto, Pyright sin errores y paridad exacta de las tres cabezas en una consulta TRAIN40 guardada. La extensión RGB+eventos registra 16 tests, Ruff correcto, Pyright sin errores y la paridad real CPU/GPU con tolerancia. La preparación R1 conserva 28 tests en su QA y 256 casos de paridad CPU exacta. Son resultados de los gates de esas rutas, no una afirmación de haber reejecutado ahora todos los tests del repositorio.

Véanse [QA event-only](evidence/event_only/QA.json), [QA full](evidence/rgb_event/QA.json) y [QA R1](evidence/r1/QA.json). Las revisiones cubrieron causalidad temporal, selección RGB, continuidad de modelos heredados, soporte pareado independiente, recuperación y hashes. La generación de estos informes vuelve a verificar métricas y contabilidad; no vuelve a entrenar modelos ni a leer cientos de GB de medios.

La evaluación full utilizó RTX 5070 Ti Laptop, aproximadamente 12 GB VRAM, driver 591.86, Python 3.11.15, PyTorch 2.11 con CUDA 12.8, NumPy 2.4.6 y h5py 3.16.0; FP32, TF32 desactivado, cuatro threads y dos interop. El [recibo de entorno](evidence/rgb_event/RUN_ENVIRONMENT.json) fija los detalles. No se actualizó globalmente el entorno para producir esta entrega.

## Cómo verificar este informe

`SOURCE_INVENTORY.json` enumera cada original local, su hash, su snapshot publicado y su hash publicado. La transformación JSON elimina prefijos personales de rutas y reformatea; por eso los dos hashes pueden diferir sin que cambien los datos científicos. Los CSV se copian byte a byte. El generador valida los hashes de snapshots antes de calcular resultados.

```powershell
python docs/research_review_20261004_08/regenerate.py
git diff -- docs/research_review_20261004_08/tables
```

No debe aparecer diferencia en las tablas con el entorno numérico documentado. Se verifican 1.024 IDs únicos y herencia exacta, 946 etiquetas válidas, 32 secuencias, cinco juegos de métricas y cuatro intervalos bootstrap. Se comprueba el total de updates a partir de sus componentes. La verificación usa las predicciones publicadas: acredita la tabla a partir de ellas, no repite inferencia de los pesos.

`SHA256SUMS.txt` cubre todos los archivos de esta entrega excepto a sí mismo. Su comprobación en PowerShell puede hacerse con `Get-FileHash -Algorithm SHA256`; las rutas del manifiesto son relativas a esta carpeta. El commit Git identifica el conjunto completo. Los enlaces a `artifacts/...` dentro de evidencias son referencias de procedencia local, no promesas de disponibilidad en GitHub.

## Recuperación de R1

La [instantánea de supervisor](evidence/r1/CHAIN_STATE.json) y el [progreso de medición](evidence/r1/measurement/PROGRESS.json) son observaciones fechadas, no telemetría en vivo. El supervisor reintenta, respeta el marcador de pausa y no debe duplicarse. Sólo si se ha comprobado que no existe supervisor activo, se puede reanudar desde la raíz del repositorio con el entorno local ya existente:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
& ../e-jepa-ttc/.venv/Scripts/python.exe -m operational.efficient_context.r1_gib_supervisor
```

Este comando usa los protocolos y fragmentos locales conservados. Un clon nuevo necesita restaurar primero esas dependencias; no es una orden autónoma de descargar y recrear el experimento. La ingesta/replay inicial tras reanudar puede tardar antes de producir nuevos pares. No se borran fragmentos para acelerar una aparente recuperación.
