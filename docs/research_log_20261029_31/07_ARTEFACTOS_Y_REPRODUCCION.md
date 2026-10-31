# 07. Qué se creó, dónde está y cómo reproducirlo

## Mapa de implementación

| Capa | Directorio o scripts | Responsabilidad |
|---|---|---|
| Auditoría de transferencia | `operational/evttc_transfer_audit/`, `scripts/audit_evttc_transfer.py` | Identidad, genealogía, protocolo, targets y diagnóstico |
| Adaptación F1 | `operational/phase_adapt_*`, scripts de refit/adaptación | Features, caché, entrenamiento de cabeza, runtime y entrega |
| Campaña agrupada | `operational/phase_campaign/` | Datos, métricas, perturbaciones, controles y resultados por grupos |
| Gate robusto | `operational/phase_robust/` | Bundle portable, geometría/calidad, caché y runtime |
| Router de arranque | `operational/phase_startup/` | Geometría desde dos cajas, experto inverso, frescura y ensemble |
| Nuevo envío | `operational/phase_submission/` | Exclusión final, política del scorer, loader e integridad cruzada |
| Auditoría histórica final | `scripts/audit_phase_submission.py` | ZIP original, hardware, cámaras y recálculo de negativos |
| Fit final | `scripts/train_phase_submission.py` | Tres semillas sobre 2.608 filas y exportado |
| Replay final | `scripts/run_phase_submission.py` | Las 434 consultas desde sensores, tiempos y sellado |
| Evaluación final | `scripts/evaluate_phase_submission.py` | Verificación previa al join, cobertura y scores locales |
| Validación final | `scripts/validate_phase_submission.py` | Fallback fuente, causalidad y perturbaciones cacheadas |
| Auditoría ampliada | `scripts/audit_phase_sources.py` | Código retenido frente al commit anterior |
| Informe y entrega | `scripts/report_phase_submission.py`, `scripts/package_phase_submission.py` | Tablas desde recibos y ZIP con manifiesto verificable |

La tabla no pretende listar cada helper del repositorio. El registro Git conserva
los archivos añadidos/modificados por cada commit. El propósito de separar módulos
es que un cambio en el scorer no se oculte dentro de inferencia, o que la selección
de una variante no quede implícita en el nombre de una carpeta.

## Variantes entregadas

Los cinco paquetes de modelos/campañas son fotografías diferentes:

1. `PHASE_TTC_ADAPT_F1_S7_2026-10-30.zip`: entrega compacta de F1; el nombre conserva
   la semilla de entrega predefinida. Sus resultados de fit no son la nueva submission.
2. `PHASE_TTC_CAMPAIGN_FINAL_2026-10-30.zip`: evidencia agrupada, externa y cruda
   posterior a F1, incluidos resultados negativos y comparadores.
3. `PHASE_TTC_ROBUST_FINAL_2026-10-31.zip`: gates de calidad evaluated/all-data,
   dependencias upstream y evidencia de esa iteración.
4. `PHASE_TTC_STARTUP_FINAL_2026-10-31.zip`: ensayos en segundos e inversa, los
   117 ajustes, variantes y auditoría de arranque; no reemplazo universal del gate previo.
5. `PHASE_TTC_ADAPT_FINAL_2026-10-31.zip`: un candidato final identificado para
   nuevo envío, 2.608 filas de fit y replay completo de competición.

El ZIP pequeño `PHASE_TTC_Adapt_EvTTC_results_2026-10-31.zip` contiene únicamente
los diez archivos de resultados para adjuntar al benchmark. El paquete grande
contiene además código, pesos, informes, inputs numéricos pequeños y verificaciones.
La [tabla de assets](09_PUBLICACION_E_INVENTARIO.md) indica tamaño y hash de cada uno.

## Interior del paquete final

```text
START_HERE.md
MANIFEST.json
repository/
  pyproject.toml, uv.lock, README.md, LICENSE
  src/, operational/, scripts/, configs/
  docs/evttc_submission_20261031/
phase_submission/
  FINAL_MODEL/              # upstream + router + MODEL.json
  training/                # población elegible, seeds, freezes y export
  replay/                  # ledger, resultados, runtime y COMPLETE.json
  audit/                   # historia, código, instalación y reproducción
  evaluation/              # LOCAL_EVALUATION.json
  validation/              # causalidad, fallback y robustez
  FINAL_CHECKS.json
inputs/                    # features y targets pequeños necesarios
historical_run/            # ledgers/recibos del original
verification_logs/
```

`router.ts` exporta la nueva fusión, no el pipeline completo de sensores. Para
inferir desde eventos/RGB se necesitan también los componentes upstream, sus
normalizadores y el adaptador. Los datos crudos y la gran caché eAP fuente se
obtienen aparte y no se redistribuyen con estos ZIP.

La población previa incluida como procedencia contiene filas Slider ya expuestas;
el script las retira antes del fit final. La población elegible guardada dentro de
`training` solo tiene las 2.608 filas utilizadas. No se confunde «el archivo existe
en el paquete» con «sus etiquetas entraron en el entrenamiento».

## Instalación y verificación de features

Desde `repository/` del ZIP extraído, con Python 3.11:

```powershell
uv sync --group dev --no-editable
$env:PYTHONPATH='src;.'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
uv run --no-sync python -m scripts.evaluate_phase_submission --replay ../phase_submission/replay --bundle ../phase_submission/FINAL_MODEL --historical-run ../historical_run --road-targets ../inputs/road_targets.json --slider-contract ../inputs/slider --output ../evaluation_reproduced
```

En Linux, `PYTHONPATH=src:.`. La instalación necesita una versión de PyTorch
compatible con el CUDA elegido si se ejecuta la ruta GPU; el lock y los recibos
indican las versiones de la entrega. La comprobación sobre features guardadas
puede hacerse sin el dataset sensorial.

Para repetir los tres ajustes pequeños:

```text
uv run --no-sync python -m scripts.train_phase_submission --population ../inputs/POPULATION.npz --exclusions ../inputs/exclusions.json --previous-bundle ../phase_submission/FINAL_MODEL --output ../retrained
```

Los hashes de configuración cambiarán al cambiar rutas; eso no significa que
cambien pesos. La reproducción local en otra instalación comprobó igualdad de
los checkpoints. Una salida existente con recibos incompatibles no debe
sobrescribirse: se utiliza una carpeta nueva para otro experimento.

## Replay real desde sensores

La raíz sensorial debe contener las diez carpetas originales, cada una con sus
cajas, HDF5 o bag y calibración. La ejecución valida identidad y registra hashes:

```text
uv run --no-sync python -m scripts.run_phase_submission --bundle ../phase_submission/FINAL_MODEL --data-root /ruta/EvTTC-Benchmark --output ../replay_new --device cuda
```

`--resume` continúa el mismo contrato; `--limit` permite un tramo parcial sin
crear un ZIP completo hasta alcanzar 434. La garantía de cambios de código se
limita a la lista del freeze original; el suplemento posterior amplía la auditoría
de la entrega, como se explica en el capítulo de verificación. Reproducir desde
otro checkout requiere conservar los bytes del código enlazado o crear una
ejecución nueva y declararla como tal.

## Contrato de submission y material preparado para EvTTC

Cada directorio contiene un `results.txt` con tabuladores y cabecera exacta:

```text
Index	timestamp (s)	ttc (s)	cost time(s)
```

Index y timestamp proceden de la consulta suministrada; TTC y coste se expresan
en segundos. El coste pertenece a esa estimación individual. No se rellena una
columna de tiempos usando una media para toda la secuencia. El verificador
compara los valores con el ledger, no solo el número de filas.

Se prepararon dos borradores en
[docs/evttc_submission_20261031](../evttc_submission_20261031/README.md). Uno
completa la ficha del modelo del 28; el otro describe la versión adaptada y pide
evaluarla como versión separada. Ambos indican único autor y ninguna afiliación.
La consulta sobre targets abarca CCRs1-low, CCRs1-medium y Slider-750. La publicación
en GitHub no envía esos correos ni implica aceptación del benchmark.
