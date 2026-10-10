# 07 — Reproducción desde GitHub y mapa de evidencia

## Checkout y versiones

El repositorio público es [Kripta-Studios/e-jepa-ttc](https://github.com/Kripta-Studios/e-jepa-ttc). Este paquete está en la rama `scientific-recovery-v12-efficient-context`; no asumir que aparece en la rama por defecto.

```bash
git clone --branch scientific-recovery-v12-efficient-context https://github.com/Kripta-Studios/e-jepa-ttc.git
cd e-jepa-ttc
python docs/agent_handoff_20261010/verify_bundle.py
python docs/agent_handoff_20261010/replay_public.py --output /tmp/ejepa-fcwd-review
```

Solo se necesita Python 3.11+ para estos dos comandos; no instalar PyTorch ni reservar GPU para recalcular CSV. En Windows sustituir `/tmp/ejepa-fcwd-review` por un directorio de resultados, por ejemplo `E:/EJEPA_results/public_handoff_replay`. Los scripts resuelven la raíz del checkout desde su propia ubicación.

El verificador comprueba hashes y referencias locales de Markdown. Los archivos históricos que Git normaliza entre CRLF y LF llevan una entrada explícita `normalization: lf` en el manifiesto: se comprueba el contenido con esos finales normalizados. Las evidencias congeladas y copias de resultados mantienen hash de bytes exactos. El replay exige la identidad exacta del scored CSV, comprueba query_id/variante y GT emparejado, recalcula MiD, compara cada fila y los agregados con sus fuentes y reconstruye la diferencia por bandas. Genera CSV y un anexo Markdown. Un resultado de verificación correcto acredita estas operaciones; no acredita haber vuelto a ejecutar los modelos.

Para inspeccionar V13 sin alterar este checkout:

```bash
git fetch origin scientific-recovery-v13-rgb-port
git show da068ebdeb2d967ee10b2afa348e1317b8720ad0:operational/rgb_port_streaming/runtime.py
git worktree add --detach ../ejepa-v13-review da068ebdeb2d967ee10b2afa348e1317b8720ad0
```

## Archivos nuevos en este paquete

| Ruta | Contenido y uso |
|---|---|
| `evidence/comparison/DEV32_PREDICTIONS.csv` | 3.947 consultas; GT, H8 por semilla/mediana, Direct, Garl eventos/full, paridad y diagnósticos |
| `evidence/comparison/FCWD_PREDICTIONS.csv` | 630 consultas; mismos campos aplicables, con razones de indisponibilidad preservadas |
| `evidence/train40/METRICS.csv` | overall y bandas de los cinco métodos TRAIN40 |
| `evidence/train40/RESULT.json` | Estado, commit, hashes de entradas/fragmentos/scorer y limitaciones |
| `evidence/train40/calculate.py` | Script original de cálculo; requiere activos locales no incluidos |
| `evidence/recomputed/FCWD_SUMMARY.csv` | Métricas nuevas por secuencia, variante y banda desde scored público |
| `evidence/recomputed/FCWD_GAP_CONTRIBUTIONS.csv` | Descomposición del gap mean_MiD frente a Garl |
| `evidence/recomputed/FCWD_PAIRED_QUERIES.csv` | Las 597 consultas comparadas y ordenadas por desventaja H8 |
| `evidence/recomputed/REPLAY_RESULT.json` | Resultado de comprobación, hashes y límites |
| `evidence/recomputed/TABLES.md` | Anexo generado desde CSV, incluido el resto de tablas resumidas |
| `evidence/V13_STATUS_SNAPSHOT.json` | Lectura temporal de journals y recibos de checkpoints V13, no estado vivo |
| `MANIFEST.json` | Hashes del paquete y de entradas públicas utilizadas |

Los dos CSV nuevos de comparación se copian byte a byte de los resultados congelados. Hash DEV32: `815e8925d350aabf77e7020f453ca3fae30a19d9467f156378f2e331aaec43c8`; hash FCWD: `3afe9dcba1ad59bc2535d438026ce0c9b97c2a7df882610776a824ca2ea32082`. No se vuelven a generar predicciones para esta publicación. Incluyen valores GT de las consultas de evaluación y salidas derivadas; no incluyen medios crudos ni pesos. Leer celdas vacías como ausencia, no como cero.

## Mapa de evidencia ya pública

| Pregunta | Archivo/directorio público |
|---|---|
| Qué se midió el 8 de octubre | [Informe original](../sota_campaign_20261008/README.md), [métricas](../sota_campaign_20261008/tables/metrics.csv), [costes](../sota_campaign_20261008/tables/system_cost.csv) |
| Qué cambió en la cabeza y preparación | [Protocolo revisión](../ttc_revision_20261009/PROTOCOL.md), [informe](../ttc_revision_20261009/README.md) |
| Sesgo de entrenamiento, upstream y distribución de features | [Tablas de revisión](../ttc_revision_20261009/tables): `train_fit_metrics.csv`, `upstream_metrics.csv`, `feature_shift.csv`, `error_concentration.csv` |
| Comparación exacta con Garl y sensibilidad | [Evidencia Garl](../sota_evidence_20261010/evidence/garl): `METRICS.csv`, `MID_OFFICIAL_SCORER.csv`, `MID_RESULT.json`, bootstrap y errores mayores |
| Reproducción FCWD raw cronológica | [FCWD CPU](../sota_evidence_20261010/evidence/streaming/fcwd_stream_cpu): `RESULT.json`, `ROWS.jsonl`, `PREDICTIONS.csv`, scored, bands y reuse |
| H4/H2 y destilación sin warp | [External](../sota_evidence_20261010/evidence/streaming/external): `METRICS.csv`, `RESULT.json` |
| Latencia y fallos de pilotos | [Streaming](../sota_evidence_20261010/evidence/streaming): cada piloto contiene su recibo y filas; los fallidos no tienen el mismo nivel de completitud |
| Preparación y predicción test12 | [Test12](../sota_evidence_20261010/evidence/test12): freezes, resultados, estado de empaquetado |
| Reanudación V13 | [Evidencia inmutable V13](https://github.com/Kripta-Studios/e-jepa-ttc/tree/da068ebdeb2d967ee10b2afa348e1317b8720ad0/docs/rgb_port_streaming_20261010/evidence) |

## Mapa de código para investigar FCWD

| Módulo | Símbolos/punto de entrada | Qué revisar |
|---|---|---|
| [ttc_revision/inputs.py](../../operational/ttc_revision/inputs.py) | `EventPreparer`, `PersistentReader` | Índices temporales, buffers, crop y ventanas nativas |
| [ttc_revision/runtime.py](../../operational/ttc_revision/runtime.py) | `device_features`, `H8Runtime` | Productores y construcción de features |
| [ttc_revision/head.py](../../operational/ttc_revision/head.py) | `DirectTTCHead`, `symmetric_ttc_loss` | Parametrización y reducción de pérdida |
| [ttc_revision/train.py](../../operational/ttc_revision/train.py) | `run` | Receta/endpoint de Direct, no confundir con entrenamiento completo |
| [ttc_revision/diagnose.py](../../operational/ttc_revision/diagnose.py) | `diagnose` | Upstream, distribución y concentración de errores |
| [ttc_revision/score.py](../../operational/ttc_revision/score.py) | `metrics`, `paired_interval`, `score` | Elegibilidad, avisos y bootstrap |
| [garl_comparison/mid.py](../../operational/garl_comparison/mid.py) | `load_scorer`, `scores`, `paired_mid` | Scorer fijado y MiD inválido explícito |
| [streaming_revision/state.py](../../operational/streaming_revision/state.py) | `Policy`, `Observation`, `FeatureState` | Identidad, causalidad y tolerancias |
| [streaming_revision/preparation.py](../../operational/streaming_revision/preparation.py) | `warp_voxel`, `Query`, `IncrementalPreparer` | Pérdida de soporte por ROI y reutilización de voxels |
| [streaming_revision/packets.py](../../operational/streaming_revision/packets.py) | `PacketRing`, `packet_bounds` | Representación compacta y selección exacta |
| [streaming_revision/runtime.py](../../operational/streaming_revision/runtime.py) | `StreamRuntime` | Cálculo incremental, productores y cabezas |
| [streaming_revision/kernels.py](../../operational/streaming_revision/kernels.py) | `correlation`, `GraphModule`, `EncoderPrecision` | Vectorización, graphs, memoria y precisión |
| [streaming_revision/distill.py](../../operational/streaming_revision/distill.py) | `FeatureStudent`, `train`, `load_student` | Objetivo de imitación y padres del estudiante |
| [streaming_revision/external_stream.py](../../operational/streaming_revision/external_stream.py) | `run` | Replay cronológico completo y unión posterior de etiquetas |
| [streaming_revision/diagnostics.py](../../operational/streaming_revision/diagnostics.py) | `run` | Scored CSV, bandas y reutilización |

## Qué puede reproducir el agente solo con GitHub

**Sí:** leer implementación, reconstruir métricas de FCWD, estudiar consultas H8/Direct/Garl en Dev32/FCWD, comprobar sesgo y contribuciones, auditar configuraciones/recibos, comparar tiempos guardados y localizar el cambio de código.

**No sin activos adicionales:** inferencia nueva desde eventos, entrenamiento, actualización de pesos, perfiles GPU nuevos, recalcular las 88.744 filas TRAIN40 desde todos sus fragmentos, reconstruir un ZIP test12 solo desde los recibos, obtener etiquetas privadas o score CodaBench. Que un JSON contenga la ruta a un archivo no significa que dicho archivo esté incluido en Git.

El script original TRAIN40 se ejecutó contra `artifacts/train40_system_20261005`: necesita `TRAIN40_INDEX.npz`, `DATA_AUDIT.json`, todos los fragmentos `train_predictions/batch_*.npz` y sus recibos, `BINDING.json`, la predicción Garl consolidada, su manifest y el scorer fijado. Estos archivos grandes/externos no se publican en este handoff. El resultado agregado y la lista de hashes sí. La comprobación pública del overall a partir de sus cuatro medias es posible; la reproducción desde todas las predicciones TRAIN40 requiere esos activos.

Los comandos antiguos de informe pueden esperar `artifacts/...` no presente en un clon limpio. Para la revisión pública usar primero los scripts de este paquete. Si se prepara un entorno de investigación completo, instalar según `pyproject.toml` y la documentación raíz, fijar datos/checkpoints por sus hashes y usar `--help` del runner antes de ejecutarlo; no crear archivos vacíos para sortear validadores.

## Rutas originales para quien posteriormente obtenga los activos

Base histórica del host: `C:/Users/Álvaro Schwiedop/Desktop/KriptaStudios/EVOCON_JEPA_Codex_Handoff/`. Worktrees `e-jepa-ttc-v12-efficient-context` y `e-jepa-ttc-v13-rgb-port`. Son referencias de procedencia, no requisitos de instalación.

- Datos públicos descargados: `E:/eAP_dataset`, `E:/GarlTTC_dataset`.
- Recálculo TRAIN40: `E:/EJEPA_results/v12_train40_mid_20261010`.
- Port/reanudación V13: `E:/EJEPA_results/v13_streaming_port_20261010`.
- Campaña V13: `artifacts/rgb_port_20261008` dentro de su worktree.
- Comparación previa: `artifacts/ttc_revision_20261009` y `artifacts/garl_checkpoint_comparison_20261010` en V12.

Los paths absolutos conservados en recibos acreditan de dónde proceden; el mapa de arriba identifica la copia pública cuando existe. No incluir credenciales, medios crudos o checkpoints en un commit para «arreglar» una ruta ausente.
