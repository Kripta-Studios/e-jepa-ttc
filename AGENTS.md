# AGENTS.md — E-JEPA-TTC

## 0. Propósito de este documento

Este archivo es la especificación operativa completa para que un agente de desarrollo como Codex construya, pruebe, documente y entregue de principio a fin el proyecto **E-JEPA-TTC**.

El agente debe tratar este documento como contrato de ingeniería y de integridad científica. No debe limitarse a producir una demo visual. El resultado debe ser un repositorio reproducible, auditable y defendible ante una comisión de selección o en una entrevista técnica.

El proyecto persigue un MVP de investigación sobre:

> Predicción multihorizonte en espacio latente mediante Joint-Embedding Predictive Architecture para estimar Time-to-Contact/Time-to-Collision a partir de streams de cámaras de eventos, con evaluación de robustez, eficiencia e incertidumbre.

Nombre de trabajo del repositorio:

```text
E-JEPA-TTC
```

Nombre largo sugerido:

```text
E-JEPA-TTC: Multi-Horizon Joint-Embedding Prediction for Uncertainty-Aware Time-to-Contact Estimation from Event Streams
```

---

## 1. Objetivo de producto y de investigación

El sistema debe recibir eventos asíncronos con forma:

```text
(x, y, timestamp_us, polarity)
```

y producir:

1. una estimación continua de TTC en segundos;
2. probabilidades de contacto dentro de horizontes discretos;
3. una medida de incertidumbre o confianza;
4. una inferencia reproducible y medible en términos de latencia;
5. representaciones latentes útiles, evaluadas mediante probes y ablations;
6. una demo de reproducción sobre secuencias reales;
7. un informe técnico con metodología, resultados y limitaciones.

La pregunta científica principal es:

> ¿El preentrenamiento JEPA multihorizonte sobre eventos no etiquetados mejora la precisión, robustez, eficiencia en régimen de pocas etiquetas y generalización respecto a baselines supervisados o representaciones manuales?

Preguntas secundarias:

- ¿Qué representación de eventos funciona mejor: event count, time surface, voxel grid o tokens sparse?
- ¿Predicción de embedding futuro de una ventana completa o predicción de regiones enmascaradas?
- ¿Es preferible predecir TTC, log(TTC) o inverse TTC?
- ¿La pérdida geométrica basada en expansión aparente mejora generalización?
- ¿La incertidumbre está calibrada y aumenta bajo perturbaciones fuera de distribución?
- ¿Qué horizonte de contexto y predicción ofrece mejor equilibrio entre precisión y latencia?
- ¿Qué parte de la mejora procede del encoder, del objetivo JEPA o del predictor multihorizonte?

---

## 2. Definición de terminado

El proyecto se considera terminado únicamente cuando se cumplan todos estos puntos:

- [ ] Repositorio instalable con un único comando documentado.
- [ ] Cargador real para EvTTC HDF5.
- [ ] Adaptador genérico para nuevos datasets de eventos.
- [ ] Validación de timestamps, polaridades, resolución y calibración.
- [ ] Partición por secuencia, sin fuga temporal o de escenario.
- [ ] Al menos tres representaciones de eventos.
- [ ] Baseline supervisado reproducible.
- [ ] Baseline geométrico o analítico reproducible.
- [ ] Modelo E-JEPA funcional con target encoder y predictor.
- [ ] Preentrenamiento autosupervisado sin usar etiquetas TTC.
- [ ] Fine-tuning supervisado con cabeza TTC.
- [ ] Ablations esenciales ejecutadas.
- [ ] Tres semillas para los resultados principales.
- [ ] Robustness suite automatizada.
- [ ] Evaluación de latencia y memoria.
- [ ] Exportación ONNX o TorchScript.
- [ ] Demo offline de streaming.
- [ ] Tests unitarios e integración.
- [ ] GitHub Actions o CI equivalente.
- [ ] Resultados guardados en CSV/JSON, no solo capturas.
- [ ] Informe técnico regenerable.
- [ ] README profesional.
- [ ] Model card y dataset card local.
- [ ] Limitaciones y riesgos explícitos.
- [ ] No existen métricas inventadas ni resultados escritos a mano.

La entrega mínima debe poder ejecutarse así:

```bash
make setup
make smoke-data
make train-baseline
make pretrain-jepa
make finetune-ttc
make evaluate
make demo
make report
```

Si el host no soporta `make`, debe existir una alternativa equivalente mediante `uv run`, `python -m` o scripts de shell.

---

## 3. Principios no negociables

### 3.1 Integridad científica

- No inventar resultados.
- No copiar cifras de artículos como si fueran propias.
- No afirmar “SOTA” sin una comparación reproducida bajo el mismo protocolo.
- No elegir hiperparámetros usando el test.
- No mezclar fragmentos de la misma secuencia entre train, validation y test.
- No usar etiquetas TTC en el preentrenamiento autosupervisado.
- No ocultar fallos o runs negativos.
- Guardar configuración, semilla, commit y entorno de cada experimento.
- Toda tabla del informe debe regenerarse desde archivos de resultados.

### 3.2 Alcance

El MVP debe ser pequeño y defendible. No intentar entrenar un foundation model. No descargar datasets gigantes sin necesidad. No añadir dependencias o módulos sin un experimento o entregable asociado.

### 3.3 Reproducibilidad

Cada experimento debe registrar:

```text
experiment_id
run_name
git_commit
config_hash
seed
dataset_manifest_hash
split_version
host
python_version
torch_version
cuda_version
gpu_name
start_time
end_time
status
checkpoint_path
metrics_path
```

### 3.4 Calidad de código

- Python tipado.
- Formato con Ruff.
- Type checking con Pyright o mypy.
- Pytest.
- Docstrings en APIs públicas.
- Errores explícitos y mensajes accionables.
- Sin rutas absolutas hardcodeadas.
- Sin secretos en el repositorio.
- Sin notebooks como única implementación.

---

## 4. Stack recomendado

### Núcleo

- Python 3.11.
- PyTorch.
- NumPy.
- h5py.
- pandas o Polars para resultados.
- pydantic para configuración y esquemas.
- Hydra u OmegaConf para experimentos.
- einops.
- scipy.
- scikit-learn para probes y calibración.
- OpenCV para visualización, no para la lógica central.
- matplotlib para figuras regenerables.
- TensorBoard, MLflow o Weights & Biases opcional.

### Calidad

- uv para entorno y lockfile.
- Ruff.
- Pyright.
- Pytest.
- pre-commit.
- GitHub Actions.

### Exportación

- ONNX.
- onnxruntime.

### Opcional

- tonic para transforms y datasets de eventos.
- hdf5plugin si el dataset usa compresiones especiales.
- lightning solo si simplifica el código; no es obligatorio.
- DVC para manifests y artefactos, sin subir datasets al repositorio.

No introducir CUDA custom kernels en la primera versión.

---

## 5. Estructura obligatoria del repositorio

```text
E-JEPA-TTC/
├── AGENTS.md
├── README.md
├── LICENSE
├── CITATION.cff
├── CONTRIBUTING.md
├── SECURITY.md
├── Makefile
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── .pre-commit-config.yaml
├── configs/
│   ├── data/
│   │   ├── evttc_starter.yaml
│   │   ├── evttc_full.yaml
│   │   ├── dsec_ssl.yaml
│   │   └── synthetic_events.yaml
│   ├── representation/
│   │   ├── event_count.yaml
│   │   ├── time_surface.yaml
│   │   ├── voxel_grid.yaml
│   │   └── sparse_tokens.yaml
│   ├── model/
│   │   ├── tiny_cnn.yaml
│   │   ├── e_jepa_small.yaml
│   │   ├── e_jepa_uncertainty.yaml
│   │   └── geometric_baseline.yaml
│   ├── train/
│   │   ├── pretrain.yaml
│   │   ├── finetune.yaml
│   │   └── linear_probe.yaml
│   ├── experiment/
│   │   ├── smoke.yaml
│   │   ├── baseline_suite.yaml
│   │   ├── jepa_main.yaml
│   │   ├── low_label.yaml
│   │   ├── robustness.yaml
│   │   └── ablations.yaml
│   └── default.yaml
├── src/e_jepa_ttc/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── logging.py
│   ├── reproducibility.py
│   ├── data/
│   │   ├── types.py
│   │   ├── base.py
│   │   ├── evttc.py
│   │   ├── dsec.py
│   │   ├── synthetic.py
│   │   ├── index.py
│   │   ├── split.py
│   │   ├── validation.py
│   │   ├── windows.py
│   │   ├── targets.py
│   │   └── collate.py
│   ├── representations/
│   │   ├── base.py
│   │   ├── event_count.py
│   │   ├── time_surface.py
│   │   ├── voxel_grid.py
│   │   ├── sparse_tokens.py
│   │   ├── normalize.py
│   │   └── augment.py
│   ├── models/
│   │   ├── blocks.py
│   │   ├── encoders.py
│   │   ├── predictor.py
│   │   ├── target_encoder.py
│   │   ├── e_jepa.py
│   │   ├── heads.py
│   │   ├── uncertainty.py
│   │   ├── tiny_cnn.py
│   │   └── geometric.py
│   ├── losses/
│   │   ├── predictive.py
│   │   ├── anti_collapse.py
│   │   ├── ttc.py
│   │   ├── geometry.py
│   │   └── uncertainty.py
│   ├── training/
│   │   ├── engine.py
│   │   ├── pretrain.py
│   │   ├── finetune.py
│   │   ├── probe.py
│   │   ├── checkpoint.py
│   │   └── callbacks.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── calibration.py
│   │   ├── robustness.py
│   │   ├── latency.py
│   │   ├── probes.py
│   │   ├── bootstrap.py
│   │   └── report_tables.py
│   ├── inference/
│   │   ├── streaming.py
│   │   ├── export.py
│   │   └── runtime.py
│   ├── visualization/
│   │   ├── events.py
│   │   ├── predictions.py
│   │   ├── embeddings.py
│   │   └── report.py
│   └── utils/
├── scripts/
│   ├── download_evttc_manifest.py
│   ├── build_index.py
│   ├── validate_dataset.py
│   ├── make_splits.py
│   ├── train_baseline.py
│   ├── pretrain_jepa.py
│   ├── finetune_ttc.py
│   ├── evaluate.py
│   ├── benchmark_latency.py
│   ├── export_onnx.py
│   ├── run_demo.py
│   └── build_report.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   └── fixtures/
├── data/
│   ├── README.md
│   ├── manifests/
│   ├── splits/
│   └── cache/
├── artifacts/
│   ├── checkpoints/
│   ├── metrics/
│   ├── figures/
│   ├── tables/
│   └── demos/
├── docs/
│   ├── methodology.md
│   ├── dataset_card.md
│   ├── model_card.md
│   ├── experimental_protocol.md
│   ├── limitations.md
│   ├── reproducibility.md
│   ├── technical_report.md
│   └── references.bib
└── .github/workflows/
    ├── ci.yaml
    ├── smoke-train.yaml
    └── release.yaml
```

---

## 6. Estrategia de datos

### 6.1 Dataset principal

Usar EvTTC como dataset principal supervisado y de evaluación.

La primera descarga debe contener únicamente:

- HDF5 sincronizado.
- `gt-ttc`.
- bounding boxes y segmentación si están disponibles.
- calibración incluida en HDF5.

No descargar vídeos, rosbag ni profundidad para la primera versión.

### 6.2 Subconjunto starter recomendado

Usar nueve secuencias que cubran:

- car-to-car recto;
- car-to-car lateral;
- car-to-pedestrian;
- velocidades baja, media y alta.

Propuesta inicial:

```text
CCRs-1-low-100%
CCRs-1-medium-100%
CCRs-1-high-100%
CCRs-side-low
CCRs-side-medium
CCRs-side-high
CPLA-low
CPLA-medium
CPLA-high
```

No asumir que la nomenclatura local coincide exactamente con la web. El script de manifest debe permitir mapear nombres remotos a IDs normalizados.

### 6.3 Datos adicionales para SSL

DSEC se puede usar solo para preentrenamiento autosupervisado y pruebas de domain shift. Descargar de forma individual entre dos y cuatro secuencias con eventos. No descargar los 125 GB completos de entrenamiento en la fase inicial.

El preentrenamiento sobre DSEC no debe usar etiquetas de flujo, disparidad o segmentación, salvo en probes separados.

### 6.4 eAP

Añadir adaptador cuando la descarga pública esté disponible y estable. No bloquear el MVP por eAP. Su formato esperado incluye `events.h5`, RGB, anotaciones de objeto, velocidad y TTC.

### 6.5 Datos sintéticos

Implementar un generador mínimo que simule:

- expansión de discos o rectángulos;
- traslación lateral;
- aproximación frontal;
- velocidades relativas variables;
- ruido de eventos;
- jitter temporal;
- eventos de fondo;
- contraste positivo y negativo;
- colisiones y no-colisiones.

El objetivo no es sustituir datos reales, sino:

- testear la matemática;
- crear fixtures pequeños;
- validar que el modelo aprende una señal TTC básica;
- detectar regresiones en CI.

### 6.6 Licencias y manifests

Nunca incluir datos crudos en Git. Crear manifests con:

```yaml
dataset: EvTTC
version: 2025-03-02
sequence_id: CCRs-1-low-100
local_hdf5: ${DATA_ROOT}/evttc/CCRs-1-low-100/data.hdf5
gt_ttc: ${DATA_ROOT}/evttc/CCRs-1-low-100/gt_ttc.csv
annotations: ${DATA_ROOT}/evttc/CCRs-1-low-100/annotations.pkl
sha256: ...
scenario_family: CCRs
speed_bucket: low
target_type: car
split_group: CCRs-1
```

---

## 7. Contratos de datos

### 7.1 Evento

```python
@dataclass(frozen=True)
class EventBatch:
    x: torch.Tensor          # int16/int32 [N]
    y: torch.Tensor          # int16/int32 [N]
    t_us: torch.Tensor       # int64 [N], monotónico no decreciente
    polarity: torch.Tensor   # int8/bool [N], normalizado a {-1, +1}
    width: int
    height: int
    sequence_id: str
    t_start_us: int
    t_end_us: int
```

Invariantes:

- mismo número de elementos en `x`, `y`, `t_us`, `polarity`;
- coordenadas dentro de resolución;
- timestamps monotónicos;
- polaridad solo en `{-1,+1}` tras normalización;
- ventana no vacía o marcada como vacía de forma explícita;
- no copiar todo el stream a memoria si puede indexarse por `ms_map_idx`.

### 7.2 Muestra temporal

```python
@dataclass(frozen=True)
class TTCWindowSample:
    context_events: EventBatch
    future_events: dict[int, EventBatch]  # horizonte_ms -> eventos
    ttc_seconds: float | None
    collision_within: dict[float, bool] | None
    object_bbox: torch.Tensor | None
    object_mask: torch.Tensor | None
    metadata: dict[str, Any]
```

### 7.3 Salida de modelo

```python
@dataclass
class TTCModelOutput:
    ttc_mean_seconds: torch.Tensor
    ttc_log_variance: torch.Tensor | None
    collision_logits: torch.Tensor
    context_embedding: torch.Tensor
    predicted_future_embeddings: dict[int, torch.Tensor]
    diagnostics: dict[str, torch.Tensor]
```

---

## 8. Indexación temporal y ventanas

Implementar indexación sin cargar todos los eventos.

Parámetros configurables:

```yaml
context_ms: 100
future_horizons_ms: [25, 50, 100, 250, 500]
stride_ms: 20
min_events: 500
max_events: 250000
clip_ttc_seconds: [0.1, 12.0]
```

Reglas:

1. El timestamp de referencia es el final de la ventana de contexto.
2. Cada horizonte futuro comienza después del contexto, sin solapamiento accidental salvo que el experimento lo permita.
3. El target TTC se interpola al timestamp de referencia.
4. Registrar qué método de interpolación se usa.
5. Excluir puntos sin target válido.
6. Permitir incluir no-colisiones con target censurado o clase negativa.
7. No sobremuestrear el mismo instante de forma descontrolada.
8. Limitar correlación temporal mediante stride configurable.

Implementar dos modos:

### Modo dense

Una muestra cada `stride_ms`.

### Modo event-budget

Ventanas con número aproximado de eventos fijo y duración variable. Este modo es una ablation, no el default.

---

## 9. Splits sin fuga

Prohibido hacer split aleatorio por ventanas.

Crear splits por grupos completos:

- escenario;
- sesión;
- target;
- velocidad;
- secuencia.

Propuesta starter:

- Train: parte de CCRs-1 y CPLA.
- Validation: una velocidad no usada de cada familia o secuencias laterales.
- Test in-domain: secuencias no vistas de familias conocidas.
- Test cross-scenario: familia completa no vista.
- Test cross-target: peatón no visto durante entrenamiento, si el volumen lo permite.

El script `make_splits.py` debe:

- recibir una semilla;
- generar JSON/YAML legible;
- guardar estadísticas;
- comprobar ausencia de intersección;
- fallar si una `split_group` aparece en más de un split.

Tests obligatorios:

```text
test_split_groups_are_disjoint
test_windows_do_not_cross_sequence_boundaries
test_no_timestamp_overlap_across_splits
test_target_distribution_report_is_generated
```

---

## 10. Representaciones de eventos

Implementar una interfaz común:

```python
class EventRepresentation(Protocol):
    def encode(self, events: EventBatch) -> torch.Tensor: ...
```

### 10.1 Event count

Canales:

- conteo positivo;
- conteo negativo;

Variantes:

- raw count;
- `log1p`;
- normalización por duración;
- normalización por máximo robusto.

### 10.2 Time surface

Para cada polaridad, guardar la antigüedad normalizada del último evento.

Definir claramente:

```text
surface = exp(-(t_end - t_last_event) / tau)
```

Ablations de `tau`.

### 10.3 Voxel grid

Dimensiones:

```text
[C = 2 * num_bins, H, W]
```

Usar interpolación temporal lineal entre bins. Implementar versión CPU y PyTorch vectorizada. Verificar igualdad aproximada con una implementación de referencia lenta.

Defaults:

```yaml
num_bins: 5
separate_polarity: true
normalization: robust_per_window
```

### 10.4 Sparse tokens

Cada token puede contener:

```text
x_norm
y_norm
t_norm
polarity
local_event_density
inter_event_time
```

Estrategias de reducción:

- muestreo uniforme temporal;
- reservoir sampling;
- agrupación en celdas espacio-temporales;
- top-k por densidad con una fracción aleatoria para evitar sesgo.

El MVP no debe depender de sparse CUDA ops.

### 10.5 Multi-timescale

Concatenar representaciones para ventanas:

```text
25 ms
50 ms
100 ms
200 ms
```

La ventana más larga aporta contexto; la más corta aporta velocidad instantánea.

---

## 11. Augmentations específicas de eventos

Toda augmentation debe preservar o actualizar correctamente el target.

Implementar:

- event dropout;
- polarity dropout;
- polarity flip con probabilidad baja;
- timestamp jitter;
- spatial jitter;
- background event injection;
- temporal crop;
- variable accumulation window;
- hot-pixel simulation;
- dead-pixel mask;
- horizontal flip cuando la geometría lo permita;
- contrast-rate scaling;
- coordinate quantization;
- missing packet simulation.

No aplicar augmentations que cambien implícitamente TTC sin ajustar el target. Por ejemplo, time scaling debe transformar:

```text
new_ttc = old_ttc / speed_scale
```

Cada augmentation debe tener:

- test determinista con seed;
- test de rango;
- test de preservación de timestamp monotónico;
- registro de parámetros usados.

---

## 12. Baselines

### 12.1 Baseline de media/último valor

Implementar referencias triviales:

- media de TTC de train;
- mediana de TTC de train;
- último TTC conocido si el protocolo lo permite;
- predictor por bucket de escenario.

### 12.2 Baseline geométrico

Implementar una aproximación basada en expansión aparente del objeto cuando existan bbox/máscara.

Si `s(t)` es una escala aparente:

```text
TTC ≈ s(t) / ds_dt
```

Usar variantes:

- ancho de bbox;
- altura de bbox;
- raíz cuadrada del área;
- radio equivalente de máscara.

Debe incluir filtros robustos:

- Savitzky-Golay o regresión local;
- rechazo de derivadas no positivas;
- límites físicos;
- fallback a NaN con causa registrada.

No presentar este baseline como física exacta.

### 12.3 Tiny CNN supervisada

Entrada voxel grid o event count.

Arquitectura orientativa:

```text
Conv stem
3-4 residual stages
global average pooling
MLP TTC
MLP collision thresholds
optional uncertainty head
```

Objetivo de tamaño:

```text
0.2M–5M parámetros
```

### 12.4 Temporal baseline

Opcional pero recomendado:

- encoder CNN por ventana;
- GRU/TCN pequeña sobre varias ventanas;
- salida TTC.

Este baseline separa el valor de “usar tiempo” del valor del objetivo JEPA.

---

## 13. Arquitectura E-JEPA

### 13.1 Diseño general

```text
context events
    -> representation/tokenizer
    -> context encoder f_theta
    -> context embeddings z_t
    -> predictor g_phi(z_t, horizon_embedding, optional metadata)
    -> predicted target embedding z_hat_{t+h}

future events
    -> same representation/tokenizer
    -> target encoder f_xi
    -> stop-gradient target embedding z_{t+h}
```

El target encoder se actualiza por EMA:

```text
xi <- m * xi + (1 - m) * theta
```

Programar `m` de 0.99 hacia 0.9999 o configurable.

### 13.2 Encoder small

Dos opciones implementables bajo una interfaz común:

#### Dense CNN/ConvNeXt mini

- Entrada `[B,C,H,W]`.
- Tokens espaciales antes del pooling.
- Embedding global y dense map.

#### ViT pequeño

- Patch size 8 o 16.
- 4–8 bloques.
- dim 192–384.
- 3–6 heads.

El default debe ser el encoder que entre en una GPU de consumo.

### 13.3 Dense y global embeddings

Mantener dos salidas:

```python
EncoderOutput(
    global_embedding=[B,D],
    dense_tokens=[B,N,D],
    feature_maps=[...],
)
```

Usar global embedding para TTC y dense tokens para pérdida JEPA localizada o probes espaciales.

### 13.4 Predictor

Implementar dos variantes:

1. MLP residual condicionado por horizonte.
2. Transformer pequeño que recibe tokens de contexto y tokens de horizonte.

Horizonte codificado con:

- embedding aprendido;
- Fourier features del tiempo en milisegundos;
- ambas opciones como ablation.

### 13.5 Objetivo multihorizonte

Predicciones para:

```text
25, 50, 100, 250, 500 ms
```

No exigir todos los horizontes si la secuencia termina. Usar máscara de validez.

### 13.6 Máscaras espaciales

Fase opcional tras el MVP global:

- ocultar bloques espacio-temporales;
- predecir embeddings target de regiones;
- pérdida sobre tokens visibles y enmascarados como ablation;
- asegurar que no se usan máscaras derivadas de ground truth.

### 13.7 Metadata opcional

No usar metadata de escenario en el modelo principal. Puede usarse en una ablation controlada. Ego-motion solo si existe y está disponible en inferencia.

---

## 14. Pérdidas

Pérdida total:

```text
L = lambda_pred * L_pred
  + lambda_var * L_variance
  + lambda_cov * L_covariance
  + lambda_ttc * L_ttc
  + lambda_cls * L_collision
  + lambda_unc * L_uncertainty
  + lambda_geo * L_geometry
```

### 14.1 Pérdida predictiva

Default:

```text
1 - cosine_similarity(normalize(z_hat), normalize(stopgrad(z_target)))
```

Comparar con Smooth L1 en embedding estandarizado.

### 14.2 Anti-colapso

Implementar al menos una opción estable:

- VICReg variance/covariance;
- Gaussian regularization estilo LeWorldModel;
- uniformity regularizer.

El default puede ser:

```text
L_pred + lambda_gauss * GaussianRegularizer(z)
```

pero debe existir diagnóstico de colapso.

### 14.3 TTC

Comparar:

- Huber sobre TTC;
- Huber sobre `log(TTC + eps)`;
- Huber sobre `1 / TTC`.

Default recomendado:

```text
Huber(log(TTC))
```

con transformación inversa para métricas.

### 14.4 Clasificación de riesgo

Umbrales configurables:

```text
0.5 s
1.0 s
2.0 s
4.0 s
```

Usar BCEWithLogits con pesos de clase calculados solo en train.

### 14.5 Incertidumbre

Cabeza de media y log-varianza:

```text
NLL = 0.5 * exp(-log_var) * error^2 + 0.5 * log_var
```

Clampear log-varianza a un rango razonable.

### 14.6 Pérdida geométrica

Opcional:

- estimar expansión latente o aparente;
- penalizar inconsistencias entre menor TTC y expansión positiva;
- no usar si no hay bbox/máscara válida.

Debe estar desacoplada para no contaminar el experimento principal.

---

## 15. Diagnóstico de colapso

Registrar por batch y epoch:

- desviación estándar media por dimensión;
- rango efectivo de covarianza;
- eigenvalues top-k;
- cosine similarity entre muestras aleatorias;
- ratio de dimensiones con std por debajo de umbral;
- norma de embedding;
- pérdida predictiva train/val;
- agreement entre student y target.

Abortar entrenamiento con error claro si:

```text
> 80% de dimensiones tienen std < 1e-3 durante N evaluaciones
```

No usar este umbral sin permitir configuración.

Generar figura:

```text
embedding_health.png
```

---

## 16. Fases de entrenamiento

### Fase 0 — Smoke test sintético

Objetivo:

- verificar pipeline;
- hacer overfit a 32–128 ejemplos;
- probar guardado y carga;
- comprobar descenso de loss.

Gate:

- el baseline debe sobreajustar un batch pequeño;
- E-JEPA debe reducir pérdida predictiva;
- no debe colapsar.

### Fase 1 — Baseline supervisado

Entrenar Tiny CNN con voxel grid.

Guardar:

- mejor checkpoint por validation MAE;
- último checkpoint;
- curvas;
- métricas por bucket;
- latencia.

### Fase 2 — Preentrenamiento E-JEPA

Usar eventos sin etiquetas TTC.

Curriculum sugerido:

1. contexto 100 ms, horizonte 50 ms;
2. añadir 25/100 ms;
3. añadir 250/500 ms;
4. augmentations suaves;
5. augmentations robustas.

### Fase 3 — Linear probe

Congelar encoder y entrenar cabeza TTC. Medir qué información contiene el embedding.

### Fase 4 — Fine-tuning

Comparar:

- encoder congelado;
- últimas capas descongeladas;
- full fine-tuning con LR menor.

### Fase 5 — Low-label

Subconjuntos:

```text
1%
5%
10%
25%
100%
```

Usar los mismos IDs de secuencia y sampling reproducible.

### Fase 6 — Incertidumbre

Entrenar o calibrar:

- heteroscedastic head;
- deep ensemble pequeño opcional;
- temperature scaling para thresholds.

---

## 17. Métricas

### Regresión

- MAE en segundos.
- Mediana del error absoluto.
- RMSE.
- MAPE con protección cerca de cero.
- Error relativo simétrico.
- MAE de log-TTC.
- Error por bucket de TTC.

Buckets sugeridos:

```text
0.0–0.5
0.5–1.0
1.0–2.0
2.0–4.0
4.0–8.0
>8.0
```

### Riesgo

Por cada threshold:

- AUROC.
- AUPRC.
- precision.
- recall.
- F1.
- false negative rate.
- expected warning lead time.

### Calibración

- ECE para clasificación.
- Brier score.
- NLL.
- cobertura de intervalos 50%, 80%, 95%.
- anchura media de intervalos.
- error frente a incertidumbre.

### Robustez

Reportar degradación absoluta y relativa.

### Eficiencia

- ms por ventana.
- ventanas por segundo.
- eventos por segundo.
- peak VRAM.
- RAM.
- parámetros.
- FLOPs aproximados.
- tamaño ONNX.

### Estadística

- media y desviación entre tres semillas.
- bootstrap CI por secuencia.
- no hacer bootstrap por ventanas tratándolas como independientes.

---

## 18. Robustness suite

Crear un runner que aplique perturbaciones sin reentrenar:

```yaml
perturbations:
  event_dropout: [0.1, 0.3, 0.5, 0.7]
  timestamp_jitter_us: [50, 200, 1000]
  background_event_rate: [0.01, 0.05, 0.1]
  hot_pixel_fraction: [0.001, 0.005]
  dead_pixel_fraction: [0.01, 0.05]
  polarity_drop: [positive, negative]
  temporal_window_scale: [0.5, 0.75, 1.25, 1.5]
  spatial_crop_fraction: [0.9, 0.75]
```

Registrar:

- target sin modificar cuando corresponde;
- seed;
- intensidad;
- diferencia de métricas;
- cambio de incertidumbre.

Una predicción probabilística es mejor si aumenta su incertidumbre bajo corrupción, no solo si mantiene MAE.

---

## 19. Experimentos obligatorios

### Tabla A — Baselines

```text
Mean
Median
Geometric bbox
TinyCNN event count
TinyCNN voxel grid
Temporal CNN/GRU
```

### Tabla B — JEPA

```text
Supervised from scratch
JEPA linear probe
JEPA frozen encoder
JEPA partial fine-tune
JEPA full fine-tune
```

### Tabla C — Representaciones

```text
Event count
Time surface
Voxel grid
Sparse tokens
Multi-timescale voxel
```

### Tabla D — Objetivo

```text
Single horizon
Multi-horizon
No anti-collapse
VICReg
Gaussian regularization
No geometry loss
With geometry loss
```

### Tabla E — Etiquetas

```text
1%, 5%, 10%, 25%, 100%
```

### Tabla F — Robustez

Baseline vs E-JEPA en cada perturbación.

No ejecutar todas las combinaciones cartesianas. Usar una matriz secuencial y justificada.

---

## 20. Evaluación de embeddings

Implementar probes simples:

- TTC lineal.
- velocidad relativa si existe.
- clase de escenario.
- target car/pedestrian.
- event rate.
- dirección de expansión.

Un buen embedding no debe limitarse a memorizar el ID de secuencia. Evaluar con splits por secuencia.

Visualizaciones:

- PCA.
- UMAP opcional.
- vecinos más cercanos.
- trayectorias latentes temporales.
- embedding distance frente a diferencia TTC.

No usar t-SNE como evidencia cuantitativa.

---

## 21. Inferencia en streaming

Crear una clase con ring buffer:

```python
class StreamingTTCEstimator:
    def push_events(self, x, y, t_us, polarity) -> None: ...
    def ready(self) -> bool: ...
    def predict(self, now_us: int) -> TTCModelOutput: ...
    def reset(self) -> None: ...
```

Requisitos:

- no reprocesar el stream completo;
- eliminar eventos fuera de ventana;
- manejar paquetes vacíos;
- detectar timestamp rollback;
- soportar batch offline y stream;
- medir preprocessing e inferencia por separado.

Salida de demo:

```text
TTC predicted
TTC ground truth
absolute error
confidence interval
risk state
latency
current event rate
```

Estados:

```text
SAFE
WATCH
WARNING
CRITICAL
UNKNOWN
```

Los thresholds deben ser configurables y no presentarse como estándar de seguridad.

---

## 22. Exportación

Exportar modelo con input tensor fijo para representaciones dense.

Entregables:

```text
model.onnx
model_metadata.json
normalization.json
example_input.npz
example_output.json
```

Verificar:

- salida PyTorch vs ONNX con tolerancia;
- CPU runtime;
- batch size 1;
- shapes documentadas;
- no incluir capas no soportadas.

---

## 23. CLI obligatoria

Ejemplos:

```bash
uv run e-jepa-ttc data validate --config configs/data/evttc_starter.yaml
uv run e-jepa-ttc data index --config configs/data/evttc_starter.yaml
uv run e-jepa-ttc split create --config configs/experiment/jepa_main.yaml
uv run e-jepa-ttc train baseline --config configs/experiment/baseline_suite.yaml seed=42
uv run e-jepa-ttc train pretrain --config configs/experiment/jepa_main.yaml seed=42
uv run e-jepa-ttc train finetune --checkpoint artifacts/checkpoints/pretrain_best.pt
uv run e-jepa-ttc evaluate --checkpoint artifacts/checkpoints/finetune_best.pt
uv run e-jepa-ttc robustness --checkpoint ...
uv run e-jepa-ttc export onnx --checkpoint ...
uv run e-jepa-ttc demo --sequence CCRs-side-high
uv run e-jepa-ttc report build
```

Toda CLI debe tener `--help` útil y devolver código distinto de cero ante fallo.

---

## 24. Configuración

Ejemplo:

```yaml
experiment:
  name: e_jepa_multihorizon_voxel
  seed: 42
  output_dir: artifacts/runs/${experiment.name}/${now:%Y%m%d_%H%M%S}

data:
  dataset: evttc
  manifest: data/manifests/evttc_starter.yaml
  split: data/splits/evttc_v1.yaml
  context_ms: 100
  stride_ms: 20
  horizons_ms: [25, 50, 100, 250, 500]
  min_events: 500

representation:
  name: voxel_grid
  bins: 5
  separate_polarity: true
  height: 240
  width: 320

model:
  encoder: convnext_mini
  embedding_dim: 256
  predictor_layers: 4
  predictor_heads: 4
  target_ema_start: 0.99
  target_ema_end: 0.9999
  uncertainty: true

loss:
  predictive: cosine
  anti_collapse: gaussian
  lambda_predictive: 1.0
  lambda_anti_collapse: 0.05
  lambda_ttc: 1.0
  lambda_collision: 0.25
  lambda_uncertainty: 0.1

trainer:
  epochs: 100
  batch_size: 32
  precision: bf16
  grad_clip_norm: 1.0
  optimizer: adamw
  learning_rate: 0.0003
  weight_decay: 0.05
  num_workers: 8
  early_stopping_patience: 15
```

Validar configuración antes de reservar GPU.

---

## 25. Tests

### Unitarios

- parsing HDF5;
- monotonicidad timestamps;
- indexación por ms;
- voxel grid contra referencia;
- time surface;
- augmentations;
- transforms de TTC;
- shape de modelos;
- EMA target encoder;
- pérdidas finitas;
- métricas con ejemplos conocidos;
- calibración;
- serialización config.

### Integración

- synthetic dataset → train baseline → checkpoint → evaluate;
- synthetic dataset → pretrain → linear probe;
- fixture HDF5 pequeño → index → dataloader;
- export ONNX → onnxruntime;
- demo genera MP4.

### Regresión

- hash de representación de fixture;
- métrica de smoke test dentro de rango;
- tiempo máximo razonable en CPU;
- no cambio accidental de split.

### Tests de propiedad

Con Hypothesis si procede:

- timestamps siempre monotónicos tras augmentation;
- voxel sum conserva aproximadamente eventos;
- inverse transforms de TTC son consistentes;
- split permanece disjunto.

---

## 26. CI

### En cada push

- instalación limpia;
- Ruff format/check;
- Pyright;
- unit tests;
- integration smoke CPU;
- build package;
- build docs.

### Semanal o manual

- smoke training GPU si hay runner;
- export ONNX;
- reproducibility check;
- dependency audit.

La CI no debe descargar EvTTC. Usar fixtures sintéticos y un HDF5 mínimo creado en test.

---

## 27. Logging y experiment tracking

Cada run debe crear:

```text
config_resolved.yaml
metadata.json
stdout.log
metrics.jsonl
summary.json
checkpoints/
figures/
predictions.parquet
```

`predictions.parquet` debe contener, por ventana:

```text
sequence_id
timestamp_us
ttc_true
ttc_pred
ttc_std
risk_true_1s
risk_prob_1s
num_events
scenario
speed_bucket
perturbation
seed
```

---

## 28. Informe técnico

El informe debe contener:

1. Resumen.
2. Motivación.
3. Cámaras de eventos y TTC.
4. Hipótesis JEPA.
5. Datasets y splits.
6. Representaciones.
7. Arquitectura.
8. Objetivos de entrenamiento.
9. Baselines.
10. Protocolo experimental.
11. Resultados.
12. Ablations.
13. Robustez.
14. Latencia.
15. Calibración.
16. Limitaciones.
17. Trabajo futuro.
18. Reproducibilidad.

No incluir conclusiones que no estén apoyadas por tablas.

---

## 29. README

El README debe mostrar en la primera pantalla:

- problema;
- contribución;
- diagrama;
- demo GIF o imagen;
- resultados principales reales;
- instalación;
- quickstart;
- datasets;
- cita;
- licencia.

No convertir el README en marketing vacío.

---

## 30. Hitos para Codex

Codex debe avanzar en este orden y hacer commits pequeños.

### Milestone M0 — Bootstrap

- estructura;
- pyproject;
- CI;
- config;
- CLI vacía;
- tests básicos.

### M1 — Datos sintéticos

- generator;
- fixtures;
- visualizador;
- targets TTC conocidos.

### M2 — EvTTC loader

- HDF5;
- index;
- validation;
- manifests;
- splits.

### M3 — Representaciones

- event count;
- time surface;
- voxel grid;
- benchmarks.

### M4 — Baselines

- trivial;
- geometric;
- Tiny CNN;
- metrics.

### M5 — E-JEPA core

- encoder;
- target EMA;
- predictor;
- losses;
- collapse diagnostics.

### M6 — Training stages

- pretrain;
- probe;
- fine-tune;
- checkpointing.

### M7 — Evaluation

- buckets;
- bootstrap;
- robustness;
- calibration;
- latency.

### M8 — Runtime

- streaming;
- demo;
- ONNX.

### M9 — Research package

- ablations;
- report;
- model card;
- release.

No saltar a M5 antes de que M2–M4 estén verificados.

---

## 31. Política de commits y ramas

Commits sugeridos:

```text
chore: bootstrap project and CI
feat(data): add synthetic event generator
feat(data): add EvTTC HDF5 adapter
feat(repr): implement voxel grid encoding
feat(model): add supervised TTC baseline
feat(jepa): add EMA target encoder and predictor
feat(train): add self-supervised pretraining loop
feat(eval): add sequence-level bootstrap metrics
feat(runtime): add streaming TTC estimator
feat(report): generate reproducible experiment report
```

Cada commit debe pasar tests. No hacer un único commit gigante.

---

## 32. Criterios de aceptación por fase

### Datos

- 100% de secuencias descargadas validan manifest.
- timestamps monotónicos.
- índices no salen de rango.
- split disjunto.

### Baseline

- overfit de batch pequeño.
- MAE finito.
- predicciones no constantes salvo baseline trivial.

### JEPA

- pérdida predictiva desciende.
- embedding no colapsa.
- linear probe supera predictor de media en smoke sintético.

### Evaluación

- métricas reproducibles con misma seed.
- bootstrap por secuencia.
- CSV y figuras consistentes.

### Demo

- procesa secuencia sin memory leak.
- muestra `UNKNOWN` cuando faltan eventos.
- no crashea en gaps.

---

## 33. Riesgos y mitigaciones

### EvTTC demasiado grande

Mitigación: starter subset; HDF5 únicamente; cache de ventanas; lazy loading.

### Preentrenamiento no mejora TTC

Mitigación: reportar resultado negativo; revisar horizon, representation y low-label. El objetivo es investigación reproducible, no forzar una mejora.

### Colapso

Mitigación: EMA, normalización, Gaussian/VICReg, batch mayor, diagnostics.

### Correlación temporal infla métricas

Mitigación: split por secuencia y bootstrap por secuencia.

### Modelo aprende event rate

Mitigación: probes; rate normalization; perturbación de density; cross-scenario test.

### Incertidumbre mal calibrada

Mitigación: calibration split; temperature scaling; coverage plots.

### Latencia dominada por preprocessing

Mitigación: medir etapas; vectorizar; cache; bajar resolución.

---

## 34. No hacer

- No usar frames RGB en el modelo principal salvo experimento claramente etiquetado.
- No usar test para early stopping.
- No hacer split por ventanas.
- No descargar DSEC completo al comenzar.
- No introducir un LLM en el pipeline.
- No llamar “tiempo real” sin medir hardware y latencia.
- No afirmar validez para sistemas de seguridad reales.
- No ocultar secuencias fallidas.
- No cambiar métricas después de ver test sin versionar protocolo.

---

## 35. Referencias técnicas que deben figurar en docs

Incluir y citar al menos:

- EvTTC, arXiv:2412.05053 y su página oficial.
- Event-Aided TTC, ECCV 2024.
- eAP dataset, arXiv:2603.16303.
- V-JEPA 2, arXiv:2506.09985.
- V-JEPA 2.1, arXiv:2603.14482.
- LeWorldModel, arXiv:2603.19312.
- Fast LeWorldModel, arXiv:2606.26217.
- VJEPA probabilístico, arXiv:2601.14354.
- DSEC.
- MVSEC.

No copiar código incompatible con la licencia del proyecto.

---

## 36. Prompt inicial recomendado para Codex

```text
Lee AGENTS.md completo y trátalo como contrato. Empieza por M0 y M1. No entrenes todavía con datos reales. Crea el repositorio instalable, la configuración tipada, la CLI, CI, un generador sintético de eventos con TTC conocido y tests. Haz commits pequeños. No inventes resultados. Al terminar cada milestone, ejecuta todos los tests y actualiza docs/progress.md con decisiones, comandos y limitaciones.
```

---

## 37. Prompt de continuación recomendado

```text
Continúa con el siguiente milestone incompleto de AGENTS.md. Antes de modificar código, revisa docs/progress.md, ejecuta tests y comprueba el estado del repositorio. No reescribas módulos que ya pasan sus contratos. Añade tests primero cuando corrijas bugs. Registra cualquier desviación del diseño en docs/decisions/ADR-XXXX.md.
```

---

## 38. Entregables finales

El agente debe entregar:

```text
Repositorio GitHub público
Release v0.1.0
Pesos baseline
Pesos E-JEPA
ONNX
Resultados CSV/Parquet
Demo MP4
Informe PDF o Markdown renderizable
Model card
Dataset card
Reproduction guide
CITATION.cff
Zenodo-ready release
```

El proyecto debe poder defenderse con honestidad como:

> Un MVP de investigación reproducible sobre aprendizaje predictivo latente para cámaras de eventos, no un producto de seguridad certificado ni una afirmación de estado del arte.
