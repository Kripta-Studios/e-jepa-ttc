# E-JEPA-TTC — Handoff de implementación A4: DINOv3 Dense Relational Distillation

**Repositorio objetivo:** `Kripta-Studios/e-jepa-ttc`  
**Rama:** `scientific-recovery-v3-hardening`  
**Commit científico de partida obligatorio:** `8c2ffeded4eb0f925d494b72adb670e7640edb17`  
**Fecha del protocolo:** 2026-08-10  
**Objetivo de este documento:** permitir que un agente de código implemente exactamente el siguiente experimento científico sin improvisar arquitectura, hiperparámetros, datos, gates ni criterios de interpretación.

---

# 0. Instrucción principal al agente

Implementa **A4 = A1-DF + DINOv3 ConvNeXt-Large local relational distillation**, manteniendo **idéntico** el modelo de inferencia A1-DF, sus 355.118 parámetros, el geometry head, Causal Scale, residual, temporal consensus, losses físicas, datos, splits, seed, optimizer, schedule, clipping y política de `unknown`.

El único cambio científico de A4 debe ser:

> Durante **training únicamente**, las features densas profundas del encoder de eventos reciben una pérdida relacional local que intenta reproducir la **estructura espacial** de features densas de un DINOv3 ConvNeXt-Large congelado aplicado al RGB sincronizado del mismo endpoint y del mismo common-square crop.

DINOv3:

- **NO** es input del modelo final;
- **NO** se carga en validation;
- **NO** se carga en inferencia;
- **NO** cambia el número de parámetros del student;
- **NO** utiliza TTC, bbox, SAM o labels para producir la representación teacher;
- **NO** se alinea mediante MSE directo entre canales RGB/event;
- **NO** se ejecuta durante cada epoch: debe materializarse un cache train-only de relaciones locales.

El experimento científico A4 **NO debe ejecutarse sobre validation antes de que la implementación, el cache train-only, la calibración train-only y el config preregistrado hayan sido auditados**.

El agente sí puede:

1. implementar;
2. ejecutar unit/integration tests;
3. hacer smoke con DINOv3 ConvNeXt-Tiny;
4. materializar el cache científico con ConvNeXt-Large usando **solo train**;
5. calibrar el peso de la distillation usando **solo train**;
6. dejar el YAML científico final con hashes/peso congelados;
7. hacer commits limpios.

El agente **NO debe**:

- ejecutar el entrenamiento A4 que evalúa validation;
- modificar gates tras mirar validation;
- usar SAM en A4;
- activar `foreground_pair_ratio_weight`;
- añadir JEPA todavía;
- cambiar el encoder student, hidden dim, geometry dim, decoder, residual o Causal Scale;
- escalar a 8k/16k;
- abrir test oficial, CodaBench o EvTTC;
- usar DINOv3 ViT-L como segundo brazo;
- hacer sweep Tiny/Base/Large mirando validation.

---

# 1. Por qué A4 existe: diagnóstico científico que no debe perderse

A4 no nace de una preferencia estética por DINO. Nace de la secuencia experimental cerrada A0 → A1 → A1-FR → A1-DF → A1-DF-R → A3.

Estado congelado en `STATUS.md`, aproximadamente líneas 188–238 del commit base:

| brazo | cambio | MiD macro ↓ | failure ↓ | Pearson ratio ↑ | dato mecanístico |
|---|---|---:|---:|---:|---|
| Garl matched | baseline event-only 2048/2048 | **203.6342** | **0%** | **.3722** | aprende señal temporal mucho mejor |
| A0 | weak-box BCE/Dice | 382.1905 | 12.30% | .0456 | casi nada de scale |
| A1 | geometry-only h/w/cx/cy | **346.8295** | **9.96%** | .1108 | mejora parcial |
| A1-FR | full-res raw 2-D | 380.2202 | 28.76% | -.0181 | raw 2-D empeora |
| A1-DF | **deep features + resize_conv** | 350.3020 | 21.09% | **.1865** | sube la señal física interna |
| A1-DF-R | + direct pair-ratio loss | 349.8628 | 19.82% | .1703 | no arregla el cuello |
| A3 | A1 + SAM teacher mask | 353.6351 | 10.89% | .1053 | SAM no arregla dinámica |

Los resultados mecanísticos que motivan A4 son especialmente:

- A1 absolute height Pearson ≈ `.4708`;
- A1 absolute width Pearson ≈ `.0788`;
- A1 `delta log h` vs física ≈ `.1048`;
- A1-DF absolute height ≈ `.4823`;
- A1-DF absolute width ≈ `.2428`;
- A1-DF `delta log h` vs física ≈ `.1704`;
- A1-DF ratio Pearson ≈ `.1865`;
- A1-DF analytic ratio slope ≈ `.0848`, es decir, señal real pero extremadamente comprimida;
- A1-DF produce 433 `unknown` por ratio bajo, **cero** por soporte;
- A1-DF-R no resuelve el problema;
- A3 demuestra que mejorar la máscara RGB objetivo con SAM tampoco resuelve la representación/dinámica.

La evidencia apunta a:

> El encoder de eventos actual contiene algo de información temporal y geométrica cuando usamos sus features profundas, pero esas features no están suficientemente estructuradas espacialmente para que un decoder pequeño extraiga una medida métrica de tamaño/escala estable.

Esto es la hipótesis **única** que A4 intenta probar.

---

# 2. Referencias de código congeladas en el commit base

Todas las líneas siguientes se refieren al commit:

```text
8c2ffeded4eb0f925d494b72adb670e7640edb17
```

Los números son líneas de archivo aproximadas en la versión raw del commit; tras editar pueden desplazarse. El agente debe usar además búsquedas por símbolo, no confiar solo en el número.

## 2.1 Modelo Causal Scale

Archivo:

```text
src/e_jepa_ttc/models/causal_scale_ttc.py
```

Puntos relevantes:

- `CausalScaleTTCConfig`: ~líneas 24–92.
- `SoftScaleObservation`: ~94–102.
- `CausalScaleTTCOutput`: ~105–128.
- `soft_vertical_extent_from_logits`: ~129–180.
- `_EndpointEncoder`: ~363–437.
- `_EndpointEncoder.features`: ~369–379.
- `resize_conv` decoder: ~397–424.
- global token (`AdaptiveAvgPool2d(1)`): ~428–432.
- `_EndpointEncoder.forward`: ~434–437.
- `CausalScaleTTC`: ~463 en adelante.
- `CausalScaleTTC.forward`: ~533 en adelante.
- flatten de `[B,T,C,H,W]`: ~546.
- llamada actual `lowres_logits, base_tokens = self.encoder(flat)`: ~547.
- interpolación foreground: ~548 en adelante.
- `soft_vertical_extent_from_logits`: ~572 aprox.
- ratio físico proviene de diferencia de `log(height)`.

La configuración A1-DF tiene:

```yaml
hidden_dim: 64
geometry_dim: 128
foreground_decoder: resize_conv
```

y dos convoluciones stride 2 en `_EndpointEncoder.features`.

Por tanto, con input `[B*T,12,128,128]`:

```text
student dense features = [B*T,64,32,32]
```

Ese tensor es el **único tensor student** que debe recibir la distillation DINO.

No usar el global token de 128 dimensiones para distillation.

---

## 2.2 Config del modelo A1-DF

Archivo:

```text
configs/model/e_jepa_causal_scale_event_v8_t015_resize_conv.yaml
```

Debe permanecer **byte-for-byte sin cambios** en A4.

Contenido congelado relevante:

```yaml
model: e_jepa_causal_scale_event_v8
modality: event
in_channels: 12
hidden_dim: 64
geometry_dim: 128
residual_depth: 2
dropout: 0.05
foreground_decoder: resize_conv
foreground_fullres_dim: 24
foreground_temperature: 1.0
foreground_temporal_smoothing: 0.15
max_abs_log_ratio_residual: 0.05
max_abs_log_height_correction: 0.0
temporal_inverse_ttc_blend: 0.75
min_abs_log_ratio: 0.002
min_sensor_support: 0.0001
ttc_clip_seconds: 60.0
initial_log_ratio_std: 0.03
log_ratio_log_variance_min: -12.0
log_ratio_log_variance_max: 2.0
risk_thresholds_s: [0.5, 1.0, 2.0, 4.0]
```

**No crear un nuevo model config con cambios encubiertos.**

A4 debe apuntar al mismo fichero.

---

## 2.3 Config experimental padre A1-DF

Archivo:

```text
configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a1_deep_features_v1.yaml
```

Líneas relevantes:

- 1–6: identidad y parent.
- 8–25: cache y splits.
- 27–42: training.
- 44–59: losses.
- 61–97: decision contract.

Valores que A4 debe heredar exactamente:

```yaml
training:
  seed: 7
  epochs: 18
  minimum_epochs: 8
  early_stopping_patience: 5
  foreground_warmup_epochs: 3
  batch_size: 32
  gradient_accumulation_steps: 1
  learning_rate: 0.0003
  minimum_learning_rate: 0.00003
  weight_decay: 0.0001
  grad_clip_norm: 1.0
  num_workers: 0
  precision: bf16
  maximum_runtime_hours: 6.0
  mask_t0_as_proxy: true
  foreground_supervision: bbox_geometry

loss:
  log_ratio_nll_weight: 1.0
  log_ratio_huber_weight: 0.0
  log_ratio_tail_weight: 2.0
  log_ratio_tail_fraction: 0.10
  foreground_bce_weight: 0.0
  foreground_dice_weight: 0.0
  foreground_extent_weight: 1.25
  foreground_width_weight: 1.25
  foreground_center_weight: 2.5
  foreground_pair_ratio_weight: 0.0
  risk_weight: 0.1
  auxiliary_inverse_ttc_weight: 0.05
  residual_regularization_weight: 0.1
  temporal_consistency_weight: 0.0
  smooth_l1_beta: 0.02
  supervise_pair_ratio_before_temporal_blend: true
```

**No modificar ningún valor de este bloque.**

La distillation A4 debe vivir fuera de `CausalScaleTTCLossConfig` para que sea claramente una supervisión de representación y no una mutación escondida de la loss física.

---

## 2.4 Trainer real

Archivo:

```text
src/e_jepa_ttc/training/causal_scale_eap.py
```

Símbolos/zonas:

- `CausalScaleEAPTrainingConfig`: ~41–91.
- `CausalScaleEAPTargets`: ~104–111.
- `_targets`: ~118–175.
- `_loss`: ~177–204.
- `_foreground_only_loss_config`: ~206–220.
- `_selection`: ~222–252.
- `_loader`: ~360–395.
- `train_one_real_epoch`: ~397–440.
- `evaluate_real_causal_scale`: ~443–735.
- `train_real_causal_scale`: ~737–947.
- contrato de resume: ~789–800.
- payload de resume: ~824–849.
- loop de epochs: ~854 en adelante.

A4 debe alterar el mínimo posible:

1. ampliar `CausalScaleEAPTrainingConfig`;
2. permitir que `_loss` pida dense features al student cuando representation supervision esté activa;
3. calcular una loss relacional adicional;
4. sumar esa loss al total;
5. reportarla en `components`;
6. mantener evaluation sin teacher;
7. mantener resume exacto y ligado al nuevo contrato.

---

## 2.5 Batch Object Event V4

Archivo:

```text
src/e_jepa_ttc/data/object_event_v4.py
```

Zonas:

- `ObjectEventV4Batch`: ~23–89.
- `event_inputs()`: ~83–89.
- `GarlTTCObjectEventV4Dataset`: ~110–168.
- `collate_object_event_v4`: ~174–256.
- campos SAM opcionales actuales: ~34–35 y ~219–255.
- `box_geometry_targets`: ~292 en adelante.

Regla fundamental:

```python
def event_inputs(self) -> dict[str, torch.Tensor]:
    return {
        "events": self.events,
        "delta_t_s": self.delta_t_s,
    }
```

A4 **NO puede modificar este contrato**.

Los targets DINO deben existir como campos auxiliares del batch igual que SAM, pero nunca deben aparecer en `event_inputs()` o `model_inputs()`.

---

## 2.6 Runner del screen

Archivo:

```text
scripts/train_causal_scale_eap_screen.py
```

Zonas:

- lectura y congelación del protocolo: ~160–205.
- check parámetros: ~206–215.
- creación datasets train/validation: ~218–223.
- wrapper SAM train-only existente: ~224–244.
- entrenamiento: ~247–256.
- summary/metadata: ~279–352.
- `model_input_contract`: ~306–326.

A4 debe seguir este patrón:

```text
base train dataset
    ↓
DINOv3RelationalTeacherDataset(train only)

validation dataset
    ↓
NO WRAPPER
```

---

## 2.7 Patrón de cache teacher existente

Archivo:

```text
src/e_jepa_ttc/data/sam_teacher_cache.py
```

Usar como referencia de diseño, especialmente ~24–124:

- hash físico del manifest;
- `verify_artifact_hash`;
- identidad `artifact_sha256`;
- scope train-only;
- verificación de shards;
- token exacto;
- sequence exacta;
- common crop exacto;
- fail closed ante duplicados;
- mismo número de filas que dataset train;
- `shard_index_groups` delegado.

A4 debe copiar **la filosofía de seguridad**, no necesariamente literalmente el código.

---

## 2.8 Lector RGB existente

Archivo:

```text
scripts/audit_sam_train_bbox_prompts.py
```

Función:

```python
_read_rgb_member(...)
```

~líneas 64–75.

Lee un miembro RGB exacto dentro de TAR y devuelve también SHA256 de los bytes.

A4 puede:

- refactorizar esta función a un módulo común si el cambio es limpio y tests lo cubren; o
- copiar una versión mínima y determinista al nuevo materializador.

No hacer una refactorización amplia del subsistema SAM solo para ahorrar diez líneas.

---

## 2.9 Dependencia Transformers

Archivo:

```text
pyproject.toml
```

El extra `multimodal` ya contiene:

```toml
transformers>=4.56,<6
```

Por tanto:

- no añadir una segunda dependencia `transformers`;
- no cambiar el rango salvo incompatibilidad demostrada;
- usar `uv sync --extra multimodal --all-groups --locked`.

---

# 3. Referencias científicas que justifican el diseño

El agente no debe “reinterpretar” estas referencias como permiso para copiar arquitecturas completas.

## DINOv3

Referencia primaria:

```text
Meta / facebookresearch/dinov3
paper: arXiv 2508.10104
teacher científico:
facebook/dinov3-convnext-large-pretrain-lvd1689m
```

Motivo de uso:

- DINOv3 está diseñado para producir **dense features** de alta calidad.
- Usamos ConvNeXt-Large como teacher porque la salida convolucional densa encaja naturalmente con un student CNN espacial.
- El teacher desaparece en inference.

## ScaleEvent

Referencia primaria:

```text
arXiv 2603.03969
repo de autores: zhiwen-xdu/ScaleEvent
```

Idea que reutilizamos:

- la distancia de modalidad RGB ↔ eventos hace peligrosa una alineación ingenua feature-a-feature;
- es preferible transferir **estructura** / relaciones densas.

A4 **NO afirma ser una reproducción de ScaleEvent**.

## Structured / relational KD

La idea general de transferir similitudes locales o estructuras de features en lugar de canales exactos es anterior y bien establecida.

Nuestra implementación concreta se mantiene deliberadamente pequeña: seis relaciones locales de coseno por posición.

## V-JEPA 2.1

Referencia futura, NO implementada en A4:

```text
arXiv 2603.14482
```

Solo justifica la rama A5 si A4 demuestra que la geometría por endpoint mejora pero la dinámica sigue mala.

---

# 4. Arquitectura exacta A4

## 4.1 Student — exactamente A1-DF

```text
events [B,3,12,128,128]
          │
          ▼
flatten endpoints
[B*3,12,128,128]
          │
          ▼
_EndpointEncoder.features
          │
          ▼
F_event [B*3,64,32,32]
          │
          ├─────────────────────────────────────┐
          │                                     │
          ▼                                     ▼
resize_conv foreground                   A4 relational loss
          │                             (train only, t1/t2)
          ▼
foreground logits
          │
          ▼
soft extent
h,w,cx,cy
          │
          ▼
Causal Scale + residual + consensus
          │
          ▼
signed TTC
```

Ningún bloque nuevo con parámetros debe entrar en el student.

---

## 4.2 Teacher offline

```text
RGB sincronizado t1/t2
        │
same event_v4_common_square crop
        │
resize 256x256 preserving whole crop
        │
DINOv3 ConvNeXt-Large frozen
        │
intermediate feature map 32x32
        │
L2 channel normalization
        │
six local cosine relation maps
        │
cache train-only
```

Nunca cachear TTC, bbox target, mask SAM o predictions TTC dentro del teacher cache.

La bbox no se usa para DINO.

El **common-square crop sí se usa**, porque es el marco observable del modelo de eventos y no es un input aprendido; forma parte de la definición del dato alineado.

---

# 5. Elección exacta del teacher científico

Modelo:

```text
facebook/dinov3-convnext-large-pretrain-lvd1689m
```

Tiny:

```text
facebook/dinov3-convnext-tiny-pretrain-lvd1689m
```

se usa **solo** para smoke/debug de la infraestructura.

No ejecutar A4 científica con Tiny y después Large sobre validation.

No usar Base como brazo intermedio.

No hacer un sweep de teachers.

## Verificación local obligatoria

Antes del cache Large:

```powershell
hf cache ls | Select-String "dinov3"
```

Debe existir Large.

Después:

```powershell
uv sync --extra multimodal --all-groups --locked
```

Y smoke offline:

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
uv run python -c "from transformers import AutoModel,AutoImageProcessor; m='facebook/dinov3-convnext-large-pretrain-lvd1689m'; AutoImageProcessor.from_pretrained(m,local_files_only=True); AutoModel.from_pretrained(m,local_files_only=True); print('OK')"
```

Si el model ID no resuelve offline por cómo Hugging Face almacena la snapshot, el script debe aceptar también `--model-path` físico y registrar:

- repo ID declarado;
- snapshot/path resuelto;
- revision;
- SHA256 `model.safetensors`;
- SHA256 config;
- SHA256 preprocessor config.

No descargar silenciosamente nada durante un run científico.

---

# 6. Representación teacher: no elegir a mano una capa mirando validation

El materializador debe ejecutar DINO con:

```python
output_hidden_states=True
```

Sobre input 256×256.

El código debe identificar una feature map espacial con resolución exactamente:

```text
32 x 32
```

No se debe seleccionar una capa porque “parece mejor” en validation.

Algoritmo:

```python
candidates = []
for index, hidden in enumerate(outputs.hidden_states):
    if hidden.ndim == 4 and hidden.shape[-2:] == (32, 32):
        candidates.append((index, hidden))

if len(candidates) != 1:
    raise RuntimeError(...)
```

Guardar en manifest:

```json
{
  "teacher_hidden_state_index": 2,
  "teacher_hidden_state_shape": [C, 32, 32]
}
```

El número `2` de arriba es **solo ejemplo**, NO hard-code.

Si Transformers no devuelve una única feature 32×32:

1. parar;
2. no interpolar una feature final 8×8/16×16 a 32×32 para salvar el pipeline;
3. implementar un forward hook sobre el stage ConvNeXt cuyo stride total sea 8;
4. unit-testear forma;
5. registrar el nombre exacto del módulo en el manifest.

La resolución 32×32 se elige **a priori** porque coincide con el student A1-DF después de dos strides ×2.

---

# 7. Preprocesamiento RGB exacto

Ésta es una zona de alto riesgo.

El student recibe eventos en el:

```text
event_v4_common_square_xyxy
```

Por tanto el teacher debe ver **exactamente el mismo campo de visión**.

Por cada token train y endpoint t1/t2:

1. resolver RGB original correspondiente;
2. leer bytes desde TAR;
3. convertir a RGB;
4. obtener `event_v4_common_square_xyxy`;
5. aplicar el crop common-square en coordenadas de la imagen RGB original;
6. clamp al frame;
7. verificar ancho/alto positivos;
8. resize de todo el crop a 256×256;
9. **NO center crop**;
10. **NO random crop**;
11. **NO horizontal flip**;
12. **NO color jitter**;
13. aplicar solo normalización requerida por DINO.

No delegar ciegamente en un `AutoImageProcessor` que pueda introducir center-crop/resizing adicional.

La implementación debe leer del processor/config los mean/std y usar una transformación explícita reproducible.

Ejemplo conceptual:

```python
image = read_rgb(...)
crop = image.crop((x1, y1, x2, y2))
crop = crop.resize((256, 256), Image.Resampling.BILINEAR)

array = np.asarray(crop, dtype=np.float32) / 255.0
tensor = torch.from_numpy(array).permute(2, 0, 1)

mean = torch.tensor(image_mean)[:, None, None]
std = torch.tensor(image_std)[:, None, None]
pixel_values = (tensor - mean) / std
```

El script debe registrar `teacher_input_size=256`.

---

# 8. Qué RGB endpoints usar

A4 usa únicamente:

```text
t1 y t2
```

No t0.

Motivo:

- son los endpoints con geometría real usable del screen;
- son los endpoints que generan el ratio actual;
- el teacher cache SAM ya se materializó sobre dos endpoints;
- evitamos introducir otra decisión sobre el proxy t0.

El student sigue procesando t0/t1/t2 porque A1-DF lo hacía.

La distillation se aplica a:

```python
student_features[:, 1:3]
```

después de reordenar dense features a `[B,T,C,H,W]`.

---

# 9. Loss relacional exacta

## 9.1 Offsets congelados

Usar exactamente estos seis offsets:

```python
A4_RELATION_OFFSETS = (
    (0, 1),
    (1, 0),
    (0, 2),
    (2, 0),
    (1, 1),
    (1, -1),
)
```

Interpretación:

- horizontal 1;
- vertical 1;
- horizontal 2;
- vertical 2;
- diagonal descendente;
- diagonal contraria.

No añadir radius 4, global attention, multi-scale relations ni learnable offsets en A4.

---

## 9.2 Función común student/teacher

Crear módulo:

```text
src/e_jepa_ttc/distillation/dinov3_relational.py
```

API sugerida:

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional


A4_RELATION_OFFSETS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 0),
    (0, 2),
    (2, 0),
    (1, 1),
    (1, -1),
)


@dataclass(frozen=True)
class LocalRelationMaps:
    values: torch.Tensor
    valid: torch.Tensor


def local_cosine_relation_maps(
    features: torch.Tensor,
    *,
    offsets: tuple[tuple[int, int], ...] = A4_RELATION_OFFSETS,
    eps: float = 1.0e-6,
) -> LocalRelationMaps:
    ...
```

Input:

```text
[..., C, H, W]
```

Output:

```text
values [..., K, H, W]
valid  [..., K, H, W] bool
```

Pasos:

```python
normalized = functional.normalize(features.float(), dim=-3, eps=eps)
```

Para cada `(dy,dx)`:

- calcular coordenadas fuente/destino válidas;
- coseno = sum de producto por canales;
- escribir en el mapa origen;
- zonas fuera de bounds = 0;
- `valid=False` fuera de bounds.

No usar `roll`, porque introduciría wrap-around silencioso.

---

## 9.3 Loss

API:

```python
def local_relational_distillation_loss(
    student_features: torch.Tensor,
    teacher_relations: torch.Tensor,
    teacher_valid: torch.Tensor,
    *,
    offsets: tuple[tuple[int, int], ...] = A4_RELATION_OFFSETS,
) -> torch.Tensor:
    ...
```

Student expected:

```text
[B,2,64,32,32]
```

Teacher:

```text
[B,2,6,32,32]
```

Valid:

```text
[B,2,6,32,32]
```

Calcular student relations online.

Valid final:

```python
valid = teacher_valid & student_relations.valid
```

Pérdida:

```python
error = (student_relations.values - teacher_relations.float()).abs()
loss = error[valid].mean()
```

Usar **L1**, no MSE.

Checks:

- hay al menos un elemento válido;
- teacher finito donde valid;
- student finito;
- shapes exactas;
- offsets exactos;
- no NaN.

El cálculo de la relation loss debe hacerse en `float32`, incluso bajo BF16 autocast, para estabilidad del cosine.

---

# 10. Qué se cachea

No cachear DINO raw features.

Cachear únicamente:

```text
relation_targets
relation_valid
sample_tokens
sequence_ids
common_square_xyxy
rgb endpoint hashes
```

Propuesta NPZ por shard:

```python
sample_tokens:            <U... [N]
sequence_ids:             <U... [N]
common_square_xyxy:       float32 [N,4]
relation_targets:         float16 [N,2,6,32,32]
relation_valid:           uint8   [N,2,6,32,32]
rgb_sha256:               uint8/string [N,2]
```

`relation_valid` puede ser bool dentro del runtime, pero NPZ puede guardarse uint8.

Tamaño aproximado:

```text
2048 * 2 * 6 * 32 * 32 * 2 bytes
≈ 50 MB
```

más masks y metadata.

Mucho mejor que cachear cientos de canales DINO por endpoint.

---

# 11. Sharding

Usar:

```text
32 shards x 64 rows
```

nombres:

```text
shard_000.npz
...
shard_031.npz
```

Cada shard:

1. se escribe a temporal;
2. se `fsync` si la infraestructura existente lo hace;
3. `os.replace` atómico;
4. SHA256;
5. manifest registra:
   - path;
   - row start/end;
   - token first/last;
   - sha256;
   - row count.

Resume de materialización:

- si shard existe y hash esperado en estado parcial coincide, reutilizar;
- si no coincide, fallar o recomputar de manera explícita;
- no mezclar outputs Large/Tiny.

---

# 12. Manifest científico teacher

Ruta:

```text
artifacts/cache/dinov3_convnext_large_relational_a4_v1/manifest.json
```

Debe estar firmado con el mismo sistema `sign_artifact`.

Schema conceptual mínimo:

```json
{
  "artifact_type": "dinov3_relational_teacher_cache_v1",
  "status": "passed",
  "teacher": {
    "model_id": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
    "weights_sha256": "...",
    "config_sha256": "...",
    "preprocessor_sha256": "...",
    "input_size": 256,
    "hidden_state_index": 0,
    "hidden_state_shape": [0, 32, 32]
  },
  "relations": {
    "type": "local_cosine",
    "offsets_dy_dx": [[0,1],[1,0],[0,2],[2,0],[1,1],[1,-1]],
    "grid_height": 32,
    "grid_width": 32,
    "dtype": "float16"
  },
  "scope": {
    "public_train_only": true,
    "row_count": 2048,
    "endpoint_count_per_row": 2,
    "validation_or_test_opened": false,
    "ttc_labels_read": false
  },
  "claim_boundary": {
    "teacher_is_model_input": false,
    "validation_teacher_generation": false,
    "ttc_labels_read": false,
    "sam_used": false,
    "bbox_prompt_used": false
  },
  "source_event_cache": {
    "manifest_sha256": "...",
    "artifact_sha256": "..."
  },
  "shards": [],
  "artifact_sha256": "..."
}
```

No copiar `hidden_state_index: 0` o channel `0`; son placeholders que el materializador debe reemplazar con valores observados y validados.

---

# 13. Cómo seleccionar exactamente los 2.048 train tokens

El teacher cache debe corresponder **1:1** al dataset A1-DF train.

No seleccionar directamente “las primeras 2048 filas de train.parquet”.

Procedimiento correcto:

1. abrir `artifacts/cache/garl_object_event_common_roi_screen_v4/manifest.json`;
2. cargar `GarlTTCObjectEventV4Dataset(..., splits=("train",))`;
3. enumerar sus 2048 `sample_token`, `sequence_id`, `event_v4_common_square_xyxy`;
4. crear tabla canonical de 2048 tokens;
5. hacer join exacto contra `E:\GarlTTC_dataset\data\train.parquet` para obtener referencias RGB;
6. exigir `validate="one_to_one"`;
7. exigir 2048 matched;
8. exigir cero duplicados;
9. exigir las nueve train sequences exactas;
10. rechazar cualquier token cuyo sequence esté en:
   - `DGqicHUGWb`
   - `pBqGOb2vYq`
   - `qoohcdtLDH`

Esto es mucho más seguro que depender del orden original del parquet.

El materializador puede leer únicamente columnas:

```text
sequence_id
sample_token
rgb_shard_paths
rgb_member_paths
```

y, si hace falta para alineamiento del crop, cualquier metadata geométrica **no TTC**.

**NO leer columna TTC** en el script teacher.

El manifest debe declarar explícitamente:

```json
"ttc_labels_read": false
```

---

# 14. Script nuevo: audit/smoke DINO

Crear:

```text
scripts/audit_dinov3_dense_teacher.py
```

Objetivo:

- smoke de infraestructura;
- usar Tiny por defecto;
- solo train;
- 1–8 tokens;
- verificar crop/alineamiento;
- listar hidden state shapes;
- construir relation maps;
- escribir artefacto diagnóstico no científico.

CLI sugerida:

```powershell
uv run python scripts/audit_dinov3_dense_teacher.py `
  --event-cache-manifest artifacts/cache/garl_object_event_common_roi_screen_v4/manifest.json `
  --train-parquet E:\GarlTTC_dataset\data\train.parquet `
  --eap-root E:\eAP_dataset `
  --model-id facebook/dinov3-convnext-tiny-pretrain-lvd1689m `
  --device cuda:0 `
  --sample-count 8 `
  --output artifacts/metrics/dinov3_dense_teacher_smoke_a4_v1.json
```

Smoke debe comprobar:

- CUDA disponible;
- BF16 viable;
- no test path;
- no validation sequence;
- shape 32×32 encontrada;
- relation values en [-1.0001,1.0001];
- al menos cierta varianza espacial > epsilon;
- todos los hashes guardados;
- no TTC leído.

**No usar smoke para elegir hiperparámetros.**

---

# 15. Script nuevo: materializador Large

Crear:

```text
scripts/materialize_dinov3_relational_teacher.py
```

CLI sugerida:

```powershell
uv run python scripts/materialize_dinov3_relational_teacher.py `
  --event-cache-manifest artifacts/cache/garl_object_event_common_roi_screen_v4/manifest.json `
  --train-parquet E:\GarlTTC_dataset\data\train.parquet `
  --eap-root E:\eAP_dataset `
  --model-id facebook/dinov3-convnext-large-pretrain-lvd1689m `
  --device cuda:0 `
  --precision bf16 `
  --output-dir artifacts/cache/dinov3_convnext_large_relational_a4_v1
```

Opciones permitidas:

```text
--resume
```

No permitir:

```text
--split validation
--test
--ttc-column
--bbox-prompt
--sam
```

El script debe hard-fail si:

- parquet filename no es `train.parquet`;
- path contiene componente `test`;
- aparece validation sequence;
- número final != 2048;
- endpoint count != 4096;
- DINO model hash no coincide con config congelado;
- hidden 32×32 no único;
- relation map no finito;
- shard hash incorrecto;
- git code dirty si se pretende crear el cache científico final.

---

# 16. Dataset wrapper nuevo

Crear:

```text
src/e_jepa_ttc/data/dinov3_relational_teacher_cache.py
```

Clase:

```python
class DINOv3RelationalTeacherDataset(Dataset[dict[str, Any]]):
    ...
```

Inspirarse en:

```text
SAMTeacherMaskDataset
src/e_jepa_ttc/data/sam_teacher_cache.py
```

Debe verificar en `__init__`:

- manifest physical SHA;
- `verify_artifact_hash`;
- artifact identity;
- status passed;
- public_train_only;
- validation/test false;
- TTC labels false;
- model ID Large exacto;
- relation type exacta;
- offsets exactos;
- grid 32×32;
- 2048 rows;
- 2 endpoints;
- todas las shard hashes;
- shape arrays;
- finitud;
- duplicados;
- mismo número de tokens que base dataset.

En `__getitem__`:

```python
record = dict(self.dataset[index])
token = str(record["sample_token"])
...
```

Validar:

- token;
- sequence;
- common crop con `atol=1e-4`;
- relation shape;
- valid shape.

Añadir:

```python
record["dinov3_relation_targets"] = torch.from_numpy(...)
record["dinov3_relation_valid"] = torch.from_numpy(...)
```

No añadir DINO raw feature.

`shard_index_groups()` debe delegar al base dataset para no romper sampling determinista.

---

# 17. Cambios a ObjectEventV4Batch

En:

```text
src/e_jepa_ttc/data/object_event_v4.py
```

Añadir tras SAM:

```python
dinov3_relation_targets: torch.Tensor | None = None
dinov3_relation_valid: torch.Tensor | None = None
```

En `.to()`:

```python
dinov3_relation_targets=(
    self.dinov3_relation_targets.to(
        device=device,
        dtype=torch.float32,
        non_blocking=non_blocking,
    )
    if self.dinov3_relation_targets is not None
    else None
),
dinov3_relation_valid=(
    self.dinov3_relation_valid.to(
        device=device,
        dtype=torch.bool,
        non_blocking=non_blocking,
    )
    if self.dinov3_relation_valid is not None
    else None
),
```

No modificar:

```python
event_inputs()
```

En collate:

1. comprobar que targets y valid aparecen juntos;
2. si alguno de los records tiene campos, todos deben tenerlos;
3. stack;
4. shape:

```text
[B,2,K,H,W]
```

5. target y valid deben compartir shape;
6. `K=6`, `H=W=32` puede verificarse aquí o en wrapper; preferible:
   - collate: consistencia genérica;
   - wrapper: contrato A4 exacto.

No permitir mixed teacher presence dentro del mismo batch.

---

# 18. Exponer dense student features sin alterar inferencia

Éste debe ser un cambio quirúrgico.

## Opción recomendada

En `CausalScaleTTCOutput` añadir al final:

```python
endpoint_dense_features: torch.Tensor | None = None
```

Al final para mantener compatibilidad con dataclass defaults.

Cambiar `_EndpointEncoder.forward`:

```python
def forward(
    self,
    values: torch.Tensor,
    *,
    return_dense_features: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    features = self.features(values)
    foreground_input = values if self.foreground_from_input else features
    foreground = self.foreground(foreground_input)
    token = self.token(features)
    return foreground, token, features if return_dense_features else None
```

Cambiar `CausalScaleTTC.forward`:

```python
def forward(
    self,
    inputs: torch.Tensor,
    delta_t_s: torch.Tensor,
    *,
    return_dense_features: bool = False,
) -> CausalScaleTTCOutput:
```

Y:

```python
lowres_logits, base_tokens, flat_dense = self.encoder(
    flat,
    return_dense_features=return_dense_features,
)
```

Antes de output:

```python
endpoint_dense_features = (
    flat_dense.reshape(batch, steps, flat_dense.shape[1], flat_dense.shape[2], flat_dense.shape[3])
    if flat_dense is not None
    else None
)
```

`CausalScaleTTCOutput(... endpoint_dense_features=endpoint_dense_features)`.

## Regla crucial

Con:

```python
return_dense_features=False
```

las predicciones deben ser numéricamente idénticas a la versión padre.

El flag **no añade parámetros**.

No guardar dense features en checkpoints.

No usarlas en validation.

---

# 19. Config training A4

En:

```text
src/e_jepa_ttc/training/causal_scale_eap.py
```

añadir a `CausalScaleEAPTrainingConfig`:

```python
representation_supervision: Literal[
    "none",
    "dinov3_local_relational",
] = "none"

representation_teacher_cache_artifact_sha256: str | None = None
representation_distillation_weight: float = 0.0
```

`__post_init__`:

```python
if self.representation_supervision == "none":
    if self.representation_teacher_cache_artifact_sha256 is not None:
        raise ValueError(...)
    if self.representation_distillation_weight != 0.0:
        raise ValueError(...)

elif self.representation_supervision == "dinov3_local_relational":
    if not self.representation_teacher_cache_artifact_sha256:
        raise ValueError(...)
    if (
        not math.isfinite(self.representation_distillation_weight)
        or self.representation_distillation_weight <= 0.0
    ):
        raise ValueError(...)
else:
    raise ValueError(...)
```

Mantener:

```python
foreground_supervision == "bbox_geometry"
```

A4 no crea un nuevo foreground supervision string.

---

# 20. Integración de la representation loss en `_loss`

Extender firma:

```python
def _loss(
    ...,
    representation_supervision: Literal["none", "dinov3_local_relational"],
    representation_distillation_weight: float,
) -> tuple[...]:
```

Para A4:

```python
need_dense = representation_supervision == "dinov3_local_relational"

output = model(
    batch.events,
    targets.delta_t_s,
    return_dense_features=need_dense,
)
```

Después de `causal_scale_ttc_loss`:

```python
total = result.total
components = dict(result.components)

if representation_supervision == "dinov3_local_relational":
    if output.endpoint_dense_features is None:
        raise RuntimeError(...)
    if batch.dinov3_relation_targets is None or batch.dinov3_relation_valid is None:
        raise RuntimeError(...)

    student = output.endpoint_dense_features[:, 1:3].float()

    relation_loss = local_relational_distillation_loss(
        student,
        batch.dinov3_relation_targets,
        batch.dinov3_relation_valid,
    )

    total = total + representation_distillation_weight * relation_loss

    components["dinov3_relational_raw"] = relation_loss.detach()
    components["dinov3_relational_weighted"] = (
        representation_distillation_weight * relation_loss.detach()
    )

return total, components, output
```

No mutar `CausalScaleTTCLossResult`.

---

# 21. Warm-up: la DINO loss SÍ permanece activa

A1-DF apaga objetivos TTC/temporal durante las tres primeras épocas con:

```text
_foreground_only_loss_config()
```

A4 conserva exactamente esa función.

La distillation de representación vive fuera del `loss_config`, por lo que:

```text
epoch 1-3:
geometry loss + DINO relational

epoch 4+:
full A1-DF loss + DINO relational
```

Esto es intencional.

No apagar DINO durante warm-up.

No activar pair-ratio durante warm-up ni después.

---

# 22. Validation debe estar físicamente libre de teacher

En:

```python
evaluate_real_causal_scale(...)
```

añadir guard explícito al principio de cada batch:

```python
if batch.dinov3_relation_targets is not None or batch.dinov3_relation_valid is not None:
    raise RuntimeError("validation batch unexpectedly contains DINO teacher targets")
```

Y llamar:

```python
model(..., return_dense_features=False)
```

No generar features densas para validation si no se necesitan.

El validation dataset no se envuelve.

No cargar manifest DINO en un validation worker.

---

# 23. Resume

La infraestructura actual compara:

```python
asdict(training_config)
```

en el resume contract.

Como A4 añade:

- representation mode;
- artifact SHA;
- distillation weight;

estos valores quedarán automáticamente ligados al resume.

Tests obligatorios:

- cambiar artifact hash → resume falla;
- cambiar distill weight → resume falla;
- cambiar mode → resume falla.

Además, si el runner tiene metadata teacher separada, verificar también que:

```text
training_config.representation_teacher_cache_artifact_sha256
==
data.dinov3_relational_teacher.artifact_sha256
```

---

# 24. Runner A4

En:

```text
scripts/train_causal_scale_eap_screen.py
```

importar:

```python
from e_jepa_ttc.data.dinov3_relational_teacher_cache import (
    DINOv3RelationalTeacherDataset,
)
```

Después de crear base datasets:

```python
train_dataset = GarlTTCObjectEventV4Dataset(...)
validation_dataset = GarlTTCObjectEventV4Dataset(...)
```

añadir:

```python
representation_teacher_metadata = None

if training_config.representation_supervision == "dinov3_local_relational":
    teacher = data.get("dinov3_relational_teacher")
    if not isinstance(teacher, dict):
        raise ValueError("A4 requires data.dinov3_relational_teacher")

    teacher_manifest = _resolve(teacher["manifest"])

    train_dataset = DINOv3RelationalTeacherDataset(
        train_dataset,
        manifest_path=teacher_manifest,
        expected_artifact_sha256=str(teacher["artifact_sha256"]),
        expected_manifest_sha256=str(teacher["manifest_sha256"]),
    )

    if (
        training_config.representation_teacher_cache_artifact_sha256
        != teacher["artifact_sha256"]
    ):
        raise ValueError("training and representation teacher identities differ")

    representation_teacher_metadata = {
        ...
        "scope": "public_train_only",
        "validation_teacher_loaded": False,
    }
```

**No tocar validation_dataset.**

---

# 25. Metadata A4

En `summary.json` final, añadir:

```json
"model_input_contract": {
  "forward_inputs": ["event_v4_common_roi", "garl_delta_t_s"],
  "bbox_is_model_input": false,
  "sam_teacher_is_model_input": false,
  "dinov3_teacher_is_model_input": false,
  "dinov3_teacher_train_only": true,
  "dinov3_teacher_validation_loaded": false,
  "rgb_is_model_input": false
}
```

Y:

```json
"representation_teacher": {
  "mode": "dinov3_local_relational",
  "model_id": "...",
  "manifest": "...",
  "manifest_sha256": "...",
  "artifact_sha256": "...",
  "distillation_weight": 0.0,
  "offsets": [[0,1],[1,0],[0,2],[2,0],[1,1],[1,-1]],
  "grid_shape": [32,32]
}
```

`distillation_weight` de arriba es placeholder.

Debe escribirse el valor real congelado.

---

# 26. Calibración del peso sin mirar validation

No usar un sweep arbitrario.

Crear:

```text
scripts/calibrate_a4_dinov3_relational_weight.py
```

Usar:

- seed 7;
- parent A1-DF model init;
- 64 muestras train deterministas;
- cero optimizer steps;
- cero validation;
- teacher cache Large final.

Selección de 64 muestras:

```python
indices = np.linspace(0, len(train_dataset) - 1, 64, dtype=np.int64)
```

o selección por sample_token equiespaciada determinista.

El artifact debe registrar los 64 tokens.

Para cada mini-batch:

1. calcular active A1-DF geometry warmup loss;
2. calcular raw DINO relation loss;
3. acumular valores por batch.

Definir:

```python
median_geometry = median(geometry_losses)
median_relation = median(relation_losses)

lambda_raw = median_geometry / median_relation

lambda_d = clip(lambda_raw, 0.25, 4.0)
```

Guardar:

```text
artifacts/metrics/a4_dinov3_relational_weight_calibration_v1.json
```

firmado.

Debe contener:

```json
{
  "scope": "public_train_only",
  "validation_opened": false,
  "optimizer_steps": 0,
  "seed": 7,
  "sample_count": 64,
  "sample_tokens": [],
  "sample_token_sha256": "...",
  "median_geometry_loss": 0.0,
  "median_relational_loss": 0.0,
  "ratio_before_clip": 0.0,
  "clamp": [0.25, 4.0],
  "selected_weight": 0.0
}
```

El agente no debe variar la fórmula si el resultado “parece pequeño/grande”.

---

# 27. Config científico A4 final

Crear:

```text
configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a4_dinov3_relational_v1.yaml
```

Debe ser clon exacto de A1-DF salvo campos A4 explícitos.

Template:

```yaml
experiment:
  name: e_jepa_garl_event_causal_scale_eap_screen_a4_dinov3_relational_v1
  protocol_version: causal_scale_eap_a4_dinov3_relational_v1
  evidence_scope: public_eap_garl_train_validation_only
  parent_arm: A1_DF_seed7
  single_scientific_difference: train_only_dinov3_convnext_large_local_relational_distillation

model_config: configs/model/e_jepa_causal_scale_event_v8_t015_resize_conv.yaml

data:
  cache_manifest: artifacts/cache/garl_object_event_common_roi_screen_v4/manifest.json
  cache_manifest_sha256: bba9ff9b143bfd57760bd61d2b6f664202581b5dee54444f44c625975557eb72
  cache_artifact_sha256: 36c12d75c91a243f4d712831cebcd3e82f896a76196b2a65b039e680f1fac309

  train_sequence_ids:
    - 2cyv0Oedzg
    - 5ilM1PX2vz
    - 6h5yRW2LGc
    - OBneIVg4Cw
    - OYgB6RGWcq
    - WbCh1DRerJ
    - mHGFBekt7X
    - qGsgzl4Q8B
    - t79dBxj1WS

  validation_sequence_ids: [DGqicHUGWb, pBqGOb2vYq, qoohcdtLDH]

  opened_splits: [train, validation]
  official_test_opened: false
  codabench_opened: false
  evttc_test_opened: false

  dinov3_relational_teacher:
    manifest: artifacts/cache/dinov3_convnext_large_relational_a4_v1/manifest.json
    manifest_sha256: REPLACE_AFTER_MATERIALIZATION
    artifact_sha256: REPLACE_AFTER_MATERIALIZATION
    model_id: facebook/dinov3-convnext-large-pretrain-lvd1689m

training:
  seed: 7
  epochs: 18
  minimum_epochs: 8
  early_stopping_patience: 5
  foreground_warmup_epochs: 3
  batch_size: 32
  gradient_accumulation_steps: 1
  learning_rate: 0.0003
  minimum_learning_rate: 0.00003
  weight_decay: 0.0001
  grad_clip_norm: 1.0
  num_workers: 0
  precision: bf16
  maximum_runtime_hours: 6.0
  mask_t0_as_proxy: true
  foreground_supervision: bbox_geometry

  representation_supervision: dinov3_local_relational
  representation_teacher_cache_artifact_sha256: REPLACE_AFTER_MATERIALIZATION
  representation_distillation_weight: REPLACE_AFTER_TRAIN_ONLY_CALIBRATION

loss:
  log_ratio_nll_weight: 1.0
  log_ratio_huber_weight: 0.0
  log_ratio_tail_weight: 2.0
  log_ratio_tail_fraction: 0.10
  foreground_bce_weight: 0.0
  foreground_dice_weight: 0.0
  foreground_extent_weight: 1.25
  foreground_width_weight: 1.25
  foreground_center_weight: 2.5
  foreground_pair_ratio_weight: 0.0
  risk_weight: 0.1
  auxiliary_inverse_ttc_weight: 0.05
  residual_regularization_weight: 0.1
  temporal_consistency_weight: 0.0
  smooth_l1_beta: 0.02
  supervise_pair_ratio_before_temporal_blend: true

decision_contract:
  checkpoint_selection: validation_sequence_macro_MiD_then_failure_rate
  require_finite_metrics_for_all_validation_sequences: true

  primary_baseline: Garl_event_only_matched_seed7_same_2048
  baseline_sequence_macro_MiD: 203.6341709373319
  baseline_failure_rate_pct: 0.0

  parent_arm: A1_DF_seed7
  parent_sequence_macro_MiD: 350.3020
  parent_failure_rate_pct: 21.0938
  parent_log_ratio_pearson: 0.1865
  parent_absolute_height_pearson: 0.4823
  parent_absolute_width_pearson: 0.2428
  parent_delta_height_physical_pearson: 0.1704

  model_config_sha256: 265dbfd57e68d7a6aa385fbf31dc0ad41154b17afbd1d9454bbd8ddd80c6663f
  expected_parameter_count: 355118

  model_architecture_must_equal_a1_df: true
  model_parameter_count_must_equal_a1_df: true
  bbox_is_training_supervision_only: true
  bbox_is_never_forward_input: true
  foreground_pair_ratio_weight_must_remain_zero: true
  sam_teacher_must_be_disabled: true
  dino_teacher_is_training_only: true
  dino_teacher_is_never_forward_input: true
  dino_teacher_model_id: facebook/dinov3-convnext-large-pretrain-lvd1689m
  dino_teacher_relation_type: local_cosine
  dino_teacher_grid_shape: [32, 32]
  dino_teacher_offsets_dy_dx:
    - [0, 1]
    - [1, 0]
    - [0, 2]
    - [2, 0]
    - [1, 1]
    - [1, -1]

  jepa_objective_must_be_disabled: true
  unknown_threshold_must_equal_parent: true
  ttc_clip_must_equal_parent: true
  temporal_consensus_must_equal_parent: true
  residual_must_equal_parent: true
  cvar_must_equal_parent: true

  representation_gate:
    absolute_height_pearson_min_gain_vs_a1_df: 0.05
    absolute_width_pearson_min_gain_vs_a1_df: 0.05

  representation_gate_is_mechanism_gate_not_benchmark_promotion_gate: true
  temporal_metrics_are_diagnostic_for_next_branch: true
  r_iso_is_diagnostic_only: true
  diagnostic_reporting: global_and_macro_by_sequence

  no_posthoc_diagnostic_threshold_is_a_sota_gate: true
  public_validation_does_not_authorize_sota_claim: true
  private_test_remains_closed: true
```

Los valores redondeados del parent pueden sustituirse por exactos desde los artifacts existentes si están disponibles.

---

# 28. Gate mecanístico preregistrado

A4 no se juzga primero por MiD.

Pregunta A4:

> ¿La distillation relacional mejora realmente la **representación geométrica espacial** del event encoder?

Gate mecanístico:

```text
absolute log-height Pearson:
A4 - A1-DF >= +0.05

Y

absolute log-width Pearson:
A4 - A1-DF >= +0.05
```

Anclas aproximadas A1-DF:

```text
height ≈ .4823
width  ≈ .2428
```

Por tanto objetivos aproximados:

```text
height >= .5323
width  >= .2928
```

Estos NO son claims SOTA ni gates eAP oficiales.

Son un criterio de mecanismo fijado **antes** de A4.

---

# 29. Métricas que deben seguir reportándose

El trainer ya reporta gran parte de ellas.

No eliminar nada.

Obligatorio:

## Static geometry

```text
absolute_log_height
absolute_log_width
centroid_x
centroid_y
```

Cada una:

- global Pearson;
- global slope;
- MAE;
- sign si aplicable;
- `std_pred/std_target`;
- macro por secuencia;
- per-sequence.

## Temporal geometry

```text
delta_log_height_vs_bbox
delta_log_width_vs_bbox
isotropic_ratio_vs_bbox
delta_log_height_vs_physical
delta_log_width_vs_physical
isotropic_ratio_vs_physical
```

## Physical output

- log-ratio Pearson;
- log-ratio slope si existe;
- MiD;
- failure;
- known fraction;
- unknown cause counts;
- clipped ±60;
- buckets:
  - crucial;
  - small;
  - large;
  - negative;
- per-sequence;
- signed accuracy;
- balance si disponible.

## A4-specific train metrics

```text
dinov3_relational_raw
dinov3_relational_weighted
```

Nunca calcular estas dos en validation.

---

# 30. Tests unitarios exactos

Crear al menos:

```text
tests/unit/test_dinov3_relational_distillation.py
tests/unit/test_dinov3_relational_teacher_cache.py
tests/unit/test_causal_scale_a4_dino_teacher.py
```

## 30.1 local relation maps

Tests:

### Identidad

Si student y teacher relation maps derivan del mismo feature tensor:

```python
loss == 0
```

con tolerancia float.

### Invariancia a escala positiva por feature vector

Si:

```python
features2 = features * 3.7
```

el cosine relation debe ser igual.

### Offset horizontal

Crear features sintéticas donde la similitud entre vecinos horizontales sea conocida.

### Offset diagonal negativo

Comprobar `(1,-1)` explícitamente.

### Border mask

Ninguna posición fuera de bounds debe ser valid.

### No wraparound

Un valor extremo del borde derecho no puede relacionarse con borde izquierdo.

### Gradientes

```python
student.requires_grad_(True)
loss.backward()
assert finite grads
assert some grad != 0
```

### Mixed precision

Teacher fp16 + student bf16/float debe producir loss finite float32.

---

# 31. Tests de modelo

## Dense feature shape

Input:

```text
[B=2,T=3,C=12,H=128,W=128]
```

Output con flag:

```text
[2,3,64,32,32]
```

## Inference equivalence

Con modelo en eval y mismo input:

```python
out_a = model(events, delta, return_dense_features=False)
out_b = model(events, delta, return_dense_features=True)
```

Comparar todos los outputs científicos excepto nuevo campo dense:

- TTC;
- log ratio;
- foreground logits;
- geometry tokens;
- known;
- residual;
- uncertainty.

Deben ser idénticos o `allclose` muy estricto.

## Parameter count

```python
sum(p.numel() for p in model.parameters()) == 355118
```

antes y después de feature exposure.

---

# 32. Tests de cache

Cada uno debe fallar:

- manifest SHA incorrecto;
- artifact signature incorrecta;
- artifact ID incorrecto;
- status != passed;
- train_only != true;
- validation/test opened != false;
- TTC labels read != false;
- model ID != ConvNeXt-Large;
- offsets distintos;
- grid != 32x32;
- shard SHA incorrecto;
- token duplicado;
- token faltante;
- sequence mismatch;
- crop mismatch > 1e-4;
- relation NaN en valid;
- relation fuera razonablemente de [-1,1];
- relation/valid shape mismatch;
- base dataset count != cache count.

---

# 33. Tests batch/input leakage

Con fake cache A4:

```python
batch.dinov3_relation_targets is not None
batch.dinov3_relation_valid is not None
```

Pero:

```python
batch.event_inputs().keys()
==
{"events", "delta_t_s"}
```

y DINO no aparece.

Validation:

```python
batch.dinov3_relation_targets is None
batch.dinov3_relation_valid is None
```

---

# 34. Test config equivalence A1-DF vs A4

Parsear ambos YAML y assert:

## Deben ser iguales

```text
model_config
data.cache_manifest
data.cache_manifest_sha256
data.cache_artifact_sha256
train_sequence_ids
validation_sequence_ids
opened_splits
todos los campos training A1-DF preexistentes
todo el bloque loss
```

## Diferencias autorizadas

```text
experiment identity
data.dinov3_relational_teacher
training.representation_*
decision_contract A4
```

Nada más.

---

# 35. Test de resume A4

Con fake teacher cache pequeño:

### Run continuo

```text
2 epochs
```

### Run resumido

```text
1 epoch
save
resume
1 epoch
```

Deben coincidir exactamente:

- model state;
- optimizer;
- scheduler;
- RNG torch;
- RNG CUDA si aplica en test;
- numpy;
- Python;
- dataloader generator;
- history;
- best epoch;
- best model;
- A4 train metrics.

Cambiar `representation_distillation_weight` antes del resume debe fallar.

Cambiar teacher artifact SHA debe fallar.

---

# 36. Test validation fail-closed

Construir validation batch artificial con DINO fields.

`evaluate_real_causal_scale` debe lanzar error.

Esto evita que una futura refactorización cargue teacher en validation sin darnos cuenta.

---

# 37. Test de no SAM

A4 config:

```text
foreground_supervision == bbox_geometry
foreground_bce_weight == 0
foreground_dice_weight == 0
```

No envolver con `SAMTeacherMaskDataset`.

`summary.json`:

```text
sam_teacher_is_model_input == false
sam_teacher_train_only == false
```

si se conserva ese campo.

---

# 38. Test pair-ratio sigue cero

```python
assert loss.foreground_pair_ratio_weight == 0.0
```

No aceptar A4 config si distinto.

El runner debe reforzar el `decision_contract`.

---

# 39. Script de auditoría del cache científico

Puede integrarse en materializer o ser:

```text
scripts/audit_dinov3_relational_teacher_cache.py
```

Debe abrir el cache ya escrito, sin DINO, y verificar:

- 2048 tokens;
- 9 sequences;
- 0 validation sequences;
- 4096 endpoints;
- 6 offsets;
- 32×32;
- SHA todos;
- relation min/max/mean/std;
- porcentaje valid esperado por offsets;
- distribución por shard;
- common crop match exacto con event cache;
- no TTC keys.

Firmar artifact de auditoría.

---

# 40. Comandos de calidad que el agente debe ejecutar

Desde root:

```powershell
uv sync --extra multimodal --all-groups --locked
```

Después:

```powershell
uv run ruff check src tests scripts
```

```powershell
uv run pyright src tests scripts
```

```powershell
uv run pytest
```

Si el repo tarda demasiado, como mínimo:

```powershell
uv run pytest tests/unit/test_dinov3_relational_distillation.py
uv run pytest tests/unit/test_dinov3_relational_teacher_cache.py
uv run pytest tests/unit/test_causal_scale_a4_dino_teacher.py
uv run pytest tests/unit/test_causal_scale_a3_sam_teacher.py
```

pero antes del handoff final debe pasar la suite completa.

---

# 41. Orden de commits recomendado

No meter todo en un mega-commit.

## Commit A — preregistro de implementación

Solo docs/config schema placeholder, sin ejecutar validation.

Mensaje sugerido:

```text
docs(experiment): preregister A4 dense relational distillation
```

Debe dejar claro:

- parent A1-DF;
- Large teacher;
- train-only;
- relation offsets;
- no SAM;
- no pair ratio;
- no JEPA;
- no physics changes;
- gate mecanístico.

## Commit B — primitive relational loss

```text
feat(distill): add local dense relation primitives
```

Incluye:

- `dinov3_relational.py`;
- unit tests.

## Commit C — cache infrastructure

```text
feat(data): add train-only DINO relational teacher cache
```

Incluye:

- materializer;
- wrapper;
- batch fields;
- cache tests;
- smoke.

## Commit D — trainer integration

```text
feat(train): add A4 representation distillation path
```

Incluye:

- dense feature exposure;
- training config;
- `_loss`;
- runner;
- metadata;
- resume tests;
- parameter invariance.

## Commit E — scientific train-only cache + calibration

```text
data(experiment): freeze A4 DINO relational teacher cache
```

Incluye:

- manifest;
- posiblemente shards si tamaño Git razonable; si no, manifest y mecanismo reproducible;
- calibration artifact;
- YAML final con hashes/peso.

**No incluir validation A4 result.**

---

# 42. Sobre guardar los shards en Git

El cache ~50–60 MB puede ser manejable pero no es obligatorio versionarlo si el repo ya tiene política de artifacts externos.

El agente debe observar cómo están tratados los caches existentes.

Regla:

- manifest y artifact identities deben quedar en repo;
- si shards no se versionan, el handoff debe incluir comando exacto reproducible para regenerarlos;
- no usar Git LFS sin que ya forme parte del repositorio;
- no introducir archivos >100 MB.

---

# 43. Comandos que el agente puede ejecutar YA

## Verificar base

```powershell
git status --short
git branch --show-current
git rev-parse HEAD
```

Esperado al comenzar:

```text
scientific-recovery-v3-hardening
8c2ffeded4eb0f925d494b72adb670e7640edb17
```

Si la rama ya avanzó legítimamente, el agente debe rebasear mentalmente el plan y documentar el nuevo parent, pero **no** sobrescribir trabajo posterior.

## Instalar deps

```powershell
uv sync --extra multimodal --all-groups --locked
```

## Tiny smoke

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"

uv run python scripts/audit_dinov3_dense_teacher.py `
  --event-cache-manifest artifacts/cache/garl_object_event_common_roi_screen_v4/manifest.json `
  --train-parquet E:\GarlTTC_dataset\data\train.parquet `
  --eap-root E:\eAP_dataset `
  --model-id facebook/dinov3-convnext-tiny-pretrain-lvd1689m `
  --device cuda:0 `
  --sample-count 8 `
  --output artifacts/metrics/dinov3_dense_teacher_smoke_a4_v1.json
```

## Large materialization

```powershell
uv run python scripts/materialize_dinov3_relational_teacher.py `
  --event-cache-manifest artifacts/cache/garl_object_event_common_roi_screen_v4/manifest.json `
  --train-parquet E:\GarlTTC_dataset\data\train.parquet `
  --eap-root E:\eAP_dataset `
  --model-id facebook/dinov3-convnext-large-pretrain-lvd1689m `
  --device cuda:0 `
  --precision bf16 `
  --output-dir artifacts/cache/dinov3_convnext_large_relational_a4_v1
```

Si se interrumpe:

```powershell
uv run python scripts/materialize_dinov3_relational_teacher.py `
  --event-cache-manifest artifacts/cache/garl_object_event_common_roi_screen_v4/manifest.json `
  --train-parquet E:\GarlTTC_dataset\data\train.parquet `
  --eap-root E:\eAP_dataset `
  --model-id facebook/dinov3-convnext-large-pretrain-lvd1689m `
  --device cuda:0 `
  --precision bf16 `
  --output-dir artifacts/cache/dinov3_convnext_large_relational_a4_v1 `
  --resume
```

## Calibración train-only

```powershell
uv run python scripts/calibrate_a4_dinov3_relational_weight.py `
  --parent-config configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a1_deep_features_v1.yaml `
  --teacher-manifest artifacts/cache/dinov3_convnext_large_relational_a4_v1/manifest.json `
  --device cuda:0 `
  --sample-count 64 `
  --output artifacts/metrics/a4_dinov3_relational_weight_calibration_v1.json
```

Después el agente debe poner el `selected_weight` exacto en A4 YAML.

---

# 44. Comando que el agente NO debe ejecutar todavía

No ejecutar:

```powershell
uv run python scripts/train_causal_scale_eap_screen.py `
  --config configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a4_dinov3_relational_v1.yaml `
  --output-dir artifacts/runs/causal_scale_eap_screen_a4_dinov3_relational_seed7 `
  --device cuda:0
```

Ese será el comando científico **después de auditoría**.

---

# 45. Definition of Done para que yo pueda auditar el repo después

El agente termina cuando:

- [ ] branch limpia;
- [ ] commit(s) publicados;
- [ ] A4 parent documentado;
- [ ] A1-DF model config sin cambios;
- [ ] parameter count 355118;
- [ ] local relation primitive implementada;
- [ ] six offsets exactos;
- [ ] no wraparound;
- [ ] Dense student features expuestas solo bajo flag;
- [ ] inference default idéntica;
- [ ] teacher cache train-only;
- [ ] ConvNeXt-Large científico;
- [ ] Tiny solo smoke;
- [ ] same common-square crop;
- [ ] DINO no usa bbox prompt;
- [ ] no SAM;
- [ ] no TTC leído por teacher cache;
- [ ] no validation RGB;
- [ ] no validation teacher fields;
- [ ] `event_inputs` intacto;
- [ ] A4 relation loss separada de physical loss config;
- [ ] A4 distillation activa en warm-up y resto;
- [ ] pair-ratio = 0;
- [ ] clipping = 60;
- [ ] min ratio = .002;
- [ ] consensus = .15;
- [ ] residual = .05;
- [ ] temporal inverse blend = .75;
- [ ] cache signed;
- [ ] calibration train-only signed;
- [ ] A4 YAML final no tiene placeholders;
- [ ] resume ligado a cache hash + weight;
- [ ] all tests green;
- [ ] ruff green;
- [ ] pyright green;
- [ ] **NO A4 validation run todavía**.

---

# 46. Qué debe enviarme el usuario/agente para la auditoría siguiente

Después de implementar, necesito solamente:

1. URL/hash del último commit;
2. indicar si el cache Large fue materializado;
3. indicar si la calibración train-only fue ejecutada;
4. **no hace falta que ejecute A4 antes de mi revisión**.

Entonces revisaré:

```text
git diff 8c2ffed..NEW_HEAD
```

y comprobaré:

- arquitectura;
- fugas;
- hashes;
- input contract;
- loss;
- cache;
- tests;
- resume;
- config;
- gates;
- coherencia con el plan.

Después daré el comando exacto de entrenamiento científico.

---

# 47. Qué comando espero autorizar tras una implementación correcta

Previsiblemente:

```powershell
uv run python scripts/train_causal_scale_eap_screen.py `
  --config configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a4_dinov3_relational_v1.yaml `
  --output-dir artifacts/runs/causal_scale_eap_screen_a4_dinov3_relational_seed7 `
  --device cuda:0
```

Si existe `state/last.pt` válido y el run fue interrumpido:

```powershell
uv run python scripts/train_causal_scale_eap_screen.py `
  --config configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a4_dinov3_relational_v1.yaml `
  --output-dir artifacts/runs/causal_scale_eap_screen_a4_dinov3_relational_seed7 `
  --device cuda:0 `
  --resume
```

No `-Force`.

---

# 48. Qué analizaré después del resultado A4

Orden obligatorio:

## Primero: mecanismo

Comparar A4 vs A1-DF:

```text
absolute height
absolute width
cx
cy
```

Gate:

```text
height gain >= .05
AND
width gain >= .05
```

## Segundo: dinámica

```text
delta log h vs bbox
delta log h vs physical
delta log w vs bbox
delta log w vs physical
r_iso vs physical
ratio Pearson
ratio slope
std_pred/std_target
```

## Tercero: benchmark

```text
MiD
failure
unknown
clip
buckets
per sequence
signed behavior
```

## Cuarto: Garl matched

Comparación same 2048:

```text
A4 vs Garl matched
```

---

# 49. Árbol de decisión posterior A4

## Caso A — static y dynamic mejoran

Ejemplo conceptual:

```text
h .48 -> .60
w .24 -> .40
delta physical .17 -> .30+
ratio .19 -> .30+
```

Interpretación:

> la representación espacial era el cuello dominante y DINO relational funciona.

Siguiente:

1. NO meter JEPA inmediatamente;
2. verificar estabilidad;
3. probar scaling 8k;
4. después 16k si curva mejora;
5. seeds 7/13/23 cuando arquitectura quede congelada.

---

## Caso B — static mejora, dynamic sigue mala

Ejemplo:

```text
h .48 -> .60
w .24 -> .40
delta physical .17 -> .18
```

Interpretación:

> el encoder ahora entiende mejor cada endpoint, pero el problema restante es coherencia/predicción temporal.

Éste es el caso que **autoriza A5 Dense Event-JEPA**.

---

## Caso C — static no mejora

Ejemplo:

```text
h .48 -> .49
w .24 -> .25
```

Interpretación:

> relational DINO supervision no cambia de forma útil la representación student.

No hacer JEPA todavía.

Siguiente candidato:

```text
ScaleEvent / pretrained event-native backbone frozen probe
```

manteniendo geometry head y Causal Scale.

---

## Caso D — bbox scale se recupera pero physical se estanca

Si:

```text
pred ratio vs bbox alto
pred ratio vs TTC claramente menor
```

entonces height/width-only llega a su límite físico real.

Siguiente:

- anisotropic scale;
- divergence;
- affine deformation;
- rotation/shear observables.

No antes.

---

## Caso E — ratio bueno pero unknown/clip dominan

Solo entonces revisar:

- incertidumbre near-zero;
- probabilistic TTC;
- unknown policy;
- clip.

No tocar singularidad antes.

---

# 50. A5 — blueprint SOLO CONDICIONAL, NO IMPLEMENTAR AHORA

Esta sección existe para que el agente entienda hacia dónde va el programa, pero no debe codificarla todavía.

Condición:

```text
A4 static geometry good
AND
A4 temporal change still poor
```

Entonces:

```text
Dense Event-JEPA
```

## Student

```text
events t0/t1/t2
      ↓
online dense event encoder
      ↓
z0,z1,z2 [B,3,C,32,32]
```

## Target

EMA copy del encoder.

## Predictor

Pequeño predictor espacial/temporal que recibe:

```text
z0,z1
```

y predice:

```text
z2_hat
```

No global pooling.

## Loss

```math
L_JEPA =
mean_p [1 - cosine(z2_hat(p), stopgrad(z2_target(p)))]
```

## Primer A5

No DINO.

No SAM.

No pair-ratio nuevo.

No nueva physics loss.

Así se aísla:

> ¿la predicción latente temporal mejora la coherencia?

## Pretraining

Solo 2048 train.

Sin bbox.

Sin TTC.

Sin validation.

Budget fijo antes de ejecutar.

Después fine-tune exactamente el geometry/Causal Scale downstream.

---

# 51. Por qué NO implementar A5 junto con A4

Porque si hacemos:

```text
DINO relational + JEPA
```

y mejora, no sabremos si:

- DINO arregló spatial representation;
- JEPA arregló temporal consistency;
- ambos;
- la interacción.

La investigación ya aprendió con A0–A3 que combinar hipótesis hace difícil falsar causas.

A4 debe ser una intervención única.

---

# 52. Fallback ScaleEvent — blueprint, no implementar ahora

Solo si A4 falla static.

Primero:

```text
pretrained event-native encoder
      ↓
frozen
      ↓
small geometry probe
      ↓
h,w,cx,cy
```

Pregunta:

> ¿una representación preentrenada específicamente en eventos contiene mucha más geometría que `_EndpointEncoder`?

Si frozen probe supera claramente A1-DF:

- fine-tuning parcial después;
- mismo Causal Scale.

No meter JEPA antes de que un encoder pueda representar bien geometría por endpoint.

---

# 53. Riesgos técnicos específicos que el agente debe evitar

## 53.1 DINO preprocessing desalineado

Error:

```text
common square crop
→ AutoImageProcessor center-crop
```

Eso destruye correspondencia espacial.

Solución:

- crop nosotros;
- resize nosotros;
- normalize nosotros;
- no crop adicional.

---

## 53.2 Interpolar feature final de DINO

Error:

```text
DINO final 8x8
→ bilinear 32x32
```

Puede crear una falsa dense representation.

A4 exige una feature nativa 32x32.

---

## 53.3 Relation wraparound

Error con:

```python
torch.roll
```

Puede conectar borde derecho con izquierdo.

Prohibido.

---

## 53.4 DINO target en validation

Aunque sea “solo para loss”, validation no debe cargarlo.

---

## 53.5 RGB accidental como input student

No añadir pixel values al `CausalScaleTTC.forward`.

---

## 53.6 Teacher projection learnable

No añadir:

```python
student_projection = Conv2d(...)
```

Eso cambia parámetros y mezcla hipótesis.

Relational loss evita esta necesidad.

---

## 53.7 MSE de channels

No:

```python
MSE(student_features, dino_features)
```

Además de dimensiones distintas, fuerza alineamiento arbitrario entre espacios de modalidades diferentes.

---

## 53.8 Usar bbox para ponderar DINO relations

No en A4.

No foreground mask.

No box mask.

No event activation mask nuevo.

Mantener hipótesis mínima.

---

## 53.9 Cambiar weights physics

No tocar.

---

## 53.10 Optimizar lambda con validation

Prohibido.

Train-only calibration una vez.

---

# 54. Tests de reproducibilidad del materializer

Materializar Tiny smoke dos veces sobre los mismos 8 tokens debe producir:

- mismos sample tokens;
- mismos RGB hashes;
- mismos crop;
- mismos relation values dentro de tolerancia;
- mismo manifest logical payload si se excluye timestamp.

Large scientific cache idealmente usa deterministic algorithms donde sea viable, pero BF16/cuDNN puede tener pequeñas diferencias. El artifact debe congelar el cache generado y su hash: después del freeze, el training usa esos targets exactos.

---

# 55. Precision teacher

Usar:

```text
DINO forward: BF16 CUDA
relation computation: FP32
cache: FP16
```

Razonamiento:

- reduce VRAM para ConvNeXt-Large;
- relation cosines se calculan más estables en FP32;
- cache FP16 es suficiente para teacher targets;
- student relation loss convierte teacher a FP32.

Si Large OOM en BF16 con batch >1:

- bajar teacher materialization batch a 1;
- no cambiar modelo;
- no usar Tiny científicamente para “resolver” OOM sin documentarlo.

---

# 56. No confundir teacher batch size con student batch

Cache materialization puede usar:

```text
teacher batch = 1,2,4...
```

según VRAM.

Eso no cambia el experimento porque targets quedan congelados.

Student A4 training mantiene:

```text
batch_size = 32
```

---

# 57. Almacenamiento de RGB hashes

Por cada endpoint guardar SHA de bytes originales.

Motivo:

Si más tarde cambia/daña un TAR, podemos demostrar qué RGB exacto generó cada teacher relation.

No hace falta guardar la imagen.

---

# 58. Common crop

El teacher manifest debe incluir por token:

```text
common_square_xyxy
```

El wrapper compara con el event record.

Esto replica la disciplina del `SAMTeacherMaskDataset`.

---

# 59. Artifact signing

Usar:

```python
from e_jepa_ttc.artifacts.hashing import sign_artifact
```

y en loader:

```python
verify_artifact_hash
```

No inventar un segundo sistema de hash.

---

# 60. Dirty tree

La creación del **cache científico definitivo** y la calibración final deben idealmente ejecutarse desde commit limpio.

Smoke/debug puede hacerse dirty y marcarse `selectable=false`.

El runner A4 ya exige clean code state para representative screen; mantener esa garantía.

---

# 61. Documentación que debe actualizarse

Después de implementar, actualizar:

```text
README.md
STATUS.md
PLAN.md
CODEX_HANDOFF.md
docs/causal_scale_eap_screen.md
docs/experimental_protocol.md
docs/limitations.md
docs/model_card.md
docs/progress.md
```

Pero únicamente con hechos verdaderos:

Antes de A4 validation:

```text
A4 implementado/preregistrado
teacher cache train-only materializado
lambda calibrado train-only
validation A4 no ejecutada
```

No escribir:

```text
A4 mejora...
```

hasta ejecutar.

---

# 62. Model card

Debe aclarar:

```text
Training:
RGB/DINO teacher may be used in A4.

Inference:
strict event-only.

DINO parameters are not part of deployed model.
```

Y el parameter count continúa 355.118.

---

# 63. Limitations

Añadir:

- DINO teacher ve RGB, por lo que A4 es cross-modal training aunque event-only inference;
- common-square crop alinea modalidades pero sigue siendo un crop oracle definido por preprocessing;
- DINO fue preentrenado fuera de eAP;
- A4 evalúa transferencia de estructura visual, no aprendizaje self-supervised event-only puro;
- solo tres validation sequences públicas;
- no test/SOTA claim.

---

# 64. Qué NO debe decir la documentación

No decir:

- “DINO resuelve foreground”;
- “DINO introduce semántica correcta de coches”;
- “JEPA probado”;
- “SOTA”;
- “test untouched” si algún script lo abrió accidentalmente;
- “event-only training” — A4 usa RGB teacher durante training;
- “event-only model/inference” sí es correcto.

---

# 65. Auditoría que yo haré en el siguiente turno

Cuando me pases el commit final, revisaré en este orden:

1. branch/head;
2. diff completo;
3. model config unchanged;
4. parameter count;
5. feature exposure;
6. relation primitive;
7. teacher crop;
8. hidden-state selection;
9. cache schema;
10. scope train-only;
11. no TTC read;
12. no validation teacher;
13. batch/input leakage;
14. loss placement;
15. warm-up;
16. resume;
17. config equality;
18. hashes;
19. calibration;
20. tests;
21. docs;
22. si está todo bien, comandos de run.

---

# 66. Resultado final esperado del trabajo del agente

Al terminar debe poder enseñarme una estructura semejante a:

```text
src/e_jepa_ttc/
├── data/
│   ├── object_event_v4.py
│   └── dinov3_relational_teacher_cache.py
├── distillation/
│   ├── __init__.py
│   └── dinov3_relational.py
├── models/
│   └── causal_scale_ttc.py
└── training/
    └── causal_scale_eap.py

scripts/
├── audit_dinov3_dense_teacher.py
├── materialize_dinov3_relational_teacher.py
├── calibrate_a4_dinov3_relational_weight.py
└── train_causal_scale_eap_screen.py

configs/
└── experiment/
    └── e_jepa_garl_event_causal_scale_eap_screen_a4_dinov3_relational_v1.yaml

tests/unit/
├── test_dinov3_relational_distillation.py
├── test_dinov3_relational_teacher_cache.py
└── test_causal_scale_a4_dino_teacher.py

artifacts/
├── cache/
│   └── dinov3_convnext_large_relational_a4_v1/
│       └── manifest.json
└── metrics/
    ├── dinov3_dense_teacher_smoke_a4_v1.json
    └── a4_dinov3_relational_weight_calibration_v1.json
```

---

# 67. Mensaje corto que puedes dar directamente al agente

> Implementa exactamente A4 según `E_JEPA_TTC_A4_DINO_RELATIONAL_IMPLEMENTATION_HANDOFF.md`. Parte de `8c2ffeded4eb0f925d494b72adb670e7640edb17` en `scientific-recovery-v3-hardening`. A4 debe conservar A1-DF y sus 355.118 parámetros, añadir únicamente distillation relacional local train-only desde DINOv3 ConvNeXt-Large sobre el RGB del mismo common-square crop, cachear seis mapas de similitud coseno 32×32 para t1/t2, calibrar el peso solo con 64 muestras train y congelar el config. No ejecutes validation A4. No SAM, no pair-ratio, no JEPA, no cambio physics/unknown/clip. Haz tests completos, commits limpios y deja hashes/artifacts suficientes para auditoría.

---

# 68. Cierre

La idea central de todo este plan es preservar la capacidad de falsar hipótesis.

A0–A3 ya han mostrado que:

```text
mejor mask target
≠
mejor temporal scale
```

y que:

```text
deep event features
→
más señal física interna
```

A4 prueba una sola cosa:

```text
¿si estructuramos mejor las dense event features
usando relaciones espaciales de un VFM fuerte,
mejora la geometría que Causal Scale necesita?
```

Si sí:

```text
seguir representación → dinámica → scaling
```

Si no:

```text
abandonar esta vía, no tunear DINO eternamente
```

Y solo si los estados se vuelven buenos pero su evolución temporal sigue mala:

```text
Dense Event-JEPA
```

Ése es el orden que mantiene la investigación científicamente interpretable.
