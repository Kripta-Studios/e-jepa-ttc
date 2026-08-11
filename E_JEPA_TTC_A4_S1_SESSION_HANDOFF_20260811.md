# Handoff científico — A4 → scaling 8192 → selección train-only de λ → A4-S1 λ=8

**Proyecto:** `Kripta-Studios/e-jepa-ttc`  
**Fecha de cierre de esta sesión:** 2026-08-11  
**Objetivo de la sesión:** escalar A4 sin contaminar el test, seleccionar de forma train-only el peso de distillation DINOv3 y ejecutar un primer follow-up S1 sobre 8192 muestras manteniendo la arquitectura A4 constante.

---

## 0. Resumen ejecutivo

Esta sesión ha producido un resultado científico importante.

El A4 original utilizaba un student event-only de **355.118 parámetros**, 2048 muestras de train, 2048 de validation y un teacher offline DINOv3 ConvNeXt-Large extraído desde RGB sincronizado. Su mejor resultado era:

- sequence-macro MiD: **322.6813**
- failure rate: **11.084%**
- log-ratio Pearson: **0.26010**

Durante esta sesión:

1. Se modificó la infraestructura para permitir **train > 2048** manteniendo la **validation original congelada en 2048**.
2. Se creó un nuevo cache event-based de **8192 train**.
3. Se materializó un nuevo cache DINOv3 RGB teacher de **8192 train**.
4. Se verificaron los **8192/8192 joins** entre muestras event, objetos y teacher.
5. Se implementó una selección de `λ` **exclusivamente train-only**, mediante 3-fold CV por secuencias.
6. La rejilla `{4, 6, 8, 10.34168, 12}` eligió **λ*=8**, sin tocar public validation.
7. Se ejecutó A4-S1 con:
   - 8192 train;
   - λ=8;
   - misma arquitectura;
   - mismos 355.118 parámetros;
   - seed 7;
   - 18 epochs;
   - mismas 2048 validation de A4.
8. S1 obtuvo:
   - sequence-macro MiD: **262.825**
   - failure rate: **8.594%**
   - log-ratio Pearson: **0.44778**
   - best epoch: **15**

Respecto a A4:

- MiD mejora **59.856 puntos**, equivalente a **−18,55%**.
- Failure baja **2,49 puntos porcentuales**, equivalente a **−22,47% relativo**.
- log-ratio Pearson sube **+0,18768**, equivalente a **+72,16% relativo**.
- Se cierra aproximadamente **50,28% de la distancia MiD entre A4 y la referencia Garl**.

Por tanto, **A4-S1 es un éxito claro frente a A4**.

Sin embargo:

- Garl sigue en **203.634 MiD** y **0% failure** en la referencia matched.
- S1 todavía está aproximadamente **29,07% por encima del MiD de Garl**.
- No hay autorización para afirmar SOTA.
- El test oficial sigue cerrado.

La conclusión más importante es que el cuello de botella de A4 **no era únicamente arquitectónico**: aumentar datos y seleccionar correctamente el peso DINO ha recuperado una señal temporal/geométrica mucho más fuerte sin añadir un solo parámetro al student.

---

# 1. Punto de partida

## 1.1 A4 original

El A4 de referencia estaba ligado al commit:

```text
f7c6c3264edbbe9c1d94650f9bcd530f059a9d5c
experiment: bind A4 to audited DINO RGB teacher
```

Student:

```text
model_config:
configs/model/e_jepa_causal_scale_event_v8_t015_resize_conv.yaml

parameter_count:
355118
```

El student sigue siendo **event-only en inferencia**.

Inputs forward:

```text
event_v4_common_roi
garl_delta_t_s
```

No se usan como inputs forward:

```text
RGB
DINO
bbox
SAM
TTC ground truth
```

DINO se utiliza exclusivamente como teacher offline durante train.

### Resultado A4 de referencia

```text
sequence_macro_MiD = 322.6813364242674
failure_rate_pct   = 11.083984375
log_ratio_pearson  = 0.2600980103
```

Referencia Garl matched incluida en el contrato experimental:

```text
sequence_macro_MiD = 203.6341709373319
failure_rate_pct   = 0.0
```

---

# 2. Primera modificación: infraestructura de scaling

Se detectó que varias partes de la infraestructura A4 estaban congeladas a 2048 filas.

Se modificaron:

```text
scripts/materialize_dinov3_relational_teacher.py
scripts/train_causal_scale_eap_screen.py
```

y los cambios quedaron en:

```text
9de1a7910f384af864c752b41e5785d6d7dcd0e6
experiment: enable A4 train scaling with frozen validation
```

## 2.1 Cambio en el materializador DINO

Antes tenía contratos rígidos equivalentes a:

```text
2048 rows
4096 RGB endpoints
```

Se añadió soporte para:

```text
--expected-rows <N>
```

de forma que el teacher DINO puede materializarse sobre un train escalado sin relajar los checks de identidad/provenance.

## 2.2 Cambio en el runner

El runner dejó de asumir que train y validation tenían que proceder del mismo manifest con exactamente:

```json
{"train": 2048, "validation": 2048}
```

Ahora soporta:

```yaml
expected_train_rows: ...
expected_validation_rows: ...

validation_cache_manifest: ...
validation_cache_manifest_sha256: ...
validation_cache_artifact_sha256: ...
```

Esto permite el protocolo correcto:

```text
TRAIN:
nuevo cache escalado

VALIDATION:
cache A4 original congelado
```

Esta separación es crucial para que el scaling de train no cambie el conjunto de comparación.

---

# 3. Nuevo cache Event-v4 de 8192 train

Se construyó:

```text
artifacts/cache/garl_object_event_common_roi_train8192_v1/manifest.json
```

Identidad:

```text
manifest SHA256:
063980fdae5fda0b2836befc662fdd1cd5659bf06f10d9760dfc0d566fac8e39

artifact SHA256:
f64b4348820b3ed83307413fd52c2cb39dae3c70deae8e65dc24996fe9ef25ec
```

Conteos:

```text
train      = 8192
validation = 4735
```

**Importante:** las 4735 validation del cache expandido NO se utilizaron para S1.

S1 utiliza como validation el cache A4 original:

```text
artifacts/cache/garl_object_event_common_roi_screen_v4/manifest.json
```

con:

```text
train      = 2048
validation = 2048
```

Hashes del cache A4 original:

```text
manifest SHA256:
bba9ff9b143bfd57760bd61d2b6f664202581b5dee54444f44c625975557eb72

artifact SHA256:
36c12d75c91a243f4d712831cebcd3e82f896a76196b2a65b039e680f1fac309
```

## 3.1 Secuencias train

Las 8192 muestras se distribuyen sobre las nueve secuencias:

```text
2cyv0Oedzg
5ilM1PX2vz
6h5yRW2LGc
OBneIVg4Cw
OYgB6RGWcq
WbCh1DRerJ
mHGFBekt7X
qGsgzl4Q8B
t79dBxj1WS
```

Validation congelada:

```text
DGqicHUGWb
pBqGOb2vYq
qoohcdtLDH
```

Overlap:

```text
[]
```

---

# 4. Nuevo cache teacher DINOv3 de 8192

Se materializó:

```text
artifacts/cache/dinov3_convnext_large_relational_a4_train8192_rgb_v1/manifest.json
```

Identidad:

```text
manifest SHA256:
6ee4b205d29a07d3aabf7137f2a312376711b5e5d9d7930ea36328390023eab3

artifact SHA256:
6511c684881f3360efb7c8718976ef6b37de36668a89d9ea2f8d4bdf6f620b20
```

Rows:

```text
8192
```

Teacher:

```text
facebook/dinov3-convnext-large-pretrain-lvd1689m
```

La generación mantiene el contrato:

```text
teacher source modality = rgb
event tensor used as teacher input = false
validation/test opened = false
TTC labels read = false
public train only = true
```

---

# 5. Gate completo de 8192

Se ejecutó un gate que recorrió las 8192 entradas completas.

Resultado:

```text
PASS: A4-S1 DATA + DINO TEACHER GATE

train = 8192
frozen validation = 2048
train/validation sequence overlap = 0
all 8192 DINO joins verified
RGB-only / train-only teacher verified
```

Se verificaron explícitamente:

```text
8192 / 8192
```

joins entre:

```text
sample
sample_token
track_id
sequence
common crop
DINO teacher target
```

Este gate cierra una fuente importante de errores silenciosos de cache o alineamiento.

---

# 6. Problema detectado con λ del DINO endpoint

La calibración original A4 había producido aproximadamente:

```text
raw λ ≈ 10.34168
```

pero el protocolo A4 tenía un clamp:

```text
max λ = 4
```

por lo que A4 utilizó:

```text
λ = 4
```

Esto no demostraba que 4 fuese óptimo; únicamente era el máximo permitido por la preregistración A4.

Por ello se decidió no lanzar S1 directamente con λ=4.

---

# 7. Selección train-only de λ

Se añadió un protocolo específico en el commit:

```text
e8f233ea353bb32c4234a6c8c29c2a89a8fd103b
experiment: preregister train-only A4 DINO lambda CV
```

Archivos añadidos:

```text
scripts/select_a4_dinov3_relational_weight_cv.py

configs/experiment/
e_jepa_garl_event_causal_scale_a4_lambda_cv_train8192_v1.yaml

tests/unit/test_a4_dinov3_lambda_cv.py
```

## 7.1 Regla experimental

Candidatos:

```text
4
6
8
10.341683...
12
```

Folds:

```text
3 folds
3 secuencias held-out por fold
9 secuencias train cubiertas exactamente una vez
```

Total:

```text
5 λ × 3 folds = 15 runs
```

Public validation abierta durante selección:

```text
0 muestras
```

Official test:

```text
false
```

Criterio preregistrado:

1. menor nine-sequence macro MiD;
2. menor failure rate;
3. menor λ como desempate.

Guard:

```text
si gana 4  -> boundary_hit
si gana 12 -> boundary_hit
```

---

# 8. Resultado de la CV de λ

Artifact final:

```text
artifact SHA256:
68a8d049509167be5db7217acb9cbd69dbd4845cf93833fcab324cc6da919318
```

Resultado:

```text
selected_lambda_candidate = 8
lambda_grid_boundary_hit   = false
promotion_ready            = true
```

## 8.1 Comparación completa

| λ | nine-sequence macro MiD ↓ | failure ↓ | best epochs por fold |
|---:|---:|---:|---|
| 4.0 | 280.771 | 9.521% | 18 / 17 / 12 |
| 6.0 | 277.841 | 9.717% | 14 / 14 / 16 |
| **8.0** | **272.517** | **8.728%** | **18 / 17 / 17** |
| 10.3417 | 276.902 | 9.119% | 18 / 17 / 17 |
| 12.0 | 275.890 | 9.521% | 12 / 17 / 14 |

Interpretación:

- λ=8 gana el criterio primario.
- También obtiene la menor failure rate.
- No gana por desempate.
- No está en un extremo de la rejilla.
- Por tanto la promoción está justificada por el protocolo.

No debe interpretarse como que `8.000` sea una constante universal exacta. La zona `8–12` es relativamente competitiva, pero **8 es el ganador preregistrado**.

---

# 9. Rama aislada para S1

Durante la sesión, la rama principal local siguió evolucionando con trabajo A5/A6.

Antes de S1, el estado local reportado era:

```text
branch:
scientific-recovery-v3-hardening

local HEAD:
8da438bdd0ac2200f2544cf58704cc782d63be90
```

Para evitar mezclar A5/A6 con el experimento limpio A4-S1, se creó una rama aislada desde el commit de la CV:

```text
base:
e8f233ea353bb32c4234a6c8c29c2a89a8fd103b
```

Nueva rama:

```text
a4-s1-train8192-lambda8-v1
```

Commit preregistrado y pusheado:

```text
946f7a8a20ef1e30c8a2c9fb50e8da9ee1caf859
experiment: preregister A4-S1 8192 lambda8 follow-up
```

Remote:

```text
origin/a4-s1-train8192-lambda8-v1
```

---

# 10. Archivos añadidos para S1

El commit `946f7a8...` añadió exactamente cuatro archivos trackeados:

```text
configs/experiment/
e_jepa_garl_event_causal_scale_eap_screen_a4_s1_train8192_lambda4_control_v1.yaml

configs/experiment/
e_jepa_garl_event_causal_scale_eap_screen_a4_s1_train8192_lambda8_v1.yaml

scripts/verify_a4_s1_lambda8_prerequisites.py

tests/unit/test_a4_s1_lambda8_contract.py
```

Commit:

```text
4 files changed
608 insertions
```

## 10.1 Config primaria

```text
configs/experiment/
e_jepa_garl_event_causal_scale_eap_screen_a4_s1_train8192_lambda8_v1.yaml
```

SHA256:

```text
1d2eb1cae3748cc06cf92e1cbe510415798ebeed329ec97867297f01ad56848d
```

## 10.2 Control λ=4

También quedó **preregistrado antes de ver S1**:

```text
configs/experiment/
e_jepa_garl_event_causal_scale_eap_screen_a4_s1_train8192_lambda4_control_v1.yaml
```

Este control todavía NO se ha ejecutado.

Su función es separar:

```text
efecto de escalar 2048 -> 8192
```

de:

```text
efecto de cambiar λ 4 -> 8
```

---

# 11. Gates de código antes de S1

Antes de entrenar:

```text
Ruff:
PASS

Pyright:
0 errors
0 warnings

pytest:
15 tests PASS
```

Prerrequisite gate:

```json
{
  "status": "passed",
  "git_head": "946f7a8a20ef1e30c8a2c9fb50e8da9ee1caf859",
  "train_rows": 8192,
  "validation_rows": 2048,
  "teacher_rows": 8192,
  "selected_lambda": 8.0,
  "epochs": 18,
  "seed": 7,
  "student_parameter_count_expected": 355118,
  "teacher_source_modality": "rgb",
  "official_test_opened": false,
  "train_validation_sequence_overlap": []
}
```

---

# 12. Protocolo exacto S1

```text
student params: 355118
architecture: exactly A4
architecture scaling: false
student capacity change: false

train rows: 8192
validation rows: 2048

λ DINO endpoint: 8.0

seed: 7
epochs: 18
minimum epochs: 8
early stopping patience: 5
foreground warmup: 3

batch size: 32
learning rate: 3e-4
minimum LR: 3e-5
weight decay: 1e-4
precision: bf16

official/private test: CLOSED
```

Student forward contract:

```text
forward inputs:
- event_v4_common_roi
- garl_delta_t_s
```

Teacher contract:

```text
DINO teacher:
train only
offline
RGB source
not a model forward input
not loaded for validation
```

---

# 13. Resultado S1

Run:

```text
artifacts/runs/
causal_scale_eap_screen_a4_s1_train8192_lambda8_seed7
```

Estado:

```text
completed_public_validation_only
```

Runtime:

```text
4421.218 s
≈ 73.69 min
```

Peak VRAM reportado por el proceso S1:

```text
1312.82 MiB
```

Checkpoint seleccionado:

```text
best_epoch = 15
```

Resultado principal:

```text
sequence_macro_MiD = 262.8249876110039
failure_rate_pct   = 8.59375
known_coverage     = 0.9140625
log_ratio_pearson  = 0.4477787018
```

---

# 14. Comparación S1 vs A4

| Métrica | A4 | S1 | Cambio |
|---|---:|---:|---:|
| sequence-macro MiD ↓ | 322.681 | **262.825** | **−59.856 / −18.55%** |
| failure ↓ | 11.084% | **8.594%** | **−2.490 pp / −22.47% rel.** |
| known coverage ↑ | 88.916% aprox. | **91.406%** | +2.49 pp |
| log-ratio Pearson ↑ | 0.26010 | **0.44778** | **+0.18768 / +72.16% rel.** |

El success contract de S1 exigía:

```text
lower MiD than A4
AND
non-worse failure than A4
```

S1 cumple ambas condiciones claramente.

---

# 15. Comparación con Garl

Referencia matched incluida en el contrato:

```text
Garl MiD     = 203.6341709373319
Garl failure = 0%
```

Gap A4 → Garl:

```text
322.681 - 203.634 = 119.047
```

Gap S1 → Garl:

```text
262.825 - 203.634 = 59.191
```

Porcentaje del gap A4→Garl cerrado por S1:

```text
≈ 50.28%
```

S1 todavía tiene un MiD aproximadamente:

```text
29.07%
```

superior al de Garl.

Por tanto:

```text
S1 = gran avance
S1 != SOTA
```

El propio `summary.json` mantiene:

```text
garl_comparison_pending = true
sota_claim_authorized   = false
```

---

# 16. Resultado por secuencia

| Secuencia | MiD | Failure |
|---|---:|---:|
| `DGqicHUGWb` | 284.005 | 8.492% |
| `pBqGOb2vYq` | 248.488 | 7.613% |
| `qoohcdtLDH` | 255.981 | 9.677% |

Las tres secuencias tienen métricas finitas.

No existe una única secuencia que explique todo el resultado.

---

# 17. Resultado por régimen TTC

| Bin | N | MiD | Failure |
|---|---:|---:|---:|
| crucial | 513 | 350.372 | **4.678%** |
| large | 548 | 126.652 | 11.314% |
| negative | 335 | 185.640 | **12.239%** |
| small | 652 | 181.782 | 7.515% |

Diagnóstico:

- `negative` y `large` siguen siendo los bins con más fallos.
- `crucial` mantiene un MiD alto pero una failure rate relativamente baja.
- Hay margen específico para mejorar estabilidad de signo y escala en los casos negativos.

---

# 18. Diagnósticos geométricos S1

Checkpoint seleccionado, epoch 15.

## 18.1 Altura absoluta

```text
global Pearson = 0.564665
macro Pearson  = 0.612543
```

## 18.2 Dinámica de altura vs física

```text
global Pearson = 0.440649
macro Pearson  = 0.439104
```

## 18.3 Dinámica de altura vs bbox

```text
global Pearson = 0.345812
macro Pearson  = 0.338123
```

## 18.4 Anchura absoluta

```text
global Pearson = 0.405905
macro Pearson  = 0.291237
```

## 18.5 Dinámica de anchura

```text
vs physical:
global = 0.124311
macro  = 0.101804
```

La altura sigue siendo la señal geométrica temporal más útil.

## 18.6 Isotropic scale

```text
delta isotropic vs physical:
global = 0.358794
macro  = 0.347230
```

La escala isotrópica contiene información útil, aunque inferior a la altura.

---

# 19. El antiguo gate mecanístico de A4 ahora PASA

Los umbrales antiguos eran aproximadamente:

```text
log-ratio Pearson       >= 0.290098
Δheight physical global >= 0.277614
Δheight macro           >= 0.282632
abs-height macro        >= 0.526904
```

S1 epoch 15:

| Gate | Umbral | S1 | Margen |
|---|---:|---:|---:|
| log-ratio Pearson | 0.290098 | **0.447779** | +0.157681 |
| Δheight physical | 0.277614 | **0.440649** | +0.163035 |
| Δheight macro | 0.282632 | **0.439104** | +0.156472 |
| abs-height macro | 0.526904 | **0.612543** | +0.085639 |

Esto es una de las conclusiones más importantes de toda la sesión.

A4 había fallado el mecanismo temporal. S1 lo supera ampliamente **sin aumentar capacidad ni añadir transport**.

Por tanto, la hipótesis anterior de que era imprescindible introducir inmediatamente un mecanismo cross-time para obtener una dinámica razonable queda debilitada.

Transport puede seguir siendo útil para cerrar el gap restante, pero ya no es necesario para explicar el fallo fundamental de A4.

---

# 20. Trayectoria durante las 18 épocas

| Epoch | MiD ↓ | Failure | log-r Pearson | Δheight phys. |
|---:|---:|---:|---:|---:|
| 1 | 391.760 | 5.957% | 0.0875 | 0.0956 |
| 2 | 396.842 | 4.053% | 0.1175 | 0.1324 |
| 3 | 408.624 | 4.932% | 0.1193 | 0.1401 |
| 4 | 343.981 | 24.512% | 0.2344 | 0.2344 |
| 5 | 324.027 | 16.650% | 0.2521 | 0.2563 |
| 6 | 321.253 | 15.479% | 0.2882 | 0.2771 |
| 7 | 299.464 | 12.598% | 0.3420 | 0.3467 |
| 8 | 302.008 | 14.209% | 0.3227 | 0.3263 |
| 9 | 298.071 | 14.062% | 0.3760 | 0.3767 |
| 10 | 287.316 | 12.061% | 0.4058 | 0.4016 |
| 11 | 273.286 | 9.912% | 0.4112 | 0.4121 |
| 12 | 278.091 | 10.596% | 0.4410 | 0.4385 |
| 13 | 269.138 | 10.303% | 0.4309 | 0.4330 |
| 14 | 264.905 | 9.375% | 0.4441 | 0.4417 |
| **15** | **262.825** | **8.594%** | 0.4478 | 0.4406 |
| 16 | 263.728 | **7.666%** | 0.4551 | **0.4582** |
| 17 | 268.188 | 9.375% | 0.4599 | 0.4549 |
| 18 | 267.206 | 9.229% | **0.4677** | **0.4582** |

Observación importante:

- MiD selecciona epoch 15.
- Los mecanismos continúan mejorando hasta epoch 18.
- Epoch 16 incluso tiene menor failure que epoch 15.
- No hay colapso al final.
- La mejora de mecanismo no se traduce monótonamente en MiD.

Esto sugiere que el horizonte de 18 epochs no es absurdamente largo y que podría existir interés futuro en un `long-control`, pero **no debe cambiarse retrospectivamente el protocolo S1**.

---

# 21. DINO endpoint durante S1

Train DINO raw:

```text
epoch 1  ≈ 0.074325
epoch 15 ≈ 0.064562
epoch 18 ≈ 0.064191
```

El loss DINO endpoint sí disminuye de forma consistente.

El weighted loss con λ=8 pasa aproximadamente de:

```text
0.5946 -> 0.5135
```

Esto refuerza la conclusión de que el teacher endpoint DINO está siendo aprendido y no es únicamente ruido regularizador.

---

# 22. Predicciones TTC crudas

Análisis directo de:

```text
run/validation_predictions.csv
```

Resultado:

```text
total                = 2048
finite predictions   = 1872
unknown / NaN        = 176
known coverage       = 91.40625%

raw TTC MAE          ≈ 14.098 s
raw TTC Pearson      ≈ 0.05497
sign accuracy        ≈ 72.97%
predictions at ±60 s = 147
```

Este apartado es importante porque muestra que el modelo todavía no está “resuelto”.

Aunque MiD y geometría mejoran mucho:

- el TTC bruto sigue teniendo Pearson muy bajo;
- siguen existiendo 176 unknown;
- siguen existiendo muchas saturaciones en ±60 s.

Por tanto, el siguiente objetivo no debería ser únicamente bajar MiD medio; también debe estabilizar el mapa de `log_ratio -> TTC`.

---

# 23. Caveat de ejecución: A6 estaba compartiendo GPU

Durante una parte de S1 había otro proceso:

```text
A6 transport adapter
```

utilizando la misma RTX 5070 Ti.

Esto provocó contención de:

```text
GPU compute
VRAM
power / thermal budget
```

No hay evidencia de que invalide las métricas S1.

El run S1:

- terminó correctamente;
- generó checkpoint y predicciones;
- mantiene provenance;
- no produjo OOM;
- no modificó el contrato experimental.

Sin embargo:

**el runtime de 73.7 min NO debe considerarse un benchmark limpio de velocidad.**

Las replicaciones futuras deben ejecutarse con S1 como único entrenamiento GPU para obtener runtimes y reproducibilidad operativa más limpios.

---

# 24. Artifacts S1

Run summary:

```text
run/summary.json

artifact SHA256 interno:
77056b32388a8bef8699ace53a4456c0821f2feaae24dba71eec50589f8d9b6b

file SHA256:
4688dfac2e78468f95784bed5d6c87ef0b3430ca30dc342e5791e1a206287c6d
```

Predicciones:

```text
run/validation_predictions.csv

SHA256:
271e4d270a013bf3d15320852e78103768fe190896474e0058a0205db1afcc9b
```

Checkpoint:

```text
model_best.pt

SHA256:
95f6758cf74278d28ac18b6475299ef1ba19c6b3c88d7424b8d7193a6c038035
```

---

# 25. ZIPs importantes de esta sesión

## 25.1 Auditoría scaling 8192

```text
a4_s1_train8192_audit_bundle.zip

SHA256:
c06987f258d597af938cde187f001f76615a018fc1361bf8eb2129190b56d645
```

Contiene la evidencia de:

- patch scaling;
- cache event 8192;
- teacher DINO 8192;
- manifests/hashes;
- provenance.

## 25.2 Source bundle para λ

```text
a4_lambda_selection_sources.zip

SHA256:
28c5fc8e99ab973e889538eb57945dc2e8536a538d5e9089c282d3bccb2897d6
```

Sirvió para construir el selector CV exacto contra el source snapshot.

## 25.3 Resultados λ-CV

```text
a4_lambda_cv_results_20260810.zip

SHA256:
0a2e868d4c7096c65e265a35c41ed33ca75518cf647c37efef08d41c80157dbd
```

Contiene:

- 15 runs;
- fold summaries;
- candidato ganador;
- artifact firmado;
- λ*=8.

## 25.4 Resultado S1

```text
a4_s1_lambda8_results_20260811.zip

SHA256:
a134017f5fb3714733fd38c309a7a39fb2fe62d708c76e9215f93fdce1738da5
```

Contenido relevante:

```text
run/summary.json
run/validation_predictions.csv
lambda_cv_summary.json
primary_config.yaml

audit/04_s1_prerequisites.log
audit/05_s1_training.log
audit/03_git_commit.log
audit/03_git_push.log

GIT_HEAD.txt
GIT_LOG.txt
GIT_STATUS.txt
GIT_HEAD.diff
OUTPUT_HASHES.txt
```

Este es el bundle principal que debe conservarse para reproducibilidad.

---

# 26. Estado Git al terminar

## 26.1 Rama S1

```text
branch:
a4-s1-train8192-lambda8-v1

HEAD:
946f7a8a20ef1e30c8a2c9fb50e8da9ee1caf859

remote:
origin/a4-s1-train8192-lambda8-v1
```

La rama fue pusheada correctamente.

## 26.2 Rama de trabajo original

Tras terminar S1, el script volvió a:

```text
scientific-recovery-v3-hardening
```

HEAD local reportado:

```text
8da438bdd0ac2200f2544cf58704cc782d63be90
```

El terminal reportó:

```text
Your branch is ahead of 'origin/scientific-recovery-v3-hardening' by 12 commits.
```

También se restauraron:

```text
19 untracked files
```

Por tanto, el estado conceptual actual es:

```text
scientific-recovery-v3-hardening
    |
    +-- trabajo local posterior A5/A6
    |
    +-- 12 commits locales aún no publicados según el terminal

a4-s1-train8192-lambda8-v1
    |
    +-- rama experimental aislada
    +-- commit 946f7a8
    +-- pusheada a origin
```

**No mezclar automáticamente ambas ramas todavía.**

---

# 27. Diagnóstico científico actualizado

## 27.1 Qué hemos demostrado

Se ha demostrado que:

```text
más datos train
+
un peso DINO seleccionado correctamente
```

pueden mejorar de forma importante tanto:

```text
TTC downstream
```

como:

```text
mecanismo geométrico temporal
```

sin aumentar la capacidad del student.

Esto es especialmente importante porque S1 mantiene:

```text
355118 parámetros
```

exactamente igual que A4.

## 27.2 Qué hipótesis queda reforzada

A4 estaba:

```text
data-limited / supervision-weight-limited
```

en una medida importante.

El salto 2048 → 8192 no fue un simple aumento cosmético.

La representación event-only actual sí tiene capacidad para aprender bastante más dinámica de escala de lo que mostraba A4.

## 27.3 Qué hipótesis queda debilitada

Antes de S1 era plausible concluir que:

> sin correspondencia cross-time explícita el modelo no podría pasar el gate temporal.

S1 demuestra que esa afirmación sería demasiado fuerte.

S1 supera el gate sin:

```text
cost volume
optical flow
cross-time attention
transport field
student más grande
```

Por tanto:

> transport puede ser útil, pero ya no puede presentarse como la única solución al fallo A4.

## 27.4 Qué sigue sin resolverse

Aún quedan problemas claros:

1. MiD sigue lejos de Garl.
2. Failure sigue en 8.59% vs 0% de Garl.
3. TTC Pearson bruto sigue muy bajo.
4. Hay 176 unknown.
5. Hay 147 predicciones saturadas en ±60.
6. Width dynamics sigue bastante débil.
7. Sólo tenemos seed 7 para S1.
8. El run mezcló dos cambios frente a A4:
   - 2048 → 8192;
   - λ 4 → 8.

Por tanto aún no sabemos cuánto de la mejora proviene de cada factor.

---

# 28. Mejor forma de continuar

## Fase 1 — ejecutar el control λ=4 ya preregistrado

Esta es la acción inmediata más limpia.

Ya existe:

```text
configs/experiment/
e_jepa_garl_event_causal_scale_eap_screen_a4_s1_train8192_lambda4_control_v1.yaml
```

y quedó comprometido en Git **antes de ver el resultado S1**.

Ese run debe mantener:

```text
8192 train
2048 validation
355118 params
18 epochs
seed 7
λ = 4
```

Comparación:

```text
A4:
2048 / λ4

S1-control:
8192 / λ4

S1-primary:
8192 / λ8
```

Esto permite separar aproximadamente:

```text
ganancia por datos
vs
ganancia por λ
```

No hace falta crear un nuevo patch para definir ese brazo; la config ya existe en `946f7a8`.

---

# 29. Fase 2 — replicar S1 λ=8

Si el objetivo es publicar o hacer una afirmación robusta, seed 7 no es suficiente.

Después del control:

```text
seed 13
seed 23
```

manteniendo TODO lo demás congelado:

```text
8192 train
λ=8
18 epochs
same validation
same architecture
same checkpoint selection
```

No volver a ajustar λ.

Objetivo:

```text
median / mean MiD across seeds
failure stability
log-ratio stability
per-sequence stability
```

Las replicaciones deberían ejecutarse sin A6 u otros procesos GPU concurrentes.

---

# 30. Fase 3 — decidir capacidad vs transport usando evidencia

No recomiendo crear ahora a ciegas otro patch de arquitectura.

La rama:

```text
scientific-recovery-v3-hardening @ 8da438...
```

ya contiene trabajo A5/A6 posterior que no está representado completamente en el ZIP S1.

Antes de decidir:

```text
student más grande
vs
A5/A6 transport
```

hay que analizar los resultados reales que ya existan en esa rama.

Especialmente interesa:

```text
A5 transport preflight
A5 transport preflight v2/v3
A6 transport adapter
```

El `GIT_STATUS.txt` del bundle S1 confirma la existencia de artifacts del tipo:

```text
artifacts/metrics/a5_transport_preflight_a4_checkpoint_v1/
artifacts/metrics/a5_transport_preflight_v1/
artifacts/metrics/a5_transport_preflight_v2/
artifacts/metrics/a5_transport_preflight_v3_confirm/
```

pero este ZIP no contiene sus métricas.

---

# 31. Qué necesito antes de crear el siguiente patch arquitectónico

Para un patch de capacity/A5/A6 contra el HEAD real actual, pasar:

```text
git rev-parse HEAD
git status --short
git log -20 --oneline --decorate
```

y los artifacts finales de:

```text
A5 preflight
A5/A6 training
A6 transport adapter
```

Además, si han cambiado respecto a `946f7a8`:

```text
src/e_jepa_ttc/models/causal_scale_ttc.py
src/e_jepa_ttc/models/local_transport.py

src/e_jepa_ttc/training/causal_scale_eap.py
src/e_jepa_ttc/losses/causal_scale_ttc.py

scripts/train_causal_scale_eap_screen.py

configs/model/*.yaml
configs/experiment/*a5*.yaml
configs/experiment/*a6*.yaml

tests/unit/*a5*
tests/unit/*a6*
```

No necesito volver a recibir:

```text
cache event 8192
teacher DINO 8192
lambda CV completo
S1 checkpoint completo
```

si los hashes siguen siendo los documentados aquí.

---

# 32. Si capacity scaling es el siguiente eje

No saltaría directamente a un student enorme.

Haría presupuestos de capacidad claramente separados, por ejemplo conceptualmente:

```text
A4/S1-small:
355k params

medium:
~1M

large:
~2–4M
```

y aumentaría de forma controlada:

```text
hidden channels
geometry token capacity
residual depth
encoder depth
```

sin cambiar simultáneamente:

```text
loss
λ
transport
dataset
epochs
```

La selección de capacidad debería volver a hacerse **train-only**, no eligiendo tamaños mirando continuamente las mismas 2048 validation.

---

# 33. Si transport es el siguiente eje

S1 obliga a reinterpretar A5.

La pregunta correcta ya no es:

> “¿necesitamos transport porque A4 no aprende dinámica?”

sino:

> “¿transport aporta señal adicional sobre un baseline S1 que YA aprende dinámica?”

Esto es una prueba mucho más fuerte.

A5/A6 debería compararse contra:

```text
S1 λ8 / 8192
```

y no únicamente contra A4 / 2048.

Si A5/A6 no supera 262.825 MiD o no mejora mecanismos sobre S1, entonces su utilidad arquitectónica queda debilitada.

---

# 34. Política de validation/test desde ahora

La public validation de 2048 ya ha sido observada en múltiples experimentos.

Por tanto los siguientes trabajos deben declararse explícitamente como:

```text
adaptive post-public-validation follow-ups
```

La selección de nuevos hiperparámetros/arquitecturas debe hacerse dentro de train cuando sea posible.

Debe permanecer cerrado:

```text
official eAP test
EvTTC test
CodaBench/private test
```

hasta tener:

```text
arquitectura final
hiperparámetros congelados
replicación suficiente
claim definido
```

---

# 35. Recomendación final

Orden recomendado:

```text
1. conservar S1 como anchor
       ↓
2. ejecutar control 8192 / λ4
       ↓
3. replicar S1 λ8 con seeds 13 y 23
       ↓
4. analizar A5/A6 ya existente contra S1
       ↓
5. elegir UN eje:
      capacity scaling
      o transport
       ↓
6. selección train-only
       ↓
7. confirmación public-validation
       ↓
8. sólo entonces considerar test oficial
```

La prioridad inmediata NO es hacer el student más grande.

La prioridad inmediata es cerrar dos preguntas que ahora son muy baratas y científicamente valiosas:

```text
¿cuánto del +18.55% viene de 4× datos?
¿se replica S1 λ8 en seeds adicionales?
```

Una vez respondidas, el baseline contra el que deben competir A5/A6 y cualquier student más grande será mucho más sólido.

---

# 36. Estado final en una frase

> **A4-S1 demuestra que el mismo student event-only de 355k parámetros, entrenado con 8192 muestras y λ-DINO=8 seleccionado exclusivamente en train, mejora A4 en ~18.6% MiD, reduce failures y supera ampliamente el antiguo gate temporal, cerrando aproximadamente la mitad del gap A4→Garl; el siguiente paso correcto es atribución λ4 + replicación multi-seed antes de escalar arquitectura o declarar que transport es necesario.**
