# PLAN_v6.md — OGE-JEPA-TTC: baseline Garl-TTC con datos locales, pretraining eAP-40 y evaluación EvTTC sellada

**Estado:** gate v6 cerrado: BASE histórico reproducido exactamente, grouped CV 5 folds × 3 seeds selecciona A0 global, A1/AttnRes/KDA/geometría rechazados por sus gates actuales, diagnóstico family-OOD ejecutado tras freeze y Benchmark-10 sellado
**Fecha:** 30 de julio de 2026
**Repositorio base:** `Kripta-Studios/e-jepa-ttc`
**Rama de referencia:** `scientific-recovery-v3-hardening`
**Commit histórico de referencia:** `574c4c898866d1ef9c1e03be2e3b8d6e885a95ac`; la implementación v6 se versiona en commits posteriores
**Restricción principal:** no descargar ni incorporar nuevos datasets; solo se permiten eAP Hugging Face train-40 ya completo, EvTTC-32 etiquetado, Benchmark-10 sellado y los teachers DINO/SAM ya descargados; el pseudo-TTC nunca es ground truth
**Objetivo:** construir una baseline Garl-TTC reproducible usando exclusivamente los datos ya disponibles, fine-tunearla con TTC oficial de EvTTC-32 y después demostrar, módulo a módulo, que OGE-JEPA-TTC mejora precisión, robustez a ego-motion y capacidad bbox-free.

---

## 0. Principio rector

Este plan no presupone que se alcanzará SOTA. Define una arquitectura, un protocolo y unos **gates de promoción** que solo permiten declarar SOTA si el resultado oficial lo demuestra.

La hipótesis central es:

> EvTTC puede mejorarse combinando representación densa JEPA, localización automática del objetivo, geometría TTC diferenciable, compensación explícita del movimiento ego y una corrección neuronal limitada, en vez de predecir TTC directamente desde un embedding global.

La versión v6 cierra el inventario de datos. No existe dentro de este proyecto un segundo release eAP utilizable con TTC explícito y no se planifica buscarlo ni descargarlo. EvTTC-32 es la única fuente de supervisión TTC oficial; Benchmark-10 solo evalúa. El eAP público de Hugging Face con 40 secuencias se usa para pretraining RGB-eventos, foreground, seguimiento, altura aparente y geometría, mientras que su pseudo-TTC derivado de tracks permanece como una señal auxiliar experimental.

La versión v6 incorpora las siguientes decisiones derivadas de Patch Policy y Kimi K3:

1. **Patch Policy permanece como principio obligatorio:** los patches de un mismo instante conservan interacción espacial bidireccional antes de cualquier compresión.
2. **Attention Residuals se incorpora como candidato principal de bajo riesgo:** cada cabeza puede recuperar de forma selectiva features tempranas, intermedias y profundas.
3. **Kimi Delta Attention no sustituye a Patch Policy:** solo se prueba como mezclador temporal factorado para aumentar resolución u horizontes; nunca ordena causalmente los patches dentro de un mismo frame.
4. **Stable LatentMoE y MOPD no se copian literalmente:** se adaptan como un router geométrico estable y una destilación multi-teacher condicionada por las predicciones del estudiante.

La nueva arquitectura se denominará provisionalmente:

# **OGE-JEPA-TTC**

**O**bject-centric **G**eometry-**E**mbedded **JEPA** for **TTC**.

---

# 1. Evidencia experimental de partida

## 1.1 Resultado que se conserva como baseline

El modelo de referencia será `BASE`, no una variante FlowMimic.

Resultados históricos auditados de seed 7:

| Brazo | MAE validation | RMSE | Error relativo |
|---|---:|---:|---:|
| `BASE` | **0,322892 s** | **0,584432 s** | 8,1554 % |
| `NO_VARIANCE` | 0,313 s | 0,602 s | **7,90 %** |
| `NO_MOTION` | 0,346 s | 0,589 s | 9,86 % |
| `ALIGN` | 0,375 s | 0,629 s | 9,73 % |
| `BOTH` | 0,442 s | 0,662 s | 11,23 % |
| `INVERSE` | 0,480 s | 0,734 s | 11,02 % |
| `NO_NAV` | 0,555 s | 1,018 s | 12,03 % |

Conclusiones de diseño obligatorias:

1. `BASE` es la inicialización segura.
2. `NO_VARIANCE` no se promociona todavía: mejora ligeramente el MAE, pero empeora RMSE y presenta contracción latente.
3. `ALIGN`, `INVERSE` y `BOTH` quedan rechazados **en su formulación global actual**.
4. La navegación es necesaria, pero debe auditarse si se usa físicamente o como shortcut.
5. El mejor checkpoint SSL aparece temprano; entrenar más no equivale a mejorar la representación.
6. El SSL loss actual no predice de forma fiable el TTC downstream.
7. `CCRm-medium-0-overlap-0` es el principal escenario de fallo.
8. El problema principal parece ser una mezcla de:
   - pérdida prematura de detalle espacial;
   - ausencia de objeto explícito;
   - compresión del rango TTC;
   - señal geométrica global mal anclada;
   - posible dependencia excesiva de navegación.

## 1.2 Lo que queda prohibido en la nueva arquitectura

No se reutilizarán como objetivo principal:

```text
flowmimic_alignment_weight = 0.25
flowmimic_inverse_ttc_weight = 0.10
```

Tampoco se volverá a aplicar inverse-TTC sintético a un embedding global.

FlowMimic solo podrá reintroducirse más adelante si:

1. supervisa explícitamente máscara, movimiento y geometría por objeto;
2. sus pérdidas están normalizadas;
3. ninguna pérdida auxiliar supera el 20 % de la norma total de gradiente;
4. supera un gate ablation contra la misma arquitectura sin FlowMimic.


## 1.3 Baseline SOTA obligatoria: Garl-TTC

Garl-TTC no puede quedar únicamente como una cita. Debe convertirse en la **baseline arquitectónica principal** que OGE-JEPA-TTC tiene que reproducir y superar.

### 1.3.1 Qué hace Garl-TTC

Configuración descrita por el artículo:

```text
entrada por objeto:
2 RGB ROI + eventos entre ambos instantes
ROI redimensionada: 128×128
intervalo: Δt = 0,1 s
ventana temporal: K = 1
```

Arquitectura final:

```text
RGB ROI t1,t2 ──→ RGB ResNet-50 ─┐
                                 ├─→ late fusion
event voxel ROI ─→ Event ResNet ─┘
                                 ↓
                     regresión de h_t1 y h_t2
                                 ↓
             TTC = Δt / (1 - h_t1 / h_t2)
```

Durante entrenamiento añade:

```text
features fusionadas
→ decoder de foreground
→ máscara de objeto
→ supervisión SAM
```

El decoder de máscara y SAM desaparecen durante inferencia.

La función de pérdida conceptual es:

```text
L = L_TTC + λ_hr L_height_ratio + λ_fg L_foreground
```

La contribución decisiva del paper no es únicamente usar RGB y eventos, sino **cambiar el objetivo intermedio**: aprender el cociente de alturas visibles en vez de pedir a una capa fully-connected que adivine TTC directamente.

### 1.3.2 Resultados de referencia publicados

En eAP test, la ablación publicada informa:

| Variante | MiD overall ↓ |
|---|---:|
| RGB baseline directo | 160,6 |
| Event baseline directo | 79,7 |
| RGB+Event early fusion directo | 130,1 |
| RGB + LHR | 68,3 |
| Event + LHR | 66,2 |
| RGB+Event + LHR early fusion | 69,7 |
| RGB+Event + LHR late fusion | 53,0 |
| Garl-TTC completo: late fusion + foreground supervision | **45,0** |

La supervisión de foreground reduce MiD de 53,0 a 45,0. La late fusion supera con claridad la early fusion, lo que confirma que RGB y eventos no deben concatenarse ingenuamente en la entrada.

En el experimento cross-dataset sobre tres secuencias EvTTC, sin fine-tuning:

| Secuencia | RTE Garl-TTC |
|---|---:|
| `CCRs2-medium` | 8,31 % |
| `CCRs2-high` | 10,56 % |
| `CCRm-medium` | 12,93 % |
| media de esas tres | **10,60 %** |

Esta referencia **no es el resultado del Benchmark-10 completo**. No debe presentarse como media oficial de las diez secuencias.

Runtime publicado:

```text
A100-40G, PyTorch:
RGB encoder      7,11 ms
Event encoder    7,08 ms
Height head      0,15 ms
total            21,05 ms

Orin NX 16 GB, ONNX/TensorRT MAXN:
total             4,55 ms
≈220 FPS
```

El artículo también informa aproximadamente 12,67 ms de media en la comparación EvTTC de tres secuencias. Los números de hardware distintos no se comparan directamente.

### 1.3.3 Fortalezas que v6 debe conservar

1. objetivo geométrico simple y fuerte;
2. late fusion separando modalidades;
3. entrada ROI compacta;
4. dos instantes y una única pasada;
5. SAM solo como teacher de entrenamiento;
6. inference graph pequeño;
7. baseline ResNet-50 fácil de reproducir;
8. evaluación por rangos TTC y escenarios negativos.

### 1.3.4 Limitaciones que OGE-JEPA-TTC debe atacar

El propio artículo identifica:

- sensibilidad a rotación y ego-motion;
- hipótesis de que el cambio de altura procede principalmente de looming traslacional;
- dependencia de ROIs proporcionadas por detecciones 2D;
- solo dos instantes;
- evento representado como voxel convencional;
- eventos poco informativos cuando la velocidad relativa es baja;
- fusión multimodal todavía simple;
- ausencia de incertidumbre explícita.

OGE-JEPA-TTC solo se justifica si mejora al menos una de esas limitaciones sin destruir la velocidad.

### 1.3.5 Niveles de comparación permitidos

```text
GARL_PAPER_REFERENCE
= números publicados por el artículo
= referencia externa
= no se presenta como reproducción local

GARL_EVTTC_FROM_SCRATCH
= arquitectura Garl-TTC reimplementada
= supervisión TTC exclusiva de EvTTC-32
= baseline apples-to-apples principal

GARL_EAP40_SSL_EVTTC
= misma arquitectura Garl-TTC
= pretraining no-TTC sobre eAP Hugging Face train-40
= fine-tuning TTC oficial sobre EvTTC-32
= candidato principal Garl local

GARL_EAP40_PSEUDO_EVTTC
= GARL_EAP40_SSL_EVTTC
+ pseudo-TTC track-derived de eAP con peso bajo
= experimento exploratorio, nunca reproducción oficial
```

No existe en v6 un experimento de reproducción oficial de Garl sobre eAP. Ningún resultado entrenado con pseudo-TTC puede describirse como una reproducción del ground truth TTC original del artículo.

### 1.3.6 Gate de paridad antes de OGE completo

La paridad local no significa igualar el `MiD=45,0` publicado, porque el release TTC supervisado usado por los autores no forma parte del inventario disponible. La paridad significa reproducir las decisiones arquitectónicas y sus tendencias bajo el mismo grouped CV de EvTTC-32:

```text
G0 direct regression
→ G3/G4 learned height ratio
→ G5 early fusion
→ G6 late fusion
→ G7 late fusion + foreground teacher
```

Gate mínimo:

```text
LHR supera regresión directa
late fusion supera early fusion
foreground supervision mejora o iguala G6
G7 es estable en ≥4/5 folds
```

Gate para abrir OGE completo:

```text
GARL_EVTTC_FROM_SCRATCH está reproducido
y
GARL_EAP40_SSL_EVTTC ha sido comparado con el mismo protocolo
```

OGE-BBOX debe superar al mejor Garl local, no a una cifra de eAP que no podemos reproducir con los datos disponibles.

# 2. Objetivo oficial de benchmark

El leaderboard oficial EvTTC evalúa diez secuencias:

1. `CCRs1-low`
2. `CCRs1-medium`
3. `CCRs1-high`
4. `CCRs2-low`
5. `CCRs2-medium`
6. `CCRs2-high`
7. `CCRm-low`
8. `CCRm-medium`
9. `Slider-750`
10. `Slider-1000`

## 2.1 Referencias que hay que superar

Valores oficiales consultados el 29 de julio de 2026:

| Secuencia | Mejor valor oficial | Método | Target de margen 5 % |
|---|---:|---|---:|
| CCRs1-low | 2,56 % | CMax | ≤ 2,43 % |
| CCRs1-medium | 3,44 % | CMax | ≤ 3,27 % |
| CCRs1-high | 5,12 % | Image’s FoE | ≤ 4,86 % |
| CCRs2-low | 3,76 % | Image’s FoE | ≤ 3,57 % |
| CCRs2-medium | 3,85 % | Image’s FoE | ≤ 3,66 % |
| CCRs2-high | 2,92 % | Image’s FoE | ≤ 2,77 % |
| CCRm-low | 5,60 % | Image’s FoE | ≤ 5,32 % |
| CCRm-medium | 3,86 % | Image’s FoE | ≤ 3,67 % |
| Slider-750 | 4,16 % | CMax | ≤ 3,95 % |
| Slider-1000 | 2,74 % | CMax | ≤ 2,60 % |

Referencias agregadas:

- media de `Image’s FoE`: **5,45 %**;
- media de CMax: **7,16 %**;
- media de STRTTC: **10,02 %**;
- media del mejor método distinto por cada secuencia: **3,80 %**.

## 2.2 Gates de éxito


### Gate de paridad Garl-TTC

Sobre `GARL_EVTTC_REIMPLEMENTED`:

```text
late fusion LHR mejora ≥ 10 % frente a early fusion LHR
foreground supervision mejora ≥ 5 % frente a late fusion sin foreground
ninguna seed diverge
```

Sobre las tres secuencias usadas en el artículo, reportar por separado:

```text
CCRs2-medium
CCRs2-high
CCRm-medium
```

La referencia literaria es 10,60 % RTE medio; no se declara superada hasta ejecutar el mismo subconjunto y métrica.

### Gate OGE frente a Garl reimplementado

```text
OGE-BBOX mejora OOF EvTTC ≥ 5 % relativo
frente a GARL_EVTTC_REIMPLEMENTED

y

latencia single-model no supera 2× la baseline Garl
salvo mejora OOF ≥ 10 %
```

### Gate SOTA principal

```text
media oficial e_TTC < 5,00 %
```

Esto debe superar con margen el promedio del mejor método individual actual.

### Gate SOTA fuerte

```text
media oficial e_TTC < 4,50 %
y ganar al menos 7 de las 10 secuencias
```

### Stretch goal absoluto

```text
media oficial e_TTC < 3,80 %
y no superar 6 % en ninguna secuencia
```

### Gate de tiempo real

```text
latencia end-to-end mediana ≤ 15 ms
latencia p95 ≤ 25 ms
```

La latencia debe incluir:

- construcción de la representación de eventos;
- encoder;
- queries y máscara;
- geometría;
- fusión;
- predicción final.

---

# 3. Dos tracks de evaluación

No se debe mezclar el problema de localizar el objeto con el de estimar su TTC.

## Track A — `OGE-JEPA-TTC-BBOX`

Comparación apples-to-apples con el leaderboard:

```text
eventos/RGB + bbox oficial
→ cabeza geométrica object-centric
→ TTC
```

Objetivo:

- aislar la calidad del estimador TTC;
- competir directamente con CMax, STRTTC, Image’s FoE y Garl-TTC;
- producir el primer candidato de submission.

## Track B — `OGE-JEPA-TTC-FULL`

Sistema completo bbox-free:

```text
frame completo
→ localización automática
→ máscara
→ geometría
→ TTC
```

Objetivo:

- demostrar que no se necesita bbox externa;
- medir por separado error de detección y error TTC;
- proporcionar el claim más útil para un sistema real.

## Regla de claims

Nunca se presentará el resultado del Track A como bbox-free.

Nunca se comparará el Track B directamente con métodos que reciben una bbox sin declarar la diferencia de protocolo.

---

# 4. Política v6 de datos cerrados

## 4.1 Inventario único permitido

El proyecto queda limitado a los siguientes recursos, ya disponibles o cuya descarga ya está en curso.

### `EAP_HF_TRAIN40`

```text
repo: NAIL-HNU/eAP-dataset
split permitido: data/train/**
secuencias: 40
sample index: data/train.parquet
por secuencia:
  events.h5
  labels.parquet
  rgb_shards/*.tar
```

Campos auditados:

```text
sample_token
sequence_id
frame_name
instance_id
track_id
category
bbox_3d_ego
translation
size
yaw
rotation
velocity
velocity_3d
ego_translation
num_pts
```

Convenciones confirmadas localmente:

```text
bbox_3d_ego = [x, y, z, length, width, height, yaw]
size        = [height, length, width]
translation == ego_translation en el 100 % de la auditoría
```

Este release no contiene una columna TTC directa y `velocity_x` no se interpreta como velocidad de cierre.

### `EVTTC32_LABELLED`

```text
32 secuencias públicas etiquetadas
uso:
  entrenamiento TTC oficial
  grouped CV
  selección de arquitectura
  calibración
  fine-tuning final
```

Esta es la única fuente de ground truth TTC supervisado del plan v6.

### `BENCHMARK10_SEALED`

```text
10 secuencias oficiales independientes
uso:
  inferencia final
  submission
```

No se usa para entrenamiento, pseudoetiquetado, early stopping o selección.

### Teachers locales

```text
DINOv3 / DINOv2
SAM ViT-L
```

Solo se usan los pesos ya descargados y versionados. No se incorporan nuevos teachers o datasets sin una futura revisión explícita del protocolo.

## 4.2 Política de cero nuevos datasets

A partir de v6:

```text
PROHIBIDO:
buscar assets TTC adicionales para completar 46 secuencias
incorporar otro dataset TTC
descargar eAP test
obtener labels privados
```

La única descarga permitida es reanudar y completar el train-40 de Hugging Face que ya está en curso.

Toda referencia del artículo a 58 secuencias, 46 train o TTC oficial se conserva como descripción del trabajo publicado, no como un recurso operativo local.

## 4.3 Cómo entrenar Garl-TTC con lo que ya tenemos

Ruta supervisada principal:

```text
EvTTC-32 RGB + eventos + bbox/ROI + TTC oficial
→ Garl-TTC from scratch
→ grouped CV
```

Ruta con pretraining externo:

```text
eAP train-40 RGB + eventos + cajas 3D + tracks
→ pretraining no-TTC de encoders y foreground
→ EvTTC-32 TTC oficial
→ grouped CV
```

Ruta experimental:

```text
eAP train-40
→ pretraining no-TTC
→ pseudo-TTC track-derived con confianza y peso bajo
→ EvTTC-32 TTC oficial
```

El pseudo-TTC nunca sustituye el fine-tuning supervisado EvTTC.

## 4.4 Objetivos Garl no-TTC sobre eAP-40

### Foreground

```text
ROI proyectada desde bbox_3d_ego
+ SAM offline
→ máscara teacher
→ L_foreground
```

La proyección requiere:

1. generar los ocho vértices de `bbox_3d_ego`;
2. verificar la dirección real de `T_event_ego`;
3. proyectar mediante `K_event`;
4. visualizar overlays en al menos 100 frames de varias secuencias;
5. rechazar cajas detrás de cámara o degeneradas;
6. usar el mapping RGB-evento únicamente tras validar el error de alineamiento.

No se acepta una convención de transformación por su nombre; debe pasar la auditoría visual y geométrica.

### Learned Height Ratio sin TTC

Para pares del mismo track:

```text
altura proyectada t1
altura proyectada t2
→ target de h1/h2
```

Esto preentrena la cabeza geométrica de Garl sin calcular TTC.

### Tracking y consistencia temporal

```text
mismo track_id → features próximas
tracks distintos → separación contrastiva
```

### Alineamiento RGB-eventos

```text
RGB ROI
event ROI
→ embeddings de objeto coherentes
```

### JEPA temporal

```text
contexto pasado de la ROI
→ representación futura del mismo objeto
```

## 4.5 ROI para Garl en cada dataset

### En EvTTC

Usar bbox/ROI oficial del protocolo, con redimensionado a `128×128`, para reproducir Garl de forma apples-to-apples.

### En eAP-40

No hay bbox 2D directa auditada. Se construye una ROI derivada:

```text
bbox_3d_ego
→ vértices 3D
→ transformación a cámara de eventos
→ proyección
→ bbox 2D
→ margen configurable
→ clamp a 1280×720
```

Para RGB, el proyecto oficial describe cámaras con baseline estrecha y mapping RGB-evento. En v6 esto se trata como una hipótesis de alineamiento que debe validarse localmente; no se asume perfecta.

Si la proyección RGB falla el gate de overlay, eAP se utiliza para event-only SSL, tracking 3D y geometría, y la rama RGB/SAM de eAP queda desactivada sin bloquear Garl supervisado en EvTTC.

## 4.6 Pseudo-TTC permitido

Solo se permite el artefacto ya auditado:

```text
track-derived pseudo-TTC
```

Reglas:

```text
official_ground_truth = false
confidence por fila
R² local mínimo
velocidad mínima
gaps máximos
rango TTC limitado
peso inicial ≤ 0,05
gradiente total ≤ 15 %
```

No usar:

```text
TTC = depth / velocity_3d[0]
```

## 4.7 Tracks de reproducibilidad

```text
STRICT_EVTTC_GARL
= Garl-TTC entrenado solo con EvTTC-32

EAP40_SSL_GARL
= pretraining no-TTC eAP-40
+ fine-tuning EvTTC-32

EAP40_PSEUDO_GARL
= EAP40_SSL_GARL
+ pseudo-TTC experimental
+ fine-tuning EvTTC-32

FOUNDATION_TEACHER_OGE
= DINO/SAM ya locales
+ eAP-40 SSL
+ fine-tuning EvTTC-32

OGE_BBOX
= comparación directa contra el mejor Garl local

OGE_FULL
= variante bbox-free
```

## 4.8 Claims permitidos

Permitido:

```text
Garl-TTC reimplementado y entrenado en EvTTC-32
Garl-TTC con pretraining no-TTC en eAP Hugging Face train-40
Garl-TTC con pseudo-TTC eAP experimental
```

Prohibido:

```text
reproducción oficial de Garl-TTC en eAP
Garl-TTC entrenado con ground truth TTC eAP
paridad MiD eAP oficial
```

---

# 5. Protocolo de datos sin contaminación y sin desperdiciar secuencias

## 5.1 Corrección fundamental: EvTTC-32 y Benchmark-10 son conjuntos distintos

El protocolo definitivo separa dos paquetes independientes:

```text
EVTTC32_LABELLED
= las 32 secuencias públicas con ground truth TTC
= desarrollo, validación cruzada y entrenamiento final

BENCHMARK10_SEALED
= las 10 secuencias independientes de la competición
= evaluación externa final con ground truth no utilizado para entrenar
```

No se extraerán diez secuencias de EvTTC-32 para construir el benchmark oficial.

La partición histórica `train/validation/family-holdout` se conserva únicamente para cerrar y comparar la matriz experimental ya iniciada. No se reutiliza como protocolo final del modelo SOTA.

## 5.2 Raíces de datos obligatorias

```text
datasets/
├── evttc_complete_staging/           # EvTTC-32 etiquetado
└── evttc_official_benchmark_sealed/  # Benchmark-10 independiente

E:\eAP_dataset/
├── README.md
├── data/
│   ├── train.parquet
│   └── train/<sequence_id>/
│       ├── events.h5
│       ├── labels.parquet
│       └── rgb_shards/*.tar
├── derived/
│   ├── pseudo_ttc_track_v1/
│   ├── sam_masks_rle/
│   ├── dinov3_object_tokens/
│   ├── projected_rois/
│   └── manifests/
├── inventory/
└── logs/
```

No se crea ninguna raíz para otro release eAP.

Manifiestos obligatorios:

```text
data/manifests/eap_hf_train40_manifest.yaml
data/manifests/evttc32_labelled_manifest.yaml
data/manifests/benchmark10_sealed_manifest.yaml
data/splits/evttc32_grouped_cv_5fold.yaml
data/splits/evttc32_final_all32.yaml
```

Los datos originales permanecen inmutables. Los derivados de eAP se escriben exclusivamente en `E:\eAP_dataset\derived\`.

## 5.3 Desarrollo: grouped cross-validation sobre las 32 secuencias

Durante el desarrollo no se entrena directamente con 32/32 y después se consulta el leaderboard.

Se ejecuta validación cruzada agrupada de cinco folds:

```text
Fold 1: ~25–26 train | ~6–7 validation
Fold 2: ~25–26 train | ~6–7 validation
Fold 3: ~25–26 train | ~6–7 validation
Fold 4: ~25–26 train | ~6–7 validation
Fold 5: ~25–26 train | ~6–7 validation
```

Cada una de las 32 secuencias:

- aparece como validation exactamente una vez;
- aparece como train en los otros folds;
- nunca comparte ventanas con train y validation dentro del mismo fold;
- no queda descartada permanentemente.

Los folds deben equilibrar, en la medida posible:

- familia física (`CCRs-*`, `CCRm`, `CPLA`, `CPNA`, `CPNAO`, `CCRs-side`);
- velocidad (`low`, `medium`, `high`);
- overlap (`0`, `50`, `100`);
- objetivo móvil o estático;
- coche o peatón;
- distribución de rangos TTC;
- disponibilidad de RGB, máscara y navegación.

No se permite split aleatorio por muestra o por ventana.

## 5.4 Predicciones out-of-fold

Cada candidato debe producir predicciones out-of-fold para las 32 secuencias:

```text
artifacts/runs/oge_sota/cv/<candidate>/oof_predictions.npz
```

La selección se basa en:

- promedio de los cinco folds;
- bootstrap agrupado por secuencia;
- peor familia;
- peor secuencia;
- TTC bajo;
- TTC alto;
- RMSE;
- error relativo;
- calibración;
- latencia;
- estabilidad entre seeds.

Un candidato no se promociona por ganar solo un fold o una secuencia.

## 5.5 Congelación antes del benchmark

Antes de ejecutar Benchmark-10 se congelan:

- arquitectura;
- resolución;
- streams y modalidades;
- número de horizontes;
- preprocessing;
- pérdidas y pesos;
- profesores;
- augmentations;
- optimizer;
- learning rates;
- número de épocas;
- regla de checkpoint averaging;
- seeds;
- regla de ensemble;
- clipping y calibración;
- commit Git;
- hashes de configuración;
- versión de CUDA/PyTorch;
- método de medida de tiempo.

Se crea:

```text
artifacts/audit/oge_sota/final_freeze_manifest.json
```

El manifiesto debe contener SHA-256 de todos los elementos anteriores.

## 5.6 Entrenamiento final 32/32

Una vez elegida y congelada la configuración mediante CV:

```text
train final = las 32 secuencias de EvTTC-32
test externo = Benchmark-10 independiente
```

Como ya no existe un validation permanente para early stopping, la duración se fija usando exclusivamente los folds:

```text
E_ssl_final = mediana o media recortada de mejores épocas SSL en CV
E_ft_final  = mediana o media recortada de mejores épocas downstream en CV
```

Alternativas permitidas si fueron decididas antes:

- EMA;
- checkpoint averaging de un rango fijo;
- SWA;
- tres seeds;
- ensemble de tres seeds.

No se puede elegir la época, seed o combinación observando el resultado de Benchmark-10.

## 5.7 Candidatos finales

Congelar como máximo los siguientes candidatos independientes:

```text
SINGLE_REALTIME
= un único modelo
= candidato principal de latencia

ENSEMBLE_ACCURACY
= seeds 7, 13 y 21
= regla de fusión fijada por CV
= candidato principal de precisión

EVENT_ONLY
= eventos + navegación

RGBE
= RGB + eventos + navegación
```

El ensemble debe declarar su coste real completo. No se puede informar el tiempo de un único miembro si se ejecutan tres.

## 5.8 Benchmark-10 sellado

Las secuencias oficiales son:

```text
CCRs1-low
CCRs1-medium
CCRs1-high
CCRs2-low
CCRs2-medium
CCRs2-high
CCRm-low
CCRm-medium
Slider-750
Slider-1000
```

El código debe impedir:

- incluir Benchmark-10 en train;
- usarlo para early stopping;
- calcular pérdidas supervisadas sobre él;
- generar pseudolabels TTC;
- cambiar hiperparámetros tras recibir puntuaciones;
- seleccionar la mejor seed mediante el leaderboard;
- calibrar clipping con resultados oficiales.

Registrar:

- manifest SHA-256;
- commit;
- checkpoint SHA-256;
- número y fecha de submissions;
- candidato enviado;
- respuesta oficial;
- cualquier cambio posterior.

## 5.9 Presupuesto de submissions

El leaderboard no se utiliza como validation iterativa.

Política:

```text
1 submission principal de precisión
1 submission principal de tiempo real
variantes adicionales solo si fueron congeladas como claims distintos
```

No se permite:

```text
enviar → mirar score → ajustar → reenviar repetidamente
```

Si se necesita corregir un fallo de formato, se documentará que no hubo cambio de modelo.

## 5.10 Integridad entre ambos paquetes

Los manifests deben demostrar:

- raíces independientes;
- hashes independientes;
- ausencia de rutas compartidas;
- Benchmark-10 sin ground truth TTC consumido por el pipeline;
- ningún cache de teacher o TTC del benchmark en training;
- ningún nombre de secuencia usado como feature;
- ningún identificador de dataset usado como atajo.

---
# 6. Arquitectura final propuesta

## 6.1 Diagrama general v6

```text
Eventos full-frame / RGB opcional / navegación
                        ↓
              Event-JEPA BASE encoder
                        ↓
      memoria de capas H0, H1, ... HL
                        ↓
       Task-Specific Attention Residuals
        ├─ features para máscara
        ├─ features para movimiento
        └─ features para geometría
                        ↓
     Spatial Patch Mixer por cada instante
  atención bidireccional entre patches del mismo frame
                        ↓
              Temporal Mixer causal
        ├─ referencia: block-causal Patch Policy
        └─ candidato: KDA temporal factorada
                        ↓
       TargetQuery + BackgroundQuery
                        ↓
           máscara coarse event-based
                        ↓
        ROIAlign diferenciable high-resolution
                        ↓
          máscara y contorno refinados
                        ↓
 motion tokens + ego-motion compensation
                        ↓
          Stable Geometry Router
   ├─ shared robust path
   ├─ height-ratio expert
   ├─ area-rate expert
   ├─ affine-expansion expert
   └─ event-contrast expert
                        ↓
             geometric inverse-TTC
                        ↓
             bounded neural residual
                        ↓
        TTC final + incertidumbre calibrada
```

Durante entrenamiento, GT/SAM/DINO y los oracles geométricos actúan como profesores mediante **Student-Conditioned Reliability-Gated Multi-Teacher Distillation**. Durante inferencia bbox-free, esos profesores desaparecen.

## 6.2 Principio Patch Policy: condición no negociable

No aplicar global pooling antes de localizar el objetivo.

Secuencia conceptual:

```text
[t-500 ms: P patches]
[t-240 ms: P patches]
[t-100 ms: P patches]
[t-60 ms:  P patches]
[t-20 ms:  P patches]
[t:        P patches]
```

Reglas:

- todos los patches de un mismo instante pueden comunicarse bidireccionalmente;
- un instante actual puede consultar instantes pasados;
- ningún instante pasado puede consultar el futuro;
- la navegación solo existe hasta `context_end`;
- la `TargetQuery` accede a los patches densos, no a un promedio global.

Esto preserva la información espacial que el modelo actual puede perder al resumir toda la escena.

## 6.3 Kimi Delta Attention sin conflicto con Patch Policy

### El conflicto si se aplica de forma ingenua

KDA es causal sobre una secuencia ordenada. Si se aplana un frame así:

```text
patch_1 → patch_2 → patch_3 → ... → patch_P
```

`patch_1` no puede ver `patch_P`, aunque ambos pertenezcan al mismo instante. Eso rompe la propiedad esencial de Patch Policy: el frame no es una secuencia temporal de patches.

### Solución: atención espaciotemporal factorada

KDA solo puede actuar sobre el eje temporal, después de resolver la interacción espacial:

```text
Por cada tiempo t:
    Z_t = SpatialSelfAttention(all patches at t)

Después:
    historial temporal de cada región/slot
    → Temporal KDA

Periódicamente:
    Global Block-Causal Attention
```

Variantes autorizadas:

```text
T0_BLOCK_CAUSAL
    todos los patches con máscara block-causal

T1_OBJECT_KDA
    Patch Policy espacial
    → TargetQuery/object tokens
    → KDA temporal solo sobre object/region tokens

T2_ALIGNED_PATCH_KDA
    Patch Policy espacial
    → patches alineados por ego-motion/movimiento
    → KDA temporal por posición/región
    → atención global block-causal periódica
```

`T1_OBJECT_KDA` es la primera variante que se probará porque conserva toda la información espacial y usa KDA únicamente para recordar la evolución del objeto.

### Decisión de inclusión

KDA **no forma parte obligatoria del primer candidato de precisión**. Es un candidato de eficiencia y contexto largo.

Se promociona solo si cumple al menos uno:

```text
A) mejora e_TTC relativo ≥ 3 % a igual resolución y latencia aceptable
B) permite ≥ 2× tokens o ≥ 2× horizonte con pérdida de precisión ≤ 1 %
C) reduce memoria pico ≥ 30 % sin empeorar peor secuencia
```

Si no cumple esos gates, el modelo final conserva block-causal attention estándar.

## 6.4 Attention Residuals: inclusión recomendada

Las capas tempranas suelen conservar bordes y posición; las intermedias, movimiento y partes; las profundas, objeto y contexto. Usar solo la última capa crea un cuello de botella y puede ser especialmente dañino cuando la última representación deriva hacia baja varianza.

Para cada token espacial se almacena:

```text
H0 = embedding inicial
H1 = salida bloque 1
...
HL = salida bloque L
```

Cada cabeza dispone de una pseudo-query propia:

```text
MaskAttnRes     → prioriza bordes y semántica
MotionAttnRes   → prioriza correspondencia temporal
GeometryAttnRes → prioriza escala y expansión
RiskAttnRes     → prioriza contexto profundo
```

Las salidas de capas se normalizan con RMSNorm antes de calcular pesos, evitando que una capa domine únicamente por tener mayor magnitud.

Primera implementación:

```text
TaskSpecificFullAttnRes
```

Como el backbone es pequeño, conservar todas las capas es asequible. Si la profundidad aumenta, usar `BlockAttnRes`.

Gate de promoción:

```text
mejora e_TTC ≥ 3 %
o mask IoU +3 puntos con TTC no peor
latencia p95 aumenta ≤ 15 %
ninguna capa recibe >90 % del peso en todas las muestras
```

Attention Residuals no entra en conflicto con Patch Policy: Patch Policy mezcla información en espacio y tiempo; AttnRes selecciona información a través de la profundidad.

## 6.5 Backbone Event-JEPA

Punto de partida:

```text
EventTubeletTransformer de BASE
```

Cambios:

1. exponer tokens densos antes del pooling;
2. devolver outputs de todas las capas para AttnRes;
3. añadir salidas multi-escala;
4. conservar posiciones espaciales;
5. permitir activación de bloques finales durante fine-tuning;
6. devolver health metrics por capa:
   - std;
   - effective rank;
   - covariance;
   - norm;
   - token diversity;
   - pesos de AttnRes.

No se sustituye el backbone hasta demostrar que la cabeza densa mejora al head global.

## 6.6 Diseño global + detalle

Para mantener resolución sin atención cuadrática sobre 1280×720:

### Stream global

```text
320×180 o 640×360
```

Funciones:

- localizar el objetivo;
- contexto de escena;
- navegación;
- trayectoria;
- máscara coarse.

### Stream de detalle

Extraer desde la representación nativa:

```text
crop 256×256 o 384×384
```

centrado en la máscara predicha mediante ROIAlign diferenciable.

Funciones:

- bordes;
- altura;
- área;
- expansión;
- normal flow;
- contorno fino.

Esto permite usar información nativa sin atención global sobre decenas de miles de tokens.

## 6.7 Decisión sobre Stable LatentMoE

No se implementará un MoE neuronal masivo en el backbone.

Motivos:

- EvTTC tiene muy pocas secuencias para aprender cientos de especializaciones;
- un router grande puede memorizar familias o velocidades;
- aumenta la complejidad sin atacar directamente el cuello de botella TTC;
- la ganancia de Kimi K3 está ligada a una escala de modelo y datos completamente distinta.

Sí se adoptan sus principios de estabilidad en el router geométrico:

```text
RMSNorm antes del router
shared expert siempre activo
cuatro expertos físicos interpretables
latent router width pequeño: 64–96
soft routing durante entrenamiento
top-2 opcional en inferencia
activación acotada
balance débil y auditable
```

Esta adaptación se denomina `StableGeometryRouter`, no `StableLatentMoE`.

# 7. Object queries y enmascaramiento

## 7.1 Fase inicial: una `TargetQuery`

EvTTC define un objetivo TTC principal. La primera versión usará:

```text
1 TargetQuery
1 BackgroundQuery
```

La query produce:

- score de objeto;
- centro;
- escala;
- máscara suave;
- embedding;
- confianza.

Máscara:

```math
m_i = sigmoid(q_target^T W z_i)
```

No se usará un crop binario duro.

## 7.2 Máscara suave

La máscara pondera:

```text
eventos_objeto = m ⊙ eventos
eventos_fondo  = (1-m) ⊙ eventos
```

También pondera el solver:

```text
w_i = mask_i × motion_confidence_i × event_quality_i
```

Ventajas:

- conserva gradientes;
- tolera bordes inciertos;
- reduce contaminación de fondo;
- evita que el rectángulo bbox se trate como silueta exacta.

## 7.3 Máscara teacher

Orden de supervisión:

1. segmentación GT de EvTTC;
2. bbox GT como fallback;
3. SAM con bbox como prompt para refinar el contorno;
4. DINO para destilación densa y consistencia temporal.

SAM y DINO no son el detector final.

Solo enseñan al `MaskDecoder` event-based.

## 7.4 Multi-query posterior

Solo después de superar el gate de una query:

```text
4 object queries + 1 background query
```

Se añadirá:

- Hungarian matching;
- identidad temporal;
- `no-object`;
- TTC por objeto;
- selector de riesgo.

No introducir multi-query antes de demostrar que `TargetQuery` funciona.

---

# 8. Uso de RGB y profesores

## 8.0 Entorno local y presupuesto de recursos

Hardware disponible:

```text
GPU: NVIDIA GeForce RTX 5070 Ti Laptop
VRAM: aproximadamente 12,8 GB
RAM: 32 GB
CPU: Ryzen 9, 32 threads lógicos
disco del proyecto C:: ~300 GB libres
disco externo E:: ~700 GB libres antes de eAP
```

Entorno detectado:

```text
Python: 3.11.15
PyTorch: 2.11.0+cu128
CUDA build: 12.8
CUDA disponible: True
```

Pareja requerida para los image processors y ops:

```text
torchvision: 0.26.0+cu128
```

Debe validarse con `torchvision.ops.nms` antes de iniciar la extracción de teachers.

### Regla de utilización

No cargar simultáneamente DINOv3 ViT-L y SAM ViT-L en GPU.

Pipeline:

```text
fase DINO
→ cargar DINOv3
→ recorrer una secuencia
→ escribir cache
→ liberar proceso

fase SAM
→ cargar SAM
→ recorrer una secuencia
→ escribir máscaras comprimidas
→ liberar proceso
```

El aislamiento por proceso evita fragmentación de VRAM y garantiza liberación completa.

## 8.0.1 Modelos ya descargados

Repositorios observados en la caché local:

| Rol | Repo Hugging Face | Uso |
|---|---|---|
| teacher principal | `facebook/dinov3-vitl16-pretrain-lvd1689m` | features densas y correspondencia |
| teacher alternativo | `facebook/dinov3-vitl16-pretrain-sat493m` | ablación de dominio |
| teacher rápido | `facebook/dinov3-vits16-pretrain-lvd1689m` | smoke tests y desarrollo |
| teacher ligero | `facebook/dinov3-convnext-tiny-pretrain-lvd1689m` | extracción económica |
| ablación histórica | `facebook/dinov2-large` | comparación DINOv2 vs DINOv3 |
| teacher de máscaras | `facebook/sam-vit-large` | pseudomáscaras offline |

Snapshots ya verificados anteriormente:

```text
DINOv3 ViT-L LVD:
ea8dc2863c51be0a264bab82070e3e8836b02d51

SAM ViT-L:
6851e0441005b0fb96f2cc4dfac472f3d1b14af1
```

La caché Hugging Face por defecto en Windows suele resolver a:

```text
C:\Users\<usuario>\.cache\huggingface\hub
```

pero el plan no debe asumir esa ruta. Hay que resolver la localización real:

```powershell
$Py = ".\.venv\Scripts\python.exe"

@'
from pathlib import Path
from huggingface_hub import scan_cache_dir

roots = [
    Path.home() / ".cache" / "huggingface" / "hub",
    Path(r"E:\eAP_dataset\.hf\hub"),
    Path(r"E:\eAP_dataset\.cache\huggingface\hub"),
]

wanted = {
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "facebook/dinov3-vitl16-pretrain-sat493m",
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "facebook/dinov2-large",
    "facebook/sam-vit-large",
}

seen = set()

for root in roots:
    if not root.exists():
        continue

    print(f"\nCACHE: {root}")
    info = scan_cache_dir(root)

    for repo in info.repos:
        if repo.repo_id not in wanted:
            continue

        seen.add(repo.repo_id)
        revisions = sorted(
            repo.revisions,
            key=lambda item: item.last_modified,
            reverse=True,
        )

        print(f"\n{repo.repo_id}")
        print(f"  repo_path: {repo.repo_path}")
        print(f"  size: {repo.size_on_disk / 1024**3:.3f} GiB")

        for revision in revisions:
            print(f"  snapshot: {revision.snapshot_path}")
            print(f"  commit:   {revision.commit_hash}")

missing = wanted - seen
if missing:
    print("\nNO LOCALIZADOS EN LAS RAÍCES INSPECCIONADAS:")
    for repo_id in sorted(missing):
        print(" ", repo_id)
'@ | & $Py -
```

Guardar la salida en:

```text
artifacts/audit/oge_sota/foundation_model_cache_inventory.txt
```

## 8.0.2 Política de selección de teacher

Orden:

```text
1. DINOv3 ViT-L LVD    → teacher principal final
2. SAM ViT-L           → máscaras offline
3. DINOv3 ViT-S LVD    → smoke y depuración
4. DINOv2 Large        → ablación, no teacher principal
5. DINOv3 SAT          → ablación de dominio
6. ConvNeXt Tiny       → fallback de velocidad
```

No se ejecutará una matriz completa con todos los teachers. Solo se promueve una alternativa si supera a DINOv3 ViT-L LVD en OOF o reduce sustancialmente coste sin pérdida relevante.

## 8.0.3 Extracción DINOv3 en 12 GB

Configuración inicial:

```text
dtype del modelo: FP16 o BF16
inference_mode: true
gradient: desactivado
microbatch inicial: 4 imágenes
microbatch fallback: 2
resolución global: 320×180 o 384×216
salida: tokens objeto/borde reducidos a 256 dimensiones
workers de lectura: 4
pin_memory: true
prefetch_factor: 2
```

No guardar por defecto todos los hidden states de todas las capas para todos los frames. Una caché densa FP16 de ViT-L puede consumir decenas de GB.

Guardar:

```text
object token
background token
boundary tokens muestreados
feature grid final opcional
projection 1024 → 256 FP16
teacher confidence
modelo, commit y preprocessing hash
```

Ablación puntual:

```text
full final-layer dense tokens
```

solo para un subconjunto y para medir si la compresión 256D pierde información.

## 8.0.4 Extracción SAM en 12 GB

Configuración:

```text
microbatch: 1
dtype: FP16
prompt prioritario: bbox 3D proyectada o bbox 2D pública
salida: RLE/bit-packed mask
postproceso: componente conectada compatible con bbox
temporal filtering: opcional y offline
```

No guardar logits SAM FP32 de resolución completa para todo eAP. Guardar:

```text
mask binaria comprimida
bbox refinada
IoU score
stability score
teacher validity
hash del prompt
```

## 8.0.5 Caches y capacidad de disco

Dado que el train completo de eAP ocupa aproximadamente 0,58 TB, todos los caches masivos deben permanecer en E:.

Presupuesto orientativo:

```text
eAP train original:              ~576 GB, confirmar con --dry-run
pseudo-TTC + Parquets:            <5 GB
SAM masks comprimidas:           5–20 GB
DINO object tokens 256D FP16:    10–30 GB
event voxels/indices:            generar bajo demanda o 20–60 GB
margen libre obligatorio:        ≥50 GB
```

No extraer todos los TAR de RGB a disco. Leer cada miembro PNG directamente desde el TAR o extraer una secuencia temporalmente y borrarla.


## 8.0.6 Prohibiciones duras de almacenamiento

Estas reglas son obligatorias, no recomendaciones.

### Prohibición 1 — DINO todas las capas

```text
PROHIBIDO:
guardar hidden states de todas las capas de DINO
para todos los frames de eAP
```

Permitido:

```text
última capa compactada a 256D
object/background/boundary tokens
full dense final-layer solo en un subconjunto auditado
```

### Prohibición 2 — logits SAM completos

```text
PROHIBIDO:
guardar logits SAM FP32/FP16 de resolución completa
para todo eAP
```

Guardar únicamente:

- máscara RLE o bit-packed;
- bbox refinada;
- scores;
- hashes.

### Prohibición 3 — voxel cache completo

```text
PROHIBIDO:
precomputar y conservar voxels de eventos
para todas las muestras, horizontes y resoluciones
```

Usar HDF5 + índices temporales y generar voxels bajo demanda. Solo se permite una caché parcial si un benchmark de loader demuestra que I/O limita la GPU y el cache queda bajo presupuesto.

### Prohibición 4 — extracción masiva de RGB

```text
PROHIBIDO:
extraer todos los rgb_shards/*.tar a PNG
```

Leer miembros directamente desde TAR o extraer una secuencia temporalmente y borrarla tras verificar el cache.

### Prohibición 5 — nuevos datasets o releases

```text
PROHIBIDO:
crear una segunda raíz eAP
descargar assets alternativos
descargar eAP test
incorporar otro dataset TTC
```

El guard de configuración debe aceptar únicamente:

```text
EAP_HF_TRAIN40
EVTTC32_LABELLED
BENCHMARK10_SEALED
```

### Prohibición 6 — todos los checkpoints

```text
PROHIBIDO:
guardar un checkpoint completo por epoch
en todos los folds, seeds y ablations
```

Por run:

```text
last.pt
best.pt
weights_only.pt
```

Tras rechazar un brazo:

- conservar config, métricas y manifest;
- eliminar optimizer/scheduler state salvo necesidad de auditoría;
- conservar pesos solo si aportan una ablación publicable.

### Presupuestos

```text
E:\eAP_dataset\derived total:  ≤55 GiB
DINO cache:                    ≤25 GiB
SAM cache:                     ≤15 GiB
event indices/cache:            ≤8 GiB
pseudo-TTC y metadata:          ≤5 GiB
espacio libre E obligatorio:   ≥50 GiB

checkpoints activos en C:      ≤80 GiB
```

El pipeline debe abortar antes de escribir cuando la estimación exceda el presupuesto.

## 8.0.7 Separación teacher/student


Durante entrenamiento final:

```text
teachers precomputados
→ no cargar DINOv3/SAM
→ GPU dedicada al Event-JEPA student
```

Esto permite:

```text
batch student: 1–2
gradient accumulation: 16–32
gradient checkpointing: true
BF16/FP16
solver geométrico: FP32
```

DINOv3 y SAM no forman parte de la latencia de inferencia salvo en una variante RGBE explícita distinta.


## 8.1 Teacher DINO

Durante entrenamiento:

```text
RGB → DINO frozen → dense RGB tokens
eventos → Event-JEPA → dense event tokens
```

Aplicar una proyección aprendida y destilación:

```math
L_DINO = 1 - cosine(P(z_event), stopgrad(z_DINO))
```

La pérdida se calcula principalmente:

- dentro del objeto;
- en el borde;
- en correspondencias temporales;
- no como alineamiento global de toda la escena.

## 8.2 Teacher SAM

Uso:

```text
RGB + bbox GT → SAM frozen → pseudomáscara precisa
```

La máscara se almacena offline. SAM no se carga durante cada batch.

## 8.3 Student-Conditioned Reliability-Gated Multi-Teacher Distillation

La Multi-Teacher On-Policy Distillation de un LLM no se puede copiar literalmente: aquí no hay generación token a token ni rollouts de RL. Se adapta su principio útil:

> Un solo estudiante debe consolidar capacidades de varios profesores especializados y debe entrenarse sobre los estados que él mismo producirá en inferencia.

Profesores:

| Profesor | Enseña | No debe enseñar |
|---|---|---|
| GT mask/bbox | localización correcta | atajos de secuencia |
| SAM | contorno y frontera | TTC |
| DINO | features densas y correspondencia | TTC directo |
| Geometry oracle | altura, área, expansión y confianza | máscara predicha |
| EMA Event-JEPA | estabilidad temporal de features | etiquetas TTC |

### Componente “student-conditioned”

No alimentar siempre al solver con el crop GT. El estudiante predice su máscara y ROI, y los profesores supervisan ese estado producido por el estudiante:

```text
student mask
→ student ROI
→ teacher feedback sobre esa ROI
```

Esto reduce el train–inference mismatch.

Curriculum:

```text
Etapa A: 100 % ROI GT
Etapa B: mezcla ROI GT / ROI predicha
Etapa C: 100 % ROI predicha + teacher consistency
```

### Reliability gating

Cada señal teacher recibe un peso por muestra:

```text
r_GT       = 1 si anotación válida
r_SAM      = acuerdo con bbox/GT + estabilidad temporal
r_DINO     = confianza de correspondencia
r_geometry = soporte de eventos + condición del solver
r_EMA      = salud latente y consistencia temporal
```

La loss teacher final es una combinación normalizada y con `stop_gradient` en todos los profesores.

Gates:

```text
ningún teacher > 20 % de grad norm total
teacher disagreement alto → bajar peso, no promediar ciegamente
PredROI curriculum mejora ≥ 3 % frente a teacher-forcing puro
```

Nombre del módulo:

```text
SC-RGMTD = Student-Conditioned Reliability-Gated Multi-Teacher Distillation
```

## 8.4 Variante RGB+eventos de inferencia

Para intentar superar también Image’s FoE:

```text
OGE-JEPA-TTC-RGBE
```

Usará DINO/RGB features ligeras durante inferencia o un encoder RGB destilado.

Motivo:

- los eventos son fuertes a TTC bajo;
- RGB ayuda a TTC alto, cuando hay poca actividad;
- el actual fallo CCRm concentra mucho error en TTC altos.

El modelo final debe publicar dos resultados:

```text
E-NAV      = eventos + navegación
RGBE-NAV   = RGB + eventos + navegación
```

El candidato absoluto SOTA será `RGBE-NAV`, salvo que `E-NAV` lo supere.

# 9. Geometría diferenciable

La red no debe predecir TTC directamente como única ruta.

## 9.1 Variable principal

Trabajar con:

```math
q = 1 / TTC
```

porque la aproximación se relaciona linealmente con expansión.

La salida final:

```math
TTC = 1 / (softplus(q) + epsilon)
```

## 9.2 Experto 1: height ratio

A partir de alturas aparentes:

```math
TTC_h = Δt / (1 - h_1 / h_2)
```

La altura se calcula desde la máscara o una cabeza de extremos verticales.

## 9.3 Experto 2: area expansion

```math
q_A ≈ 0.5 × d(log A)/dt
```

Robusto a cambios de anchura y útil cuando la altura es imperfecta.

## 9.4 Experto 3: affine radial expansion

Para cada token del objeto:

```text
desplazamiento =
traslación
+ expansión radial
+ rotación
```

Resolver mediante mínimos cuadrados ponderados:

```math
θ = (Xᵀ W X + λI)^(-1) Xᵀ W y
```

Parámetros:

```text
tx, ty, kappa, omega
```

`kappa` produce inverse-TTC geométrico.

## 9.5 Experto 4: event contrast alignment

Compensar eventos con el movimiento estimado.

La solución correcta debe aumentar el contraste/alineamiento del objeto.

No se ejecutará una optimización CMax lenta completa. Se usará:

- una o dos iteraciones diferenciables;
- initialization de la cabeza neuronal;
- regularización por contraste;
- sin bucle de varios segundos.

## 9.6 Stable Geometry Router

No confiar en una sola fórmula y no introducir un MoE neuronal masivo.

El router recibe únicamente features físicas y de calidad:

- confianza y área de máscara;
- soporte de eventos;
- condición numérica de cada solver;
- rotación ego;
- oclusión;
- estabilidad temporal;
- acuerdo entre horizontes;
- rango TTC preliminar.

Arquitectura:

```text
router_input
→ RMSNorm
→ latent projection 64–96
→ bounded GLU/MLP
→ soft routing weights
```

Salida:

```text
q_shared = experto robusto común
q_routed = g_h q_height + g_A q_area + g_aff q_affine + g_evt q_event
q_geo    = q_shared + q_routed
```

Durante entrenamiento se usa soft routing. `top-2` solo se habilita en inferencia si no reduce precisión.

Audits:

- utilización por experto;
- entropía del router;
- distribución por rango TTC;
- distribución por familia;
- sensibilidad a `sequence_id` eliminado;
- colapso a un único experto;
- especialización física interpretable.

Gate:

```text
mejora e_TTC ≥ 2 % frente a confidence-weighting determinista
ningún experto > 85 % de uso global
router no predice la identidad de secuencia por atajo
```

Si falla, sustituir el router aprendido por pesos deterministas basados en confianza geométrica.

## 9.7 Residual neuronal limitado

```math
q_final = softplus(q_geo + α × tanh(Δq))
```

Restricciones:

- `α` pequeño y aprendible;
- el residual no puede borrar arbitrariamente la geometría;
- registrar por muestra:
  - `q_geo`;
  - `Δq`;
  - `q_final`;
  - pesos de expertos.

---

# 10. Compensación explícita de movimiento ego

La navegación no se concatenará únicamente al embedding final.

## 10.1 Uso físico

Usar:

- velocidad;
- aceleración;
- yaw rate;
- intrínsecos;
- intervalo temporal;

para estimar el flujo aparente inducido por ego-motion.

Pipeline:

```text
movimiento observado
− componente explicada por ego
= movimiento residual del objeto
```

## 10.2 Audits anti-shortcut

Ejecutar obligatoriamente:

| Ablation | Qué prueba |
|---|---|
| `NAV_ONLY` | cuánto TTC puede predecirse sin eventos |
| `NAV_SHUFFLED` | si navegación actúa como identificador de escenario |
| `NAV_TIME_SHIFTED` | sensibilidad a sincronización causal |
| `NAV_NOISY` | robustez a sensores |
| `EVENT_ZERO` | dependencia real de eventos |
| `RGB_ZERO` | dependencia real de RGB |
| `MASK_RANDOM` | dependencia real de localización |

Gate:

```text
NAV_SHUFFLED debe empeorar ≥ 20 %
NAV_ONLY no debe acercarse a menos del 80 % del modelo completo
```

Si no se cumple, hay shortcut.

---

# 11. Funciones de pérdida

## 11.1 Supervisión TTC

Usar combinación:

```math
L_TTC =
Huber(log(1+TTC_pred), log(1+TTC_gt))
+ λ_rel × |TTC_pred - TTC_gt| / max(TTC_gt, ε)
```

Añadir pesos por rango para evitar dominancia de TTC medios:

```text
0,8–2 s
2–4 s
4–6 s
6–10+ s
```

Cada bin debe contribuir de forma equilibrada.

## 11.2 Máscara

```math
L_mask = L_BCE + L_Dice + L_boundary
```

## 11.3 Tracking temporal

```math
L_track =
1 - cosine(slot_t, slot_t-1)
+ mask_warp_consistency
```

## 11.4 Movimiento

```math
L_motion = robust_huber(predicted_motion, pseudo_or_geometry_motion)
```

No requiere ground truth de optical flow si se usa consistencia temporal y contraste.

## 11.5 Geometría

```math
L_geo =
Huber(q_geo, 1/TTC_gt)
+ consistency(q_height, q_area, q_affine)
```

Solo durante entrenamiento supervisado.

## 11.6 Multi-teacher distillation

```math
L_{teachers} = Σ_k r_k · normalize(L_k)
```

Donde cada `r_k` es la fiabilidad del teacher y cada profesor está detenido con `stop_gradient`. La selección de ROI pasa gradualmente de GT a predicción del estudiante.

Incluye:

- DINO distillation en tokens objeto y bordes;
- SAM/GT mask distillation;
- geometry-oracle distillation;
- EMA temporal consistency.

## 11.7 Contraste de eventos

```math
L_contrast = -contrast(warp(events_object, θ))
```

con límites para evitar soluciones degeneradas.

## 11.8 Varianza adaptativa

No usar:

```text
variance_weight = 0
```

ni mantener ciegamente:

```text
variance_weight = 1.0
```

Usar hinge adaptativo:

```math
L_var = mean(ReLU(min_std - std_dimension))
```

y peso programado.

Gate de salud:

```text
context std ≥ 0,05
predictor std ≥ 0,05
effective rank no cae > 20 %
```

## 11.9 Estabilidad del router

Regularización débil de entropía/load balance únicamente para impedir colapso completo. No se fuerza uniformidad: se permite que un experto domine cuando la evidencia física lo justifica.

Penalizar además dependencia espuria de metadatos de secuencia.

## 11.10 Balance de pérdidas

Cada pérdida se normaliza por EMA de su magnitud.

Regla obligatoria:

```text
ninguna pérdida auxiliar puede aportar >20 % de la norma de gradiente total
```

Implementar:

- logging de grad norm por loss;
- clipping por componente;
- GradNorm o weighting por incertidumbre;
- alarmas si una loss domina tres epochs consecutivos.

Esto evita repetir el fallo de `INVERSE`.

---


## 11.11 Pseudo-TTC eAP con confianza

Solo en `EAP_PSEUDO_TTC_EXPLORATORY`:

```math
L_{pseudo} =
confidence
\times
Huber(log(1+|TTC_{pred}|), log(1+|pseudoTTC|))
```

Reglas:

```text
peso inicial: 0,05
confidence mínima: 0,50
|range_rate| mínimo: 0,5 m/s
local R² mínimo: 0,70
|pseudo-TTC| máximo inicial: 10 s
máximo gradiente total: 10–15 %
```

Separar acercamiento y alejamiento mediante signo o una cabeza binaria; no aplicar `log(1+TTC)` a valores negativos sin transformación explícita.

Gate:

```text
EAP_SSL + pseudo-TTC
debe superar
EAP_SSL sin pseudo-TTC

en ≥4/5 folds EvTTC y sin empeorar el peor fold >5 %
```

Si falla, eliminar totalmente esta loss y conservar eAP solo para SSL/percepción.

# 12. Selección de checkpoint

No seleccionar únicamente por SSL validation loss.

## 12.1 Pretraining

Composite score:

```text
future prediction loss
+ collapse penalty
+ token diversity penalty
+ mask-free temporal consistency
```

Early stopping si:

- no mejora en 4 epochs;
- std cae bajo umbral;
- effective rank cae de forma persistente.

## 12.2 Downstream

Seleccionar por grouped-validation:

```text
score =
mean_relative_error
+ 0.25 × normalized_RMSE
+ 0.25 × high_TTC_error
+ 0.25 × low_TTC_safety_error
```

Desempate:

1. peor secuencia;
2. RMSE;
3. seed variance;
4. latencia.

---

# 13. Fases de implementación y gates



## Fase G0 — congelar la especificación Garl-TTC

```text
ROI: 128×128
dos instantes
Δt objetivo: 0,1 s cuando el dataset lo permita
RGB encoder separado
event encoder separado
late fusion
height-ratio head
foreground decoder solo en training
```

Registrar cualquier diferencia necesaria por el formato EvTTC.

## Fase G1 — adapters de datos existentes

Implementar:

```text
EvTTCGarlDataset
Eap40GarlPretrainDataset
ProjectedEapRoiBuilder
GarlPairSampler
```

Gate de proyección eAP:

```text
≥100 overlays inspeccionados
≥95 % de ROIs válidas en el subconjunto auditado
mediana de objeto dentro de ROI ≥95 %
```

Si falla, desactivar RGB/SAM eAP y mantener event-only SSL.

## Fase G2 — baselines directas EvTTC

```text
G0_DIRECT_RGB
G1_DIRECT_EVENT
G2_DIRECT_RGBE_EARLY
```

## Fase G3 — Learned Height Ratio EvTTC

```text
G3_LHR_RGB
G4_LHR_EVENT
G5_LHR_RGBE_EARLY
G6_LHR_RGBE_LATE
```

Gate:

```text
LHR supera regresión directa
late fusion supera o iguala early fusion
sin empeorar peor fold >10 %
```

## Fase G4 — foreground supervision EvTTC

```text
G7_GARL_EVTTC
= G6 + SAM/GT foreground teacher
```

SAM se ejecuta offline; el decoder no existe en inferencia.

## Fase G5 — pretraining Garl sobre eAP-40 sin TTC

Preentrenar:

```text
RGB/event encoders
foreground decoder
height prediction
height-ratio consistency
track consistency
RGB-event alignment
JEPA temporal por objeto
```

Resultado:

```text
G8_GARL_EAP40_SSL_EVTTC
= pretraining eAP-40
+ fine-tuning G7 en EvTTC-32
```

Gate:

```text
G8 supera G7 en ≥4/5 folds
mejora media OOF ≥3 %
peor fold no empeora >5 %
```

## Fase G6 — pseudo-TTC eAP opcional

```text
G9_GARL_EAP40_PSEUDO_EVTTC
= G8
+ pseudo-TTC weight 0,05 durante pretraining
+ fine-tuning EvTTC idéntico
```

Si G9 no supera G8, se elimina.

## Fase G7 — export y runtime

Exportar el mejor de G7, G8 o G9 promocionado. Medir preparación de ROI, voxelización, encoders, head y postproceso.

## Fase E0 — completar e inventariar la descarga eAP-40 ya iniciada

No iniciar nuevas fuentes. Reanudar únicamente `data/train/**`.

Gate:

```text
40 labels.parquet
40 events.h5
0 archivos test
inventario local completo
margen libre ≥50 GiB
```

## Fase E1 — pseudo-TTC y ROIs derivados

```text
projected_rois/
pseudo_ttc_track_v1/
projection_audit/
```

## Fase E2 — caches SAM y DINOv3

```text
SAM masks RLE
DINOv3 object/boundary tokens 256D
```

## Fase E3 — pretraining general eAP para OGE

```text
masked future JEPA
RGB-event alignment
object consistency
tracking
box geometry
foreground
```

Primero sin pseudo-TTC.

## Fase E4 — pseudo-TTC auxiliar OGE

Comparar `EAP_SSL` frente a `EAP_SSL + PSEUDO_TTC_0.05` y promocionar solo mediante OOF EvTTC.

---

## Fase 0 — congelar el experimento actual

Entregables:

- resumen multisemilla final;
- hashes;
- ranking;
- confirmación de holdout cerrado;
- baseline `BASE` seleccionado.

No cambiar el pipeline v6 a mitad de ejecución.

Gate:

```text
matriz actual completa y auditada
```

---

## Fase 1 — protocolo EvTTC-32 CV + Benchmark-10 sellado

Implementar:

```text
scripts/build_evttc32_grouped_cv.py
scripts/build_benchmark10_manifest.py
scripts/verify_benchmark10_seal.py
```

Entregables:

- manifest completo de las 32 secuencias etiquetadas;
- cinco folds grouped sobre las 32;
- reporte de distribución por familia, velocidad, overlap y rango TTC;
- manifest independiente del Benchmark-10;
- guard que impida utilizar Benchmark-10 durante train o selección;
- prueba de que cada secuencia de EvTTC-32 aparece una vez como validation;
- esquema de entrenamiento final 32/32.

Gate:

```text
32/32 secuencias cubiertas por CV
cada secuencia aparece una vez como validation
0 ventanas cruzan train/validation
0 rutas de Benchmark-10 entran en train
100 % de hashes válidos
```

---
## Fase 2 — oracle geométrico

Sin object queries.

Variantes:

```text
GEO_BBOX_HEIGHT
GEO_MASK_HEIGHT
GEO_MASK_AREA
GEO_MASK_AFFINE
GEO_MASK_MIXTURE
```

Usar bbox/máscara GT para responder:

> Si el objeto estuviera perfectamente localizado, ¿la geometría puede batir el baseline?

Gate mínimo EvTTC-32 grouped CV:

```text
mean relative error < 7 %
CCRm-like < 12 %
```

Gate de promoción ambicioso:

```text
mean relative error < 5,5 %
```

Si `GEO_MASK_MIXTURE` no supera `BASE`, no implementar todavía detección completa. Primero corregir geometría.

---

## Fase 3A — Dense Patch TTC

Comparar:

```text
BASE_GLOBAL_HEAD
vs
BASE_DENSE_PATCH_HEAD
```

Encoder congelado inicialmente.

Sin máscaras ni geometría nueva.

Gate:

```text
mejora emparejada ≥ 5 % en 4/5 folds
sin degradar peor secuencia > 5 %
```

Esto prueba directamente si el pooling global era cuello de botella.

---

## Fase 3B — Task-Specific Attention Residuals

Comparar:

```text
LAST_LAYER_ONLY
LEARNED_LAYER_SUM
TASK_SPECIFIC_ATTNRES
```

Cabezas independientes para máscara, movimiento y geometría.

Gate:

```text
e_TTC mejora ≥ 3 %
o mask IoU +3 puntos sin empeorar TTC
latencia p95 +≤15 %
```

AttnRes se incorpora al candidato principal solo si pasa este gate.

---

## Fase 3C — Hybrid Temporal KDA, ablation condicional

Comparar:

```text
T0_BLOCK_CAUSAL
T1_OBJECT_KDA
T2_ALIGNED_PATCH_KDA_GLOBAL_REFRESH
```

No sustituir interacción espacial por recurrencia causal.

Gate de promoción:

```text
precisión +≥3 % a igual compute
o memoria −≥30 % sin pérdida
o 2× resolución/horizonte con degradación ≤1 %
```

Si no pasa, KDA queda fuera del modelo final.

---

## Fase 4 — TargetQuery y máscara event-only

Entrenar:

```text
dense tokens
→ TargetQuery
→ máscara coarse
```

Teachers mediante SC-RGMTD:

- GT segmentation/bbox;
- SAM-refined mask;
- DINO dense features;
- geometry oracle;
- EMA Event-JEPA.

Curriculum obligatorio:

```text
GT ROI → mixed ROI → predicted ROI
```

Métricas:

- IoU;
- Dice;
- target recall;
- center error;
- temporal consistency.

Gates:

```text
target recall ≥ 98 %
mask IoU medio ≥ 0,75
center error ≤ 5 % de la diagonal
```

---

## Fase 5 — refiner high-resolution

Añadir:

```text
coarse query
→ differentiable ROI
→ high-resolution event/RGB tokens
→ refined mask
```

Gate:

```text
IoU +5 puntos
o error geométrico −10 %
```

---

## Fase 6 — geometría diferenciable por objeto

Integrar:

- height;
- area;
- affine;
- contrast;
- confidence;
- ego compensation.

Primero con máscara GT y después con máscara predicha.

Gate:

```text
PredMask-GEO ≤ GTMask-GEO + 10 % relativo
```

---

## Fase 6B — Stable Geometry Router

Comparar:

```text
DETERMINISTIC_CONFIDENCE_MIX
SOFT_STABLE_ROUTER
TOP2_STABLE_ROUTER
```

Gate:

```text
SOFT_STABLE_ROUTER mejora ≥2 %
sin colapso de experto
sin dependencia de identidad de secuencia
```

`TOP2` solo se usa si reduce latencia sin perder precisión.

---

## Fase 7 — residual y uncertainty

Añadir residual limitado y NLL/calibración.

Gate:

```text
MAE mejora ≥ 3 %
RMSE no empeora
ECE/confidence mejora
residual medio < 30 % de q_geo
```

Si el residual domina, reducirlo o eliminarlo.

---

## Fase 8 — RGBE

Añadir RGB en inferencia:

```text
RGB dense tokens + event dense tokens
→ late fusion object tokens
```

No usar early concatenation como primera opción.

Gate:

```text
mejora TTC alto ≥ 15 %
media global mejora ≥ 5 %
latencia p95 ≤ 25 ms
```

---

## Fase 9 — multi-object bbox-free

Solo después de superar todas las fases anteriores.

Añadir:

- 4 queries;
- Hungarian matching;
- tracking;
- selector de riesgo;
- TTC por objeto.

No es requisito para el primer submission bbox-assisted.

---

## Fase 10 — freeze, entrenamiento 32/32 y evaluación oficial

Procedimiento:

1. cerrar grouped CV sobre EvTTC-32;
2. elegir una única configuración mediante métricas OOF;
3. congelar arquitectura, épocas, seeds, ensemble y runtime protocol;
4. crear `final_freeze_manifest.json`;
5. entrenar los candidatos finales sobre EvTTC-32 completo;
6. exportar ONNX cuando corresponda;
7. medir latencia local con el protocolo congelado;
8. ejecutar inferencia sobre Benchmark-10 sin ground truth;
9. escribir y validar los archivos `results.txt`;
10. solicitar autorización de submission;
11. enviar únicamente candidatos previamente congelados;
12. registrar score y respuesta oficial;
13. no reajustar el modelo a partir del leaderboard.

---
# 14. Matriz experimental mínima

No ejecutar una matriz gigantesca desde el principio.



## Tier G — Garl-TTC con datos locales

| ID | Variante | Supervisión |
|---|---|---|
| G0 | RGB direct regression | EvTTC TTC |
| G1 | Event direct regression | EvTTC TTC |
| G2 | RGBE early direct | EvTTC TTC |
| G3 | RGB Learned Height Ratio | EvTTC TTC |
| G4 | Event Learned Height Ratio | EvTTC TTC |
| G5 | RGBE LHR early fusion | EvTTC TTC |
| G6 | RGBE LHR late fusion | EvTTC TTC |
| G7 | G6 + foreground supervision | EvTTC TTC |
| G8 | eAP-40 SSL pretrain + G7 fine-tune | eAP no-TTC + EvTTC TTC |
| G9 | G8 + pseudo-TTC auxiliar | eAP pseudo + EvTTC TTC |

Baseline Garl local principal:

```text
max_OOF(G7, G8, G9 promocionado)
```

## Tier E — pretraining externo cerrado

| ID | Variante |
|---|---|
| E0 | BASE sin eAP |
| E1 | eAP Event-JEPA SSL |
| E2 | E1 + RGB-event DINOv3 |
| E3 | E2 + SAM/projected ROI/geometry |
| E4 | E3 + pseudo-TTC 0,05 |
| E5 | mejor eAP-pretrained + EvTTC fine-tuning |

No existe ningún brazo que dependa de otro dataset o release.

## Tier 1 — cadena principal, seed 7

| ID | Variante |
|---|---|
| A0 | BASE global |
| A1 | BASE dense patch block-causal |
| A2 | A1 + task-specific AttnRes |
| A3 | A2 + TargetQuery + SC-RGMTD |
| A4 | A3 + GT-mask geometry |
| A5 | A3 + predicted-mask geometry |
| A6 | A5 + ego compensation + deterministic mix |
| A7 | A6 + Stable Geometry Router |
| A8 | A7 + bounded residual + uncertainty |
| A9 | A8 + RGB late fusion |

Promover solo brazos que superen gates.

## Tier K — KDA, separado de la cadena principal

| ID | Variante |
|---|---|
| K0 | A2 con block-causal |
| K1 | A2 con Object-KDA |
| K2 | A2 con aligned-patch KDA + global refresh |

KDA no se combina con todos los demás módulos hasta demostrar primero valor independiente.

## Tier 2 — seeds 7, 13 y 21

Ejecutar únicamente:

- mejor event-only;
- mejor RGBE;
- mejor bbox-assisted;
- BASE;
- mejor KDA solo si fue promocionado.

## Tier 3 — ablations del paper

1. global vs dense;
2. block-causal vs Object-KDA;
3. última capa vs AttnRes;
4. bbox vs predicted mask;
5. teacher forcing vs SC-RGMTD;
6. sin DINO;
7. sin SAM;
8. sin geometry oracle teacher;
9. sin navegación;
10. nav shuffled;
11. sin ego compensation;
12. height only;
13. area only;
14. affine only;
15. contrast only;
16. deterministic mix vs Stable Geometry Router;
17. sin residual;
18. sin JEPA pretraining;
19. frozen vs fine-tuned backbone;
20. event-only vs RGB-only vs RGBE.

# 15. Estructura de archivos propuesta


La versión v6 añade:

```text
configs/
  sota/
    garl_ttc_replica.yaml
    garl_ttc_evttc_cv.yaml

  eap/
    download_train.yaml
    closed_dataset_registry.yaml
    teacher_cache_dinov3_vitl.yaml
    teacher_cache_sam_vitl.yaml
    pretrain_eap_ssl.yaml
    pretrain_eap_pseudo_ttc.yaml

src/e_jepa_ttc/
  data/
    closed_dataset_registry.py
    eap_hf40_adapter.py
    eap_projected_roi.py
    garl_pair_sampler.py
    eap_manifest.py
    eap_tar_rgb_reader.py
    eap_events_reader.py
    eap_labels.py
    eap_pseudo_ttc.py
    external_pretrain_guard.py

  teachers/
    foundation_cache_registry.py
    dinov3_offline_extractor.py
    sam_offline_extractor.py

scripts/
  build_garl_ttc_spec.py
  train_garl_ttc_replica.py
  pretrain_garl_eap40_ssl.py
  pretrain_garl_eap40_pseudo.py
  evaluate_garl_ttc_cv.py
  export_garl_ttc_onnx.py
  audit_eap_projection_overlays.py
  audit_eap_labels.py
  inspect_eap_ttc_semantics.py
  audit_eap_velocity_semantics_global_fixed.py
  build_eap_pseudo_ttc_track_v1.py
  build_eap_download_manifest.py
  build_eap_sam_cache.py
  build_eap_dinov3_cache.py
  pretrain_eap_ssl.py
  pretrain_eap_pseudo_ttc.py
  validate_eap_cache.py

tests/
  test_eap_train_only_guard.py
  test_eap_no_test_paths.py
  test_eap_bbox_convention.py
  test_eap_pseudo_ttc_not_official.py
  test_eap_teacher_cache_hashes.py
  test_external_pretrain_manifest.py
```


```text
configs/
  sota/
    protocol_evttc32_cv_benchmark10.yaml
    oracle_geometry.yaml
    dense_patch.yaml
    target_query.yaml
    object_geometry_event.yaml
    object_geometry_rgbe.yaml
    official_submission.yaml

src/e_jepa_ttc/
  data/
    evttc_sota_protocol.py
    benchmark10_guard.py
    teacher_cache.py
    mask_targets.py
    grouped_cv.py

  models/
    garl_ttc_replica.py
    height_ratio_head.py
    foreground_training_decoder.py
    dense_patch_ttc.py
    spatial_patch_mixer.py
    block_causal_transformer.py
    temporal_kda.py
    hybrid_spatiotemporal_mixer.py
    attention_residual_router.py
    target_query.py
    object_queries.py
    mask_decoder.py
    highres_refiner.py
    motion_head.py
    geometry_mixture.py
    stable_geometry_router.py
    residual_ttc.py
    uncertainty_head.py
    risk_selector.py
    object_geo_jepa_ttc.py

  geometry/
    height_ratio_ttc.py
    area_rate_ttc.py
    affine_expansion_ttc.py
    event_contrast.py
    ego_motion_compensation.py
    weighted_solver.py
    geometry_confidence.py

  teachers/
    dino_teacher.py
    sam_teacher.py
    rgb_event_alignment.py
    reliability_gated_multiteacher.py

  training/
    object_geo_trainer.py
    student_conditioned_curriculum.py
    loss_balancer.py
    grad_norm_monitor.py
    health_monitor.py
    checkpoint_selector.py

  evaluation/
    evttc_official_metric.py
    per_sequence_report.py
    calibration_report.py
    latency_benchmark.py
    submission_writer.py

scripts/
  build_evttc_sota_protocol.py
  build_teacher_masks.py
  build_dino_cache.py
  run_oracle_geometry.py
  train_dense_patch_ttc.py
  train_target_query.py
  train_object_geo_ttc.py
  run_object_geo_matrix.ps1
  evaluate_dev22_cv.py
  evaluate_official10.py
  export_object_geo_onnx.py
  write_evttc_submission.py

tests/
  test_benchmark10_guard.py
  test_block_causal_mask.py
  test_kda_spatial_independence.py
  test_kda_temporal_causality.py
  test_attention_residual_weights.py
  test_mask_no_future_leakage.py
  test_geometry_closed_form.py
  test_weighted_solver_gradients.py
  test_ego_motion_causality.py
  test_loss_balance_limits.py
  test_teacher_stop_gradient.py
  test_teacher_reliability_gating.py
  test_student_roi_curriculum.py
  test_geometry_router_no_collapse.py
  test_submission_format.py
  test_checkpoint_provenance.py
```

---

# 16. Configuración inicial recomendada

```yaml
model:
  name: oge-jepa-ttc-v5
  event_backbone: event-tubelet-transformer
  initialize_from: base
  global_resolution: [320, 180]
  detail_resolution: [384, 384]
  native_event_resolution: [1280, 720]
  dense_dim: 256
  target_queries: 1
  background_queries: 1
  temporal_horizons_ms: [20, 60, 100, 240, 500]

patch_policy:
  enabled: true
  spatial_attention: bidirectional_per_frame
  temporal_causality: strict
  global_pool_before_query: false

attention_residuals:
  enabled: true
  mode: task_specific_full
  sources: [embedding, block_outputs]
  normalize_keys: rmsnorm
  heads: [mask, motion, geometry, risk]

# La referencia de precisión sigue siendo block_causal.
temporal_mixer:
  primary: block_causal
  layers: 3
  heads: 8
  kda_candidate:
    enabled: false
    mode: object_kda
    spatial_attention_first: true
    global_refresh_every: 2
    promote_only_after_gate: true

mask:
  soft: true
  teacher_gt: true
  teacher_sam: true
  teacher_dino: true
  min_target_recall: 0.98

multi_teacher:
  method: student_conditioned_reliability_gated
  roi_curriculum: [gt, mixed, predicted]
  teachers: [gt_mask, sam, dino, geometry_oracle, ema_event_jepa]
  max_teacher_gradient_fraction: 0.20
  disagreement_policy: downweight

geometry:
  height_ratio: true
  area_rate: true
  affine_expansion: true
  event_contrast: true
  ego_compensation: true
  solver_dtype: float32
  router:
    type: stable_geometry_router
    latent_dim: 64
    shared_path: true
    training_routing: soft
    inference_top_k: null
    rmsnorm: true
    bounded_activation: true
  residual_limit: 0.30

navigation:
  enabled: true
  inject_in_geometry: true
  inject_global_embedding: false

loss:
  ttc_log_huber: 1.0
  relative_ttc: 0.5
  mask_bce: 1.0
  mask_dice: 1.0
  boundary: 0.25
  temporal_mask: 0.25
  geometry: 0.5
  dino_distill: 0.1
  sam_distill: 0.1
  geometry_teacher: 0.1
  ema_temporal: 0.05
  event_contrast: 0.1
  variance_adaptive: 0.05
  router_stability: 0.01
  max_aux_gradient_fraction: 0.20

training:
  precision: bf16
  batch_size: 2
  gradient_accumulation: 16
  gradient_checkpointing: true
  seeds: [7, 13, 21]
  early_stopping_patience: 4
```


Configuración adicional v5:

```yaml
external_pretraining:
  enabled: true
  dataset: eap_public_detection_train
  repo_id: NAIL-HNU/eAP-dataset
  root: E:/eAP_dataset
  sequences_expected: 40
  samples_expected: 118247
  allow_test_split: false
  objectives:
    jepa_future: 1.0
    rgb_event_dinov3: 0.10
    object_tracking: 0.25
    box_geometry: 0.25
    sam_mask: 0.10
    pseudo_ttc: 0.0

foundation_models:
  dino_primary:
    repo_id: facebook/dinov3-vitl16-pretrain-lvd1689m
    frozen: true
    offline_cache_only: true
    dtype: float16
    microbatch: 4
    output_dim: 256
  sam_mask:
    repo_id: facebook/sam-vit-large
    frozen: true
    offline_cache_only: true
    dtype: float16
    microbatch: 1
  dino_ablation:
    repo_id: facebook/dinov2-large
    enabled: false

eap_pseudo_ttc:
  enabled: false
  label_source: track_derived_pseudo_ttc_v1
  official_ground_truth: false
  loss_weight: 0.05
  min_confidence: 0.50
  min_abs_range_rate_mps: 0.5
  min_local_r2: 0.70
  max_abs_ttc_s: 10.0
  max_gradient_fraction: 0.15

storage:
  raw_eap_hf40: E:/eAP_dataset/data/train
  evttc32: datasets/evttc_complete_staging
  benchmark10: datasets/evttc_official_benchmark_sealed
  allow_new_dataset_roots: false
  derived_eap: E:/eAP_dataset/derived

  hard_guards:
    forbid_dino_all_layers_full_dataset: true
    forbid_sam_full_resolution_logits: true
    forbid_full_dataset_event_voxel_cache: true
    forbid_bulk_rgb_tar_extraction: true
    forbid_duplicate_eap_raw_release: true
    forbid_all_epoch_checkpoints: true

  budgets_gib:
    derived_total_max: 55
    dino_max: 25
    sam_max: 15
    event_cache_max: 8
    pseudo_ttc_max: 5
    active_checkpoints_c_max: 80

  minimum_free_gib:
    drive_e: 50
    drive_c: 100
```

Los pesos iniciales son puntos de partida. Deben normalizarse por magnitud y norma de gradiente.

# 17. Comandos previstos


## Completar la descarga eAP train-40 ya iniciada

Este comando no incorpora un dataset nuevo. Reanuda la descarga actualmente en curso y limita el contenido a `data/train/**`.

```powershell
$HF = ".\.venv\Scripts\hf.exe"
$EapRoot = "E:\eAP_dataset"

$env:HF_HOME = "$EapRoot\.hf"
$env:HF_HUB_CACHE = "$EapRoot\.hf\hub"
$env:HF_XET_CACHE = "$EapRoot\.hf\xet"
$env:HF_XET_CHUNK_CACHE_SIZE_BYTES = "0"

& $HF download `
    NAIL-HNU/eAP-dataset `
    --repo-type dataset `
    --local-dir "$EapRoot" `
    --include "README.md" `
    --include "sample_submission.json" `
    --include "data/train.parquet" `
    --include "data/train/**"
```

No usar `--dry-run` si la versión local del CLI no lo soporta.

Gate final:

```text
Labels = 40
Events = 40
TestFiles = 0
FreeGiB ≥ 50
```

Después de completar esta descarga:

```text
NO ejecutar comandos de descarga de otros datasets
NO descargar eAP test
NO crear otra raíz eAP
```

## Localizar y verificar DINO/SAM
## Localizar y verificar DINO/SAM

```powershell
@'
import torch
import torchvision

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
'@ | & $Py -
```

Carga offline, uno por proceso:

```powershell
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_OFFLINE = "1"

$env:DINO_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"

@'
import os
import torch
from transformers import AutoImageProcessor, AutoModel

repo_id = os.environ["DINO_ID"]
processor = AutoImageProcessor.from_pretrained(
    repo_id,
    local_files_only=True,
)
model = AutoModel.from_pretrained(
    repo_id,
    local_files_only=True,
    torch_dtype=torch.float16,
).cuda().eval()

print(type(processor).__name__)
print(type(model).__name__)
print("parameters:", sum(p.numel() for p in model.parameters()))
print("PASS DINO")
'@ | & $Py -
```

Cerrar ese proceso y probar SAM por separado:

```powershell
$env:SAM_ID = "facebook/sam-vit-large"

@'
import os
import torch
from transformers import SamProcessor, SamModel

repo_id = os.environ["SAM_ID"]
processor = SamProcessor.from_pretrained(
    repo_id,
    local_files_only=True,
)
model = SamModel.from_pretrained(
    repo_id,
    local_files_only=True,
    torch_dtype=torch.float16,
).cuda().eval()

print(type(processor).__name__)
print(type(model).__name__)
print("parameters:", sum(p.numel() for p in model.parameters()))
print("PASS SAM")
'@ | & $Py -
```

Después:

```powershell
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
Remove-Item Env:TRANSFORMERS_OFFLINE -ErrorAction SilentlyContinue
```

---


## Crear rama

```powershell
git switch scientific-recovery-v3-hardening
git pull
git switch -c object-geo-sota-v1
```

## Crear protocolo

```powershell
$env:PYTHONPATH = "src"

.\.venv\Scripts\python.exe scripts\build_evttc32_grouped_cv.py `
  --evttc32-root datasets\evttc_complete_staging `
  --benchmark-root datasets\evttc_official_benchmark_sealed `
  --output data\splits\evttc32_grouped_cv_benchmark10_sealed.yaml `
  --folds 5 `
  --group-by sequence family `
  --seal-benchmark
```

## Oracle geométrico

```powershell
.\.venv\Scripts\python.exe scripts\run_oracle_geometry.py `
  --config configs\sota\oracle_geometry.yaml `
  --cv-manifest data\splits\evttc32_grouped_cv_benchmark10_sealed.yaml `
  --output artifacts\runs\oge_sota\oracle
```

## Teacher masks

```powershell
.\.venv\Scripts\python.exe scripts\build_teacher_masks.py `
  --config configs\sota\target_query.yaml `
  --splits dev `
  --output artifacts\features\oge_teacher_masks
```

## Dense patch

```powershell
.\.venv\Scripts\python.exe scripts\train_dense_patch_ttc.py `
  --config configs\sota\dense_patch.yaml `
  --seed 7 `
  --output artifacts\runs\oge_sota\dense_patch\seed7
```

## Object geometry

```powershell
.\.venv\Scripts\python.exe scripts\train_object_geo_ttc.py `
  --config configs\sota\object_geometry_rgbe.yaml `
  --seed 7 `
  --output artifacts\runs\oge_sota\rgbe\seed7
```

## Evaluación oficial final

Este comando debe exigir autorización explícita y manifest sellado:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_official10.py `
  --config configs\sota\official_submission.yaml `
  --checkpoint artifacts\runs\oge_sota\final\best.pt `
  --authorize-final-test `
  --output artifacts\official_submission
```

---

## Entrenamiento final con las 32 secuencias

Las épocas y reglas deben proceder del resumen de CV:

```powershell
.\.venv\Scripts\python.exe scripts\train_object_geo_ttc.py `
  --config configs\sota\object_geometry_rgbe.yaml `
  --cv-summary artifacts\runs\oge_sota\cv\selection_summary.json `
  --train-manifest data\splits\evttc32_final_all32.yaml `
  --seeds 7 13 21 `
  --freeze-manifest artifacts\audit\oge_sota\final_freeze_manifest.json `
  --output artifacts\runs\oge_sota\final
```

## Inferencia en Benchmark-10

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_benchmark10.py `
  --config configs\sota\official_submission.yaml `
  --benchmark-manifest data\manifests\benchmark10_sealed_manifest.yaml `
  --freeze-manifest artifacts\audit\oge_sota\final_freeze_manifest.json `
  --checkpoint-root artifacts\runs\oge_sota\final `
  --authorize-benchmark-inference `
  --output artifacts\official_submission
```

## Formato oficial de `results.txt`

La documentación oficial especifica cuatro campos:

```text
Index    timestamp (s)    ttc (s)    cost time(s)
```

Significado:

- `Index`: índice correspondiente a la estimación TTC;
- `timestamp`: timestamp correspondiente a la estimación;
- `ttc`: TTC estimado por el método;
- `cost time`: tiempo consumido para una única estimación TTC.

Ejemplo de escritura interna recomendada:

```text
Index	timestamp (s)	ttc (s)	cost time(s)
0	0.000000	5.284731	0.012843
1	0.050000	5.231904	0.012517
```

La página oficial no concreta:

- si la cabecera es obligatoria;
- el delimitador exacto;
- la precisión decimal;
- si se entrega un archivo por secuencia o un paquete conjunto;
- el convenio exacto para excluir o incluir I/O en `cost time`.

Por ello:

1. se genera una versión tabulada con la cabecera mostrada;
2. se conserva un archivo por secuencia y un manifest global;
3. se confirma el empaquetado definitivo con los organizadores durante la revisión manual;
4. no se altera ninguna predicción al cambiar solamente el empaquetado.

Estructura local:

```text
artifacts/official_submission/
├── SINGLE_REALTIME/
│   ├── CCRs1-low/results.txt
│   ├── CCRs1-medium/results.txt
│   ├── ...
│   ├── Slider-1000/results.txt
│   ├── submission_manifest.json
│   └── runtime_environment.json
└── ENSEMBLE_ACCURACY/
    ├── CCRs1-low/results.txt
    ├── ...
    ├── Slider-1000/results.txt
    ├── submission_manifest.json
    └── runtime_environment.json
```

## Medición reproducible de `cost time(s)`

El tiempo reportado debe corresponder al candidato realmente enviado.

Protocolo interno:

- warm-up separado y no incluido;
- sincronización CUDA antes y después;
- reloj monotónico de alta resolución;
- medición por estimación, no promedio inventado;
- incluir voxelización/ventana online, encoder, queries, geometría y postproceso;
- excluir carga única del modelo y lectura de checkpoint;
- no excluir un miembro del ensemble;
- guardar tiempo individual por fila;
- informar hardware, software, precisión y batch real.

Pseudocódigo:

```python
torch.cuda.synchronize()
start = time.perf_counter()

prediction = complete_online_ttc_estimation(sample)

torch.cuda.synchronize()
cost_time_s = time.perf_counter() - start
```

Este convenio debe confirmarse con los organizadores, ya que la web solo define el campo como tiempo consumido por una estimación individual.

## Validador antes del envío

```powershell
.\.venv\Scripts\python.exe scripts\validate_evttc_submission.py `
  --submission-root artifacts\official_submission\ENSEMBLE_ACCURACY `
  --benchmark-manifest data\manifests\benchmark10_sealed_manifest.yaml `
  --require-sequences 10 `
  --require-finite `
  --require-positive-ttc `
  --require-nonnegative-runtime `
  --require-index-match `
  --require-timestamp-match `
  --write-report artifacts\official_submission\validation_report.json
```

El validador debe rechazar:

- columnas ausentes o adicionales;
- `NaN` o infinito;
- TTC no positivo;
- runtime negativo;
- índices duplicados;
- timestamps duplicados o desordenados;
- filas ausentes;
- timestamps que no correspondan a las consultas oficiales;
- mezcla accidental de candidatos;
- resultados generados con checkpoint distinto al freeze manifest.

## Solicitud de submission

La competición exige solicitar primero acceso y cada aplicación se revisa manualmente.

Enviar la solicitud a:

```text
kaizhen@hnu.edu.cn
```

Información requerida por la web oficial:

- posición: Undergrad, PhD, PostDoc o Researcher;
- conferencia o revista y año prevista o ya publicada;
- descripción del proyecto en 3–5 frases;
- motivo para evaluar en el servidor EvTTC.

Plantilla:

```text
Subject: Application for EvTTC benchmark submission — OGE-JEPA-TTC

Position:
[Researcher / PhD / ...]

Conference or journal:
[venue and year]

Project description:
[3–5 sentences describing OGE-JEPA-TTC, modalities, geometry and evaluation.]

Reason for requesting evaluation:
[Explain that Benchmark-10 is the untouched external test set and that
the architecture was selected only with grouped CV on the labelled EvTTC-32.]
```

Adjuntar o tener preparados:

- nombre del método;
- paper/preprint cuando exista;
- repositorio o commit;
- modalidades usadas: eventos, RGB y/o navegación;
- bbox-assisted o bbox-free;
- entorno de runtime;
- freeze manifest;
- SHA-256 de los archivos;
- aclaración de si el resultado es single o ensemble.

## Política para “los mejores resultados finales”

La regla de fusión se decide con CV, no con Benchmark-10.

Candidato de precisión:

```text
ENSEMBLE_ACCURACY
= tres seeds congeladas
= fusión robusta en inverse-TTC o regla seleccionada por OOF
= runtime real del ensemble
```

Candidato de tiempo real:

```text
SINGLE_REALTIME
= mejor configuración single según OOF
= ONNX/TensorRT si no modifica las predicciones fuera de tolerancia
```

No se escoge entre ambos mirando cuál obtiene mejor score oficial. Ambos deben representar claims diferentes definidos antes de la submission.

---


# 18. Tests científicos obligatorios


## Paridad Garl-TTC

- fórmula TTC por height ratio contra casos sintéticos;
- denominador próximo a cero protegido;
- alturas siempre positivas;
- mismo `Δt` en labels y fórmula;
- late fusion realmente separa encoders;
- decoder foreground ausente en inference export;
- SAM tiene `stop_gradient`;
- ROI exactamente 128×128 en baseline;
- runtime incluye preparación de ROI y eventos;
- comparación de tres secuencias Garl con métrica idéntica;
- no llamar Benchmark-10 al subconjunto de tres secuencias.

## Inventario cerrado de datos

- el registry acepta únicamente `EAP_HF_TRAIN40`, `EVTTC32_LABELLED` y `BENCHMARK10_SEALED`;
- cualquier raíz adicional produce error;
- no existe loader para eAP test;
- no existe adapter de un segundo release eAP;
- `translation == ego_translation` no se interpreta como pose ego mundial;
- `size` se convierte correctamente desde `[h,l,w]`;
- `velocity_x` no se usa como closing speed directo;
- las pseudoetiquetas declaran `official_ground_truth=false`;
- Garl recibe TTC oficial únicamente desde EvTTC;
- el pretraining eAP de Garl puede ejecutarse con `pseudo_ttc_weight=0`;
- cada ROI eAP conserva provenance de proyección;
- la dirección de `T_event_ego` se valida por overlay;
- eAP y EvTTC nunca comparten sampler ni fold.

## Almacenamiento

- abortar si DINO all-layers full-dataset está activado;
- abortar si SAM logits full-resolution están activados;
- abortar si voxel cache global está activado;
- abortar si extracción masiva TAR está activada;
- abortar si cache derivada supera 55 GiB;
- abortar si quedan menos de 50 GiB libres en E:;
- política de checkpoints conserva como máximo `last`, `best`, `weights_only`.



## eAP y teachers externos

- el manifest contiene exactamente 40 secuencias train;
- no existe ninguna ruta `data/test/` en el loader de entrenamiento;
- `translation == ego_translation` no se interpreta como pose ego mundial;
- `size` se convierte correctamente desde `[h,l,w]`;
- `velocity_x` no se usa como closing speed directo;
- todas las pseudoetiquetas declaran `official_ground_truth=false`;
- teacher cache incluye repo ID y snapshot commit;
- DINO/SAM se ejecutan con `stop_gradient`;
- no se leen teachers online en el benchmark de latencia;
- los TAR RGB se leen sin extracción total;
- reanudación de descarga no cambia hashes;
- eAP y EvTTC nunca comparten sampler o folds.


## Causalidad

- ningún token ve eventos futuros;
- KDA nunca impone orden causal entre patches del mismo frame;
- Spatial Patch Mixer se ejecuta antes de KDA;
- el estado KDA solo avanza entre tiempos causales;
- la máscara teacher no usa frame futuro;
- navegación solo hasta `context_end`;
- crops solo usan máscara producida con contexto causal;
- target encoder no recibe navegación futura.

## Geometría

- gradcheck del solver;
- casos sintéticos con TTC conocido;
- expansión cero produce TTC infinito/cap;
- contracción no produce colisión positiva;
- estabilidad con máscaras parciales;
- estabilidad FP32.

## Datos

- no overlap EvTTC-32 CV / Benchmark-10 sellado;
- no ventanas cruzan secuencia;
- no augmentations cambian TTC sin actualizar geometría;
- hashes de máscaras teacher;
- timestamps RGB-eventos dentro de tolerancia;
- `results.txt` contiene exactamente Index, timestamp, TTC y cost time;
- todos los TTC son finitos y positivos;
- todos los runtimes son finitos y no negativos;
- índices y timestamps coinciden con las consultas del benchmark;
- el runtime del ensemble mide el ensemble completo.

## Shortcuts y routing

- nav shuffled;
- teacher shuffled;
- DINO/SAM disagreement;
- router expert collapse;
- router sequence-family predictability;
- AttnRes layer collapse;
- RGB shuffled;
- mask shuffled;
- event-zero;
- sequence-ID removal;
- metadata removal.

## Reproducibilidad

- seeds;
- deterministic sampler;
- resume exacto;
- hash de config;
- hash de split;
- hash de checkpoint;
- registro de entorno CUDA/PyTorch.

---

# 19. Criterios de rechazo

Rechazar una arquitectura si ocurre cualquiera:

1. mejora solo train y no grouped validation;
2. empeora el peor escenario más de 10 %;
3. `context/pred std < 0,05` de forma persistente;
4. una loss auxiliar domina más del 20 % del gradiente;
5. `NAV_ONLY` se acerca demasiado al modelo completo;
6. el residual explica más del 50 % de `q_final`;
7. la máscara predicha no alcanza 98 % recall;
8. la mejora depende de una sola seed;
9. el test oficial se ha utilizado para tuning;
10. el runtime p95 supera 25 ms sin mejora clara;
11. no supera `BASE` por al menos 5 % en CV;
12. no puede reproducirse desde un manifest sellado;
13. KDA rompe la interacción espacial o no aporta precisión/eficiencia;
14. AttnRes colapsa siempre a una capa sin ventaja medible;
15. un teacher domina o empeora frente al curriculum sin teacher;
16. el router selecciona expertos por identidad de secuencia y no por evidencia física;
17. el pretraining eAP solo mejora con pseudo-TTC y falla sin esa pseudoetiqueta;
18. se detecta cualquier ruta test eAP en entrenamiento;
19. se presenta `velocity_x` como velocidad de cierre sin documentación;
20. los caches externos carecen de snapshot hash o manifiesto;
21. OGE no supera a la reimplementación Garl en OOF;
22. la ganancia de OGE procede únicamente de más compute o una ROI más favorable;
23. se usa un release eAP sin declarar su formato;
24. se usa eAP test o se intenta obtener otro release;
25. Garl recibe pseudo-TTC como si fuera ground truth oficial;
26. se superan los presupuestos duros de almacenamiento.

---

# 20. Estrategia de publicación

## Claims separados


### Claim 0 — paridad Garl-TTC

> Reimplementamos una baseline Garl-TTC bajo el mismo protocolo EvTTC-32 utilizado para seleccionar OGE-JEPA-TTC.

Este claim es obligatorio antes de afirmar superioridad.

### Claim 0B — pretraining eAP-40

> Preentrenamos Garl-TTC con objetivos no-TTC sobre el eAP público train-40 y lo fine-tuneamos con TTC oficial exclusivamente en EvTTC-32.

El pseudo-TTC, cuando se use, se informa como ablación separada.

### Claim 1 — estimación TTC bbox-assisted

> OGE-JEPA-TTC supera los métodos oficiales bajo el protocolo bbox/ROI del benchmark.

### Claim 2 — bbox-free

> OGE-JEPA-TTC localiza automáticamente el objetivo y estima TTC desde frame completo.

### Claim 3 — geometría

> La geometría diferenciable por objeto supera la regresión TTC directa.

### Claim 4 — dense patches

> Conservar tokens espaciales mejora frente al pooling global.

### Claim 5 — ego-motion

> La compensación explícita mejora y no funciona como shortcut.

### Claim 6 — depth routing

> Attention Residuals permite que máscara, movimiento y geometría recuperen niveles de representación diferentes.

### Claim 7 — conditional efficient temporal mixing

> KDA solo se reivindica si permite más resolución/horizonte o mejor precisión sin romper Patch Policy.

### Claim 8 — multi-teacher consolidation

> SC-RGMTD mejora el paso de masks/crops oracle a estados producidos por el propio estudiante.


### Claim 9 — external multimodal pretraining

> El pretraining público eAP mejora la representación object-centric RGB-eventos antes del fine-tuning TTC en EvTTC.

### Claim 10 — pseudo-TTC auxiliary ablation

> El pseudo-TTC derivado de tracks solo se reivindica si aporta una mejora OOF reproducible frente al mismo pretraining sin esa loss; nunca se presenta como ground truth oficial.

## Tablas mínimas del artículo

1. leaderboard oficial 10 secuencias;
2. event-only vs RGB vs RGBE;
3. bbox vs predicted mask;
4. geometría experts;
5. dense vs global;
6. navegación y shortcut audits;
7. three-seed uncertainty;
8. runtime;
9. high-TTC y low-TTC bins;
10. casos cualitativos de máscaras y expansión.

---

# 21. Orden de prioridad real

```text
P0   terminar y congelar la matriz actual
P1   completar la descarga eAP Hugging Face train-40 ya iniciada
P2   congelar manifest de datos cerrado y bloquear nuevas raíces
P3   congelar especificación Garl-TTC
P4   implementar projection/ROI audit para eAP-40
P5   implementar direct-regression G0-G2 en EvTTC
P6   implementar LHR G3-G6 y comparar early vs late fusion
P7   añadir foreground teacher y cerrar G7
P8   preentrenar Garl sobre eAP-40 sin TTC y cerrar G8
P9   probar pseudo-TTC opcional y conservar G9 solo si gana
P10  exportar mejor Garl local a ONNX/TensorRT
P11  precomputar SAM masks compactas
P12  precomputar DINOv3 object/boundary tokens compactos
P13  pretraining eAP SSL de OGE sin pseudo-TTC
P14  ablación OGE SSL vs SSL+pseudo-TTC
P15  construir grouped CV 5-fold sobre EvTTC-32
P16  sellar Benchmark-10
P17  oracle geométrico multi-experto
P18  dense patch block-causal
P19  Attention Residuals
P20  TargetQuery + SC-RGMTD
P21  refiner high-resolution
P22  ego-motion compensation
P23  Stable Geometry Router
P24  residual limitado + uncertainty
P25  RGBE
P26  KDA solo si supera gate
P27  multi-object bbox-free
P28  congelar configuración final
P29  entrenar 32/32
P30  generar y validar submission
```

# 22. Definición de “proyecto terminado”

El proyecto se considerará terminado solo cuando exista:

- un checkpoint final reproducible;
- un manifest completo del train público eAP-40;
- un inventario de DINOv3/DINOv2/SAM con snapshots;
- caches SAM/DINO validados y dentro de los presupuestos duros;
- reimplementación Garl-TTC reproducible y evaluada;
- manifest cerrado que demuestre que solo se usaron eAP-40, EvTTC-32 y Benchmark-10;
- comparación Garl G7 vs G8 vs G9;
- una ablación `sin eAP` vs `eAP SSL` vs `eAP SSL + pseudo-TTC`;
- evaluación de tres seeds;
- manifest EvTTC-32 CV / Benchmark-10 sellado sellado;
- resultado oficial en diez secuencias;
- diez `results.txt` válidos por candidato o el empaquetado que confirme el organizador;
- validation report sin errores;
- solicitud de submission aprobada;
- latencia end-to-end;
- ONNX exportado;
- máscara visualizable;
- `q_geo`, residual y confidence auditables;
- comparación bbox-assisted y bbox-free;
- paper actualizado;
- tests y CI verdes;
- commit y artefactos registrados;
- respuesta o entrada oficial del leaderboard.

## Resultado de éxito

```text
mean e_TTC oficial < 5,00 %
runtime p95 ≤ 25 ms
sin fuga de test
sin bbox en el track FULL
tres semillas estables
```

## Resultado SOTA fuerte

```text
mean e_TTC oficial < 4,50 %
ganar ≥ 7/10 secuencias
runtime p95 ≤ 25 ms
```

---

# 23. Referencias de diseño

- **EvTTC: An Event Camera Dataset for Time-to-Collision Estimation**, arXiv:2412.05053.
- **Event-Aided Time-to-Collision Estimation for Autonomous Driving / STRTTC**, arXiv:2407.07324.
- **A Unifying Contrast Maximization Framework for Event Cameras**, CVPR 2018.
- **Toward Deep Representation Learning for Event-Enhanced Visual Autonomous Perception: the eAP Dataset / Garl-TTC**, arXiv:2603.16303. eAP train público se utiliza como pretraining externo declarado; su release de detección no se confunde con ground truth TTC oficial.
- **Patch Policy: Efficient Embodied Control via Dense Visual Representations**, arXiv:2607.18236.
- **Kimi K3: Open Frontier Intelligence**, arXiv:2607.24653. Se adaptan selectivamente KDA, Attention Residuals, estabilidad de routing y consolidación multi-teacher; no se replica el MoE masivo.
- **Slot Attention for Object-Centric Learning**, NeurIPS 2020.
- **DINOv2 / DINOv3**, features visuales densas. Teacher principal local: `facebook/dinov3-vitl16-pretrain-lvd1689m`.
- **Segment Anything**, máscaras teacher. Teacher local: `facebook/sam-vit-large`.
- Leaderboard y descarga de Benchmark-10: https://nail-hnu.github.io/EvTTC/competition/
- Formato oficial de datos y submission: https://nail-hnu.github.io/EvTTC/download/data_format/#submit%20format
- Solicitud manual de submission: https://nail-hnu.github.io/EvTTC/submit/
- Información consultada el 29 de julio de 2026.

---

# 24. Resumen ejecutivo final

La nueva arquitectura no intentará mejorar `BASE` añadiendo otra pérdida global.

Cambiará la estructura del problema:

```text
ANTES
full-frame → embedding global → TTC directo

V5 FINAL
full-frame
→ dense patches Patch Policy
→ AttnRes por tarea
→ mezcla temporal causal
→ TargetQuery
→ máscara student-conditioned
→ detalle high-resolution
→ movimiento compensado por ego
→ Stable Geometry Router
→ residual limitado
→ TTC
```

## Decisión final sobre las cuatro ideas evaluadas

| Idea | Decisión | Razón |
|---|---|---|
| Patch Policy | **Obligatoria** | Conserva detalle espacial y ataca el pooling prematuro |
| Attention Residuals | **Candidato principal** | Bajo riesgo; recupera bordes, movimiento y semántica de distintas capas |
| KDA | **Condicional** | Útil para resolución/horizonte, pero puede romper la lógica espacial si se aplana ingenuamente |
| Stable LatentMoE | **No literal** | Demasiado dato/escala; se adapta como Stable Geometry Router |
| Multi-Teacher On-Policy Distillation | **No literal** | Se adapta como SC-RGMTD sobre máscaras y ROIs producidas por el estudiante |

La ruta más probable hacia SOTA en v4 combina pretraining eAP declarado con selección TTC estricta en EvTTC:

1. aprovechar el pretraining BASE ya auditado;
2. descargar únicamente el train público eAP y congelar su manifest;
3. precomputar DINOv3 y SAM secuencia por secuencia sin cargarlos durante el training student;
4. entrenar eAP SSL/object-centric sin pseudo-TTC como baseline externo;
5. probar el pseudo-TTC solo como loss auxiliar de baja confianza y eliminarlo si no supera la ablación;
6. conservar detalle espacial mediante Patch Policy;
7. recuperar features de varias profundidades mediante AttnRes;
8. usar bbox/máscara como teacher, no como requisito de inferencia;
9. entrenar sobre ROIs producidas progresivamente por el estudiante;
10. emplear RGB para TTC alto y eventos para TTC bajo;
11. imponer la geometría en la arquitectura de salida;
12. usar navegación mediante compensación física;
13. estabilizar el router geométrico sin un MoE masivo;
14. activar KDA solo si demuestra valor medible;
15. impedir que pérdidas, teachers, navegación o experts dominen el aprendizaje;
16. seleccionar toda decisión TTC mediante grouped CV en EvTTC-32;
17. congelar arquitectura, épocas, seeds y ensemble antes de Benchmark-10.

El objetivo no es construir un modelo más grande, sino un modelo cuya estructura obligue a resolver correctamente:

```text
qué objeto
qué información de cada capa necesita
cómo evoluciona en el tiempo
cuánto se expande
qué parte se debe al ego
qué experto geométrico es fiable
cuánto falta para la colisión
```

# 25. Fuentes operativas verificadas para v6

## Fuentes de referencia

- Artículo Garl-TTC/eAP: `https://arxiv.org/abs/2603.16303`
- Proyecto oficial: `https://nail-hnu.github.io/eAP_dataset/`
- Dataset que sí usa este proyecto: `https://huggingface.co/datasets/NAIL-HNU/eAP-dataset`
- EvTTC: `https://nail-hnu.github.io/EvTTC/`

La web del artículo describe el dataset completo y TTC por objeto como parte del trabajo publicado. En v6 esa información se utiliza únicamente para comprender Garl-TTC. No se interpreta como prueba de que exista un segundo paquete descargable dentro de nuestro inventario.

## Inventario operativo v6

```text
EAP_HF_TRAIN40
  40 labels.parquet
  804.510 anotaciones de objeto
  5.116 tracks comparados
  610.348 muestras temporales auditadas
  sin TTC directo

EVTTC32_LABELLED
  única supervisión TTC oficial

BENCHMARK10_SEALED
  evaluación externa final

DINO/SAM
  teachers locales ya descargados
```

## Decisión irreversible del protocolo

```text
No se buscan nuevos datasets.
No se descarga un segundo release eAP.
No se descarga eAP test.
Garl se supervisa con TTC de EvTTC-32.
eAP-40 solo preentrena o aporta pseudo-TTC experimental.
```

# 26. Estado de implementación verificable — 30 de julio de 2026

## 26.1 Controles cerrados

`B0_HISTORICAL_BASE_EXACT` reproduce el checkpoint histórico sobre su cache
original. Las predicciones son equivalentes byte a byte y las métricas de
validation son MAE `0,3228917687 s`, RMSE `0,5844324448 s` y error relativo
medio `8,1553575311 %`. El checkpoint downstream corresponde a la época 26/30
y el encoder SSL a la época 6/30.

El control histórico se reporta por separado. `A0_MATCHED_GLOBAL` es el control
justo de la matriz object-cache y comparte muestras, inicialización, cabeza,
optimizador, épocas y early stopping con Dense Patch, AttnRes y Object-KDA.

## 26.2 Implementación corregida

- Dense Patch preserva interacción espacial por instante antes del mezclador
  temporal.
- AttnRes RMS-normaliza las claves y mezcla valores originales por tarea.
- Object-KDA implementa recurrencia delta solo sobre el eje temporal.
- Garl G0–G7 replica el protocolo público de tres instantes, dos intervalos de
  eventos, ROI 128x128, ResNet-50, LHR, late fusion y foreground training-only.
- Height/area/affine estiman inverse-TTC en el endpoint actual; event contrast
  usa eventos object-centric.
- La navegación NEU/heading se transforma a la cámara de eventos mediante la
  cadena calibrada navegación–LiDAR–RGB–evento.
- La compensación traslacional física exige profundidad. Cuando usa la
  distancia oficial EvTTC se etiqueta como oracle/teacher y no como entrada de
  inferencia.

## 26.3 Matrices y gates

Core y Garl escriben en árboles independientes:

```text
artifacts/runs/evttc32_architecture_v4_<split>_<mode>/
├── core/fold-<n>/matrix_summary.json
└── garl/fold-<n>/matrix_summary.json
```

Los smokes de dos épocas solo validan integración. La promoción requiere screen
304/80, grouped CV de cinco folds y tres seeds únicamente para BASE y máximo
dos finalistas. TargetQuery, máscaras predichas, refiner, router, residual e
incertidumbre permanecen bloqueados hasta que la geometría bbox-GT supere el
gate frente a BASE.

## 26.4 Resultado de los screens y confirmación larga

El screen Core de ocho épocas no se usa ya para decidir Dense porque fue
demasiado corto. La confirmación mantiene idénticos checkpoint inicial,
muestras, batch efectivo, optimizador, máximo de 40 épocas y early stopping:

| Brazo | Mejor época | Épocas completadas | Error rel. macro | Score | MAE macro |
|---|---:|---:|---:|---:|---:|
| A0 global | 17 | 23 | 16,129 % | 0,32523 | 0,701 s |
| **A1 Dense** | **20** | 26 | **15,210 %** | **0,30543** | **0,628 s** |
| A2 AttnRes | 10 | 16 | 16,136 % | 0,32503 | 0,653 s |
| K1 Object-KDA | 7 | 13 | 16,960 % | 0,34139 | 0,731 s |

Decisión:

```text
A1 Dense/Patch Policy  → promover a grouped CV
A2 AttnRes             → no combinar; no mejora A1
K1 Object-KDA          → no combinar; no mejora A1
```

A1 necesita 20 épocas para alcanzar su mejor checkpoint; por eso podía parecer
peor a ocho épocas. A0 no recibió más cómputo: A1 completó tres épocas más por
la misma regla de early stopping. A1 mejora 5,70 % el error relativo y 10,45 %
el MAE frente a A0, con una latencia 1,91 veces mayor.

El screen Garl ResNet-50 G0–G7 se ejecutó con batch efectivo 24. G5 RGBE-LHR
early fue el mejor brazo local con 36,52 % de error relativo macro. No es
paridad Garl: el repositorio público usa 50 épocas y late fusion inicializada
desde ramas LHR unimodales preentrenadas.

## 26.5 Gate geométrico

La expansión causal de bbox con ventana 21 y calibración log-afín train-only
produce 311/314 predicciones. El fallback determinista a A1 en las tres filas
inválidas obtiene:

```text
A1 Dense                       rel 15,210 %   score 0,30543
geometría causal + fallback    rel 14,790 %   score 0,31144
```

Mejora ligeramente el error relativo, pero no alcanza el 5 % y empeora el
score por RMSE. Además usa hasta 21 cajas pasadas frente a tres frames de A1.
No desbloquea TargetQuery, máscaras predichas, refiner, router, residual o
uncertainty.

El port causal trazable al código público STRTTC implementa NLTS, contornos,
normal flow local, RANSAC y solver de tres parámetros. En el screen de 40
muestras resuelve 27 y falla 13; las exitosas tienen 112,96 % de error relativo
macro. La cobertura y los fallos se conservan, por lo que tampoco se promociona.

## 26.6 Grouped CV sin contaminación SSL

Solo A0 y A1 pasan al CV. Mientras no existan checkpoints SSL entrenados
exclusivamente con el train de cada fold, ambos se inicializan desde los mismos
pesos EventTubelet aleatorios mediante `-RandomControl`. Esto evita que un
checkpoint SSL histórico haya visto eventos de una secuencia usada como
validation. La confirmación con el BASE auditado y el CV desde cero responden a
preguntas distintas y se reportan por separado.

## 26.7 Cierre ejecutado del gate A0/A1

Se completaron los 30 runs predeclarados: cinco folds, seeds 7/13/21 y dos
arquitecturas. Todas las parejas pasaron la auditoría de selección de muestras,
backbone, cabeza y trainer.

| Brazo | Score ± sd seeds | Error rel. ± sd | MAE ± sd | Latencia |
|---|---:|---:|---:|---:|
| **A0 global** | **0,58452 ± 0,00853** | **30,25 % ± 0,52** | 1,011 ± 0,039 s | 4,54 ms |
| A1 Dense | 0,59312 ± 0,00349 | 30,55 % ± 0,06 | **1,007 ± 0,013 s** | 9,82 ms |

A1 empeora el score un 1,47 % y el error relativo un 0,99 %; su mejora de MAE
es solo 0,41 %. A0 gana 10/15 pares en score/error relativo y 8/15 en MAE. Los
bootstrap OOF pareados por secuencia cruzan cero. A1 requiere 1,58× tiempo de
entrenamiento y 2,16× latencia. Por tanto, A0 es la arquitectura final y A1 se
conserva únicamente como hipótesis object-centric.

Con A0 fijo, tres seeds compararon los perfiles `matched` y `throughput`.
`matched` mejora el score validation medio un 10,95 %, aunque tarda 4,02× más,
y selecciona seed 13. El checkpoint se congeló antes de abrir family-OOD:

| Split | Secuencias / ventanas | Score | Error rel. macro | MAE macro |
|---|---:|---:|---:|---:|
| validation | 5 / 314 | 0,28992 | 14,46 % | 0,541 s |
| family-OOD reutilizado | 8 / 481 | 0,53784 | 30,56 % | 0,805 s |

El diagnóstico OOD degrada 85,5 % el score, 111,4 % el error relativo y 48,8 %
el MAE. No es un test virgen del proyecto y no habilita un claim SOTA.
Benchmark-10 permanece sellado.

eAP train-40 está completo (40 secuencias, 216 archivos, 536,64 GiB), pero no
incluye TTC oficial. Su uso aprobado es un piloto SSL de 2–4 secuencias y el
mismo fine-tuning EvTTC; solo se escala a 40 si el piloto mejora ese gate. El
pseudo-TTC local cubre 195.024/804.510 filas (24,24 %) y no participa en
selección, calibración ni métricas TTC oficiales.

## 26.8 Enmienda ejecutada: bbox-ROI y CARLA DVS Looming

La hipótesis de que bastaba con localizar el objeto se probó mediante
`R1_MATCHED_BBOX_ROI`. Mantiene backbone, cabeza, ventanas, batch y optimizador
de A0, y cambia solo el pooling por una selección de tokens dentro de la bbox
GT. Los cinco folds seed 7 obtuvieron:

| Brazo | Score | Error rel. | MAE | Tiempo |
|---|---:|---:|---:|---:|
| A0 seed 7 | 0,58125 | 30,16 % | 0,966 s | 1.443 s |
| R1 bbox-ROI | 0,59814 | 30,99 % | 1,010 s | 2.410 s |

R1 empeora 2,90 % el score, 2,74 % el error relativo y 4,55 % el MAE, con
1,67× tiempo. Queda rechazado sin seeds 13/21. La próxima hipótesis debe medir
expansión/FoE explícitamente; no basta con pasar tokens localizados a la misma
cabeza global.

Se incorpora `CARLA_DVS_LOOMING_1406` como fuente sintética autorizada para SSL,
looming y riesgo. No reemplaza EvTTC ni desbloquea Benchmark-10. El archivo
local coincide con el MD5 oficial `21a3e72a1c1d9c441a7426393f4e545f`. El
manifest audita 1.406 secuencias y 7.692.448.155 eventos; 1.395 secuencias son
válidas con 100 ms de contexto. Los splits bloqueados contienen 803 train, 298
validation y 294 test, sin compartir bloques contiguos de 25 IDs.

Restricciones obligatorias:

1. leer `events.npy` mediante mmap y `allow_pickle=False`;
2. no crear un cache voxel global ni una segunda copia de los 71,64 GiB;
3. tratar negativos como TTC censurado, nunca asignarles un TTC inventado;
4. no usar `vel` ni `diameter_object` como features;
5. considerar el test CARLA solo out-of-sample sintético;
6. medir utilidad por transferencia CARLA→EvTTC con fine-tuning fold-local;
7. mantener Benchmark-10 sellado hasta superar el gate OOF.

Preparación reproducible:

```powershell
uv run --no-sync python scripts/prepare_carla_looming.py `
  --root datasets/CARLA_DVS_Looming_Dataset/random_spawn `
  --manifest data/manifests/carla_dvs_looming_v1.json `
  --split data/splits/carla_dvs_looming_blocked_v1.json
```

## 26.9 Enmienda ejecutada: entrenamiento y transferencia CARLA reanudables

El brazo externo ya tiene una ruta cerrada de ejecución. El encoder
EventTubelet de 21 canales se preentrena con predicción JEPA densa futura; los
diez canales de eventos son reales y los once auxiliares se fijan a cero para
no inventar navegación o geometría. No se lee ninguna etiqueta TTC, clase de
colisión, velocidad ni diámetro.

Perfil full congelado para el primer gate:

```text
contexto / stride                   100 / 50 ms
horizontes                          50, 100, 250 ms
ventana target                      100 ms
máximo ventanas por secuencia       16
pares train / validation / test     12.020 / 4.457 / 4.297
resolución / bins                   160x90 / 5
batch / acumulación                24 / 2
workers / prefetch                  8 / 2
precisión                           BF16
optimización                        AdamW fused, warm-up + cosine, clip 1,0
target encoder                      EMA 0,99 → 0,9999
early stopping                      mínimo 8, paciencia 6, máximo 30
```

Los probes medidos seleccionan batch 24/acumulación 2/ocho workers con 8,46
observaciones/s. Batch 16/32/48/96 y seis/doce workers no lo superaron porque el
cuello está en lectura/voxelización CPU/SSD, no en capacidad VRAM. La
proyección es 32,5 min por época y 16,2 h al máximo. El smoke de dos épocas
bajó validation loss de 0,02563 a 0,02247 sin colapso. Un test de contrato de
16 pares produjo 0,02195; ambas son losses SSL, no error TTC.

Artefactos obligatorios:

```text
history.jsonl
metrics.json firmado
carla_jepa_encoder_best.pt
carla_jepa_encoder_last.pt
resume.pt atómico durante ejecución
validation_evaluation.json
test_evaluation.json
logs por etapa
orchestration_status.json
```

El checkpoint externo solo se acepta en EvTTC si coincide el hash del split,
usa EventTubelet/21 canales/5 bins, fue seleccionado por validation y declara
falso para TTC, colisión, velocidad, diámetro y Benchmark-10. La comparación
CARLA→EvTTC usa A0 en los mismos cinco folds y seeds 7/13/21 que el control.
Se auditan samples, cache, cabeza común y trainer; la inferencia se agrega OOF
y el bootstrap se agrupa por secuencia.

Comando canónico:

```powershell
.\scripts\run_carla_evttc_complete.ps1 -Profile Full -Resume
```

El pipeline completo no autoriza abrir Benchmark-10. El siguiente gate solo
pregunta si la inicialización CARLA mejora A0 OOF de manera consistente; un
buen test sintético CARLA por sí solo no promueve el modelo.

## 26.10 Enmienda ejecutada: piloto eAP-12 y pipeline eAP→EvTTC

Los pilotos pareados posteriores rechazan CARLA bajo el presupuesto corto:
CARLA-SSL empeora A0 en fold 0/seed 7 un 1,72 % de RTE y 3,60 % de MAE; el
brazo JEPA+TTC sintético empeora RTE un 17,3 %. A1 tampoco rescata estas
inicializaciones. El full CARLA se detiene: no se elimina la evidencia, pero no
se gasta el presupuesto largo sin una hipótesis nueva.

El camino activo usa 12 secuencias eAP elegidas sin consultar EvTTC. El
inventario firmado `eap_train40_inventory_v1.json` describe las 40 secuencias;
`eap_pilot12_v1.json` fija nueve train y tres validation. Solo se abren eventos
HDF5 bajo demanda mediante `ms_to_idx`; RGB, pseudo-TTC, EvTTC y Benchmark-10
quedan fuera del pretraining.

Dos brazos comparten encoder, seed, ventanas y optimización:

```text
eAP-SSL = JEPA denso a 100/250/500 ms
eAP-Geo = eAP-SSL + centro/tamaño bbox + cierre radial + expansión
          + objectness por patch
```

Las cajas 3D se proyectan con `K_event` y `T_event_ego`. Geo usa derivadas
locales como señal geométrica débil, no `TTC = -depth/velocity` como etiqueta.
La cabeza auxiliar no se transfiere; solo se inicializa el EventTubelet de A0 o
A1. El checkpoint debe ser `best` por validation y firmar inventario/split.

Perfiles canónicos:

```text
Analysis: eAP-12 máximo 3 épocas, early stop 2/1, 1.024/256;
          EvTTC fold 0/seed 7 por defecto
Full:     eAP-40 32/8, 16.384/4.096, máximo 30, early stop 8/6;
          EvTTC 5 folds × 3 seeds, máximo 40 y early stop 10/6
Hardware: BF16, batch 24/accum 2, 8 workers eAP, 12 workers EvTTC,
          pinned memory, persistent workers, prefetch 2, TF32, AdamW fused
```

Comandos únicos:

```powershell
.\scripts\run_eap_evttc_complete.ps1 -Profile Analysis -DryRun
.\scripts\run_eap_evttc_complete.ps1 -Profile Analysis -Resume
.\scripts\run_eap_evttc_complete.ps1 -Profile Full -Resume
```

El orquestador guarda logs, estado, best/last/resume, métricas y predicciones
OOF; compara A0 y A1 por separado y falla si control/transferencia no comparten
folds, seeds, samples, cache, cabeza y trainer. `--resume` exige además el
conjunto exacto de pares solicitado. El piloto real terminó tres épocas sin
colapso: SSL seleccionó loss `0,002358`; Geo loss total `0,087108` e IoU patch
`0,2867`. Son pérdidas eAP, no métricas TTC.

El screen se amplió a folds 0/1, seed 7. Geo mejora A0 en RTE/MAE en 2/2
(+3,66 %/+4,30 % agregado) y A1 en RTE en 2/2 (+6,57 %), aunque MAE de A1
queda 1/2 (+7,95 % agregado). SSL no es consistente. A0-Geo satisface el gate
12→40; el full usa el split firmado `eap_train40_v1.json` con las 40 secuencias
en 32/8, conservando validation piloto y completándola solo mediante hash de ID.
Los bootstrap RTE aún cruzan cero: no existe claim SOTA ni autorización para
abrir Benchmark-10.
