# CODEX HANDOFF — E-JEPA-TTC

Fecha de corte: 2026-08-10 (Europe/Madrid)

Rama: `scientific-recovery-v3-hardening`

Remote publicado antes de este handoff: `origin/scientific-recovery-v3-hardening`
Base al empezar este lote: `cb25c0ff9344b17f23d8e4793a5390d1ec5d6a3b`

Este archivo es el punto de entrada para una sesión nueva. El objetivo activo no está
terminado: construir y evaluar honestamente un estimador TTC event-only, después
RGB-only y finalmente multimodal, con la aspiración de superar Garl-TTC en eAP y el
SOTA comparable en EvTTC. No existe todavía un claim SOTA.

## 1. Qué se consiguió en la sesión que termina

### 1.1 Cierre sintético V8 y CVaR

La arquitectura causal-scale event-only predice foreground por endpoint, extrae una
altura visible diferenciable y deriva TTC mediante la identidad física:

```text
events t0/t1/t2
  -> foreground logits por endpoint
  -> temporal consensus reversible w=0.15
  -> altura normalizada h0/h1/h2
  -> r = log(h_current / h_previous)
  -> inverse_ttc = expm1(r) / delta_t
  -> TTC, incertidumbre y riesgo
```

El residual aprendido está acotado y es antisimétrico; la cabeza directa de TTC es
solo auxiliar y no alimenta la predicción principal. Implementación:

- `src/e_jepa_ttc/models/causal_scale_ttc.py`
- `src/e_jepa_ttc/losses/causal_scale_ttc.py`
- `configs/model/e_jepa_causal_scale_event_v8_t015.yaml`
- `configs/train/causal_scale_v8_tail_cvar.yaml`
- `docs/causal_scale_v8.md`

CVaR sobre el 10% de mayores errores de log-ratio, peso `2.0`, produjo el mejor V8:

| Métrica validation multigrupo | Resultado | Gate |
|---|---:|---:|
| Pearson macro | `.946212649` | `>= .95`, falla por `.003787351` |
| Pearson 801/802/803 | `.9481158/.9456668/.9448553` | cada grupo `>= .95` |
| TTC symmetric relative error macro | `.2954736` | `<= .30`, pasa |
| foreground IoU | `.868659` | `>= .60`, pasa |
| sign accuracy | `.982978` | `>= .95`, pasa |
| slope | `.917236` | `[.8,1.2]`, pasa |
| translation leakage p95 | `.005864` | `<= .02`, pasa |

V8 quedó cerrado sin abrir test seeds `901/902/903`. No se cambiaron gates. Artefacto
firmado: `artifacts/metrics/causal_scale_v8_diagnostic_comparison_v1.json`, identidad
`71bb1d8299141180ff964154e3440b971014e50953e174b9fb489ba9bbe1ef79`.

Commits publicados durante V8:

```text
683a4f0 docs(results): record v7 held-out correlation failure
46f9d61 feat(train): add multigroup causal scale selection
c681d34 feat(model): add temporal foreground consensus arms
1731ae4 feat(loss): add causal scale tail-risk optimization
cb25c0f docs(results): close v8 with tail-risk diagnostics
```

### 1.2 Auditoría local de eAP y Garl-TTC

Fuentes externas, solo lectura:

```text
E:\eAP_dataset       ~691.493 GiB, 334 ficheros, 40/46 secuencias locales
E:\GarlTTC_dataset   parquets públicos Garl, 88,744 filas train
E:\Garl-TTC           release oficial, commit 256661242b8a7f5e56aa3c1c02348b30f6e89de6
```

Checkpoints oficiales auditados en `E:\Garl-TTC\checkpoints`:

```text
paper_event_only_lhr.pth
paper_visual_only_lhr.pth
paper_ours_full.pth
```

La comparación primaria debe ser nuestro event-only frente a
`paper_event_only_lhr.pth`. El multimodal oficial se reportará solo como referencia de
modalidad distinta, no como comparación apples-to-apples.

No hay GT privado del test oficial local. La cifra del paper no puede presentarse como
reproducida. No ejecutar CodaBench sin freeze y autorización explícita.

### 1.3 Subconjunto representativo congelado

Cache materializado existente:

```text
artifacts/cache/garl_object_event_common_roi_screen_v4/manifest.json
artifact identity: 36c12d75c91a243f4d712831cebcd3e82f896a76196b2a65b039e680f1fac309
file SHA256: bba9ff9b143bfd57760bd61d2b6f664202581b5dee54444f44c625975557eb72
```

Cobertura exacta:

- train: 2,048 muestras, 9 secuencias;
- validation: 2,048 muestras, 3 secuencias diferentes;
- ningún ID de secuencia cruza splits;
- buckets totales: crucial `979`, small `1160`, large `1053`, negative `904`;
- cada una de las tres secuencias validation contiene los cuatro buckets;
- input real `[B,3,12,128,128]`, `delta_t=0.1 s`;
- ROI cuadrada común preserva escala absoluta;
- no contiene RGB ni máscaras de segmentación;
- las cajas oficiales están en coordenadas ROI, rango observado `21.333..106.667`.

Secuencias train:

```text
2cyv0Oedzg 5ilM1PX2vz 6h5yRW2LGc OBneIVg4Cw OYgB6RGWcq
WbCh1DRerJ mHGFBekt7X qGsgzl4Q8B t79dBxj1WS
```

Secuencias validation:

```text
DGqicHUGWb pBqGOb2vYq qoohcdtLDH
```

### 1.4 Adaptación real implementada

Nuevos componentes:

- `src/e_jepa_ttc/training/causal_scale_eap.py`
  - trainer BF16;
  - warm-up foreground;
  - CVaR 10%/peso 2;
  - gradient accumulation y clipping;
  - early stopping después de `minimum_epochs=8`, paciencia `5`;
  - selección lexicográfica por MiD macro por secuencia y failure rate;
  - límite total duro `6.0 h`;
  - `last.pt` atómico cada época con modelo, optimizer, scheduler, RNG, historial,
    estado del DataLoader y paciencia;
  - `best.pt` resumible y `model_best.pt` de inferencia.
- `scripts/train_causal_scale_eap_screen.py`
  - valida hashes y split antes de reservar GPU;
  - falla si Git/código está dirty;
  - abre solo train/validation;
  - genera `summary.json`, `validation_predictions.csv`, checkpoints e historial;
  - soporta `--resume`.
- `configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_v1.yaml`
  - 18 épocas máximo, batch 32, seed 7, BF16;
  - contrato exacto de datos, pérdida y claim boundary.
- `src/e_jepa_ttc/data/object_event_v4.py::weak_box_masks`
  - rasteriza cajas como supervisión débil declarada;
  - no son segmentación GT;
  - `t0` se invalida porque su caja es proxy en este cache;
  - `t1/t2` son supervisión oficial;
  - las cajas nunca entran en `CausalScaleTTC.forward`.

Benchmark real ya ejecutado, solo para throughput, 128 train + 128 validation:

```text
train: 3.4567 s
validation: 1.8323 s
total: 5.2890 s
peak VRAM: 395.6 MiB con batch 8
MiD macro tras un único epoch parcial: 345.18 (diagnóstico, no resultado)
known coverage: .421875 (diagnóstico, no resultado)
```

La extrapolación lineal conservadora de batch 8 es aproximadamente 85 s por época
completa train+validation y ~26 min para 18 épocas. Batch 32 debe ser más rápido. Hay
margen muy amplio bajo 6 h, pero el run completo aún no se ha ejecutado.

## 2. Integridad científica y filosofía de trabajo

1. Un resultado sintético no es un resultado eAP.
2. Validation puede seleccionar arquitectura/checkpoint; test no puede hacerlo.
3. No mover gates después de ver resultados.
4. Un fallo se conserva y diagnostica; no se oculta mediante media recortada.
5. MiD/RTE se calculan con el protocolo firmado de Garl y se reportan counts/failures.
6. Event-only se compara primero con Garl event-only sobre los mismos tokens.
7. Las cajas GT usadas para crop/supervisión se declaran como oracle; el modelo no es
   bbox-free.
8. Una caja rasterizada es weak supervision, no una máscara de segmentación real.
9. No afirmar SOTA con una seed, validation local o modalidades distintas.
10. Cada run debe guardar commit, hashes, entorno, seed, split, historial, checkpoint,
    tiempo, VRAM y predicciones regenerables.

## 3. Estado exacto al abrir la sesión nueva

Ejecutar primero:

```powershell
git status --short
git branch --show-current
git log -5 --oneline --decorate
git fetch origin
git rev-parse HEAD
git rev-parse origin/scientific-recovery-v3-hardening
```

Hay cuatro borrados tracked ajenos detectados al final de esta sesión y no incluidos
deliberadamente en el commit del handoff:

```text
APPLY.md
E_JEPA_TTC_CODEX_HANDOFF_2026-08-08.md
README_STABLE_SCREEN_V3_FIXED.txt
RESULTS_INVALIDATION.md
```

No asumir si deben restaurarse o eliminarse. Preguntar al usuario o confirmar su
intención. El runner real exige tracked/code clean, así que estos cambios deben
resolverse antes del run. Al inicio de esta sesión había siete patches y cuatro ZIPs
untracked del usuario; ya no aparecen en el status ni en el directorio al cierre y
este agente no los borró. No intentar reconstruirlos, restaurarlos o versionarlos sin
una petición explícita.

## 4. Primeras verificaciones obligatorias

```powershell
uv run ruff check src/e_jepa_ttc/data/object_event_v4.py `
  src/e_jepa_ttc/training/causal_scale_eap.py `
  scripts/train_causal_scale_eap_screen.py `
  tests/unit/test_causal_scale_ttc.py

uv run pyright src/e_jepa_ttc/data/object_event_v4.py `
  src/e_jepa_ttc/training/causal_scale_eap.py `
  scripts/train_causal_scale_eap_screen.py

uv run pytest tests/unit/test_causal_scale_ttc.py `
  tests/unit/test_garl_signed_metrics_v4.py -q

uv run pytest -q
```

Antes del full run, añadir y ejecutar un test pequeño de interrupción/reanudación que
demuestre que `state/last.pt` continúa desde la siguiente época y conserva el best.
La serialización está implementada, pero ese camino exacto no se ha probado end-to-end
en este corte.

## 5. Ejecución siguiente: entrenamiento real event-only

Una vez limpio y con tests verdes:

```powershell
uv run python scripts/train_causal_scale_eap_screen.py `
  --config configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_v1.yaml `
  --output-dir artifacts/runs/causal_scale_eap_screen_v1_seed7 `
  --device cuda
```

Si se interrumpe:

```powershell
uv run python scripts/train_causal_scale_eap_screen.py `
  --config configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_v1.yaml `
  --output-dir artifacts/runs/causal_scale_eap_screen_v1_seed7 `
  --device cuda --resume
```

Revisar al terminar:

```text
artifacts/runs/causal_scale_eap_screen_v1_seed7/summary.json
artifacts/runs/causal_scale_eap_screen_v1_seed7/validation_predictions.csv
artifacts/runs/causal_scale_eap_screen_v1_seed7/model_best.pt
artifacts/runs/causal_scale_eap_screen_v1_seed7/state/last.pt
artifacts/runs/causal_scale_eap_screen_v1_seed7/state/best.pt
```

Diagnóstico mínimo:

- curva train/validation y época elegida;
- MiD global y macro por secuencia;
- MiD/RTE/failure por cuatro buckets;
- known coverage;
- log-ratio Pearson, MAE y slope si se añade al resumen;
- weak bbox IoU, recordando que no es segmentation IoU;
- distribución de predicciones por secuencia/bucket;
- outliers de cola y error top 10%;
- tiempo, throughput y peak VRAM;
- NaN/unknown no sustituidos por cero.

## 6. Comparación exacta con Garl-TTC oficial

Falta implementar un builder de subset, sugerido:

```text
scripts/build_garl_validation_subset_from_predictions.py
```

Debe:

1. leer `validation_predictions.csv` y extraer exactamente los 2,048 sample tokens;
2. filtrar, sin modificar originales:
   - `E:\GarlTTC_dataset\data\train.parquet`;
   - `E:\GarlTTC_dataset\annotations\train.parquet`;
3. preservar orden/joins oficiales y fallar si no hay cobertura 2048/2048;
4. escribir parquets y asset list bajo
   `artifacts/subsets/garl_validation_common_roi_v1/`;
5. guardar hashes de inputs/outputs, sequences, tokens y counts en manifest firmado;
6. verificar que no se abre test privado.

Después ejecutar Garl event-only:

```powershell
uv run python scripts/evaluate_official_garl_validation.py `
  --release-root 'E:\Garl-TTC' `
  --config 'E:\Garl-TTC\configs\ablation\event_lhr.yaml' `
  --checkpoint 'E:\Garl-TTC\checkpoints\paper_event_only_lhr.pth' `
  --dataset-root 'E:\GarlTTC_dataset' `
  --data-parquet artifacts/subsets/garl_validation_common_roi_v1/data.parquet `
  --labels-parquet artifacts/subsets/garl_validation_common_roi_v1/labels.parquet `
  --asset-list artifacts/subsets/garl_validation_common_roi_v1/assets.txt `
  --output-dir artifacts/runs/garl_official_event_only_same2048 `
  --device cuda
```

Crear después un comparador firmado, sugerido:

```text
scripts/build_causal_scale_eap_garl_comparison.py
artifacts/metrics/causal_scale_eap_garl_event_only_comparison_v1.json
```

Comparar por los mismos sample tokens y secuencias:

- paper MiD overall;
- sequence-macro MiD;
- MiD/RTE/failure por bucket;
- bootstrap por secuencia, nunca por ventanas;
- latencia/preprocesamiento por separado;
- modalidad y representación declaradas.

Garl usa 2 endpoints × 20 canales con preprocessing oficial; nuestro modelo usa 3
endpoints × 12 canales en ROI común. Es una comparación de modalidad y muestras
igualadas, no una equivalencia de representación. Reportarlo explícitamente.

## 7. Criterio de decisión tras la comparación

No escalar automáticamente. Primero clasificar el fallo:

- `weak bbox IoU` bajo: localización/foreground;
- IoU alto pero log-ratio Pearson/slope malos: operador de escala/temporal;
- log-ratio bueno pero MiD malo: singularidad/clipping/known policy;
- positivo bueno y negative malo: dirección de movimiento/post-contact;
- una secuencia domina: domain shift, no sesgo global;
- cola top 10% domina: inspección de outliers, CVaR o sampler, sin mirar test.

Si nuestro event-only supera Garl event-only en validation con margen consistente por
secuencia, repetir seeds `13` y `23` bajo config preregistrada. Solo después congelar
un candidato. Si no, modificar una sola hipótesis por ablation y conservar el negativo.

## 8. Ruta arquitectónica posterior

### RGB-only

Reutilizar `CausalScaleTTC` con `modality: rgb` y encoder RGB específico. Mantener el
mismo contrato de outputs: foreground, escala, log-ratio, uncertainty/risk. Entrenar y
comparar RGB-only contra `paper_visual_only_lhr.pth` sobre los mismos tokens.

### Multimodal

No concatenar RGB y eventos desde el principio. Usar dos encoders y dos observaciones
geométricas con fusión tardía condicionada por incertidumbre/soporte:

```text
event encoder -> r_event, sigma_event, support_event
RGB encoder   -> r_rgb,   sigma_rgb,   support_rgb
              -> gated precision-weighted fusion
              -> TTC/risk common physical head
```

Debe funcionar en tres modos: event-only, RGB-only y RGB-E, con modality dropout en
training. Comparar multimodal contra `paper_ours_full.pth`, no contra el checkpoint
event-only.

### EvTTC

EvTTC test permanece sellado. Primero implementar/adaptar inferencia label-free y
congelar checkpoint/config. Separar predict y score. Ningún resultado eAP validation
autoriza por sí mismo una apertura EvTTC test o un claim SOTA.

## 9. Qué no hacer

- no abrir seeds sintéticas V8 test 901/902/903;
- no reutilizar seeds consumidas 303 o 603 como evidencia nueva;
- no comparar nuestro event-only con Garl multimodal como si fuese igualdad;
- no usar cajas/targets/IDs como inputs del forward;
- no llamar mask GT a los rectángulos weak-box;
- no seleccionar con eAP test, EvTTC test o CodaBench;
- no descargar/procesar los 691 GiB completos para este screen;
- no reconstruir ni versionar los patches/ZIPs del usuario que desaparecieron fuera
  de las acciones de este agente;
- no resolver los cuatro borrados tracked ajenos sin confirmar intención;
- no escribir “SOTA” hasta reproducir protocolos oficiales, tres seeds y evaluación
  externa comparable.

## 10. Definición de salida de la próxima sesión

Como mínimo:

1. test resume end-to-end verde;
2. entrenamiento seed 7 terminado y artefacto firmado;
3. baseline Garl event-only sobre los mismos 2,048 tokens;
4. comparación y diagnóstico publicados en `.md` y JSON/CSV;
5. decisión explícita: repetir seeds, ablation única o rechazar brazo;
6. Ruff/Pyright/Pytest verdes;
7. `git add` selectivo, commit y push;
8. ningún test privado abierto y ningún claim inflado.
