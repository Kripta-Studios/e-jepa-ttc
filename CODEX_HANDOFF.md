# CODEX HANDOFF — E-JEPA-TTC

Fecha de corte: 2026-08-10 (Europe/Madrid)

## Resultado activo: Garl matched fijado y A1 geometry-only negativo

A0 seed 7 terminó en 16 épocas, seleccionó época 11 y no se promueve: MiD global
`383.8549432`, MiD macro `382.1905104`, failure `12.3046875%`, weak-box IoU
`0.4997858107` y Pearson log-ratio `0.0456290990`. La referencia release sobre los
mismos 2.048 tokens obtuvo MiD macro `117.4281582` y cero failures, pero sus tres
secuencias validation aparecen en train oficial y tuvo 88.744 filas/50 épocas.
Etiquetar la tabla: **official release reference; unequal training budget and
sequence exposure**.

Comparador firmado:
`artifacts/metrics/causal_scale_eap_garl_event_only_comparison_v1.json`, identidad
`9f2bebde05729b7ace6fdbc0a990e6b75bf180ec87220924219ed7095105281c`.
La diferencia A0 menos release es `+264.4246879` MiD, IC95% por secuencia
`[228.8007775, 302.7170041]`.

Matched training terminó desde cero en GPU: época seleccionada 11/16, MiD global
`203.0982270`, MiD macro `203.6341709`, failure `0%`, Pearson log-ratio `.372213`.
A0 queda `+180.7031360` MiD por detrás, IC95% por secuencia
`[131.7444284, 215.3146093]`, y gana solo el `35.6904%` de los pares finitos.
Garl matched mejora las tres secuencias. Su punto débil es negative: MiD `437.5957`
y predicciones siempre positivas; A0 obtiene `210.1439` ahí pero con `20%` failures.
La comparación firmada completa tiene identidad
`e63447135e2b09c5c6a7e2afb996bb70cce8cbba4a112afc87069e2f60c254de`.
A1 fue preregistrado antes de ejecutarse en
`configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a1_geometry_v1.yaml`,
SHA256 `bc3fe3daabb8f205b1dda81f6da442c2d7452253330960d0c3ff65af7795ba28`.
Mantiene el mismo model config y 344.591 parámetros entrenables. La única diferencia
científica es reemplazar BCE+Dice+extent contra la weak-box rasterizada por Huber
sobre `log h`, `log w`, `cx`, `cy` derivados numéricamente de bbox. Pesos congelados:
`1.25/1.25/2.5` (centro promedia x/y), suma nominal 5. `pair_ratio` permanece cero.
El forward sigue aceptando solo eventos y delta; A1 no invoca el rasterizador.

A1 seed 7 terminó las 18 épocas en GPU y seleccionó la época 18: MiD global
`346.1117485`, MiD macro `346.8294571`, failure `9.9609375%`, known coverage
`.900390625` y Pearson log-ratio `.1108212322`. Mejora A0 en `35.3610532` MiD
macro y en las tres secuencias, pero sigue `143.1952862` por detrás de Garl
matched. El IC95% por secuencia de A1−Garl es
`[115.1041790, 166.6704803]`; A1 gana `39.0998%` de los 1.844 pares finitos.

La geometría explica por qué no basta: `corr(log h_pred,log h_bbox)=.470828`,
pero anchura `.078759`, centro x `.063569` y centro y `.031956` permanecen débiles.
La dinámica de altura solo alcanza `.059130` frente a bbox-ratio y `.104778`
frente al ratio físico; `r_iso` es esencialmente nulo (`-.000826` frente a física).
Por tanto, quitar la weak-box ayuda parcialmente a altura y MiD, pero no confirma
que el rectángulo fuese la causa principal: la representación espacial/temporal
event-native sigue siendo el cuello de botella. No se promueve A1 ni se escala.

Comparación A1/Garl firmada:
`artifacts/metrics/causal_scale_eap_garl_event_only_a1_geometry_comparison_v1.json`,
identidad `471fa106f4137f71ecfa4165abec696e5f83644830ded14a82abff8fb7ba485d`.
La siguiente intervención debe actuar sobre representación densa event-native,
manteniendo la cabeza geométrica y Causal Scale como controles; A1-R no es el
siguiente paso porque A1 tampoco aprendió `w,cx,cy` de forma suficiente.

Auditoría de observabilidad firmada
`artifacts/metrics/causal_scale_eap_a1_geometry_observability_v1.json`, identidad
`737a3663c13dc083b918e0101f4954bcfc22b23257255e0d183f8e09f0aa635d`:
t1/t2 repiten el mismo fallo (`h r=.478/.493`, `w r=.048/.105`). La anchura bbox
sí varía (`std=.096/.095`), por lo que no es un target constante. La masa absoluta
de eventos cruda ocupa casi todo el ROI (`extent h=.9965`, `w=.9984`, std
`~.003`) y su cambio no reproduce bbox (`r=-.052/-.078`). El decoder separable
actual opera sobre el input full-resolution y colapsa cada eje con `amax`; la
hipótesis siguiente, aún no demostrada, es que ese colapso pierde coocurrencia 2-D
en un ROI con actividad difusa. El control mínimo es `equivariant_fullres` 2-D con
la misma loss A1, no DINO/SAM/JEPA ni `pair_ratio` simultáneamente.

El control se denomina A1-FR y está preregistrado en
`configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a1_fullres_v1.yaml`,
SHA256 `7ceb114963e8aad8f4c7edeb70344759543d3ac58abc6a47b862d3acf772c42e`.
Su model config SHA256 es `97232184d7fb00520136319f5e902c726e26766ddaae236459b6d42d9596d39a`.
Tiene 340.870 parámetros, 3.721 menos que A1. Fuera del decoder foreground, el
model config es exactamente igual; data/training/loss son byte-semánticamente
iguales a A1. Debe publicarse antes de ejecutar y correr una sola vez en CUDA.

La primera ejecución de infraestructura A1-FR queda invalidada, no borrada. En su
checkpoint elegido, `DGqicHUGWb` tenía MiD `NaN` (100% failure en negative) y el
macro promedió solo las otras dos secuencias. El selector no exigía cobertura
finita de todas las secuencias. Artifact de invalidación firmado
`causal_scale_eap_a1_fullres_invalid_selection_v1.json`, identidad
`fd5bde50328080975781a8fc2cdae1e0a198bb5a878430fde3e1f87c9be8f19b`.
El selector ahora exige MiD finito en las tres secuencias y no cuenta candidatos
incompletos para best/stale. `minimum_epochs=8` sigue significando suelo de early
stopping; la selección comienza después del warm-up. Repetir desde cero en un output
nuevo; no reanudar el estado contaminado.

La repetición válida A1-FR terminó 16 épocas y seleccionó la 11 con cobertura 3/3:
MiD global `380.3621495`, macro `380.2202364`, failure `28.7597656%`, known
`.7124023` y Pearson log-ratio `-.0181180`. Es peor que A1 por `+33.3908` MiD
macro y `+18.7988` puntos de failure; solo mejora A0 en `1.9703` MiD, a costa de
muchos más unknowns. Altura/anchura absolutas `.271/.090` y deltas contra bbox
`-.020/-.028` rechazan la hipótesis de que conservar 2-D en una cabeza superficial
raw fuese suficiente. Comparador firmado identidad
`b02518601497907ae2ca41a345c8719298f97047ee59e9b7fad8909bd53c3c35`.

Ambos decoders probados (`separable` y `fullres`) reciben el input crudo y no las
features profundas de `_EndpointEncoder`. La siguiente hipótesis mínima es usar el
decoder existente `resize_conv`, que sí consume features aprendidas, manteniendo la
loss A1. No introducir pretraining/teachers hasta probar ese control.

Ese control queda preregistrado como A1-DF en
`configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a1_deep_features_v1.yaml`.
Experiment SHA256 `dddfb393bb0ce2c3245335cc459948c0830a435ffeaf5a76e710679a8180b284`;
model SHA256 `265dbfd57e68d7a6aa385fbf31dc0ad41154b17afbd1d9454bbd8ddd80c6663f`;
355.118 parámetros. Ejecutar una sola vez seed 7 en GPU y no tocar pair-ratio,
teachers, unknown/clip ni loss A1.

A1-DF terminó 18 épocas y seleccionó la 14: MiD global `350.0584595`, macro
`350.3020204`, failure `21.09375%`, known `.7890625` y Pearson log-ratio
`.1864874`. Mejora claramente la señal de A1 (ratio `.1108 -> .1865`, delta altura
vs física `.1048 -> .1704`, anchura absoluta `.0788 -> .2428`), pero empeora MiD
en `3.4726` y failure en `11.1328` puntos. No se promueve ni se escala.

La descomposición firmada `5a9c42934f3335ddbe4fe679f3e53f4187926fa46749321fb27ac1e3775141da`
muestra ratio analítico/físico `r=.1703`, slope `.0848`, residual/físico `r=.0622`,
slope `.0038`, y 433 unknown por ratio bajo, cero por soporte. La representación
profunda recupera señal, pero el cambio queda muy subescalado. El siguiente control
mínimo justificable es A1-DF-R: mismo A1-DF y una única supervisión pair-ratio
directa, preregistrada antes de ejecutarse. Teachers/JEPA siguen aplazados.

A1-DF-R queda congelado en
`configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a1_deep_features_ratio_v1.yaml`,
SHA256 `b3f9eb9eb05b028c57e578f0962701647dcc29b8708f7b5a0a8ac9870de43f67`.
Único cambio experimental: pair-ratio `0 -> 5`. El target deriva de altura bbox
numérica training-only, sin rasterización; warm-up lo mantiene en cero. El peso se
fijó con 2.048 train: contribución `.09044`, comparable a width `.08547`, sin usar
validation ni sweep.

A1-DF-R terminó 18 épocas, best 17: global `349.5329377`, macro `349.8628324`,
failure `19.82421875%`, known `.8017578`, ratio `.1702691`. Frente a A1-DF solo
mejora `.4392` MiD y `1.2695` puntos de failure; DGq/pBq empeoran y únicamente
qooh mejora, así que la ganancia está dominada por una secuencia. Analítico/físico
baja de `.1703` a `.1499`; slope sube `.0848 -> .0893` y unknown canónico baja
432 -> 406. Resultado negativo/insuficiente; no sweep ni escalado.

Comparador firmado `0560154542b06c12d20a24ed719ec9461ebaab9d7e4fa080afe964cda2dd6205`;
descomposición `0bc741a5f732f705e4862a2e40e45413027fa6b28f3e326468ed30cea49900e7`.
La auditoría de recursos ya terminó: 0/64.629 máscaras únicas son materiales; los
64.629 RGB únicos sí existen en 135 TAR. SAM ViT-L y DINOv3 ConvNeXt-Tiny pasan
config/processor/licencia/pesos locales. Artefacto firmado
`garl_foreground_resource_audit_v2.json`, identidad `6e910ec2…f1e246`, generado por
el commit publicado `4f5cc46`. No se abrió test ni se cargaron teachers.

El smoke SAM bbox-prompt sobre una fila train ya pasó desde `e4969f1`: BF16/CUDA,
inferencia `.4207 s`, peak VRAM `1691.39 MiB`, máscara finita `6.63%` y score interno
`1.0`. Artefacto `be097e6c…2af5e9`. No es evidencia de calidad ni TTC. El siguiente
trabajo es preregistrar un audit train-only multisequence de scores/geometría y, si
pasa, precomputar solo train para A3 `event-only inference with RGB distillation`;
validation no recibe pseudo-máscaras. No escalar 2.048 ni abrir test.

El audit multisequence ya pasó desde `4400dd7`: 36 pares/72 endpoints, nueve
secuencias, sin TTC; bbox–mask IoU mediana `.5761`, área temporal Pearson `.6471`,
signo `.8286`, una degenerada. Artefacto `0922d540…73dd44`, endpoints CSV
`bf659472…84eaf2`. El subset exacto train ya expone las 2.048 filas y rutas RGB en
`artifacts/subsets/garl_event_only_matched_screen_v1/train_data.parquet`. Próximo
hito: preregistrar/materializar solo esas 4.096 máscaras con filtros train-derived;
no abrir `validation_data.parquet` para teacher generation.

La materialización terminó en `6f9c92a`: 32 shards, 2.048 tokens exactos,
4.096 endpoints; `.9492` válidos individualmente y `.7822` pares/`3204` máscaras
usables tras filtro temporal. Tiempo `1784.63 s`, inferencia `.17110 s`, VRAM
`1691.89 MiB`, cache ~2.18 MB. Manifest firmado `aaa60090…0426b0`; todos los NPZ,
sidecars y orden de tokens fueron verificados. Próximo hito: integrar A3 como A1 +
SAM BCE/Dice train-only en masks válidas, con fallback geometry-only y validation
sin teacher.

El subset para matched training está materializado en
`artifacts/subsets/garl_event_only_matched_screen_v1`, identidad firmada
`dd08ecc983f30e38a939204f9a2df09e4966bbe73bd764c972f7726e5d4e34d3`.
Contiene exactamente 2.048 train/9 secuencias y 2.048 validation/3 secuencias,
con igualdad de tokens, join keys y TTC contra cache y parquets públicos.

### Diagnóstico A0 reproducible

`artifacts/metrics/causal_scale_eap_a0_failure_decomposition_v1.json`, identidad
`75918c58cd91258fac5aac11f8d6fca00ce6cf43014e5ee19ab3a30d7c91beb7`:

- bbox-ratio vs ratio físico: Pearson `.759753`, slope `.862189`;
- altura predicha vs altura bbox: Pearson `.372040`, slope `.170747`;
- ratio analítico vs bbox: Pearson `.014517`;
- ratio analítico vs físico: Pearson `.036820`;
- residual vs físico: Pearson `-.033979`;
- ratio efectivo vs físico: Pearson `.045641`;
- 252/2.048 unknown por `|pair_ratio| < .002`; cero por soporte bajo;
- 151 predicciones conocidas saturadas en ±60 s.

Conclusión observacional: las cajas contienen señal de escala útil y los eventos
tienen soporte, pero el mapa no aprende una extensión temporal fiel. La inversión
física amplifica ratios malos cerca de cero. La weak-box rectangular es la hipótesis
principal, no una causa confirmada; A1 es el experimento que debe resolverlo.

El preprocessing oficial del baseline matched ya está completo en
`artifacts/cache/garl_official_event_only_matched_preprocessing_v1`, identidad
`92af281030170733411ef9d65b19e88ebc8019c729dd6743e02ae9c40f564b52`.
Contiene 2.048 train/2.048 validation, tensores FP32 `[40,128,128]`, sin RGB ni
bbox como input. Declara el crop bbox oracle oficial. Train tardó `166.7501 s` y
validation `155.3283 s`; el error máximo de target fue menor de `4.8e-7 s`.
Garl matched y A1 están cerrados; no reinterpretar la referencia release como
comparación causal ni reabrir A1 con pesos ajustados post-hoc.

El runner `scripts/run_garl_matched_screen.py` desactiva explícitamente ambos
pretrained checkpoints release y escribe todo fuera de `E:\Garl-TTC`. Su smoke de
2 batches, batch 32/8 workers, terminó en 59.51 s sin OOM y dejó el release intacto.
No usar el checkpoint smoke como resultado matched.

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
margen muy amplio bajo 6 h. Esta estimación histórica queda supersedida por los runs
A0 (`541.49 s`) y A1 (`631.88 s`) completos.

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

El test `tests/integration/test_causal_scale_eap_resume.py` ya demuestra igualdad
exacta entre cuatro épocas continuas y dos + resume: modelo, optimizer, scheduler,
RNG Torch/Python/NumPy, generador del DataLoader, historial y best checkpoint. También
demuestra rechazo fail-closed si cambia config o tamaño del dataset.

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

El builder exacto ya está implementado:

```text
scripts/build_garl_validation_subset_from_predictions.py
```

Valida y firma:

1. leer `validation_predictions.csv` y extraer exactamente los 2,048 sample tokens;
2. filtrar, sin modificar originales:
   - `E:\GarlTTC_dataset\data\train.parquet`;
   - `E:\GarlTTC_dataset\annotations\train.parquet`;
3. preservar orden/joins oficiales y fallar si no hay cobertura 2048/2048;
4. escribir parquets y asset list bajo
   `artifacts/subsets/garl_validation_common_roi_v1/`;
5. guardar hashes de inputs/outputs, sequences, tokens y counts en manifest firmado;
6. verificar que no se abre test privado;
7. igualdad TTC cache/predictions frente al parquet público con tolerancia `1e-6`;
8. orden exacto tras roundtrip Parquet y manifest con hash canónico de tokens.

Ejecutarlo tras el entrenamiento:

```powershell
uv run python scripts/build_garl_validation_subset_from_predictions.py `
  --predictions artifacts/runs/causal_scale_eap_screen_v1_seed7/validation_predictions.csv `
  --output-dir artifacts/subsets/garl_validation_common_roi_v1 `
  --expected-count 2048
```

Después ejecutar Garl event-only:

```powershell
uv run python scripts/evaluate_official_garl_validation.py `
  --release-root 'E:\Garl-TTC' `
  --config 'E:\Garl-TTC\configs\ablation\event_lhr.yaml' `
  --checkpoint 'E:\Garl-TTC\checkpoints\paper_event_only_lhr.pth' `
  --dataset-root 'E:\eAP_dataset' `
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

1. entrenamiento seed 7 terminado y artefacto firmado;
2. subset exacto construido con el builder ya probado;
3. baseline Garl event-only sobre los mismos 2,048 tokens;
4. comparación y diagnóstico publicados en `.md` y JSON/CSV;
5. decisión explícita: repetir seeds, ablation única o rechazar brazo;
6. Ruff/Pyright/Pytest verdes;
7. `git add` selectivo, commit y push;
8. ningún test privado abierto y ningún claim inflado.

## 11. Prompt listo para la siguiente sesión

```text
Trabaja en el repo
C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\EVOCON_JEPA_Codex_Handoff\e-jepa-ttc
en la rama scientific-recovery-v3-hardening. Lee primero AGENTS.md y
CODEX_HANDOFF.md completos y trata el estado actual del worktree como autoritativo.
Usa Sol-Advisor conforme a su skill, sin sustituir sus roles si su lane no está
disponible.

Continúa la ruta honesta hacia superar Garl-TTC en eAP y después EvTTC. No abras
test privado eAP, CodaBench, EvTTC test ni las seeds sintéticas V8 901/902/903. No
reutilices 303/603. No afirmes SOTA con validation local o una seed.

Primero inspecciona los cuatro borrados tracked ajenos que CODEX_HANDOFF.md registra;
no los restaures ni los confirmes sin mi autorización. El runner exige código limpio,
así que pregúntame qué hacer con ellos si siguen presentes. No toques otros cambios
ajenos.

Cuando el worktree esté limpio, ejecuta el entrenamiento event-only seed 7:
uv run python scripts/train_causal_scale_eap_screen.py --config
configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_v1.yaml --output-dir
artifacts/runs/causal_scale_eap_screen_v1_seed7 --device cuda
Usa --resume si existe state/last.pt. El trainer ya tiene early stopping, límite 6 h
y resume determinista probado end-to-end.

Analiza summary.json y validation_predictions.csv por secuencia, bucket, MiD, RTE,
failure rate, known coverage, log-ratio, weak-box IoU y cola top 10%. Después construye
el subset exacto con scripts/build_garl_validation_subset_from_predictions.py y evalúa
el checkpoint oficial EVENT-ONLY de E:\Garl-TTC sobre exactamente esos 2.048 tokens
mediante scripts/evaluate_official_garl_validation.py, usando
E:\Garl-TTC\configs\ablation\event_lhr.yaml y
E:\Garl-TTC\checkpoints\paper_event_only_lhr.pth. No uses el multimodal como baseline
apples-to-apples; repórtalo solo como referencia separada.

Implementa un comparador firmado por token/secuencia/bucket con bootstrap por
secuencia, diagnostica los fallos y cambia una sola hipótesis por ablation si nuestro
modelo no gana. Si gana consistentemente en validation, preregistra y ejecuta seeds
13/23 antes de freeze. Después continúa RGB-only y fusión tardía RGB-E; EvTTC solo
tras freeze label-free. Actualiza los .md y artefactos regenerables conforme avances,
ejecuta Ruff/Pyright/Pytest, y haz git add selectivo, commit y push en hitos lógicos.
```
