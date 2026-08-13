# E-JEPA-TTC

> Estado vigente (2026-08-13): Scientific Recovery V6 cerró seis entrenamientos
> fold-locales. V6.1 radio 2 mejora A8.0 de 197,69 a 194,12 MiD, pero su IC95%
> incluye cero y falla el gate 175. A5 causal obtiene 155,47 y es el mejor E-JEPA
> TTC limpio, aunque no es promocionable porque no preserva geometría. Garl mantiene
> 144,35 y cero failures. Public validation y private/test permanecen cerrados.
> Estado canónico: [Scientific Recovery V6](docs/SCIENTIFIC_RECOVERY_V6_STATUS.md).
> Contrato operativo V7/V8:
> [CODEX_HANDOFF.md](CODEX_HANDOFF.md). V7 mantiene test y CodaBench cerrados.

> Estado event-only (2026-08-10): Garl matched desde cero fija la métrica a batir
> en MiD macro 203,63. A0 obtuvo 382,19, A1 geometry-only 346,83 y A3
> SAM-distilled 353,64. A3 empeora A1; no existe claim SOTA.

El diagnóstico de A0 localiza el defecto antes de la física: la expansión bbox sí
correlaciona con TTC (`r=.760`), pero la expansión extraída del mapa no correlaciona
con bbox (`r=.015`) ni TTC (`r=.037`). La inversión TTC agrava ratios cercanos a
cero, pero no crea el error. A1 probó de forma aislada la hipótesis weak-box: la
altura absoluta mejoró, pero anchura, centros y cambio temporal siguieron débiles.
La weak-box no era la explicación suficiente.

El preprocessing oficial del baseline Garl matched ya está materializado y firmado
(`92af2810…f564b52`): 2.048 train/2.048 validation, event-only FP32, sin pesos
release ni fuentes selladas.

Ese baseline matched ya terminó: Garl obtiene MiD macro `203,63` y `0%` failures,
frente a A0 `382,19` y `12,30%`. La ventaja aparece en las tres secuencias y el
IC95% agrupado por secuencia de A0−Garl es `[131,74, 215,31]`. Garl matched,
sin embargo, no modela receding: todas sus predicciones son positivas y su MiD
negative es `437,60`; esta limitación sigue visible.

A1 se congeló antes del run: mismo CNN y prediction path, sin weak-box densa, con
supervisión training-only de `h,w,cx,cy`, pair-ratio cero y sin parámetros nuevos.
Terminó best 18/18 con MiD `346,83`, failure `9,96%` y Pearson log-ratio `.111`.
`log h` alcanza `.471`, pero `log w/cx/cy` solo `.079/.064/.032`; `delta log h`
contra bbox es `.059`. El siguiente control actuará sobre representación densa
event-native, no sobre clipping ni la inversión TTC.

La auditoría por endpoint confirma que la anchura bbox no es constante, pero el
decoder separable no la sigue en t1/t2 (`r=.048/.105`). Como la masa absoluta de
eventos es difusa en casi todo el ROI y el decoder actual usa `amax` por eje, el
siguiente control cambia solo a convolución full-resolution 2-D. DINO, SAM, JEPA y
pair-ratio permanecen fuera para no contaminar la ablation.

Ese control full-resolution ya terminó y fue negativo: MiD macro `380,22`, failure
`28,76%` y Pearson log-ratio `-.018`. La estructura 2-D superficial sobre eventos
crudos no basta. El siguiente paso probará si el foreground debe consumir features
profundas del encoder (`resize_conv`) antes de recurrir a pretraining RGB/event.
Ese brazo A1-DF se preregistró antes de ejecutarse. Su run terminó con más señal
temporal (`r=.186`) pero MiD macro `350,30` y
`21,09%` failures: no supera A1 ni Garl matched. El ratio aprendido está
subescalado, por lo que el siguiente control aislado será supervisarlo directamente
sin cambiar arquitectura. A1-DF-R está preregistrado con peso train-normalized
`5.0`. Terminó con macro `349,86` y `19,82%` failures: mejora marginalmente A1-DF,
pero dos de tres secuencias empeoran y no supera A1. Se cierra sin sweep/escalado.

La auditoría material posterior descarta el uso inmediato de máscaras oficiales:
las 88.744 filas de train declaran 177.488 referencias `.npy` (64.629 únicas), pero
ninguna existe bajo los seis roots locales auditados. En cambio, los 64.629 frames
RGB únicos están presentes dentro de 135 TAR. Los snapshots locales SAM ViT-L y
DINOv3 ConvNeXt-Tiny tienen config, processor, licencia y pesos verificados por hash.
Esto permite preregistrar un smoke SAM con bbox prompts **solo sobre train**, pero el
brazo resultante se llamará `event-only inference with RGB distillation`; no es
event-only puro y no puede generar targets teacher en validation. Artefacto firmado:
`artifacts/metrics/garl_foreground_resource_audit_v2.json`, identidad
`6e910ec2…f1e246`.

El smoke preregistrado posterior pasó en `cuda:0`: una bbox train real produjo tres
máscaras finitas; la seleccionada ocupa `6,63%` de la imagen, con score IoU interno
`1,0`, inferencia `0,421 s` y peak VRAM `1.691 MiB`. Es solo un gate de viabilidad,
no una evaluación de calidad ni una métrica TTC. Identidad firmada
`be097e6c…2af5e9`.

El audit train-only posterior cubrió las nueve secuencias (36 pares/72 endpoints)
y pasó los gates congelados: bbox–mask IoU mediana `.576`, cambio de área Pearson
`.647`, signo `.829` y una máscara degenerada. Autoriza precomputar únicamente las
2.048 filas train exactas, con filtros explícitos; no autoriza generar teachers en
validation ni afirmar calidad de segmentación.

La materialización exacta ya terminó: 4.096 endpoints en 32 shards, `94,92%`
válidos individualmente y `78,22%` de pares válidos después del filtro temporal.
El cache packbits ocupa ~2,18 MB, conserva cada rechazo y mantiene cobertura en las
nueve secuencias. Preprocessing total `1.784,6 s`, inferencia media `.1711 s`, peak
VRAM `1.691,9 MiB`. Manifest firmado `aaa60090…0426b0`; sigue siendo train-only y
la materialización por sí sola no era evidencia TTC. A3 se implementó/preregistró sin cambiar
la arquitectura A1: añade BCE `1.0` + Dice `.5` solo sobre las 3.204 máscaras
aceptadas y mantiene geometry-only en las demás. Validation no carga el cache.
Config SHA-256 `83e8c716…9b7754`. A3 terminó best 8/13 con MiD macro `353,64`,
failure `10,89%` y ratio Pearson `.105`: peor que A1 en las tres secuencias. Solo
mejora negative; su IC95% A3−A1 por secuencia es `[1,55,10,64]`. Se cierra sin
sweep, escalado ni seeds adicionales. La siguiente rama será event-native y no
añadirá otra dependencia de máscara RGB.

Pipeline reproducible para estimar Time-to-Contact/Time-to-Collision a partir de
cámaras de eventos, con una ruta event-only high-resolution y una futura ablación
RGB-E multimodal.

Estado: Causal Scale V8 alcanza Pearson sintético multigrupo `.94621` con CVaR pero
falla el gate `.95`; sus tests permanecen sellados. El screen eAP/Garl
train/validation de 2.048/2.048 completó A0, Garl matched y A1. La hipótesis
científica todavía no está demostrada. No existe claim SOTA ni resultado oficial
eAP/CodaBench. Consulta el
[estado operativo](STATUS.md) antes de ejecutar experimentos largos.

## Qué produce

- TTC continuo firmado en segundos;
- logits de riesgo por horizontes configurables;
- incertidumbre cuando la cabeza correspondiente está activa;
- embeddings globales/densos y diagnósticos de colapso;
- métricas macro por secuencia, robustez, calibración y latencia;
- checkpoints auditables, export ONNX, demo offline e informe regenerable.

## Instalación

Requisitos recomendados: Python 3.11, `uv`, PyTorch con CUDA para entrenamiento y
una GPU de consumo con aproximadamente 12 GiB de VRAM.

```powershell
uv sync --locked --all-groups --no-editable
uv run --no-sync python -m e_jepa_ttc --help
```

Validación completa:

```powershell
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pyright
uv run --no-sync pytest -q
```

## Datos

Las rutas locales nunca se hardcodean en el paquete. Las fuentes principales son:

```text
EAP_ROOT=E:\eAP_dataset
GARLTTC_ROOT=E:\GarlTTC_dataset
GARLTTC_RELEASE_ROOT=E:\Garl-TTC
```

- EvTTC-32 local se usa para desarrollo, grouped CV y evaluación controlada.
- El benchmark EvTTC permanece sellado hasta congelar candidato.
- eAP/Garl se lee bajo demanda desde HDF5/parquet.
- CARLA se retiró del camino activo tras una transferencia negativa.

## Object Event TTC v4

V4 corrige los fallos falsados por las auditorías v3:

```text
eventos t0/t1/t2
  -> una sola ROI cuadrada común (unión temporal + margen)
  -> 12 canales activos, sin el tail de 9 canales constantes
  -> encoder event-only online + target encoder EMA
  -> predicción JEPA local de tokens futuros, sin cajas ni motion embedding
  -> cabeza de expansión event-only exactamente antisimétrica
  -> cabeza motion-only independiente
  -> fusión tardía con gate event mínimo
  -> TTC firmado derivado de g = delta_t / TTC
```

El trainer aplica warm-up event-only y dropout de modalidad. Un checkpoint no se
selecciona si los eventos pueden ponerse a cero o barajarse sin degradación
medible. V4 conserva scratch, Level-transfer y cada rama por separado en las
métricas; no permite presentar el atajo geométrico como aprendizaje visual.

Screen completo:

```powershell
uv run --no-sync python scripts/run_e_jepa_object_event_v4.py `
  --profile screen `
  --stages preflight cache scratch level `
  --eap-root 'E:\eAP_dataset' `
  --garlttc-root 'E:\GarlTTC_dataset' `
  --pretrained artifacts\runs\level_dynamics_pilot256\pretrain\level\seed-7\checkpoint.pt `
  --device cuda
```

Consulta [el contrato y los gates de v4](docs/object_event_v4.md) antes de
escalar a full o abrir EvTTC.

## Causal Scale TTC v5

V4.31 falsó el matcher congelado. V5 reemplaza su mecanismo por un contrato común
para event-only, RGB-only y fusión tardía RGB-E:

```text
foreground causal -> altura visible -> log-ratio firmado
                   -> TTC + incertidumbre + riesgo derivados físicamente
```

El primer core event-only está implementado sin coordenadas bbox, categoría o ID de
secuencia como inputs. Su residual es antisimétrico y acotado; la cabeza TTC libre es
solo auxiliar. El gate ideal-foreground versionado pasó. Nueve diagnósticos de
aprendizaje train/validation llevaron el candidato a Pearson `.9560`, pendiente
`.9686`, signo `.9957`, IoU `.8640` y error TTC simétrico `.2639`; sigue no promovido
porque leakage de traslación `.02399` falla el gate `.02`. El test sintético se abrió
una vez desde el commit limpio `d9d20af` y confirmó el fallo: Pearson `.92135` y
translation `.02749`. Todos los datos reales permanecen cerrados.

V6 no reutiliza ese test: una cabeza foreground separable sin strides, evaluada solo
en train/validation 401/502, reduce translation leakage a `.00462` y alcanza IoU
`.89323`. Pearson `.92042` sigue por debajo de `.95`, por lo que test 603 no se abre.
La comparación firmada está documentada en [Causal Scale v6](docs/causal_scale_v6.md).

V7 transporta causalmente el TTC del par anterior y lo combina con el par actual sin
añadir parámetros. En validation 502 pasa todos los gates: Pearson `.96126`, TTC
`.24345` y translation `.00351`. El test limpio 603 posterior falló Pearson `.92014`;
todos los demás gates pasaron. V7 queda cerrado y real-data continúa sellado.

V8 mantiene esa arquitectura y el presupuesto, pero selecciona checkpoints sobre
tres grupos validation mediante media macro y peor grupo. Las seeds test 901/902/903
están preregistradas y selladas. El primer diagnóstico da Pearson macro `.80631` y
CVaR+consenso lo eleva a `.94621`; sigue sin ser pass y test no se abrió. Detalles en
[Causal Scale v8](docs/causal_scale_v8.md).

```powershell
uv run --no-sync python scripts/evaluate_causal_scale_v5_operator.py --require-clean
```

Consulta [el contrato, resultado y siguiente gate](docs/causal_scale_v5.md).

## Screen eAP/Garl causal-scale event-only

El protocolo autorizado abre solo train/validation públicos y compara modalidades
event-only sobre los mismos tokens. Config, trainer y runner:

```text
configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_v1.yaml
src/e_jepa_ttc/training/causal_scale_eap.py
scripts/train_causal_scale_eap_screen.py
```

```powershell
uv run python scripts/train_causal_scale_eap_screen.py `
  --config configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_v1.yaml `
  --output-dir artifacts/runs/causal_scale_eap_screen_v1_seed7 `
  --device cuda
```

Usa early stopping validation-only, CVaR top 10%, límite 6 h y checkpoints atómicos.
Para reanudar se repite el comando con `--resume`; la equivalencia continua/resume
está probada end-to-end. Tras el run,
`scripts/build_garl_validation_subset_from_predictions.py` crea el subset Garl exacto
y firmado por tokens. Las cajas oficiales son supervisión weak-box y crop oracle,
nunca entrada del modelo. Ese screen se completó históricamente; Scientific Recovery
V5 lo sustituyó por cadenas fold-locales autocontenidas. Véase
[el protocolo del screen](docs/causal_scale_eap_screen.md) y
[el handoff](CODEX_HANDOFF.md).

## Entrenamiento high-resolution histórico

Preflight del perfil full, sin reservar GPU:

```powershell
uv run --no-sync python scripts/run_e_jepa_garl_final.py `
  --profile full --stages train freeze `
  --eap-root 'E:\eAP_dataset' `
  --garlttc-root 'E:\GarlTTC_dataset' `
  --dry-run
```

Screen corto y acotado:

```powershell
uv run --no-sync python scripts/run_e_jepa_garl_final.py `
  --profile screen --stages train `
  --eap-root 'E:\eAP_dataset' `
  --garlttc-root 'E:\GarlTTC_dataset' `
  --output-root artifacts/runs/e_jepa_garl_event_screen_v1
```

El perfil full exige un commit limpio. Omitir `--max-samples-per-split` es parte
del contrato: usa todas las filas válidas del split firmado. `--resume` restaura
el checkpoint y el orden de muestras se deriva de seed+época.

## EvTTC Tabla VI y submission

El runner mantiene separados:

```text
train -> freeze -> evttc-predict -> evttc-score -> submission-validate
```

`evttc-predict` no puede abrir TTC/depth/categoría/máscaras. `evttc-score` recibe
los targets en otro proceso. La creación del manifest EvTTC real y el paquete
eAP/CodaBench siguen pendientes.

## Evidencia actual

| Evidencia | Resultado | Decisión |
|---|---:|---|
| BASE histórico EvTTC | 8,1554 % error relativo | ancla, no SOTA |
| A0 grouped CV 5x3 | 30,25 % ± 0,52 | arquitectura EvTTC histórica |
| A1 Dense grouped CV | 30,55 % ± 0,06 | rechazado |
| bbox-ROI / AttnRes / KDA | regresión o gate fallido | rechazados |
| high-res raw smoke 16/16 | MiD macro 1868,3186 | integración solamente |
| Object Expansion v3 | usa casi exclusivamente motion | falsado como event-TTC |
| Object Event v4 | v4.30 OOF negativo; v4.31 train-only estable pero no físicamente equivariante | no promocionado; full cerrado |
| Causal Scale v5 | test sintético: Pearson .92135, translation .02749; gates fallidos | no promovido; real-data cerrado |
| Causal Scale v6 | validation: Pearson .92042, translation .00462, IoU .89323 | test 603 sellado; no promovido |
| Causal Scale v7 | test: Pearson .92014 (falla), TTC .24576, translation .00338 | no promovido; seed 603 consumida |
| Causal Scale v8 | validation Pearson .94621 con CVaR; falla por .00379 | test 901/902/903 nunca abierto |
| Scientific Recovery V5 A8.0 | grouped-dev fold-local MiD 197,69 vs A6 211,51 y Garl 144,35 | mejora A6, falla gate 175; no promovido |

El smoke high-resolution valida integración, no precisión. Los resultados v3
muestran que la expansión firmada y la supervisión de ratio son útiles, pero
que el crop independiente, la ausencia de t0 y el atajo de cajas impiden
atribuir el resultado a eventos. V4 existe precisamente para falsar o corregir
esa ruta.

## Arquitectura base

```text
eventos raw HDF5
  -> voxel temporal causal [T,C,H,W]
  -> patch embedding high-resolution
  -> atención espacial por ventanas
  -> merge 2x2 opcional
  -> mixer temporal block-causal
  -> query pooling / tokens densos
  -> cabeza TTC firmada
```

KDA permanece como resultado negativo. La configuración RGB-E existe como
contrato de investigación, pero el trainer event-only la rechaza hasta que la
fusión causal esté implementada y probada.

## Estructura útil

- `src/e_jepa_ttc/`: datos, modelos, pérdidas, evaluación e inferencia;
- `scripts/build_eap_object_event_v4_cache.py`: caché común t0/t1/t2;
- `scripts/train_e_jepa_object_event_v4.py`: trainer y gates v4;
- `scripts/run_e_jepa_object_event_v4.py`: orquestación screen/full;
- `configs/experiment/e_jepa_garl_object_event_{screen,full}_v4.yaml`: perfiles v4;
- `data/protocols/garl_evttc_table_vi_v1.yaml`: frontera predict/score;
- `artifacts/metrics/`: evidencia compacta versionada;
- `PLAN.md`: plan científico y gates;
- `STATUS.md`: handoff operativo actual.

`artifacts/runs`, `artifacts/features`, datasets y caches son locales/ignorados y
se pueden regenerar. No deben subirse a Git.

## Integridad científica

- splits por secuencia, nunca aleatorios por ventana;
- selección de modelo macro por secuencia;
- pretraining SSL sin etiquetas TTC;
- EvTTC no se usa para seleccionar el supuesto zero-shot;
- resultados negativos y fallos relevantes se conservan;
- no se declara SOTA desde smokes, una semilla o cifras copiadas de artículos;
- v4 exige dependencia observable de eventos antes de congelar candidato.

## Documentación

- [estado canónico Scientific Recovery V5](docs/SCIENTIFIC_RECOVERY_V5_STATUS.md)
- [auditoría científica y de código V5](docs/E_JEPA_TTC_V5_SCIENTIFIC_CODE_AUDIT.md)

### v4.30 authoritative negative result and v4.31 next action

The authoritative full v4.30 SHA256 is `9722202A4D33F6B5D1B933EEDA1F9143E13E4E2FD64B21356E93783AFAA1C689`, status `completed_oof_gate_failed`. Stabilization passed `.0010116798/.0423071422/.1308624286`; rank-only winner `stable_multiscale_similarity` has no champion. Its best-arm Pearson `.4791568608`, negative accuracy `0`, balanced `.5`, std ratio `.3731916487`, slope `.1788173388`, high-bucket Pearson `-.1972577670`, and ratios `.92439/.58893/.48926/.30467` failed the frozen objective; both arms failed with no sealed data opened. The target-free saved-NPZ post-hoc audit (not preregistered) found forward-vs-swap `log_eta` correlation `+.53338`, zero sign flips, and 95.8% coverage at `|log_eta| >= .005`. The next action after Sol's rethink is a TTC-label-free but train-box-conditioned common-object-ROI v4.31 redesign: TTC/sign/bucket-independent selection, immutable sequence/time-disjoint train-only stabilization/audit pools, sanitized event/ROI-only artifacts, exact physical reversal controls, and no development/test/EvTTC. The direct full-frame v4.31 draft was rejected before execution and is not evidence.

The v4.31 implementation and negative 512-row diagnostic are documented in [the causal-audit handoff](docs/object_event_v4_31.md).  Cache preflight passed and no sealed data were opened.  Stability passed, but the operator failed analytic zoom, slope, sign, oddness, translation leakage and swap coverage; stage 2 was absent and the recorded worktree was dirty, so the result is explicitly non-selectable and non-authoritative.  Full remains closed.

### Superseded historical v4.30 executable protocol

The following is superseded diagnostic history, not current v4.30 state. A 96-row post-fix train-only diagnostic was `diagnostic_only`: JS median `.010237284936010838` passed; JS p95 `.19495552778244019` and BASE-pixel displacement p95 `.5500071191315064` failed. The earlier `D9DE07…` diagnostic is superseded. The authoritative completed v4.30 summary is the SHA and negative result stated above; historical diagnostics cannot modify it.

- [plan de ejecución](PLAN.md)
- [estado actual](STATUS.md)
- [protocolo experimental](docs/experimental_protocol.md)
- [Object Event TTC v4](docs/object_event_v4.md)
- [Object Event TTC v4.29 preregistration](docs/object_event_v4_29.md)
- [Object Event TTC v4.30 stable similarity preregistration](docs/object_event_v4_30.md)
- [Object Event TTC v4.31 causal audit handoff](docs/object_event_v4_31.md)
- [Causal Scale TTC v5](docs/causal_scale_v5.md)
- [Causal Scale TTC v6](docs/causal_scale_v6.md)
- [Causal Scale TTC v7](docs/causal_scale_v7.md)
- [ADR-0001: geometry-bound causal scale](docs/decisions/ADR-0001-causal-scale-v5.md)
- [dataset card](docs/dataset_card.md)
- [model card](docs/model_card.md)
- [limitaciones](docs/limitations.md)
- [reproducibilidad](docs/reproducibility.md)
- [informe técnico](docs/technical_report.md)
- [informe PDF](docs/e_jepa_ttc_paper.pdf)

El sistema es experimental y no está validado para control de seguridad.
