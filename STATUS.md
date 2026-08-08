# Estado del repositorio

Actualizado: 2026-08-03.

Branch activa: `scientific-recovery-v3-hardening`.

Base experimental observada en los artefactos locales: commit `6e4ad4b29a805dc26a88a4ca1f3368ba1bcf952a`, con worktree `dirty` durante los screens Dense Level–Dynamics. Este documento registra resultados de desarrollo; no constituye una afirmación SOTA.

## Objetivo

Construir un estimador TTC por objeto que supere a Garl-TTC bajo protocolos eAP y EvTTC comparables, primero event-only y después RGB-E. Ningún checkpoint está promovido todavía: faltan evaluación multisemilla, test oficial eAP/CodaBench y EvTTC zero-shot sellado.

CPLA-high is diagnostic only; it is never an official final test split.

## Estado ejecutivo

Funciona y está validado:

- lectura raw/on-demand de eventos eAP y unión con anotaciones Garl-TTC;
- split por secuencia y sampling balanceado;
- backbone high-resolution compatible entre Dense Level–Dynamics JEPA y el downstream;
- transferencia exacta y fail-closed del backbone;
- pretraining label-free de los brazos `level`, `temporal_residual`, `nce` y `nce_visreg`;
- entrenamiento supervisado, checkpoints `best`/`last`, resume y métricas firmadas;
- auditoría de inputs, embeddings, gradientes, perturbaciones y micro-overfit;
- capacidad del modelo para memorizar 16 ejemplos desde scratch y desde inicialización JEPA.

Bloqueado o no demostrado:

- ningún brazo JEPA mejora todavía el downstream TTC de forma científica;
- los screens supervisados v1 colapsaron a predicciones casi constantes;
- no se ha ejecutado el nuevo `stable-screen-v2` sobre validation sequence-disjoint;
- NCE puro no modificó materialmente el encoder ni mejoró downstream;
- RGB-E, full multisemilla, EvTTC Tabla VI y test oficial eAP siguen pendientes.

## Dense Level–Dynamics: pretraining real

Se entrenaron cuatro brazos con el mismo backbone `192/16/6/2/no-merge`, las mismas filas y 1.000 updates:

| Brazo | Resultado mecanístico |
|---|---|
| `level` | resuelve casi por completo el objetivo absoluto, pero no demuestra dinámica |
| `temporal_residual` | modifica fuertemente el encoder y sigue aprendiendo el residuo |
| `nce` | pérdida prácticamente plana y encoder casi idéntico a `level` |
| `nce_visreg` | VISReg modifica la geometría; NCE permanece estancado |

La auditoría de distancia entre checkpoints mostró aproximadamente:

- `level` vs `temporal_residual`: encoder relative-L2 `0,721`;
- `level` vs `nce`: encoder relative-L2 `0,0075`;
- `level` vs `nce_visreg`: encoder relative-L2 `0,305`.

Estos resultados son mecanísticos, no una mejora TTC demostrada.

## Downstream compatible v1: resultado negativo

Los cinco downstreams compatibles usaron el mismo backbone, 2.048 muestras por split y seed 7:

| Inicialización | MiD macro validation | Diagnóstico |
|---|---:|---|
| scratch | `201,864049` | predicción casi constante |
| level | `201,862249` | indistinguible de scratch |
| temporal_residual | `202,008482` | mejor RTE, peor MiD primario |
| nce | `201,864323` | indistinguible de scratch |
| nce_visreg | `201,830460` | mejora nominal `0,0166 %`, no promocionable |

La desviación de las predicciones respecto al target fue extremadamente pequeña:

- scratch/level/nce: ratio `prediction_std / target_std` alrededor de `5e-6`;
- temporal residual: alrededor de `1,6e-4`;
- nce_visreg: alrededor de `1e-3` a `3e-3` según checkpoint/muestra.

Ningún brazo pasa un gate científico de transferencia.

## Diagnóstico del colapso supervisado

### Datos

Los eventos no están vacíos ni son idénticos:

- fracción no nula media `0,208`;
- desviación global `4,283`;
- diferencia absoluta media entre muestras adyacentes `0,190`;
- actividad distribuida en los cinco pasos temporales.

Por tanto, el colapso no procede de una lectura HDF5 vacía ni de una desalineación evidente evento-label.

### Geometría y gradientes

Los embeddings downstream terminaron casi unidimensionales:

- rango efectivo aproximado `1,05–1,21` en dimensión 192;
- la primera dirección explica entre `95,8 %` y `99,3 %` de la varianza;
- la norma de gradiente de la cabeza TTC es cientos o miles de veces superior a la del patch embedding.

El modelo aprende rápidamente un readout casi constante mientras el backbone recibe una señal muy pequeña.

### Perturbaciones

Poner los eventos a cero cambia fuertemente el embedding, pero invertir el orden temporal produce cambios diminutos. El modelo colapsado detecta contenido/actividad, pero apenas explota el orden temporal.

## Micro-overfit: conclusiones

Los micro-overfits deterministas descartan un bug fundamental de capacidad:

- scratch full-batch memoriza 16 ejemplos con error prácticamente cero;
- `level` con LR uniforme alcanza Pearson `0,9973`, pero más lentamente que scratch;
- `level` con LR discriminativo alcanza Pearson aproximadamente `1,0` y MAE `0,039 s`;
- `level` head-only no funciona bien: Pearson `0,35`, MAE `4,35 s`;
- `level` pool+head sí memoriza: Pearson `0,9995`, MAE `0,259 s`;
- scratch con backbone aleatorio congelado y pool+head también memoriza perfectamente.

Conclusión:

1. el modelo y la alineación de datos pueden aprender TTC;
2. el query pooling debe adaptarse junto con la cabeza;
3. memorizar con un backbone congelado no demuestra que JEPA codifique TTC, porque un backbone aleatorio también lo hace;
4. el protocolo v1 de batch 2, LR único `3e-4`, BF16 y fine-tuning completo inmediato favorece el atractor constante;
5. el beneficio de JEPA sigue sin demostrarse.

## Stable fine-tuning v2

Se introduce un perfil nuevo sin modificar los perfiles históricos:

- módulo `src/e_jepa_ttc/training/tubelet_finetuning.py`;
- grupos AdamW separados: backbone, query pooling y TTC head;
- `collision_head` excluida mientras no exista loss de colisión;
- warm-up de pooling+cabeza durante 32 optimizer steps;
- LR posterior: backbone `1e-5`, pooling `1e-4`, cabeza `3e-4`;
- batch efectivo 16 mediante acumulación 8;
- FP32 para el gate de estabilidad;
- métricas de salud de predicción por época;
- checkpoints colapsados no pueden convertirse en `best.pt`;
- resume valida la identidad exacta de los grupos del optimizador.

El umbral inicial de colapso es:

```text
prediction_std / target_std < 0.01
```

El perfil es únicamente un screen de desarrollo de 256/256 muestras y seed 7.

## Archivos del perfil estable

```text
src/e_jepa_ttc/training/tubelet_finetuning.py
scripts/train_e_jepa_tubelet_lhr.py
scripts/run_e_jepa_garl_final.py
configs/train/garl_highres_stable_screen_v2.yaml
configs/experiment/e_jepa_garl_event_dense_level_dynamics_stable_screen_v2.yaml
tests/unit/test_tubelet_lhr_finetuning.py
tests/integration/test_tubelet_lhr_stable_screen.py
```

## Primer gate que debe ejecutarse

### Tests

```powershell
uv run --no-sync ruff check `
  src/e_jepa_ttc/training/tubelet_finetuning.py `
  scripts/train_e_jepa_tubelet_lhr.py `
  scripts/run_e_jepa_garl_final.py `
  tests/unit/test_tubelet_lhr_finetuning.py `
  tests/integration/test_tubelet_lhr_stable_screen.py

uv run --no-sync pytest `
  tests/unit/test_tubelet_lhr_finetuning.py `
  tests/integration/test_tubelet_lhr_stable_screen.py
```

### Scratch estable

```powershell
uv run --no-sync python scripts/run_e_jepa_garl_final.py `
  --profile stable-screen `
  --stages train `
  --eap-root 'E:\eAP_dataset' `
  --garlttc-root 'E:\GarlTTC_dataset' `
  --output-root artifacts/runs/stable_screen_v2/scratch
```

### Level estable

```powershell
uv run --no-sync python scripts/run_e_jepa_garl_final.py `
  --profile stable-screen `
  --stages train `
  --eap-root 'E:\eAP_dataset' `
  --garlttc-root 'E:\GarlTTC_dataset' `
  --pretrained artifacts/runs/level_dynamics_pilot256/pretrain/level/seed-7/checkpoint.pt `
  --output-root artifacts/runs/stable_screen_v2/level
```

## Gate de promoción

No ejecutar `temporal_residual`, `nce_visreg`, seeds 13/23 ni full hasta que scratch y level cumplan:

1. `prediction_std_ratio >= 0.01` en validation;
2. Pearson validation finito y claramente no nulo;
3. MiD mejor que el baseline constante bajo el mismo subset;
4. ausencia de regresión de failure rate;
5. comportamiento reproducible al repetir seed 7;
6. ventaja de `level` sobre scratch suficientemente grande para justificar más semillas.

Si level no mejora scratch, la hipótesis JEPA no se promociona aunque ambos modelos dejen de colapsar.

## Object Event v4.29

Implemented and ready for preregistered execution; full attribution/OOF has not run. The wrapper runs a
sealed-state preflight and a train-only local-affine OOF analyzer. No promotion or
performance claim has been made; development validation, eAP official and EvTTC are
not opened unless its complete all-seed OOF champion passes every fixed gate.

Current corrected verification passes the full Pytest suite, compilation, Ruff,
Pyright, 14 targeted v4.29 tests, PowerShell parsing and the real sealed-state preflight. The corrected
fixed balanced 64-sample, six-epoch train-only diagnostic had 64/64 valid fits for
both arms: LHR loss `1.7021 → 0.3224` (Pearson `0.9631`, peak `926.0 MiB`) and
geometry-teacher loss `1.7453 → 0.3659` (Pearson `0.9623`, peak `930.4 MiB`).
These diagnostics are not OOF evidence and cannot satisfy any promotion gate.
