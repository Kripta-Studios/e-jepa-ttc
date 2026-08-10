# Estado del repositorio

## 2026-08-10 — screen causal-scale A0: negativo

- A0 seed 7 terminó: época seleccionada 11/16, MiD macro `382.1905104`, failure
  `12.3046875%`, weak-box IoU `0.4997858107`, Pearson log-ratio `0.0456290990`.
- Referencia release sobre los mismos 2.048 tokens: MiD macro `117.4281582`, cero
  failures. Es una referencia desigual y expuesta, no un baseline matched.
- Las tres secuencias validation estaban en train oficial; 4.735 de las 88.744
  filas públicas pertenecen a ellas.
- Comparador firmado: `9f2bebde05729b7ace6fdbc0a990e6b75bf180ec87220924219ed7095105281c`.
- No hay promoción ni claim SOTA. Siguiente hipótesis única: A1 geometry-only.
- Subset matched materializado y firmado: 2.048 train/9 secuencias y 2.048
  validation/3 secuencias, identidad
  `dd08ecc983f30e38a939204f9a2df09e4966bbe73bd764c972f7726e5d4e34d3`.

## Addendum eAP causal-scale screen v1 (2026-08-10)

V8 CVaR cerró como fallo sintético honesto (`.94621 < .95`) sin abrir test
901/902/903. Por autorización explícita del usuario se implementó un screen separado
eAP/Garl exclusivamente train/validation. Usa un cache firmado de 2.048/2.048 filas,
9/3 secuencias disjuntas y los cuatro buckets TTC. El modelo recibe solo eventos
`[3,12,128,128]` y delta temporal; las cajas t1/t2 son supervisión débil declarada,
nunca inputs, y t0 proxy queda inválido.

El trainer real está en `src/e_jepa_ttc/training/causal_scale_eap.py`, el runner en
`scripts/train_causal_scale_eap_screen.py` y el protocolo congelado en
`configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_v1.yaml`. Implementa
CVaR, BF16, early stopping, límite 6 h y `best/last` resumibles por época. Un benchmark
128+128 tardó 5,289 s y 395,6 MiB de VRAM; no es resultado eAP. Resume ya pasa un
test end-to-end exacto, incluido optimizer/scheduler/RNG/historial/best. También está
implementado el builder firmado del subset Garl por tokens. Falta el entrenamiento
completo y ejecutar la comparación oficial.

La comparación primaria será contra
`E:\Garl-TTC\checkpoints\paper_event_only_lhr.pth` sobre exactamente los mismos
2.048 tokens. Aún no existe resultado comparable, freeze, test oficial ni claim SOTA.
El plan operativo completo está en `CODEX_HANDOFF.md` y
`docs/causal_scale_eap_screen.md`.

## Addendum v7 (2026-08-10)

V7 combines the v6 equivariant foreground with parameter-free causal transport of the
previous inverse-TTC estimate. The selected train/validation run passes every frozen
validation gate: Pearson `.9612550`, slope `.9274405`, sign `.9911504`, IoU `.8926791`,
ratio MAE `.0169565`, TTC symmetric relative error `.2434549`, 80% coverage `.7994100`
and translation leakage `.0035127`. Test seed 603 and all real data remain closed.

The signed diagnostic artifact is
`artifacts/metrics/causal_scale_v7_diagnostic_comparison_v1.json`, identity
`eb3497fafad8a4d23284b263303628be8ad025fd61bac57ad5f54580d142ee82`.

The clean one-shot test at published commit `0bc781f` subsequently failed only
correlation: Pearson `.9201432 < .95`. Translation `.0033841`, IoU `.8896096`, TTC
`.2457614`, slope `.9278544`, sign `.9911308` and calibration `.7827051` passed.
Artifact identity is `97e52b2a9d3463d6a2e57d12e9408f80bb6a3b8e0d491beeb3546c2d1586a52b`.
V7 is closed, seed 603 consumed, and no SOTA/Garl-TTC claim is authorized.

V8 was preregistered before its validation-only diagnostic. It keeps equal total
sample budget while selecting across train seeds 701/702/703 and validation seeds
801/802/803. Tests 901/902/903 remain sealed. Selection combines macro and worst
group, and a future full pass requires every group. No real data is authorized.

The first V8 diagnostic at `46f9d61` selected epoch 16 but failed correlation:
macro Pearson `.80631`, with groups 801/802/803 at `.64269/.89168/.88458`. Macro IoU
`.89016`, TTC `.28206` and translation `.00469` pass. A 5%-trim diagnostic reaches
`.96612` on group 801, isolating rare endpoint-mask catastrophes. Parameter-free,
reversal-equivariant temporal logit consensus arms `.10/.15` are next; test stays closed.

Those arms improve macro Pearson to `.93388/.93804`; `.15` has balanced group
Pearson `.94338/.93594/.93479` but TTC `.30191` narrowly fails. A frozen 10%-tail
CVaR arm is the final synthetic ablation. The user explicitly authorized a separate
eAP train/validation exploratory screen; official test/CodaBench/EvTTC remain closed.

CVaR is the selected V8 arm: macro Pearson `.94621`, score `.38670`, TTC `.29547`;
group Pearson `.94812/.94567/.94486`. It remains a gate failure and tests 901–903
were never opened. Signed comparison identity: `71bb1d8299141180ff964154e3440b971014e50953e174b9fb489ba9bbe1ef79`.

## Addendum v6 (2026-08-10)

V6 preregisters new synthetic groups 401/502/603 and does not reuse consumed v5 test
303. Four train/validation diagnostics are signed in
`artifacts/metrics/causal_scale_v6_diagnostic_comparison_v1.json` (identity
`e00a64a90aee5c302ad486763ed147a2af590a7d3575191395e0f0d374d6191f`). The selected
stride-free separable row/column foreground head reaches validation Pearson `.92042`,
slope `.94878`, sign `.99115`, IoU `.89323`, TTC symmetric relative error `.26628`,
coverage `.79941` and translation leakage `.00462`. It solves the v5 translation
failure but still misses Pearson `.95`; test 603 was not opened and v6 is not promoted.

Pair-ratio mask supervision, learned height correction and frozen residual refinement
did not improve correlation. The next event-only version must use multiple synthetic
train/validation groups and macro/worst-group temporal-scale selection. All real data,
RGB, EvTTC and CodaBench remain closed.

## Addendum v5 (2026-08-09)

Implemented and published: shared foreground-scale core, event-only v5 config,
geometry-bound uncertainty/risk loss, synthetic gate runner and ADR. The clean run at
commit `7945e99` passed its ideal-foreground algebra gates with Pearson `1.0`, slope
`.9999995232`, sign `1.0`, oddness `0/0`, translation leakage `0`, controlled square
rotation leakage `.00171029` and zero-event unknown `1.0`. The compact artifact is
`artifacts/metrics/causal_scale_v5_synthetic_operator_gate_v1.json`; serialized SHA256
is `3fd4d2a25b85173cf34bb8738f5b7e80190f31f26acc9ed9a4d3c818d10afb20`.

The learned event-foreground runner is now implemented and nine train/validation-only
diagnostics are preserved in the signed comparison artifact
`artifacts/metrics/causal_scale_v5_diagnostic_comparison_v1.json` (identity
`27053c853b93b1ff14ec32f4db79e4e216c6a5c3929f385998b549c1dee2fe80`). The selected
deconvolutional candidate reached validation Pearson `.9559997`, slope `.9686489`,
sign `.9956896`, IoU `.8639824`, ratio MAE `.0190324`, TTC symmetric relative error
`.2638807` and calibrated 80% coverage `.7974138`. Translation leakage p95 was
`.0239874`, so it missed the frozen `.02` gate and is not promoted. Test seed 303,
all real TTC data, RGB, eAP, EvTTC and CodaBench remained closed.

The one-shot clean-tree test subsequently ran at published commit `d9d20af` and
failed. Test Pearson was `.9213532` and translation leakage p95 `.0274930`; the
frozen gates require `.95` and `.02`. Slope `.9691788`, sign `.9941860`, IoU
`.8724040`, TTC symmetric relative error `.2592012`, 80% coverage `.7761628`, empty
unknown `1.0` and empty false positives `0` passed. Artifact identity is
`ce42fe957c4944a72bf38b5b134df7dfd0809ccc1c87b6cff6a749662093ea29`; serialized
SHA256 is `ac26c1b2ad87c6c991decd46b8fd2e478cd87dcd926f49edd492bcdf3214c7e8`.

V5 is therefore `completed_gate_failed`; seed 303 is consumed and must not be used
for tuning or rerun as new evidence. eAP, EvTTC, RGB and CodaBench remain closed. The
next version must preregister new synthetic group splits and improve cross-seed
equivariance before any train-only real-data screen. No benchmark improvement exists.

## Addendum v4.31 (2026-08-09)

The v4.30 full result is authoritative and negative: SHA256
`9722202A4D33F6B5D1B933EEDA1F9143E13E4E2FD64B21356E93783AFAA1C689`, status
`completed_oof_gate_failed`. Stabilization passed `.0010116798/.0423071422/.1308624286`.
`stable_multiscale_similarity` is only the rank winner; champion is null. Best-arm
Pearson `.4791568608`, negative accuracy `0`, balanced `.5`, std ratio `.3731916487`,
slope `.1788173388`, high-bucket Pearson `-.1972577670`, magnitude ratios
`.92439/.58893/.48926/.30467`; both arms failed and no sealed data opened. The
target-free saved-NPZ post-hoc audit (not preregistered) found forward-vs-swap
`log_eta` correlation `+.53338`, zero sign flips, and 95.8% coverage at
`|log_eta| >= .005`. The
next action after Sol's rethink is a TTC-label-free but train-box-conditioned
common-object-ROI v4.31 redesign, not supervised retraining. Selection is
independent of TTC/sign/buckets; train-only stabilization and audit pools are
immutable and sequence/time-disjoint; retained artifacts are sanitized
event/ROI-only; exact physical reversal controls remain mandatory. Development,
test and EvTTC remain closed. The direct full-frame v4.31 draft was rejected
before execution and is not evidence.

### v4.31 handoff status

The 512-row real-data train-only diagnostic completed after a passing sanitized-cache
preflight.  It opened no TTC, development, test, EvTTC or RGB inputs.  It is
non-selectable and dirty-tree/non-authoritative, and stage 2 is absent.  Stability
passed, but physical behavior failed decisively: analytic Pearson `.29172`, slope
`.00852`, sign `.59082`, oddness `1/1`, translation leakage `.28859`, and swap
coverage `.00391`.  Status is `not_issued_diagnostic`; full remains closed and the
frozen v4.30 operator must not be promoted.  Windows blockers fixed during execution
were SHA case canonicalization, explicit memmap closure before atomic rename,
jitter-safe non-overlapping t0, true one-record-per-line JSONL, UTF-8 child output,
and orphan-child termination.  Exact hashes and limitations are in
`docs/object_event_v4_31.md`.

## Superseded historical v4.30 preregistration (2026-08-08)

La implementación v4.30 queda preregistrada y ejecuta como un único comando el
OOF agrupado y, solo tras una promoción genuina, full-train seguido de una única
development validation. The no-authoritative wording in this historical section
is superseded by the completed v4.30 negative result above. Diagnóstico, fallo de estabilización u OOF fallido no pueden leer los
inputs development; el estado es no seleccionable. eAP oficial y EvTTC no se
abren automáticamente, incluso tras development passed.

El diagnóstico train-only v4.30 de esta historia era no seleccionable; no es el
resultado actual. Diagnóstico,
fallo de estabilización u OOF fallido no pueden leer inputs development; eAP
oficial y EvTTC no se abren automáticamente.

Historial operativo y evidencia no seleccionable:

- Antes del hotfix, `-DiagnosticSamples 12` sufrió un bug nullable de PowerShell
  que omitió `--diagnostic-samples`; empezó el OOF completo de 2.048 filas y se
  terminó de forma segura tras unos 50 minutos, antes de artefactos stage-1,
  resumen, validation o métricas. Usó alrededor de 7 GiB de GPU y no abrió datos
  sellados. El hotfix usa `PSBoundParameters` y muestra modo/output resueltos.
- El diagnóstico real de 12 filas terminó en 24,7 s: JS median `.013007` y JS
  p95 `.100864` (fallo `.08`) quedan como historia acotada. Su displacement p95
  `.306892` no es un pass válido: offsets coarse estaban infraescalados.
- El 96 filas previo (51,7 s; SHA256
  `1A607311C140D7E8A063F139C1FFDCCF826A19D99CBD2BDFF3E6B74815F73C10`) sigue
  inválido/superseded para displacement por offsets infraescalados.
- El diagnóstico post-fix histórico de 96 filas tuvo status `diagnostic_only` y
  resumen `artifacts/debug/object_event_v4_30_diagnostic/summary.json`, SHA256
  `CF9EC7D67EB421AA86304ABD4AB4582F6865CCEABD8D29F5CD7EC4EADBA06BD3`. JS median
  `.010237284936010838` pasó; JS p95 `.19495552778244019` falló; displacement p95
  BASE `.5500071191315064` falló. Las 9/9 historias KL descendieron. Caché: 96
  filas, 36 teacher batches, build count 1 y `4.1370828000363` s. Rank/champion
  null y todos los flags sellados false; se detuvo antes de brazos.
- El SHA `D9DE07…` es el diagnóstico pre-fix superseded, no evidencia actual.
  Los cuatro blockers corregidos son schema real t1/t2, endpoint firmado,
  centro multiescala único ponderado por soporte en píxeles base y `effective_seed`
  veraz. At that historical point the directory was empty; this is superseded by
  the completed authoritative summary SHA recorded above.

Los diagnósticos no pueden debilitar ni satisfacer gates. El siguiente texto es
plan histórico ya resuelto: la estabilización/OOF autoritativa v4.30 se completó,
falló sus gates congelados y no autoriza más ejecución v4.30. La acción vigente
es el rediseño v4.31 box-conditioned y TTC-label-free descrito al inicio.

Verificación actual: targeted v4.30 `30/30`; Pytest completo 100% pass, 7 skipped
y warnings heredados UTF-8/PyTorch; Ruff focalizado limpio; Pyright 0. Ruff global
no está limpio: tiene 872 hallazgos heredados.

Decisión histórica ya resuelta: v4.30 failed its frozen OOF gates. The replacement
is the box-conditioned TTC-label-free v4.31 redesign stated above, not a direct
full-frame audit or relaxed gates. SPAE
solo justifica un bottleneck compacto/canal-estructurado si esa auditoría localiza
la cola; INTACT-JEPA queda para una ablation posterior con gramática física común,
likelihood y gradientes asimétricos, no como parche del matcher actual.

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

Full attribution and grouped OOF completed with status
`completed_oof_gate_failed`. No development-validation, eAP official test or EvTTC
data was materialized. Both arms had two aggregate invalid rows out of 2,048 due
to condition number above the locked limit 100, so complete coverage failed and no
arm was promoted.

Current corrected verification passes the full Pytest suite, compilation, Ruff,
Pyright, 14 targeted v4.29 tests, PowerShell parsing and the real sealed-state preflight. The corrected
fixed balanced 64-sample, six-epoch train-only diagnostic had 64/64 valid fits for
both arms: LHR loss `1.7021 → 0.3224` (Pearson `0.9631`, peak `926.0 MiB`) and
geometry-teacher loss `1.7453 → 0.3659` (Pearson `0.9623`, peak `930.4 MiB`).
These diagnostics are not OOF evidence and cannot satisfy any promotion gate.

Valid-only OOF diagnostics nevertheless establish a material architectural signal:
`local_affine_lhr` reached Pearson `0.7628`, negative accuracy `0.8278`, balanced
sign `0.8919`, minimum-sequence Pearson `0.4724`, std ratio `1.1860`; the geometry
teacher reached `0.7671`, `0.8295`, `0.8901`, `0.4895`, and `1.1709`. Both would
pass every frozen performance gate on valid rows, but that analysis is explicitly
non-selectable. The teacher's high-magnitude ratio is `0.7363`, while the smallest
bucket is overpredicted at `1.4658`; calibration remains magnitude-dependent.

Seed attribution is `mixed_inconclusive`: backbone marginal Pearson range `0.0714`
exceeds matcher-init range `0.0416`, but crossover interaction reaches about
`0.059`. The result supports stabilizing the geometry representation and the local
solver rather than selecting seed 13. Summary SHA-256:
`6f9f59ab1dba0471c1be608d8acd270f6642dcbe4e10c3ed3cc0960eb96c86d8`.
