# Scientific Recovery V5 — canonical status

Fecha de corte: 2026-08-13. Rama:
`scientific-recovery-v5-provenance-dual-transport`. El aggregate se regeneró con
worktree tracked limpio en `c55e791c563e6f463385685e8dd3b4aa62d485a7`.

V5 está cerrado como resultado de desarrollo negativo para promoción. A8.0 mejora
A6 en el agregado de los mismos folds, pero no alcanza el gate preregistrado
`MiD <= 175`. No se ejecutan A8.1–A8.5. No hay candidato sealed, claim SOTA ni
autorización para abrir private/test.

## Contratos y correcciones P0

- El paired bootstrap namespacifica `track_id`, exige identidad de targets y tokens,
  y liga predicciones y metadata por SHA-256.
- Claim readiness rechaza artifacts stale, de tipo/status/scope incorrecto o cuyos
  hashes no correspondan al candidato y Garl actuales.
- El master runner pone en cuarentena outputs anteriores y no propaga un paired si
  el subproceso falla.
- La replicación mecanística V4 usa un parent A4 seed 7 fijo y separa stochasticity
  de transport de stochasticity de arquitectura.
- A8 grouped-dev corrige una fuga posterior: cada fold entrena su propio A4 parent
  con seis secuencias; A6 y A8 parten de ese mismo parent y Garl parte de cero.

El paired público P0 fresco A6 seed 7–Garl conserva el scope histórico de public
validation: Δ MiD `+60,952812`, IC95% `[45,446970, 79,909598]`, probabilidad de
menor MiD E-JEPA `0`, y Δ failure `+7,275391` puntos, IC95%
`[5,901790, 8,724490]`. A5–Garl es solo diagnóstico: Δ MiD `+18,323599`, IC95%
`[4,528645, 38,064085]`, probabilidad `0,0058`, Δ failure `+4,931641`.

Claim readiness queda `claims_blocked=true`, `NO_PROMOTABLE_CANDIDATE` y
`private_test_opened=false`.

## Grouped development congelado

Protocolo:
`configs/protocol/scientific_recovery_v5_train_only_grouped_dev.json`.
SHA del archivo `be48917ae52d1c77d046318bd9ed284a32e8b16258257203fff439332b547874`;
SHA del artifact `f09c688fb4991714abc9d645dda787cb27f1e02a2d1857312ce3e45519bd7a63`.

| Fold | Train rows | Dev rows | Dev sequences |
|---:|---:|---:|---|
| 0 | 5461 | 2731 | `5ilM1PX2vz`, `OYgB6RGWcq`, `qGsgzl4Q8B` |
| 1 | 5461 | 2731 | `2cyv0Oedzg`, `6h5yRW2LGc`, `mHGFBekt7X` |
| 2 | 5462 | 2730 | `OBneIVg4Cw`, `WbCh1DRerJ`, `t79dBxj1WS` |

Cada secuencia aparece una sola vez en outer-dev. Train/dev son disjuntos por
secuencia, sample token y track. Los teacher tokens usados son exactamente los de
fold-train y su intersección con fold-dev es vacía. Outer-dev sí selecciona el
checkpoint: es desarrollo, no test. `public_validation_used_for_selection=false`.

## Cadena autocontenida por fold

```text
A4-Fk (6 train) ─┬─> A6-Fk ─┐
                 └─> A8-Fk ─┼─> 3 outer-dev
Garl-Fk (scratch) ───────────┘
```

Los antiguos `scientific_recovery_v5_a6_grouped_fold*` inicializados desde el A4
global quedan `diagnostic_parent_exposed` y no son elegibles. Se conserva el MiD
observado F0 `119,500657`; F1 fue interrumpido al detectar la exposición. Sus
summaries no se reescribieron.

Los parents limpios A4 seed 7 obtuvieron MiD F0/F1/F2
`246,627289 / 323,231478 / 303,404635`. Los SHA de checkpoint son, respectivamente,
`bb9d087c…8cc9c`, `181d2db3…9f39` y `da1a498b…b879`.

## Resultados A8.0

Todos los modelos usan los 8192 tokens outer-dev exactos una sola vez, targets y
track IDs idénticos. Garl es exact-sample, target, budget, metric and oracle-ROI
matched; su preprocessing no es idéntico: E-JEPA usa tres endpoints/12 canales y
Garl dos endpoints/40 canales.

| Model | Params | F0 MiD | F1 MiD | F2 MiD | Macro 9 seq | Failure | Pearson | Coverage | Geometry | Causal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| A4 | 355118 | 246,6273 | 323,2315 | 303,4046 | 291,0878 | 9,4482% | 0,0608 | 90,5518% | source parent | model-prefix PASS |
| A6 | 498130 | 180,6823 | 218,5176 | 235,3257 | 211,5085 | 7,8491% | 0,2011 | 92,1509% | exact parent | model-prefix PASS |
| A8.0 | 627827 | 180,7953 | 201,5479 | 210,7309 | 197,6914 | 7,0190% | 0,2246 | 92,9809% | exact parent | model-prefix PASS |
| Garl | 24674178 | 130,3498 | 163,2190 | 139,4902 | 144,3530 | 0% | 0,0421 | 100% | N/A | different temporal representation |

A8.0 conserva `causal_left`, radius 1, τ `0,02`, residual bound `0,05`, event-only
en el neural forward y geometry frozen. Las seis auditorías prueban igualdad exacta
de tensors/outputs, `requires_grad=false` y ausencia del optimizador. En los tres
folds el transport encoder comienza igual al encoder A4 y cambia tras entrenar.

Paired outer-dev, bootstrap 5000 por `sequence_id+track_id`:

| Comparison (first−second) | Δ MiD | IC95% | P(first lower) | Δ failure | IC95% failure |
|---|---:|---:|---:|---:|---:|
| A8.0−A6 | −13,8171 | [−17,7245, −10,0655] | 1,0000 | −0,8301 pp | [−1,6215, −0,0370] |
| A8.0−Garl | +53,3384 | [+44,3126, +61,7827] | 0,0000 | +7,0190 pp | [+6,4143, +7,6411] |
| A6−Garl | +67,1555 | [+58,1653, +75,8915] | 0,0000 | +7,8491 pp | [+7,2119, +8,5227] |

Gate A8.0: mejora A6 `PASS`; MiD≤175 `FAIL`; objetivo fuerte≤160 `FAIL`;
geometry exacta `PASS`; model-prefix causality `PASS`; coverage `PASS`. Decisión:
`FAIL`, sin promoción a A8.1.

Artifact aggregate:
`artifacts/scientific_recovery_v5/results/aggregate.json`, SHA firmado
`b3f0fc484b16f5d503d20deb35b275ddaf7392b2c9484ebc4874b29c8bbb4fc5`.

## Límites y siguiente paso

- Las 2048 filas de public validation no participaron en selección A8; su uso
  previo sigue siendo adaptativo e histórico.
- Las 8192 filas train proceden de un universo histórico estratificado por bucket
  TTC. El fold assignment V5 no usa targets, pero el universo no es una muestra
  target-free.
- Oracle bbox/ROI es privilegio de preprocessing compartido; bbox y RGB no son
  inputs neurales. DINO/bbox solo supervisan train.
- Se demuestra causalidad del modelo, no causalidad streaming extremo a extremo.
- Solo hay parent seed 7 y transport seed 7; los tres folds no son replicación
  multiseed.
- No se ejecutaron A8.1–A8.5, robustness A8 ni public-validation final porque A8.0
  falló su gate.
- `private_test_opened=false`; no existe sealed evaluation.

Siguiente paso único: diagnosticar, exclusivamente en train-only grouped-dev y sin
cambiar el gate, por qué A8.0 no mejora F0 y mantiene un gap de 53,34 MiD con Garl;
no promover una arquitectura nueva hasta preregistrar una hipótesis que explique
esa heterogeneidad.

## Barrido documental

Se enumeraron los 72 archivos `*.md` y los 60 archivos bajo `docs/`. Se revisaron
los matches globales de V3/V4, A4–A8, causalidad, paired/bootstrap, claim readiness,
`track_id`, Garl, public validation, private/test, dual transport y grouped-dev.

Se actualizaron los documentos vigentes: `README.md`, `STATUS.md`,
`PLAN.md`, `CODEX_HANDOFF.md`, `docs/progress.md`, `docs/experimental_protocol.md`,
`docs/reproducibility.md`, `docs/limitations.md`, `docs/model_card.md`,
`docs/technical_report.md`, `docs/dataset_card.md`, `docs/methodology.md`,
`docs/causal_scale_eap_screen.md`, los tres planes/specs A8/V5 bajo
`docs/superpowers/` y `docs/E_JEPA_TTC_V5_SCIENTIFIC_CODE_AUDIT.md`.

Se dejaron intactos deliberadamente los handoffs A4/A4D/A4-S1 y los documentos
`object_event_v4_*`, `causal_scale_v5..v8`, planes/specificaciones preregistradas,
el handoff V3 y el plan V4/V5: son provenance histórica. Los handoffs V3/V4 y el
postmortem A6 locales permanecen además untracked y no se incorporan sin alterar su
origen. No quedó documentación vigente conocida con un claim V5 contradictorio;
las cifras históricas siguen etiquetadas como históricas.
