# Results invalidation and claim boundary

Updated: 2026-08-02.

## Valid evidence classes

```text
historical_exact
integration_only
screen_candidate
full_candidate
grouped_cv_valid
multiseed_valid
official_external
invalid
```

## Historical exact

`B0_HISTORICAL_BASE_EXACT` is numerically valid on its historical split because
the checkpoint, cache, metrics and prediction arrays have been reproduced
exactly. It is not a matched ablation against the new object-cache models.

## Integration only

All runs with:

- compact Garl backbone;
- two smoke epochs;
- 38/10 samples;
- cache format earlier than v6;
- output protocol earlier than architecture v4;

remain engineering diagnostics. They cannot promote modules or appear as final
paper results.

The real cache-free high-resolution run recorded in
`artifacts/metrics/e_jepa_tubelet_lhr_trainer_smoke_current_v1.json` is also
`integration_only`: it used 16/16 samples, one epoch and one seed, has
`claim_eligible=false`, and obtained a validation sequence-macro MiD of
`1868.3186`. It validates the data/model/checkpoint contract but provides no
evidence of competitiveness.

The semantic shortcut artifacts are `synthetic_diagnostic`, not
`screen_candidate`. They establish that the current variance/VISReg family can
encode a fixed-per-sequence shortcut while retaining apparently healthy latent
statistics. They do not establish that the real eAP encoder encodes that shortcut.

`r2_rate_dependence` and `residual_r2` are invalid as production promotions: the
R²-lite arm failed the predeclared 15% log-TTC improvement gate, while adding
optimization and estimator complexity. `temporal_residual` is only a conditional
candidate because it passed the slow-shortcut fixture but regressed sharply in the
frame-varying control. A real matched `level` vs `level+temporal_residual` eAP
comparison is mandatory before any model change.

## Current comparison boundary

A learned candidate can become `screen_candidate` only when:

- code and configuration belong to a committed revision;
- Core/Garl output stage is isolated;
- cache format is v6;
- sample selection and initialization hashes are present;
- early stopping and checkpoint policy are recorded;
- Benchmark-10 remains unopened.

`grouped_cv_valid` requires five complete folds. Para el programa Level–Dynamics
vigente, `multiseed_valid` requiere los seeds predeclarados 7, 13 y 23 para BASE y
máximo dos finalistas. La mención histórica 7/13/21 queda sustituida por la orden
explícita actual y no debe reaparecer en manifests nuevos.

`full_candidate` for the Garl high-resolution path requires all valid rows, a
clean committed tree, seeds 7/13/23, comparable hashes and a freeze artifact.
That class permits external evaluation; it is not itself an SOTA claim.

## Oracle boundary

Allowed as diagnostic oracle:

- bbox GT;
- segmentación GT;
- distancia oficial EvTTC para estudiar compensación traslacional.

Not allowed as final model input:

- distance/depth ground truth;
- TTC ground truth;
- future navigation;
- sequence ID;
- Benchmark-10 labels.

`translation_compensated_box_mixture_oracle` must never be compared as a
deployable candidate unless depth is replaced by a predicted causal value and
the experiment is rerun.

## Historical invalidations

- FlowMimic global, inverse-TTC sintético y su combinación no fueron promovidos
  porque empeoraron BASE.
- eAP pseudo-TTC no es oficial y usa contexto futuro.
- caches eAP anteriores a la corrección de normalización sparse no son
  comparables.
- smokes con resúmenes Core/Garl sobrescritos se conservan solo mediante sus
  `summary.json` individuales.
- CARLA checkpoints/caches were removed after negative transfer; the compact
  tracked metrics remain sufficient evidence of the rejected result.

## Official claim

`official_external` exige:

- configuración congelada antes de abrir Benchmark-10;
- commit limpio;
- freeze manifest;
- checkpoint hash;
- formato de submission validado;
- runtime end-to-end del candidato real;
- registro del número de submissions.

No existe actualmente ningún resultado `official_external` ni claim SOTA.
