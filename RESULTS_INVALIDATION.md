# Results invalidation and claim boundary

Updated: 2026-07-30.

## Valid evidence classes

```text
historical_exact
integration_only
screen_candidate
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

## Current comparison boundary

A learned candidate can become `screen_candidate` only when:

- code and configuration belong to a committed revision;
- Core/Garl output stage is isolated;
- cache format is v6;
- sample selection and initialization hashes are present;
- early stopping and checkpoint policy are recorded;
- Benchmark-10 remains unopened.

`grouped_cv_valid` requires five complete folds. `multiseed_valid` requires the
predeclared seeds 7, 13 and 21 for BASE and máximo dos finalistas.

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
