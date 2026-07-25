# FlowMimic physical-approach experiment

Updated: 2026-07-25.

## Research question

Does a physically constrained synthetic approach prior improve the causal
event-tubelet JEPA encoder for TTC estimation, especially without exposing real
TTC labels during pretraining?

This is an adaptation inspired by FlowMimic, not a reproduction of its video
editing results. FlowMimic does not evaluate event cameras or TTC.

## Metric clarification

The historical local result `0.312 +/- 0.044 s` is TTC mean absolute error
(MAE): the prediction differs from the reference TTC by 312 ms on average. The
historical scratch result `0.465 +/- 0.021 s` is the same type of error under the
local paired protocol, a 32.9% lower MAE for JEPA.

The `13 ms` reported for Garl-TTC is inference latency, not TTC error. Dividing
312 ms of error by 13 ms of runtime has no scientific meaning. The corresponding
Garl-TTC accuracy number cited in the audit is `10.60%` relative TTC error on its
official protocol; the historical local JEPA number is `6.42 +/- 0.45%` MARE on
CPLA-high. Those accuracy values are also not directly comparable because the
sequence set, ROI/bbox assistance and evaluation protocol differ.

Two separate comparisons are required:

1. accuracy: same sequences, samples, labels and MAE/RTE definition;
2. efficiency: batch size 1 on the same hardware, with preprocessing and model
   latency reported separately.

## Implemented hypothesis

For a constant-speed frontal approach with TTC `T` at the context reference
time, apparent pinhole scale at relative time `delta` is:

```text
s(delta) = T / (T - delta)
```

The simulator follows this causal order:

```text
analytic object motion
  -> log-intensity frames
  -> accumulated contrast-threshold crossings
  -> positive/negative voxel bins
  -> optional cache-compatible normalization/metadata
```

It never warps an already accumulated voxel grid. Sub-threshold contrast is
carried relative to the last emitted-event reference, matching the essential
event-camera mechanism more closely than independent frame differencing.

The pretraining additions are independently weighted:

- `flowmimic_alignment_weight`: JEPA context-to-future latent alignment on
  synthetic event windows;
- `flowmimic_inverse_ttc_weight`: a positive inverse-TTC auxiliary head trained
  only from the analytic synthetic trajectory.

The auxiliary head is present only during pretraining. The downstream TTC head
is still fitted through the existing supervised protocol.

## Scientific-integrity boundary

- No value from cache field `y_ttc` is read by the FlowMimic path.
- Real future events remain the main EMA-target JEPA signal.
- Synthetic TTC is recorded explicitly as analytic supervision, not described
  as unlabeled real-data supervision.
- Future navigation is zeroed for target embeddings as in the existing JEPA.
- Synthetic generation runs only in the training loop; model selection remains
  based on the unchanged real validation loss.
- CPLA-high is a reused diagnostic test and must not be opened for this
  architecture selection.

Relevant implementation:

- `src/e_jepa_ttc/representations/flowmimic.py`
- `src/e_jepa_ttc/training/jepa.py`
- `src/e_jepa_ttc/cli.py`
- `tests/unit/test_flowmimic.py`
- `tests/unit/test_jepa_training.py`

## Decisive validation-only matrix

Use the same cache v2, architecture, seed, batches, horizons and optimizer for
all rows:

| ID | Synthetic latent alignment | Synthetic inverse-TTC | Purpose |
| --- | ---: | ---: | --- |
| E0 | 0 | 0 | clean event-tubelet JEPA control |
| E1 | >0 | 0 | isolate synthetic future alignment |
| E2 | same as E1 | >0 | test the physics/TTC prior |

Execution is staged to avoid spending three full seeds on a bad idea:

1. one-seed short pretrain plus identical validation-only fine-tune for E0-E2;
2. promote only a variant that improves validation MAE and does not collapse;
3. rerun the promoted pair with seeds 7, 13 and 21 at the full schedule;
4. only after freezing the choice, obtain a genuinely unopened holdout or the
   complete official protocol.

Primary promotion metric: validation MAE in seconds. Secondary diagnostics:
relative error, RMSE, inverse-TTC auxiliary loss, embedding effective rank and
latency. SSL loss alone cannot establish TTC improvement.

## Current status

- Scientific-provenance hardening published in commit `416b498`.
- Simulator unit tests: passing.
- FlowMimic/JEPA integration smoke: passing.
- Existing JEPA/prober focused suite: 21 tests passing.
- Full repository QA: Ruff passing and 196 tests passing.
- Cache v2 rebuild: pending.
- E0/E1/E2 validation results: pending; no result should be filled in manually.

## Continuation checklist

1. Commit and push the tested FlowMimic implementation.
2. Build a train+validation-only cache format v2 from source manifests.
3. Audit physical array shape, sidecar counts and hashes.
4. Run E0/E1/E2 without evaluating CPLA-high.
5. Append exact commands, commit hashes, artifact hashes and generated metrics
   paths below.

## Experiment ledger

No training run has been completed yet. This section is intentionally empty
until metrics are generated by code.
