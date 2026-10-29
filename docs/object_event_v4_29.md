# Object Event TTC v4.29 — local affine event correspondence

**Status:** `completed_oof_gate_failed`. The full attribution and grouped OOF run
completed at commit `f3094f2d16ac39042627448ab706c63f593cf058` in 18,183 s.
Development validation, eAP official test and EvTTC remain sealed.

The falsifiable hypothesis is that local event-feature correspondence estimates a robust, interpretable current-to-previous affine transform. Forward accepts only `[B,3,C,H,W]` events, using v4.8 dense maps and one shared projection. No TTC, height, box, sequence or track data reaches the model.

The convention is `Y_previous = X_current A^T + t`, with feature-pixel centers in `[-1,1]^2`. Adjacent radius is 4, direct radius 7 and cosine temperature `.07`. WLS uses activity/foreground/confidence support, identity/zero ridge `.001`, then exactly one detached Huber-IRLS pass (`delta=.08`). Validity requires finite solve, `det(A)>.05`, condition at most 100 and adequate support. Invalid estimates are surfaced, not replaced. Main scale is `log(||A12 e_y||)`; expansion is `1-exp(log_eta_vertical)`.

There are exactly two fixed arms: `local_affine_lhr`, using projected visible-height LHR, and `local_affine_geom_teacher`, which adds only t1/t2 common-ROI box log-height/log-width and mapped-center residual targets. t0 boxes never enter teacher targets, raw translation is not supervised, and no TTC MLP/readout, sweep or pseudoflow exists.

Before arms, seed attribution uses checkpoint and matcher-init seeds `{7,13,23}` on the same grouped folds. It records hashes, construction and fold-only optimizer RNG schedules, no workers, histories and predictions. Pearson and log-eta Pearson are reported separately with marginal ranges and interaction/crossover diagnostics. Its dominance rule is descriptive and limited to those seeds (`.03` material range; `1.5x` ratio); it cannot choose a seed or alter arms.

The fixed objective is Pearson; ties are minimum sequence Pearson, negative accuracy, then lexical arm name. OOF must be complete, finite, and satisfy every absolute and v4.28-gain gate before one development evaluation. Otherwise status is `completed_oof_gate_failed`. The held-out zero-event, temporal-shuffle (`t0,t1,t2 -> t2,t0,t1`) and endpoint-swap controls are report-only.

Invalid affine estimates remain NaN and make the complete-coverage gate fail. The
analyzer additionally emits explicitly labelled `valid_only` diagnostics and
invalid-reason counts; these are investigative outputs and can never select or
promote an arm.

If (and only if) complete OOF passes, the analyzer reads the supplied v4.10
development files, materializes validation once, trains the fixed champion on all
train samples for 12 epochs per seed, and compares with the locked v4.10 thresholds
in YAML. It emits `completed_development_passed` or
`completed_development_failed`; neither outcome opens an official test.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_object_event_v4_29_local_affine.ps1 -Device cuda
```

Local cosine correspondence may fail under low activity, occlusion, repeated structure and large displacement; an affine image-motion approximation is not physical ground truth.

## Full train grouped-OOF result

Both arms failed the preregistered complete-coverage gate. In each arm, two of
2,048 aggregate rows were invalid (`0.0977%`), both in sequence `6h5yRW2LGc`,
because at least one constituent seed exceeded the locked condition-number limit
100. The affected aggregate conditions were approximately `125–132`; determinants,
robust masses and residuals remained finite. Predictions stayed NaN and no fallback
or threshold relaxation was applied. Consequently no arm was selectable and the
analyzer did not materialize development validation.

Valid-only diagnostics are scientifically informative but cannot satisfy gates:

| Metric | `local_affine_lhr` | `local_affine_geom_teacher` |
|---|---:|---:|
| Pearson | 0.762817 | 0.767142 |
| log-eta Pearson | 0.743956 | 0.748227 |
| expansion MAE | 0.017627 | 0.017439 |
| positive accuracy | 0.955992 | 0.950575 |
| negative accuracy | 0.827768 | 0.829525 |
| balanced sign | 0.891880 | 0.890050 |
| minimum-sequence Pearson | 0.472400 | 0.489531 |
| minimum-sequence negative accuracy | 0.685714 | 0.628571 |
| prediction std ratio | 1.186014 | 1.170889 |
| calibration slope | 0.904712 | 0.898238 |
| negative-track macro accuracy | 0.825581 | 0.837209 |
| minimum negative-track accuracy | 0.25 | 0.25 |

Applying the frozen gate function to only the 2,046 valid rows would pass every
performance/calibration/gain check for both arms, but that counterfactual is
explicitly non-selectable. The teacher improves Pearson by only `0.0043`,
minimum-sequence Pearson by `0.0171`, and negative-track macro accuracy by `0.0116`;
it does not remove the two aggregate invalid rows.

Magnitude diagnostics show real progress over v4.28 without complete calibration:
the teacher has ratios `1.466`, `1.193`, `1.032`, and `0.736` in the four fixed
buckets. Large motion is no longer compressed to roughly one quarter of GT, but
small motion is overpredicted and `|g| >= .08` remains underpredicted. Its large-
magnitude Pearson is only `0.232`. A globally reasonable std ratio therefore does
not establish full magnitude calibration.

The report-only controls behave directionally correctly: zero-event inputs are
100% invalid; temporal shuffle retains mean fold Pearson about `0.52`, while
endpoint swap falls to about `0.22`. Thus temporal order matters, but substantial
appearance/sequence signal survives shuffling and remains a generalization risk.

Seed attribution finds material effects from both factors. Backbone marginal
Pearson range is `0.0714`, matcher-initialization range is `0.0416`, and the same
ordering holds for log-eta. The fixed marginal rule labels the backbone dominant,
but strong crossover interactions (`max |interaction| ≈ 0.059`) force the final
conclusion `mixed_inconclusive`. Seed 13 is not selected.

Artifact hashes:

- config: `e850b702eb499adc777f544b3fbc7f5c4b0f4087d993ed9bd7acef0d409c4719`;
- summary: `6f9f59ab1dba0471c1be608d8acd270f6642dcbe4e10c3ed3cc0960eb96c86d8`.

## Train-only implementation diagnostics

These checks are not OOF results and were not used to change the preregistered
configuration:

- Re-run after the robust-mass correction used the fixed first 32 negative and first
  32 non-negative train samples, checkpoint seed 7, optimizer seed 42900 and six
  epochs. `local_affine_lhr` loss fell `1.7021 → 0.3224`; all 64 fits were valid,
  with Pearson `0.9631`, log-eta Pearson `0.9207`, MAE `0.0102`, std ratio `1.314`,
  calibration slope `1.266`, condition `8.72–30.73`, and `926.0 MiB` peak VRAM.
- Under the same fixed diagnostic, `local_affine_geom_teacher` loss fell
  `1.7453 → 0.3659`; all 64 fits were valid, with Pearson `0.9623`, log-eta Pearson
  `0.9203`, MAE `0.0095`, std ratio `1.275`, calibration slope `1.227`, condition
  `8.72–30.74`, and `930.4 MiB` peak VRAM.
- Neither subset contained a `|g| >= .08` sample. These are train-only
  implementation diagnostics, not OOF evidence, and did not alter any locked
  architecture, threshold, loss, or gate. The observed amplitude inflation is
  diagnostic only and says nothing about grouped-sequence generalization.
