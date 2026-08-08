# Object Event TTC v4.29 — local affine event correspondence

**Status:** implemented and preregistered; full attribution/OOF not yet executed. No improvement claim is made. Development validation, eAP official test and EvTTC remain sealed.

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
