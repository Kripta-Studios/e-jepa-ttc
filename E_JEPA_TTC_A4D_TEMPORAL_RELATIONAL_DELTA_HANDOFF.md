# A4D — DINOv3 local relational temporal-delta probe

## Status

This experiment is **adaptive after A4**. Its design and gates are frozen here
**before A4D validation is opened**. It must never be described as part of the
original A4 preregistration.

## Why this is the next smallest experiment

A4 matched the RGB DINOv3 teacher independently at t1 and t2:

`L_A4 = mean |R_s(t) - R_t(t)|`.

The selected A4 run improved the ratio mechanism and height dynamics, but width
dynamics remained weak and inconsistent across validation sequences. Endpoint
matching does not explicitly require the student to reproduce the *change* in
teacher relations between the two endpoints.

A4D therefore changes exactly one scientific term while preserving A4's model,
teacher cache, endpoint loss, geometry loss, data split and inference path:

`L_delta = mean |(R_s(t2)-R_s(t1)) - (R_t(t2)-R_t(t1))|`.

Validity is the intersection of teacher/student relation validity at both
endpoints. The operation is float32 L1. No bbox mask is applied.

## Frozen invariants

- Same model config and exactly 355,118 trainable parameters as A4.
- Same event inputs and same public train/validation split.
- Same signed A4 DINOv3 relation cache; no teacher rematerialization.
- Same six offsets and 32x32 relation grid.
- Same A4 endpoint relational weight: 4.0.
- No SAM, JEPA, pair-ratio, new backbone, TTC-input teacher, bbox model input,
  clip/unknown change, or official test access.
- Validation receives no DINO fields.

## Train-only calibration before validation

The temporal coefficient is not chosen from A4D validation. Run
`scripts/calibrate_a4d_dinov3_temporal_delta_weight.py` against the frozen A4
config on exactly 64 equispaced train rows, seed 7, zero optimizer steps.

The target contribution at random initialization is 25% of the already weighted
A4 endpoint-relational contribution:

`lambda_delta_raw = 0.25 * 4.0 * median(L_endpoint_raw) / median(L_delta_raw)`.

Clip to `[0.25, 4.0]`. The calibration artifact must be signed, hashed and bound
into the A4D YAML with `scripts/freeze_a4d_temporal_delta_config.py` before training.

The same calibration also reports teacher-only `|R_t(t2)-R_t(t1)|` statistics
per offset. Degenerate exactly-zero median teacher change is a hard failure.
No diagnostic threshold beyond non-degeneracy is used to tune the coefficient.

## Frozen A4 parent values

Selected A4 epoch 17:

- sequence-macro MiD: 322.6813364242674
- failure: 11.083984375%
- log-ratio Pearson: 0.26009801030158997
- delta-log-height vs physical Pearson: 0.2476144314489911
- delta-log-width vs physical macro-by-sequence Pearson: 0.013604921890200397

## A4D mechanism gate

Registered now, after A4 and before A4D validation. All conditions are required:

1. log-ratio Pearson >= `0.26009801030158997 + 0.03`;
2. delta-log-height vs physical Pearson >= `0.2476144314489911 + 0.03`;
3. macro-by-sequence delta-log-height vs physical Pearson >=
   `0.2526319214906502 + 0.03`;
4. macro-by-sequence absolute-log-height Pearson may drop by at most `0.02`
   from the A4 value `0.5469041860501996`.

Width dynamics remain a reported anisotropy diagnostic, but are not an A4D gate:
the current causal TTC readout is analytically height-ratio based.

MiD/failure are reported but secondary to this mechanism gate for A4D.
Thresholds must not be changed after A4D validation has been inspected.

## Decision after one seed-7 A4D screen

- **Gate passes:** temporal relation change is supported. Only then replicate on
  preregistered additional seeds before considering an 8k scale-up.
- **Gate passes but width dynamics remain weak:** temporal height/ratio change is
  supported for TTC, while anisotropy remains a secondary limitation. Replicate
  first; only then decide whether a deformation/divergence head is justified.
- **Temporal loss learns but mechanism gate fails:** DINO relation-change
  supervision is not sufficient; next representation probe should use explicit
  event-only cross-time correspondence/transport rather than fixed-patch JEPA.
- **Temporal loss does not decrease or calibration is degenerate:** stop A4D and
  inspect relation target temporal energy/optimization; do not tune validation.

## Claim boundary

A4D remains a public validation mechanistic screen. It cannot authorize a SOTA
claim and must not open official eAP/CodaBench/EvTTC test labels.
