# Scientific Recovery V9 — E-Clock X0 implementation status

Status: implementation complete; scientific OOF campaign not executed.

The X0 implementation is isolated on `scientific-recovery-v9-eclock-x0`, based on
`718e0bf7ca9950fbc0fc2a3537e4b0e0e25a72a2`. It adds the direct benchmark-phase
coordinate, the frozen A5 replay/readout controls, and one matched BASE/DYN class.
X0-DYN-W is limited to config, schema, and synthetic loss tests; its execution is
disabled.

## Scientific boundary

All X0 artifacts and future claims must declare:

```text
upstream_roi_is_box_conditioned=true
```

BASE/DYN may additionally declare only:

```text
explicit_foreground_height_interface_bypassed=true
```

This is not a claim that the system is geometry-free, bbox-free, detector-free, or
free of implicit geometric information. X0-PAIR-U is geometry-infused and is only a
readout diagnostic. Smoke outputs are engineering checks and never scientific
results.

The maximum future DYN-versus-BASE claim, and only after an authorized complete OOF
campaign, is:

> En el universo grouped-development y presupuesto preregistrado, conservar los
> nueve resúmenes de correspondencia global uniforme mejora frente al control que
> calcula y anula esos mismos slots, cuando ambos predicen benchmark phase y omiten
> la interfaz explícita de altura.

## Matched contrast

`X0-BASE-U` and `X0-DYN-U` use the same model class, endpoint trunk/token, feature
schema, phase head, seed, initialization, optimizer, scheduler, precision, sampler,
budget, checkpoint policy, matcher radius, matcher temperature, and forward/reverse
matcher calls. BASE computes both observed motion vectors and applies exactly
`m01_base = m01 * 0.0` and `m12_base = m12 * 0.0` at the audited feature boundary.
This is topology-, initialization-, data-, budget-, and matcher-call-matched; it is
not a claim of equal scientific gradient signal.

The frozen motion schema contains exactly, in order:

1. `translation_x`
2. `translation_y`
3. `divergence_x`
4. `divergence_y`
5. `divergence_isotropic`
6. `flow_magnitude`
7. `confidence_margin`
8. `entropy`
9. `cycle_error`

The transport helper is called with `foreground_weight=None`; the duplicated trailing
nine legacy fields are discarded.

## X0-DYN-W

The configured and signed reduction is literally:

```text
loss_reduction=normalized_weighted_absolute_phase_error
```

It accumulates `sum(weight * absolute phase error) / sum(weight)` in float64 and
fails on non-finite inputs or a non-positive/non-finite denominator. The config
forbids weight clipping, bucket recomputation, outer-dev weight selection, and
target/error resampling. No real-data forward, smoke, or scientific run is allowed
for this arm in the present phase.

## Execution state and remaining prerequisites

The protocol/config validators, strict 8,192-row/three-fold precheck, signed reference
builder, fixed-update resume substrate, aggregator, verifier, and PowerShell
orchestrator are implemented. The bounded synthetic BASE/DYN smoke is permitted but
does not enter the production aggregator.

The verified V8 essential-results package intentionally omits physical A5
checkpoints and tensor cache shards. Therefore PAIR/A5 real smokes and the bounded
outer-train smoke cannot run in this checkout. They must not be replaced or
reconstructed. A future scientific launch must supply the original hash-verified,
fold-local A5 checkpoints and the signed 12-channel train-only cache/split adapter;
no sealed evaluation path may be resolved.

The future coordinate control `X0-DYN-PHASE` versus `X0-DYN-INV-CTRL` remains
documented but is not implemented or executed.
