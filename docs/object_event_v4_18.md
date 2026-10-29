# Object Event TTC v4.18 — radial/divergence physics bottleneck

## Why this experiment exists

v4.12–v4.17 established three facts:

1. event features contain real recession/negative-TTC information;
2. flexible sign heads recover hard negatives but generalise poorly from train sequences to unseen validation sequences;
3. v4.17 fixes the train-prior problem with a signed anchor, yet its grouped-train OOF Pearson remains much higher than validation.

v4.18 therefore stops adding flexible classifier capacity. It asks a narrower question: does a low-dimensional, sequence-invariant physical representation of 2-D radial expansion generalise?

## Physics features

For each of the frozen true-seed v4.8 backbones (7, 13 and 23), v4.18 derives geometry only from event-driven foreground probability maps and event-activity maps at the two endpoint times. It never uses boxes, visible heights, sequence IDs or track IDs as forward features.

For both foreground and activity distributions it measures:

- half log mass change (area-to-linear-scale proxy);
- RMS radial log-scale change around a symmetric midpoint centre;
- first radial transport;
- second radial transport/divergence proxy;
- covariance-determinant log-scale change.

All five signed quantities reverse sign when the endpoint order is swapped. Centroid displacement is used only as an even reliability attenuation. The three true-seed estimates are combined by a robust median and an even seed-agreement factor, preserving oddness.

The resulting ten-dimensional vector is deliberately small.

## Predictor

Two predictions are reported:

- `raw_physics`: no label fitting; the robust normalised physical features vote directly on expansion direction;
- `monotone_physics`: a zero-bias odd linear head with non-negative feature weights. Positive geometric dilation can only increase approach evidence; a feature that is not useful can be suppressed but cannot learn a sequence-specific inverse sign.

Feature scales and model weights are fit using train only. Three grouped train folds are reported before the single development-validation evaluation.

Magnitude is frozen to the absolute v4.10 multiseed ensemble prediction. This experiment isolates direction/physics and cannot create near-zero magnitude cancellation.

## Decision, not another gate loop

v4.18 exits 0 whenever the experiment completes. It records a `decision` rather than blocking on a long gate list:

- if the physics bottleneck preserves global correlation while improving the fragile negative regime, integrate the branch into the event model and then partially unfreeze;
- if even these explicit physical features do not generalise, stop sign-head experiments and move supervision into the dense spatial encoder itself (radial/divergence auxiliary field loss).

Official eAP test and EvTTC remain unopened.
