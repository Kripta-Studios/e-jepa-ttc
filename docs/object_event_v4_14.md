# Object-event TTC v4.14 — locked true-seed dual-head replication

V4.13 repaired the fragile negative cases on the already-open seed-7 development
screen, but its three fusion constants were informed by that screen.  V4.14
therefore freezes the constants and tests the directional probe with true seeds
7, 13 and 23.

## Locked protocol

- `base_blend = 0.20`
- `override_blend = 0.51`
- `negative_override_probability = 0.985`
- no validation retuning;
- v4.8 backbones and v4.12 probe initialization are seed-specific;
- the v4.10 magnitude ensemble remains frozen;
- eAP test and EvTTC remain unopened.

The primary aggregate uses the median negative probability across seeds.  With
three probes, a sign-changing override therefore requires at least two probes
to support it at the locked confidence level.  This consensus rule is fixed
before seeds 13 and 23 are evaluated.

V4.13 can create near-zero expansions when the two heads nearly cancel.  V4.14
therefore adds a preregistered TTC-calibration safety gate: weighted RTE may not
exceed 1.5 times the frozen v4.10 baseline, and TTC saturation may increase by
at most 0.06.  These gates do not change predictions; they prevent a Pearson
improvement from hiding unstable TTC values.

## Interpretation

A pass supports integrating the odd directional head and stable magnitude head
inside one event-only model.  A failure localised to negative cases indicates
that the representation still needs an explicit radial-flow/divergence feature,
not another validation-fitted threshold.
