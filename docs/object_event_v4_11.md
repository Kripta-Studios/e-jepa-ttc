# Object Event TTC v4.11 — train-only sign/magnitude router

## Motivation

V4.10 established a robust global event-only signal across seeds 7, 13, and 23:
Pearson, MiD, balanced sign, bootstrap, and seed agreement all passed.  The sole
architectural failure was concentrated in the negative windows of one held-out
sequence.  The three seeds agreed with one another while agreeing on the wrong
positive sign.  Therefore neither seed uncertainty nor another convex average can
solve the failure.

## Hypothesis

TTC expansion should be factorized into two decisions:

1. **sign** — approach versus recession;
2. **magnitude** — absolute expansion rate.

V4.11 keeps the stable v4.10 magnitude and learns a small sign head from the v4.9
expert predictions.  When the sign head assigns high train-calibrated probability
to recession while the fixed ensemble is positive, it applies a bounded negative
residual.  This is deliberately non-convex: it can repair a unanimous false
positive from all experts.

## Leakage controls

- The router and its hyperparameters use only train predictions.
- Hyperparameters are selected by leave-one-train-sequence-out predictions.
- Sequence and track IDs are grouping variables, never features.
- Features are seed-invariant summaries of base, dense, and fixed expert outputs.
- Boxes, visible heights, RGB, validation targets, official eAP test, and EvTTC are
  not inputs.

## Interpretation

V4.11 is a falsifiable development screen, not yet the final integrated network.
If it repairs held-out negative accuracy while preserving global correlation, the
next model should put an equivalent sign head inside the event architecture and
train sign and magnitude jointly.  If it fails, expert-output routing is
insufficient and the representation itself needs explicit recession-sensitive
normal-flow/divergence supervision.
