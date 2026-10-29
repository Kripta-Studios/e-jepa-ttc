# Object-Signed-Expansion v3

## Status

This route is an additive, falsifiable candidate. It does not modify or erase
Object-LHR v1, Object-Expansion v2, or their recorded negative results. The
screen profile is not claim-eligible and EvTTC remains sealed until a clean,
seeded full candidate is frozen.

## Problem addressed

Object-Expansion v2 represented TTC as a binary direction multiplied by a
positive magnitude. The diagnostic audit showed that the direction score had
useful ranking information but was poorly calibrated under the sequence-level
change in the negative-TTC prior. Lowering the classification threshold raised
negative recall but damaged the positive TTC buckets.

V3 removes that discontinuity and predicts the signed, dimensionless expansion
coordinate

\[
 g = \frac{\Delta t}{\mathrm{TTC}}.
\]

The sign is part of the regressed physical variable. TTC is derived only for
reporting:

\[
 \widehat{\mathrm{TTC}} = \frac{\Delta t}{\hat g}.
\]

## Architecture

The input remains the two official object ROIs from the existing cache. No TTC,
depth, 3-D box, category, or future frame is passed through `model_inputs()`.
The new route combines:

1. the exact transferable high-resolution JEPA backbone;
2. query pooling plus event-activity-weighted dense token pooling;
3. endpoint residual adapters initialized as the identity;
4. explicit ordered temporal features, including endpoint difference,
   absolute difference, product, and latent prediction error;
5. a causal 2-D box-motion encoder using the same endpoint boxes already used
   to construct the ROIs;
6. a forward and reverse latent endpoint predictor trained against an EMA target encoder;
7. an antisymmetric event score;
8. a bounded residual correction around a deterministic box-height LHR prior.

The deterministic prior is

\[
 r_{box}=\log(h_1/h_2),\qquad g_{box}=1-\exp(r_{box}).
\]

It is reported as a separate baseline. The learned candidate is not eligible
unless it beats that baseline on validation; this prevents box geometry from
being presented as a JEPA gain.

## Objective

The primary target is continuous signed expansion. Auxiliary terms are:

- official TTC-implied log-height ratio;
- projected visible-height ratio when reliable;
- continuous sign-margin loss without a classification threshold;
- forward and reverse latent endpoint prediction against stop-gradient EMA targets;
- latent variance floor;
- event-activity reconstruction over valid patches;
- ordered-pair swap consistency;
- small signed-log TTC auxiliary loss.

Negative, crucial, small, and large TTC regimes receive explicit loss weights.
Checkpoint selection additionally penalizes the negative bucket and MAE.

## Scientific gates

A checkpoint must satisfy all of the following before it can become `best.pt`:

- sufficient signed-expansion variance;
- minimum expansion and log-ratio correlation;
- balanced sign accuracy;
- minimum negative recall;
- minimum sign AUC;
- bounded TTC saturation;
- at least 0.5% validation MiD improvement over the deterministic geometry prior in screen;
- at least 2% improvement over the prior in the full candidate.

The final summary reloads and evaluates `best.pt`; it keeps the last evaluation
separately.

## Fairness and interpretation

This route explicitly consumes observable 2-D box motion. That is a stronger
input contract than an event-only crop whose box coordinates are discarded.
Consequently, results must always be reported in three rows:

1. deterministic box-geometry prior;
2. scratch v3;
3. JEPA-transfer v3.

Only the improvement of rows 2/3 over row 1 measures event/JEPA value. A strong
absolute score that does not beat the prior is not evidence for JEPA.

## Execution order

Run exact-base/cache preflight, focused tests, scratch screen, and then the
seed-matched Level-transfer screen. Do not rebuild the existing screen cache.
Do not launch the full profile unless both learned arms pass all gates and the
JEPA arm improves over scratch without regressing negative recall.
