# Object-expansion TTC v2

## Motivation

The historical Object-LHR v1 route is retained unchanged.  Its screen proved
that object-centric ROIs remove the full-frame constant-output collapse, but it
also exposed three failure modes:

1. absolute visible height is not identifiable after every ROI is resized to a
   common `128 x 128` canvas unless crop scale is given to the model;
2. converting a noisy log-height ratio through `TTC = dt / (1 - exp(r))`
   amplifies small errors near `r = 0` and caused frequent `+-60 s` saturation;
3. the v1 `mid_scale: 10000` term abruptly dominated all other objectives.

## Stable coordinates

The v2 model predicts the TTC direction and the bounded magnitude

```text
log |1 / TTC|.
```

The primary target is therefore finite and non-singular.  The implied LHR
coordinate is used as a consistency objective:

```text
signed_inverse = 1 / TTC
log_ratio = log(1 - delta_t * signed_inverse).
```

Absolute heights remain available in the cache only for provenance audits; they
are not optimized by v2.

## Transfer protocol

The exact JEPA backbone is loaded when `--pretrained` is supplied.  A
zero-initialized residual endpoint adapter and a pair projector learn the ROI
shift.  During the first five transfer epochs the backbone LR is zero while
pooling, adapter and task heads train.  Scratch never freezes a random
backbone.

## Checkpoint gate

A checkpoint is eligible only after the warm-up and only when all conditions
hold:

- signed inverse-TTC prediction has non-trivial variance;
- log-ratio Pearson exceeds the configured minimum;
- balanced sign accuracy exceeds the configured minimum;
- TTC saturation remains below the configured maximum;
- sequence-macro MiD is finite.

The final report reloads `best.pt`.  `best_evaluation` and `last_evaluation` are
reported separately so a deteriorated final epoch cannot masquerade as the
selected checkpoint.

## Scientific status

This patch changes the hypothesis and must be evaluated first as a paired
screen.  It does not establish transfer benefit or SOTA performance.  The
correct comparison is the same cache, seed and v2 head with only initialization
changed: scratch versus the frozen-audited JEPA checkpoint.
