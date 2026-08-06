# Object Event TTC v4.4 — train-only geometric residual calibration

## Why this experiment is next

V4.3 established a reproducible event-only signal across seeds 7, 13, and 23:
mean validation Pearson 0.5903, ensemble Pearson 0.6215, and near-total loss of
correlation after zeroing or shuffling events.  It nevertheless failed the robust
gate because `DGqicHUGWb` classified only 1/28 negative examples correctly.

Pearson is a learnability diagnostic, not the official eAP objective.  eAP uses
motion-in-depth

```text
eta = 1 - delta_t / TTC
MiD = |log(eta_hat) - log(eta)| * 1e4
```

and weights the crucial, small, large, and negative TTC ranges by
`0.5/0.3/0.1/0.1`.  Consequently, v4.4 reports MiD directly and asks whether an
explicit event-geometry cue fixes the sign/calibration weakness without hiding it
behind RGB, boxes, observable motion, or validation-label fitting.

## What v4.4 does

1. Reuses the immutable v4.2 predictions for seeds 7/13/23.
2. Extracts box-free spatial moments from the corrected common-coordinate
   `t0/t1/t2` event tensors.
3. Builds an analytic looming proxy from the temporal change in radial event
   extent, plus activity, anisotropy, and centroid-motion diagnostics.
4. Selects ridge regularization by leave-one-training-sequence-out CV.
5. Fits two calibrators on **training rows only**:
   - geometry-only;
   - neural-ensemble plus a geometry residual.
6. Evaluates once on the unchanged held-out validation sequences.
7. Reports expansion metrics, official-formula eAP MiD/RTE, per-sequence sign
   metrics, OOF predictions, and all calibrator coefficients.

The validation subset is not the official eAP test split, and this patch does not
run EvTTC.  Therefore it cannot establish a SOTA claim.  A pass only authorizes a
v4.5 differentiable geometry head and then the official eAP/EvTTC protocols.

## Pass criteria

The hybrid must preserve Pearson and MAE while materially improving MiD,
balanced-sign accuracy, global negative accuracy, and the minimum negative
accuracy among sequences with at least 20 negative examples.  Geometry-only must
also show a positive independent signal.

A failure is scientifically useful: it means post-hoc radial geometry is not
sufficient, and the next model should use learned foreground/height-ratio
representations and explicit ego-rotation compensation rather than fusion with
the old observable-motion shortcut.
