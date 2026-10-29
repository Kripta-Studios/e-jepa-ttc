# Object Event TTC v4.10 — true-seed fixed-fusion robustness

## Motivation

V4.9 showed that a fixed 50/50 average of the v4.2 global event expert and the
v4.8 dense-motion expert improves Pearson, expansion MAE, balanced sign and MiD
on the held-out development split.  A single seed is not sufficient evidence for
an architectural promotion.

A hidden reproducibility issue also had to be closed: the v4.6-v4.8 YAML files
contain `train.seed: 7`.  Merely selecting a different v4.2 checkpoint would not
produce a true seed repeat.  V4.10 materializes deterministic seed-specific
copies of the unchanged v4.6-v4.8 configs and records their hashes.

## Protocol

For seeds 7, 13 and 23, v4.10 keeps:

- the same fixed sequence split;
- the same architecture, losses and hyperparameters;
- the same v4.9 fusion coefficient `alpha=0.5`;
- event tensors as the only inference input;
- boxes and visible heights as training-only supervision;
- the official eAP test and EvTTC sealed.

Seeds 13 and 23 are run through v4.6, v4.7, v4.8 and v4.9.  Expected negative
intermediate screens are allowed only where the next stage explicitly targets
that failure regime.  The final v4.8 and v4.9 screens must pass.

The aggregator aligns samples by `(sequence_id, sample_token, track_id)`, reports
per-seed metrics, pairwise agreement, an equal-seed ensemble, track-cluster
bootstrap uncertainty and per-sequence sign robustness.

## Decision

Only a passing v4.10 result permits development of a single integrated dual-head
model.  Failure means the v4.9 complementarity was seed-specific and the current
architecture must not be promoted to official eAP or EvTTC evaluation.

## v4.10.5: complete scientifically failed seed screens

A v4.9 screen that exits with code 2 is a completed scientific result, not an
operational failure. The multiseed runner now preserves and reuses its summary
and prediction CSVs, then includes that seed in the robustness aggregate. The
aggregate keeps `all_seed_screens=false` and therefore cannot relabel the
experiment as passed when `require_all_seed_screens=true`; it simply produces
the complete cross-seed diagnostics needed to decide the next architecture.
