# Causal Scale TTC v5

## Status

The shared geometry core and event-only configuration are implemented. The clean
synthetic ideal-foreground operator gate passed. This is mechanistic evidence only:
the event encoder has not yet learned foreground, no real TTC split was evaluated,
and no comparison with Garl-TTC was performed.

Authoritative compact artifact:

```text
path: artifacts/metrics/causal_scale_v5_synthetic_operator_gate_v1.json
git_commit: 7945e9936af7b6fa802ef2cca4172487adf0f2d6
git_dirty: false
artifact_sha256: 7f604160094831598017ae5741860a0a0702a7095fd227af9950363a9ca4b1e1
serialized_file_sha256: 3fd4d2a25b85173cf34bb8738f5b7e80190f31f26acc9ed9a4d3c818d10afb20
status: completed_passed
scope: synthetic_mechanistic_only
```

The run was executed from a clean detached worktree with no tracked or untracked
files. The unrelated root handoff files in the primary worktree were never copied
into or observed by the evidence worktree.

## Architecture contract

Each causal endpoint is processed with shared weights:

```text
event ROI [B,T,12,H,W]
  -> compact dense CNN
  -> foreground logits
  -> normalized soft vertical extent h_t
  -> endpoint geometry token

r_t = log(h_t / h_t-1) + bounded_antisymmetric_residual
inverse_TTC_t = expm1(r_t) / delta_t
```

The model accepts only endpoint tensors and elapsed times. Bounding-box coordinates,
category, target ID and sequence ID are absent from its API. The residual uses one
shared ordered scorer in both directions and is exactly antisymmetric. Its final
layer starts at zero. The default model has 336,398 parameters.

The uncertainty head predicts variance in log-ratio space. Inverse-TTC and TTC
variance use a first-order propagation. Collision probabilities are derived from the
inverse-TTC distribution rather than a separate risk shortcut. A direct inverse-TTC
head exists only as an auxiliary loss and does not feed the primary outputs.

## Executed gate

Run from the repository root:

```powershell
uv run --no-sync python scripts/evaluate_causal_scale_v5_operator.py --require-clean
```

| Metric | Result | Frozen gate | Pass |
|---|---:|---:|:---:|
| analytic zoom Pearson | 1.000000 | >= 0.95 | yes |
| through-origin slope | 0.9999995 | 0.8–1.2 | yes |
| sign accuracy | 1.000000 | >= 0.95 | yes |
| oddness median / p95 | 0 / 0 | <= 0.2 / 0.5 | yes |
| identity p95 | 0 | <= 1e-5 | yes |
| translation leakage p95 | 0 | <= 1e-4 | yes |
| square-rotation leakage p95 | 0.0017103 | <= 0.02 | yes |
| zero-event unknown fraction | 1.000000 | >= 1.0 | yes |

The control uses ideal analytic rectangle foreground logits. It proves the observable,
sign convention, reversal algebra and fail-safe path. It does not test whether the
CNN can infer foreground from event tensors. The rotation result applies only to a
square control and is not an invariance claim for arbitrary objects.

## Training objective

The implemented loss combines:

- Gaussian NLL on `log(h_current / h_previous)` derived from signed TTC;
- foreground BCE and Dice on training-only masks;
- risk BCE from the geometry-derived risk logits;
- a low-weight auxiliary inverse-TTC loss;
- bounded-residual regularization;
- constant-velocity consistency across three endpoints when both pairs are known.

Targets for which `1 + delta_t / TTC <= 0` are excluded from the scale-ratio NLL,
because a positive apparent scale cannot encode a post-contact interval under this
model. They may still contribute valid negative risk and auxiliary supervision.

## Authorization boundary and next gate

This pass authorizes only a learned-mask synthetic event experiment. The next runner
must generate causal event tensors from expanding, receding, translating and empty
shapes, train only on its training split, and evaluate predicted foreground/scale on
held-out seeds. Promotion requires all current operator gates plus non-trivial mask
IoU, finite calibration and a small-batch overfit check.

Only after that learned event path passes may a versioned train-only eAP development
screen be designed. RGB-only must then implement an independent exposure/blur
reliability contract under the same scale output. RGB-E remains a late-fusion stage
after both unimodal arms are independently valid. eAP test/CodaBench and EvTTC remain
closed until architecture, configs and checkpoints are frozen.
