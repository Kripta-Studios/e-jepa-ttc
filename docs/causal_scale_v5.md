# Causal Scale TTC v5

## Status

The shared geometry core, event-only learning dataset and train/evaluation runner are
implemented. The clean synthetic ideal-foreground operator gate passed. Nine later
train/validation-only diagnostics learned foreground and scale, but the selected
candidate remains unpromoted because one frozen validation gate still fails. No real
TTC split was evaluated and no comparison with Garl-TTC was performed.

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

## Learned event diagnostics

The signed comparison artifact
`artifacts/metrics/causal_scale_v5_diagnostic_comparison_v1.json` is generated from
the nine complete diagnostic summaries. It is explicitly non-selectable and records
dirty-tree train/validation development only. Test seed 303 and every real source
remained closed.

| Variant | Pearson | Slope | Sign | IoU | TTC sym. rel. | Translation p95 |
|---|---:|---:|---:|---:|---:|---:|
| initial joint | .2678 | .1932 | .6207 | .3619 | 1.2969 | .1157 |
| extent + warm-up | .5941 | .6813 | .8534 | .6312 | .7577 | .1013 |
| fixed selection | .9064 | .9119 | .9655 | .8493 | .3838 | .0577 |
| translation + calibration | .9329 | .9608 | .9914 | .8528 | .2706 | .0296 |
| deconv + scale + cosine | **.9560** | **.9686** | **.9957** | .8640 | **.2639** | .0240 |
| resize-conv ablation | .9430 | .9759 | .9871 | **.8651** | .2794 | .0320 |

The table is a rendering of the signed comparison artifact, not manually sourced
benchmark evidence. The Huber arm reduced correlation to `.8724`; it is rejected.
The resize-conv decoder also regressed correlation and translation. The selected
deconvolutional candidate passes 11/12 validation gates; translation p95 misses the
frozen `.02` bound by `.00399`. Thresholds were not relaxed.

The useful interventions were foreground extent supervision, a mask-only warm-up,
90 ms causal accumulation, validation selection after warm-up, a half-resolution
learned decoder, translation augmentation, cosine decay and validation-only scalar
variance calibration. The direct TTC auxiliary remains isolated from primary output.

## Authorization boundary and next gate

The diagnostics authorize only committing and verifying the exact synthetic protocol.
The full runner may then open held-out synthetic seed 303 once from a clean worktree.
Promotion requires every frozen learning gate; a failure returns to synthetic
architecture work without opening real data.

Only after that learned event path passes may a versioned train-only eAP development
screen be designed. RGB-only must then implement an independent exposure/blur
reliability contract under the same scale output. RGB-E remains a late-fusion stage
after both unimodal arms are independently valid. eAP test/CodaBench and EvTTC remain
closed until architecture, configs and checkpoints are frozen.
