# Causal Scale TTC v6 — equivariant synthetic diagnostics

Updated: 2026-08-10.

## Scope and sealed boundary

V6 follows the failed v5 held-out gate. It does not reuse consumed v5 test seed 303.
Its new groups are train 401, validation 502 and sealed test 603. All four executed
runs are `diagnostic_nonselectable`: only 401/502 were instantiated. eAP, EvTTC, RGB,
Garl-TTC labels and CodaBench remained closed.

Signed comparison artifact:

```text
path: artifacts/metrics/causal_scale_v6_diagnostic_comparison_v1.json
artifact_identity: e00a64a90aee5c302ad486763ed147a2af590a7d3575191395e0f0d374d6191f
serialized_sha256: 8b2037b7ef18f0430164bb10755e541206927d8000882b69a68b5e32e0e7b048
selectable: false
test_opened: false
```

## Architecture

The selected foreground head is coordinate-free and stride-free:

```text
event endpoint [12,H,W]
  -> small full-resolution convolutional stem
  -> max projection over columns / rows
  -> independent dilated 1-D row and column heads
  -> separable filled foreground logits [1,H,W]
  -> soft vertical moment -> log-height ratio -> TTC
```

The four dilation levels give each axis a receptive field greater than 30 pixels
without a full-resolution U-Net. Integer translation does not change sampling phase.
The existing low-resolution encoder remains responsible for geometry tokens,
uncertainty and the bounded antisymmetric residual. No box coordinate, category,
sequence ID or direct TTC feature is accepted.

## Diagnostics

| Variant | Pearson | Slope | Sign | IoU | TTC sym. rel. | Translation p95 |
|---|---:|---:|---:|---:|---:|---:|
| shallow full-resolution | .8575 | .9194 | .9646 | .1824 | .4342 | .0279 |
| separable equivariant | **.9204** | .9488 | **.9912** | **.8932** | .2663 | .00462 |
| + pair-ratio mask loss | .8717 | **.9523** | .9882 | .8720 | .2638 | **.00414** |
| + learned height correction | .8586 | .9519 | .9882 | .8659 | **.2545** | .00486 |

The separable head materially solves the spatial failure: versus the shallow
full-resolution arm, IoU rises `.1824 -> .8932` and translation leakage falls
`.0279 -> .00462`. It passes slope, sign, foreground, TTC, calibration, oddness,
known and empty controls. Pearson `.9204` still fails the unchanged `.95` gate, so
test seed 603 remains sealed and v6 is not promoted.

The pair-ratio mask objective and bounded learned log-height correction both reduce
Pearson and are rejected. A separate frozen-mask residual refinement reached only
`.9225`; it is a diagnostic observation, not a retained artifact or selected model.

## Decision

Do not enlarge the foreground decoder further: v6 shows that spatial support and
equivariance are no longer the limiting factor. The next protocol must improve
cross-group temporal scale estimation using multiple train and validation groups,
macro/worst-group selection and a preregistered new test family. It must not use seed
303 or 603 for tuning. Real data remains unauthorized until a successor passes every
synthetic gate from a clean frozen commit.
