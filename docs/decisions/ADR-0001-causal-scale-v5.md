# ADR-0001 — Geometry-bound causal scale core for v5

- Status: accepted for implementation and synthetic testing
- Date: 2026-08-09
- Scope: event-only first; reusable by RGB-only and late-fused RGB-E
- Evidence level: architecture and mechanistic tests, not eAP/EvTTC performance

## Context

Object Event v4.31 showed that the frozen correspondence matcher was stable but not
physically equivariant. Its 512-row train-only diagnostic failed zoom slope, sign,
oddness, translation leakage and reversal coverage. Adding another TTC regressor to
that matcher would preserve the failed mechanism and make any apparent improvement
harder to attribute.

The replacement must expose the same observable and output contract in event-only,
RGB-only and RGB-E experiments. Bounding boxes may define an offline crop and provide
training-only foreground targets, but their coordinates cannot enter the network as
numeric features.

## Decision

Every endpoint is encoded independently with shared weights. A foreground decoder
produces a dense mask and a differentiable, translation-invariant vertical extent.
For consecutive endpoints the primary physical coordinate is

```text
r = log(h_current / h_previous)
inverse_TTC_current = expm1(r) / delta_t
TTC_current = delta_t / expm1(r)
```

This sign convention makes expansion and approaching TTC positive. Recession is
negative. A target is physically invalid when `1 + delta_t / TTC <= 0`, because a
positive apparent scale cannot represent a post-contact interval under this model.

The learned correction is

```text
residual(a, b) = limit * tanh((score(a, b) - score(b, a)) / 2)
```

and is therefore exactly antisymmetric under temporal reversal. Its final layer is
zero-initialized and its magnitude is bounded. The main TTC and risk predictions use
only the foreground scale plus this guarded residual. A direct inverse-TTC head is
retained solely as an auxiliary training objective and cannot feed the primary
prediction.

Uncertainty is predicted in log-ratio space and propagated to inverse TTC and TTC by
a first-order delta approximation. Risk probabilities are derived from the resulting
inverse-TTC distribution. Low sensor support and ratios close to zero produce an
explicit `known_mask=false`; the finite TTC transport value must never be consumed
without that mask.

The event arm starts with three causal endpoint tensors and a 336k-parameter default
encoder. The same class can accept RGB endpoints, but an RGB experiment is not
authorized until its exposure/blur reliability contract and independent baseline
protocol are added. Multimodal fusion will operate on modality-specific geometry
tokens and distributions, not early-concatenated channels.

## Mandatory gates

Before learning from a real TTC split, the scale operator must pass:

- analytic zoom Pearson at least `0.95`;
- slope in `[0.8, 1.2]` and sign accuracy at least `0.95`;
- reversal oddness median/p95 no greater than `0.2/0.5`;
- identity, translation and controlled square-rotation leakage thresholds;
- exact unknown behavior for zero-event inputs;
- finite loss and gradients through foreground, uncertainty and auxiliary heads.

Passing the ideal-foreground gate authorizes only the next synthetic event-learning
gate. It does not show that the encoder can infer foreground from event data, does not
measure TTC accuracy and does not authorize eAP test, EvTTC or a SOTA claim.

## Consequences

Positive consequences:

- the physical sign and time convention are testable without TTC data;
- temporal reversal is structural rather than encouraged only by a loss;
- the three modality arms can be compared using one target and output contract;
- box-coordinate shortcuts are absent from the model API;
- low-support predictions can fail safely instead of returning confident clipped TTC.

Costs and open risks:

- foreground quality becomes a primary bottleneck;
- moment-based visible height may be sensitive to fragmented masks and truncation;
- delta-method uncertainty is only a local approximation near the TTC singularity;
- camera/target rotation is not generally invariant, so the current rotation gate is
  limited to a square control and must not be generalized to arbitrary objects;
- real event sparsity, RGB exposure and cross-domain calibration remain unmeasured.

The next implementation must train the event foreground/scale path on a synthetic
expansion generator, then rerun the complete operator gate on predicted masks. Only a
passing learned-mask result can authorize grouped train-only eAP development.

## Recorded evidence

The clean implementation at `7945e99` passed the ideal-foreground gate. The compact
artifact is `artifacts/metrics/causal_scale_v5_synthetic_operator_gate_v1.json`, with
artifact identity `7f604160094831598017ae5741860a0a0702a7095fd227af9950363a9ca4b1e1`
and serialized SHA256
`3fd4d2a25b85173cf34bb8738f5b7e80190f31f26acc9ed9a4d3c818d10afb20`.
This evidence changes no limitation or authorization boundary stated above.
