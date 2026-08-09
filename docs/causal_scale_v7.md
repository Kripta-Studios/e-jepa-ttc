# Causal Scale TTC v7 — causal temporal transport

Updated: 2026-08-10.

## Scope

V7 retains the v6 stride-free separable foreground and adds a parameter-free temporal
operator. Train seed 401 and validation seed 502 are open; test seed 603 remains
sealed. No eAP, EvTTC, RGB, Garl-TTC label or CodaBench input was accessed.

Signed diagnostic comparison:

```text
path: artifacts/metrics/causal_scale_v7_diagnostic_comparison_v1.json
artifact_identity: eb3497fafad8a4d23284b263303628be8ad025fd61bac57ad5f54580d142ee82
serialized_sha256: bd4b2a23968751616e455a5414b91c877a6ddba55893d3196181d4d144b26196
selectable: false
test_opened: false
```

## Temporal operator

For each pair, the geometry path produces inverse TTC `q`. The previous estimate is
transported to the current endpoint under constant relative velocity:

```text
q_previous_at_current = q_previous / (1 - delta_t * q_previous)
q_current = 0.75 * q_last_pair + 0.25 * q_previous_at_current
r_current = log(1 + delta_t * q_current)
```

Invalid transport denominators fail safely to the current-pair estimate. With only
two endpoints, the model is exactly the v6 pair operator. Reversal gates therefore
audit the T=2 pair operator; translation gates compare the complete T=3 prediction.
The blend adds no learned parameter and only scalar operations.

The weight `.75` was selected on validation 502 after a declared scan `[0,1]`.
Weights `.60–.85` all exceeded Pearson `.95` in the fixed v6 checkpoint analysis,
so the selected value is not an isolated optimum. The primary foreground/NLL remains
supervised on the unblended pair ratio; the temporal blend is applied to causal
inference and validation calibration.

## Validation diagnostics

| Variant | Pearson | Slope | Sign | IoU | TTC sym. rel. | Translation p95 |
|---|---:|---:|---:|---:|---:|---:|
| blend inside joint ratio loss | .94137 | .97459 | .99115 | .88235 | .26626 | .00578 |
| pair-supervised + gate-aware selection | **.96126** | .92744 | .99115 | **.89268** | **.24345** | **.00351** |

The selected checkpoint also has ratio MAE `.0169565`, 80% coverage `.7994100`,
known coverage `1.0`, empty unknown `1.0`, empty false-positive fraction `0`, oddness
median `3.43e-8` and p95 `2.17e-5`. It passes every frozen validation gate.

## Authorization boundary

This is validation-only evidence. It authorizes freezing code/configuration and one
clean-tree evaluation of seed 603 after the exact commit is published. It does not
authorize real data, an eAP screen, EvTTC scoring, comparison with Garl-TTC or a SOTA
claim. A held-out failure closes v7 without threshold changes or test reuse.
