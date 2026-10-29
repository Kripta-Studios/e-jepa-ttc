# Object Event TTC v4.19 — dense correspondence/divergence probe

v4.18 falsified the hypothesis that endpoint foreground geometry (mass, radius,
covariance determinant) is itself a transferable TTC direction signal. The
validation correlations of all ten features were near zero even though several
foreground features correlated on train.

v4.19 therefore does **not** add another flexible classifier. It probes whether
the frozen v4.8 deep spatial maps preserve local correspondence information.
For each endpoint pair it computes a local cosine-correlation volume, soft
feature displacement, then two translation-invariant physical quantities:

- spatial divergence `du/dx + dv/dy`;
- foreground-centred radial scale slope `(u·r)/||r||²`.

Forward and reverse endpoint scores are antisymmetrised. Seeds 7/13/23 are
combined by the median only after converting each seed to physical scalar
scores. No boxes, heights, sequence IDs or track IDs enter forward inference.
The v4.10 magnitude remains frozen so this experiment isolates direction.

There is no scientific exit-code gate. The decision record distinguishes:

1. frozen correspondence already sufficient -> train a flow decoder + partial unfreeze;
2. correspondence exists but is sequence-specific -> add event-flow pretraining;
3. no metric correspondence signal -> add event-native flow/contrast objective before TTC fine-tuning.

Official eAP test and EvTTC remain unopened.
