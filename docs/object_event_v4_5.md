# Object Event TTC v4.5 — paired reciprocal MiD fine-tuning

## Decision from v4.4

V4.4 falsified the claim that global radial moments can repair the held-out sign
failure.  Geometry-only validation Pearson was weak, the hybrid improved weighted
MiD by only about two percent, and the fragile sequence retained one correct
negative out of twenty-eight.  The next controlled variable is therefore the
training objective, not another hand-crafted fusion branch.

## Scientific question

Can the already validated v4.2 event encoder improve paper-aligned MiD and
sequence-held-out sign robustness when fine-tuned with:

1. the four official eAP TTC-range weights in log-eta space;
2. balanced sign supervision;
3. the exact reciprocal target for a reversed temporal clip; and
4. reciprocity regularisation in log-eta space?

The architecture, cache, split, inputs and seed identities remain fixed.  Each
v4.5 seed starts from its matching v4.2 checkpoint.

## Exact reversal

For forward expansion `g`, define `eta = 1 - g`.  Temporal reversal implies
`eta_rev = 1 / eta`, therefore

```text
g_rev = 1 - 1 / (1 - g) = -g / (1 - g)
```

and the consistency equation is

```text
log(eta_forward) + log(eta_reverse) = 0.
```

This replaces the first-order approximation `g_rev = -g`.

## Fail-closed advancement

V4.5 does not claim SOTA and does not open the official eAP test or EvTTC.  It
advances to a learned foreground/height-ratio geometry head only when the
three-seed ensemble materially improves v4.4 weighted MiD, preserves event
dependence and Pearson, and repairs eligible per-sequence negative recall.
