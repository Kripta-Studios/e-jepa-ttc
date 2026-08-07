# Object Event TTC v4.17 — Signed-anchor causal residual sign

## Motivation

v4.16.2 showed that removing the accidental double importance weighting does not remove the train-to-validation sign prior shift. Grouped train OOF remains strong, while validation predicts roughly the train negative rate instead of the lower validation rate. The temporal representation is useful, but an absolute sign classifier is too free to replace stable event-only evidence on unseen sequences.

## Architecture

All three true-seed v4.8 backbones remain frozen. For every sample v4.17 computes:

- the mean+median v4.12 descriptor consensus;
- a signed expansion anchor from the median of the three v4.8 signed expansions;
- a magnitude anchor from the median absolute expansion.

The signed anchor is normalised with a scale estimated only from train. Its negative-class logit is used as the current default decision. A causal exact-odd temporal head learns only a bounded residual around that anchor:

`sign_logit = signed_anchor_logit + bounded_temporal_residual`

`prediction = sign(sign_logit) * frozen_v48_magnitude`

Both the anchor logit and residual are odd, so exact sign antisymmetry is preserved. The residual is bounded symmetrically, which prevents weak train-prior evidence from overturning a strong signed event anchor. Magnitude is not learned in v4.17.

Training uses uniform without-replacement epoch sampling and an unweighted BCE sign loss. There is no sequence/sign importance reweighting in either the sampler or the loss. Track IDs are grouping metadata only.

## Scientific decision

- If v4.17 restores validation Pearson and positive accuracy while keeping the negative recovery, proceed to multiseed replication and then partial unfreezing.
- If the signed anchor itself is weak and v4.17 still fails, stop adding classifier heads and introduce explicit radial/divergence geometry inside the event representation.

Official eAP test and EvTTC remain unopened.
