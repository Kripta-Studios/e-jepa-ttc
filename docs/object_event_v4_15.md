# Object Event TTC v4.15 — shared odd sign × magnitude projection

## Motivation

V4.14 reproduced the v4.12 directional probe with true seeds 7, 13 and 23.
The signed TTC predictions were highly correlated, but the raw negative
probabilities were not (`min pairwise r = 0.439`).  Consequently the locked
absolute threshold `0.985` produced no multiseed overrides and failed to repair
the 28 receding samples in `DGqicHUGWb`.

## Architectural change

V4.15 is not another post-hoc average of probabilities.  It:

1. freezes the three v4.8 backbones;
2. extracts an exactly odd descriptor from every backbone;
3. combines the per-seed descriptors by their robust mean and median;
4. trains one shared bias-free odd sign head;
5. chooses its sign threshold only from grouped OOF predictions on the train
   sequences;
6. projects the selected sign onto the frozen v4.10 magnitude.

The final operation is `sign × |magnitude|`, not a blend of opposite signed
values.  Therefore v4.15 cannot introduce the near-zero cancellation that made
v4.13 RTE unstable.

## Scientific boundary

The eAP test split and EvTTC remain closed.  Validation is evaluated once after
all head and threshold choices are complete.  Sequence and track identifiers
are used only to form grouped train folds and metrics, never as forward inputs.


## v4.15.1 positive-tail calibration hotfix

The first v4.15 screen selected probability threshold `0.999`, while the
median-logit calibration bounded observed probabilities near `[0.268, 0.732]`.
Because the frozen train ensemble already had no positive-baseline negative
targets, projection-based OOF scoring rewarded a no-op threshold.

v4.15.1 therefore selects a small false-positive tail budget (`alpha`) from
direct grouped-OOF sign discrimination.  The candidate range is locked to
`0.001..0.003`; validation is not used.  After the final head is trained, the
operational probability threshold is recomputed as the corresponding upper
quantile of final-head probabilities on positive train samples.  The inference
rule remains positive-to-negative `sign x frozen magnitude`, so magnitude, MiD
and TTC saturation cannot be changed by cancellation.


## v4.15.2 train-calibrated magnitude-risk hotfix

v4.15.1 activated 97 validation overrides and repaired the negative-sign gates,
but 68 overrides were positive targets and Pearson fell from `0.6760` to
`0.6556`.  The false overrides carried about twice the baseline expansion
magnitude of the corrected hidden negatives.

v4.15.2 keeps the OOF positive-tail sign calibration and adds a conservative
magnitude-risk ceiling.  Its value is the locked 25th percentile of absolute
frozen-v4.10 expansion on positive train rows.  A positive-to-negative override
now requires both high sign confidence and magnitude below that train-derived
ceiling.  The 25th-percentile policy was chosen after inspecting the v4.15.1
development validation result, so this rerun is explicitly not independent; the
official eAP test and EvTTC remain closed.
