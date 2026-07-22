# Results invalidation and claim boundary

Snapshot: 2026-07-13. Historical code commit: `d29339d4093e9ba7e306eb45a35daad1d976efc8`.
The recovery worktree is intentionally not a result-producing state until it is committed cleanly.

## Allowed registry states

The only result states are `invalid_pre_fix`, `reused_test_diagnostic`,
`smoke_only`, and `valid_post_fix`. The legacy spelling `reused_test` is accepted
only while reading old split metadata and is normalized to
`reused_test_diagnostic`.

`valid_post_fix` requires a full commit SHA, a clean worktree, immutable hashes,
an exact command, timestamps, hardware, and saved metrics. No current run has
that status.

## Local data inventory

| Item | Verified value |
|---|---:|
| EvTTC source sequences | 9 |
| Source files | 1,121 |
| Source bytes | 58,137,313,248 |
| Full navigation cache | 2,402,953,055 bytes |
| Cache windows | 3,972 |
| Split windows used by the historical trainer | 3,012 train / 474 validation / 478 test |

The available sequences are the three `CCRs-1`, the three `CCRs-side`, and the
three `CPLA` speed variants. There is no additional locally available sequence
that has remained uninspected and can serve as a final test.

## Historical run decision

The strongest all-window row (`0.312034689 +/- 0.044063632 s` CPLA-high MAE)
is retained only as `reused_test_diagnostic`:

- SSL pretraining used seed 7 only;
- downstream fine-tuning used seeds 7, 13, and 21;
- the downstream models were initialized from SSL `last` at epoch 30, although
  validation selected SSL `best` at epoch 26;
- downstream evaluation used the validation-selected `best` checkpoints;
- CPLA-high had already been inspected in prior development branches;
- old checkpoints and metric JSON files do not contain all post-fix provenance
  fields now required by the registry.

The three diagnostic CPLA-high MAEs are `0.365196984`, `0.313609060`, and
`0.257298023 s`. Their spread measures downstream randomness conditional on a
single SSL seed; it is not end-to-end multi-seed uncertainty.

The historical SSL artifact itself is `invalid_pre_fix` for promotion because
it lacks the new checkpoint provenance schema. Its numerical diagnostic outputs
remain auditable through the exact files and SHA-256 values in
`artifacts/registry.jsonl`.

## Gates

- Development: train and select on `train`/`validation` only.
- Diagnostic: CPLA-high may be evaluated only after a complete freeze and must
  be labelled `reused_test_diagnostic`.
- Official/final: the aggregator fails closed without `--split-protocol` and
  rejects this split even when the protocol is supplied.
- Final recovery: blocked until an independently sourced, uninspected
  sequence-level holdout exists. Repartitioning these same nine sequences does
  not create a new final test.

Historical baselines, ROI probes, ablations, and test tables not explicitly
registered remain non-promotable historical diagnostics.

## 2026-07-22 object-cache normalization invalidation

All eAP object-cache shards and Object-JEPA checkpoints created before cache
format version 2 are invalid for scientific comparison. The earlier sparse
voxel normalizer centred occupied values by their nonzero median. Although it
kept empty voxels at zero, it could also map every occupied voxel to zero when
their magnitudes were equal, erasing sparse event evidence. Version 2 uses a
non-centred 95th-percentile magnitude scale, keeps empty voxels exactly zero,
and preserves occupancy and sign. Pre-version-2 smoke losses, ONNX files and
latency measurements remain engineering diagnostics only and must not appear
as final experimental results. Caches, checkpoints, robustness results and
deployment exports must be regenerated from version 2.
