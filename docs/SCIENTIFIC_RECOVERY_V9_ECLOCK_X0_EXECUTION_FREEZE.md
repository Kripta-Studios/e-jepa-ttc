# Scientific Recovery V9 — E-Clock X0 execution freeze

This document freezes the engineering and statistical interpretation used for the authorized
seed-7 X0 screen. It was written before any new outer-development prediction was produced.

## Scientific identity

- Branch: `scientific-recovery-v9-eclock-x0`.
- Required X0 starting ancestor: `af66f2c8ca2017059d7765b5f171e1cda866ab07`.
- Seed: 7. Outer folds: 0, 1, 2. Batch size: 32. Updates: 6840 per trained fold.
- Learning rate: `3e-4`; weight decay: `1e-4`; precision: FP32.
- The only scientific checkpoint is the immutable update-6840 milestone. Training minima are
  diagnostics and never select a checkpoint.
- Outer-development data is materialized only after training state and final checkpoint have
  been signed. It is evaluated exactly once.

## Arms and order

The fixed order is A5-REPLAY, BASE-U, DYN-U, the primary DYN-U versus BASE-U comparison, and
PAIR-U. BASE-U and DYN-U have the same train budget, optimizer, scheduler, topology and seeded
initialization. PAIR-U is a geometry-infused readout diagnostic; it is not evidence of a height,
bbox, foreground, or detector bypass.

## Primary gate frozen before results

The signed protocol names `X0-DYN-U_vs_X0-BASE-U` as the primary comparison. Its gate requires:

1. the exact 8192-row paired universe and exact token/sequence/track/fold/target/weight identity;
2. complete finite predictions, zero scientific failures, and zero coverage loss;
3. matched BASE/DYN configuration, training commit and paired bootstrap draws;
4. a disclosed positive finite-draw fraction; and
5. the upper endpoint of the paired 95% bootstrap interval for `MiD(DYN)-MiD(BASE)` strictly
   below zero.

The `-3 MiD` threshold applies only to the separate comparisons with official A5 and is not
silently transferred to the primary BASE/DYN gate.

The hierarchical bootstrap resamples sequences and then tracks. If any resampled sequence
replica loses a required TTC bucket, the entire draw is discarded. The finite-draw fraction is
always reported; no per-sequence finite-value filtering is permitted.

## Engineering controls

- The cache remains read-only and is never copied.
- All 32 signed shards are physically verified once in preflight.
- Direct, bounded shard-LRU and fold-RAM modes must return exactly equal tensors and identities
  on a fixed outer-train subset.
- `fold_ram` is selected only if projected staging leaves at least 6 GiB of available host RAM;
  otherwise `shard_lru` is selected. No target metric or outer-development result participates
  in this decision.
- Only one fold view is staged at a time, with direct preallocation rather than list-plus-stack
  duplication.
- Resume checkpoints are compact, atomic, signed and written every 100 updates. Immutable
  milestones are kept at 250, 500, 1000, 2000, 4000 and 6840 updates.
- Resume requires exact arm/fold/seed/config/protocol/reference/commit/optimizer/scheduler,
  checkpoint SHA, RNG state and sampler-order identity. A state indicating possible prior
  outer-development access fails closed instead of re-evaluating.
- The local worktree virtual environment is not used for the campaign because its Torch binary
  installation is incomplete. The verified read-only reference environment supplies Python and
  CUDA packages, while `PYTHONPATH` points to this frozen worktree's `src` directory.

## Prohibited scope

Public validation, private test, EvTTC test and CodaBench remain closed. DYN-W, seeds 13/23,
X1–X5, V9 Track M and recurrent depth are outside this campaign. No push is performed.
