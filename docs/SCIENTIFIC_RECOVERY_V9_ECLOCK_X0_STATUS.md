# Scientific Recovery V9 — E-Clock X0 corrected pre-execution status

Status: implementation and synthetic integrity QA complete; training, outer-train
smoke, OOF and sealed evaluation not executed.

The X0 worktree remains based on immutable parent
`718e0bf7ca9950fbc0fc2a3537e4b0e0e25a72a2`. This correction replaces every
self-declared production identity with values bound to the signed protocol and
the five-family signed reference artifact.

## Reference taxonomy

There are exactly five distinct families:

| Reference family | Producer identity | Recomputed MiD |
|---|---|---:|
| `official_a5_oof` | official V7 point/full-coverage OOF | 158.44857930928274 |
| `official_c2f_oof` | official V7 C2F seed-7 fold chain | 158.57314044954794 |
| `nested_router_retrained_a5_constituent` | A5 retrained inside nested V8 Router | 162.19984180136834 |
| `nested_router_retrained_c2f_constituent` | C2F retrained inside nested V8 Router | 158.92456189064018 |
| `prospective_router_r` | prospective V8 Router output | 153.87679951674625 |

The builder recalculates these values from the physical artifacts and verifies
their hashes. `X0-A5-REPLAY` and the first `X0-PAIR-U` config are bound only to
`official_a5_oof`. A nested Router constituent cannot satisfy that identity.

## Canonical production boundary

The production precheck loads signed identities rather than accepting caller
lists or hashes. It requires 8,192 unique tokens, seed 7, integer folds exactly
`{0,1,2}`, the exact nine sequences and sequence-to-fold assignment, all four
buckets per sequence, canonical token/target/fold/weight hashes, three fold
summaries and three physical checkpoints. Any missing, extra, duplicate,
non-finite or self-consistent-but-noncanonical input fails closed.

The row-level OOF contract is:

```text
sample_token, sequence_id, track_id, outer_fold,
target_ttc_s, target_benchmark_phase,
predicted_benchmark_phase, predicted_inverse_ttc_raw, predicted_ttc_raw,
predicted_ttc_clipped, is_clip_saturated,
scientific_mid_per_row, scientific_failure, sample_weight,
arm_id, seed, checkpoint_sha256, config_sha256, protocol_sha256,
cache_manifest_sha256, split_manifest_sha256
```

The primary row metric is calculated in float64 directly as
`10000 * abs(target_benchmark_phase - predicted_benchmark_phase)`. Raw inverse
TTC and raw TTC must agree with phase within strict tolerance. Clipping to ±60 s
is exported only as a deployment diagnostic.

For `X0-A5-REPLAY`, the historical clipped A5 vector is checked only to prove
that each official fold checkpoint reproduces the signed legacy artifact. The
X0 scientific MiD still uses the raw inverse-TTC-derived phase; the legacy
replay MiD is labelled diagnostic-only and cannot replace that value.

```text
zero_failure_is_partially_assisted_by_output_domain=true
deployment_clipping_not_used_for_scientific_metric=true
```

## Cache and fold execution

The read-only adapter requires an explicit `--cache-root`. Scientific identity
depends on the manifest SHA, relative shard paths, all 32 shard sizes/SHAs,
preprocessing version and canonical ordered token identity—not on the absolute
host path. It validates the declared event tensor shape/dtype, endpoint order,
label t2 anchor and delta-t before exposing model inputs.

`CollisionClockOuterTrainBatch` and `CollisionClockOuterDevBatch` are separate
types. The trainer accepts only the former; the evaluator accepts only the
latter plus a verified frozen-checkpoint capability. Outer-dev cannot enter
training statistics, early stopping, normalization, calibration or checkpoint
selection. The only checkpoint policy is `last_update_fixed_budget`.

The future OOF flow is complete for folds 0, 1 and 2: verify cache, construct
disjoint views, initialize the exact arm, train outer-train for the fixed update
budget, save/freeze the final update, evaluate outer-dev once, and write raw
row-level coordinates plus a signed fold summary. Dry-run prints this DAG and
all paths without opening shards or creating scientific results.

## Resume, aggregation and bootstrap

Resume binds commit/dirty state, arm and role, reference family where applicable,
seed/fold/motion mode, model class/topology/initialization, config/protocol/
reference/split/cache paths and hashes, canonical and subset hashes, optimizer,
scheduler, precision, update budget, checkpoint policy, completed updates, all
available RNG states, sampler/order state and the physical checkpoint SHA. Any
mismatch or truncation is fatal.

The aggregator has a closed registry:

- `X0-A5-REPLAY`
- `X0-PAIR-U`
- `X0-BASE-U`
- `X0-DYN-U`

`X0-DYN-W`, unknown arms, smoke/subset evidence and incomplete folds are rejected.
MiD, failure, clipping diagnostics, bootstrap intervals and gates are recomputed
internally. Paired bootstrap samples sequence then tracks, keeps all rows of a
selected track, uses identical draws for candidate/reference, and records the
draw identity SHA. The A5 gate names and verifies `official_a5_oof` explicitly.

## Matched BASE/DYN boundary

BASE and DYN retain the audited shared class, 308,005 parameters, input `[B,946]`,
encoder, clock head, matcher radius 1, temperature 0.02, four matcher calls and
the same nine ordered observables. Both compute `m01` and `m12`; only BASE applies
`m01 = m01 * 0.0` and `m12 = m12 * 0.0`. The duplicated 18-field historical
calculation remains intact and X0 consumes only the first nine fields.

## X0-DYN-W

`X0-DYN-W` remains `execution_authorized=false`, `status=not_executed`. Its
float64 normalized weighted absolute phase loss remains unit-tested, but no
forward, smoke, training or OOF is authorized. The closed aggregator rejects it.

## Deferred execution

No real-data model forward/backward, optimizer step, outer-train smoke, OOF,
seed 13/23 run, sealed evaluation or X1–X5/Track-M/recurrent-depth implementation
was performed during this correction. The future launch requires explicit
authorization and the external read-only cache/reference roots.

Future real-data outer-train smoke command (documented, not executed):

```powershell
python scripts/train_scientific_recovery_v9_eclock.py `
  --config configs/experiment/scientific_recovery_v9_eclock/x0_base_u.yaml `
  --mode outer-train-smoke --fold 0 `
  --cache-root <V8_READ_ONLY_TRAIN8192_CACHE_ROOT> `
  --reference-root <V8_READ_ONLY_CHECKOUT_ROOT> `
  --output-root <NONSCIENTIFIC_SMOKE_OUTPUT_ROOT> `
  --device cuda --execute-authorized-outer-train-smoke
```

This mode consumes one typed outer-train batch, performs one optimizer update,
never opens or evaluates outer-dev, and emits only
`evidence_class=smoke`/`scientific_result=false`.
