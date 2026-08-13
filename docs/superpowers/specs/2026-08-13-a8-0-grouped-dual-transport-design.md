# A8.0 train-only grouped dual-transport design

> Superseded addendum: la frase inferior “parent checkpoint is always A4 causal
> seed 7” describía un parent global y quedó invalidada por exposición de outer-dev.
> La ejecución válida usa A4-F0/F1/F2, todos seed 7 y entrenados solo en el train de
> su fold. Se conserva el texto original como provenance. Resultado:
> [Scientific Recovery V5](../../SCIENTIFIC_RECOVERY_V5_STATUS.md).

## Approval and scope

The user-provided V5 protocol is the approved design authority for A8.0. This specification narrows implementation choices; it does not change the preregistered hypothesis or gates.

A8.0 asks whether a separate trainable transport encoder can recover A5-like TTC while the A4 geometry encoder remains frozen. It keeps `causal_left`, local radius 1, temperature 0.02, residual bound 0.05, the existing geometry supervision and DINO relational teacher, event-only inference, and oracle ROI as preprocessing privilege. It does not add multiscale correlation, recurrence, ROI augmentation, confidence gating, new teachers, RGB inference, IMU, or bbox transport inputs.

## Considered implementations

1. Extend the canonical causal-scale runner with a signed grouped-development mode. This reuses the model, loss, optimizer, checkpoint selection, evaluation, and prediction paths. It needs a small dataset-view abstraction and explicit protocol validation. This is selected.
2. Create a separate A8 training runner by copying the existing runner. This isolates experiments but duplicates scientific logic and risks metric/checkpoint drift. Rejected.
3. Materialize six new fold-specific train/dev caches. This preserves sequential shards but adds expensive artifacts and another cache-provenance path without changing samples. Rejected.

## Data flow

The frozen protocol `configs/protocol/scientific_recovery_v5_train_only_grouped_dev.json` is loaded and its artifact signature, file SHA, source cache identity, fold index, sequence sets, row counts, and token hashes are validated before GPU allocation.

One base dataset opens only the `train` split of the 8192-row event cache. A single identity scan maps base indices to the selected fold's six train and three dev sequences. Two read-only indexed views preserve the base dataset's shard-local sampler groups. The DINO teacher wrapper is attached only to the train view. The dev view cannot expose teacher fields. Current public validation is not instantiated.

The model forward remains exactly:

```text
event_v4_common_roi + delta_t_s
    -> frozen primary geometry encoder
    -> trainable copied transport encoder
    -> causal local transport and bounded residual
    -> TTC/risk outputs
```

Bbox geometry and DINO relations are loss targets only. Sequence, track, token, bbox, RGB, teacher tensors, and TTC labels are not forward features.

## Training and outputs

Each fold is an independent sequential CUDA run with `num_workers=0`. The original,
superseded assumption reused one global A4 causal seed-7 parent; parent-exposure
auditing rejected it. The valid execution uses the fold-specific A4-Fk seed-7
parent. A8.0 seed identifies transport-training stochasticity conditional on that
fold parent; seed 7 is used in every fold so fold differences are not mixed with
seed differences.

The runner writes a grouped-dev summary, `dev_predictions.csv`, best/last checkpoints, progress, and effective configuration hashes. The summary must bind the protocol artifact/file hashes, fold, sequence/token contracts, parent checkpoint SHA, model/config SHA, git SHA, seed semantics, and closed public-validation/private-test state.

An aggregator computes mean and sample standard deviation across the three sequence-held-out folds. It reports MiD, failure, Pearson, geometry correlations, full coverage, parameter count, and per-fold values. It does not inspect current public validation.

## Geometry and causality evidence

For each fold, geometry protection requires all of the following:

- primary geometry parameters have `requires_grad=false` and are absent from optimizer groups;
- primary geometry state tensors are byte-exact before and after training;
- primary geometry state SHA equals the parent encoder state SHA;
- fixed-probe geometry outputs are numerically invariant before and after training;
- reported dev geometry metrics are included but do not substitute for state equality.

The existing prefix audit is extended to the dual-stream model and asserts invariance for geometry, transport features, bounded residual, pair TTC, and fused output when arbitrary future endpoints are appended or changed. This supports `model-prefix-causal`, not strict end-to-end non-oracle streaming causality.

## Gates and stopping rule

A8.0 promotion uses only the frozen grouped-dev aggregate:

- hard contracts: event-only forward, oracle ROI only as preprocessing, `causal_left`, exact geometry freeze, prefix causality, complete finite fold metrics, no public validation selection, private/test closed;
- first-stage TTC gate: mean MiD <= 175;
- strong target: mean MiD <= 160;
- descriptive aspiration only: MiD < 144.9;
- A8.0 must improve the A6 grouped-dev comparator under the same folds and budget before promotion.

If a required operational artifact is missing, the run is `BLOCKED`, not a scientific failure. If training completes and misses the TTC gate, A8.0 is `FAIL` or `INCONCLUSIVE` according to fold stability; A8.1 is not automatically promoted.

## Test plan

- protocol signature/hash/fold mismatch rejection;
- sequence-view disjointness, exhaustiveness, token hashes, and shard-group partitioning;
- validation/dev batches reject teacher fields;
- grouped mode never constructs the public-validation split;
- effective config and parent checkpoint identity are recorded;
- optimizer excludes geometry parameters;
- primary geometry tensor SHA and fixed-probe output invariance;
- dual-stream prefix causality at branch and fused outputs;
- synthetic manual metric and fold aggregation cases;
- CPU smoke training from a tiny synthetic dataset before any CUDA fold.
