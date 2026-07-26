# FlowMimic physical-approach experiment

Updated: 2026-07-26.

## Research question

Does a physically constrained synthetic approach prior improve the causal
event-tubelet JEPA encoder for TTC estimation, especially without exposing real
TTC labels during pretraining?

This is an adaptation inspired by FlowMimic, not a reproduction of its video
editing results. FlowMimic does not evaluate event cameras or TTC.

## Metric clarification

The historical local result `0.312 +/- 0.044 s` is TTC mean absolute error
(MAE): the prediction differs from the reference TTC by 312 ms on average. The
historical scratch result `0.465 +/- 0.021 s` is the same type of error under the
local paired protocol, a 32.9% lower MAE for JEPA.

The `13 ms` reported for Garl-TTC is inference latency, not TTC error. Dividing
312 ms of error by 13 ms of runtime has no scientific meaning. The corresponding
Garl-TTC accuracy number cited in the audit is `10.60%` relative TTC error on its
official protocol; the historical local JEPA number is `6.42 +/- 0.45%` MARE on
CPLA-high. Those accuracy values are also not directly comparable because the
sequence set, ROI/bbox assistance and evaluation protocol differ.

Two separate comparisons are required:

1. accuracy: same sequences, samples, labels and MAE/RTE definition;
2. efficiency: batch size 1 on the same hardware, with preprocessing and model
   latency reported separately.

## Implemented hypothesis

For a constant-speed frontal approach with TTC `T` at the context reference
time, apparent pinhole scale at relative time `delta` is:

```text
s(delta) = T / (T - delta)
```

The simulator follows this causal order:

```text
analytic object motion
  -> log-intensity frames
  -> accumulated contrast-threshold crossings
  -> positive/negative voxel bins
  -> optional cache-compatible normalization/metadata
```

It never warps an already accumulated voxel grid. Sub-threshold contrast is
carried relative to the last emitted-event reference, matching the essential
event-camera mechanism more closely than independent frame differencing.

The pretraining additions are independently weighted:

- `flowmimic_alignment_weight`: JEPA context-to-future latent alignment on
  synthetic event windows;
- `flowmimic_inverse_ttc_weight`: a positive inverse-TTC auxiliary head trained
  only from the analytic synthetic trajectory.

The auxiliary head is present only during pretraining. The downstream TTC head
is still fitted through the existing supervised protocol.

## Scientific-integrity boundary

- No value from cache field `y_ttc` is read by the FlowMimic path.
- Real future events remain the main EMA-target JEPA signal.
- Synthetic TTC is recorded explicitly as analytic supervision, not described
  as unlabeled real-data supervision.
- Future navigation is zeroed for target embeddings as in the existing JEPA.
- Synthetic generation runs only in the training loop; model selection remains
  based on the unchanged real validation loss.
- CPLA-high is a reused diagnostic test and must not be opened for this
  architecture selection.

Relevant implementation:

- `src/e_jepa_ttc/representations/flowmimic.py`
- `src/e_jepa_ttc/training/jepa.py`
- `src/e_jepa_ttc/cli.py`
- `tests/unit/test_flowmimic.py`
- `tests/unit/test_jepa_training.py`

## Decisive validation-only matrix

Use the same cache v2, architecture, seed, batches, horizons and optimizer for
all rows:

| ID | Synthetic latent alignment | Synthetic inverse-TTC | Purpose |
| --- | ---: | ---: | --- |
| E0 | 0 | 0 | clean event-tubelet JEPA control |
| E1 | >0 | 0 | isolate synthetic future alignment |
| E2 | same as E1 | >0 | test the physics/TTC prior |

Execution is staged to avoid spending three full seeds on a bad idea:

1. one-seed short pretrain plus identical validation-only fine-tune for E0-E2;
2. promote only a variant that improves validation MAE and does not collapse;
3. rerun the promoted pair with seeds 7, 13 and 21 at the full schedule;
4. only after freezing the choice, obtain a genuinely unopened holdout or the
   complete official protocol.

Primary promotion metric: validation MAE in seconds. Secondary diagnostics:
relative error, RMSE, inverse-TTC auxiliary loss, embedding effective rank and
latency. SSL loss alone cannot establish TTC improvement.

## Current status

- Scientific-provenance hardening published in commit `416b498`.
- FlowMimic implementation published in commit `647b340`.
- Simulator unit tests: passing.
- FlowMimic/JEPA integration smoke: passing.
- Existing JEPA/prober focused suite: 21 tests passing.
- Full repository QA after the multiseed/robustness hardening: Ruff and format
  checks passing; 208 tests passing, including sequence identity, paired
  bootstrap, raw-event cache corruptions and frozen-gate command guards.
- Cache v2 train+validation rebuild and exhaustive audit: passed.
- Single-seed E0/E1/E2 validation pilot: complete; E1 selected for the
  three-seed/full-schedule gate, but not yet promotable.

## Continuation checklist

1. Execute the now-frozen E0/E1 gate with independent paired SSL/downstream
   seeds 7, 13 and 21 at 30 SSL epochs; the eight-epoch pilot selected both best
   checkpoints at the schedule boundary.
2. Generate sequence-aware predictions, paired bootstrap and raw-event
   robustness with the committed gate tooling.
3. Keep CPLA-high closed; obtain a genuinely unopened holdout or complete the
   official EvTTC protocol after architecture freeze.

Exact settings, failure conditions and artifact fields are frozen in
`docs/flowmimic_multiseed_protocol_2026-07-26.md` and
`configs/experiment/flowmimic_e0_e1_multiseed.yaml`.

## Experiment ledger

### Rejected cache build C0

Command executed from commit `647b340`:

```powershell
uv run --no-sync e-jepa-ttc cache voxel `
  --manifest data/manifests/evttc_full_starter_local.yaml `
  --split data/splits/evttc_full_starter_sealed.yaml `
  --exclude-split test `
  --index data/cache/evttc_full_starter_index.json `
  --output artifacts/features/evttc_trainval_v2_voxel_160x90_b5_raw_meta_nav.npz `
  --width 160 --height 90 --bins 5 --no-normalize `
  --metadata-channels --navigation-channels
```

Observed output:

- elapsed: `928.071 s`;
- SHA-256: `077c78b5bdaef5d8c5574bc8537b464690e78896feb4bde4338ade828e2fba89`;
- physical/declared shape: `[3972, 21, 90, 160]`;
- status: **rejected before training** because all 3,972 windows remained,
  including 478 `test` windows.

Root cause: the CLI forwarded `exclude_splits`, but `build_voxel_cache()` did
not apply it. The correction filters windows before tensor allocation, rejects
unknown split names, includes exclusions in the preprocessing hash and records
them in NPZ/summary/sidecar metadata. A regression test verifies that `test` is
physically absent.

No model was trained from C0. Its numbers are infrastructure facts, not model
accuracy results.

### Accepted cache build C1

The same build command was repeated from the corrected split-filter commit
`f62b268`. Generated facts:

- input index windows: `3972`;
- physical output windows: `3494`;
- shape: `[3494, 21, 90, 160]`, dtype `float16`;
- split counts: train `3019`, validation `475`, test `0`;
- sequence counts: seven train, one validation;
- explicitly excluded splits: `["test"]`;
- elapsed: `689.082 s`;
- cache SHA-256:
  `22d3ef27018925aae62825f0a7f51d1420ae93cacf59aeb18b04758f5a35e88a`.

The cache audit was hardened and published as `80ff992`, then rerun from that
commit. Exhaustive audit artifact:

```text
artifacts/metrics/flowmimic_cache_trainval_v2_audit.json
```

Audit facts:

- status: `passed`;
- code commit: `80ff9923d085381d6a08644686cfe3cdf4a23bd3`;
- audited samples: `3494/3494`;
- cache format: v2;
- hash match: true;
- nonempty windows collapsed to zero: `0`;
- encoded all-zero samples: `0`;
- normalization: `none`;
- audit artifact SHA-256:
  `02f3f633b13f413c4bf6b49176c3e70d373af52d63ac1602e119402af3a819c2`.

This cache is eligible for E0/E1/E2 validation-only training. It is not itself
an accuracy result.

### Rejected GPU smoke S0

E2 was executed for one pretraining epoch with seed 7, batch 12, event-tubelet
encoder, transformer predictor, tubelet mask, alignment weight 0.25 and
inverse-TTC weight 0.10. Runtime was `118.977 s`; CUDA did not run out of
memory, but aggregate synthetic alignment and total train loss were `NaN`.

This run is rejected and no downstream model was trained. Root cause:

- real train navigation has `ego_navigation_valid=1` with near-zero standard
  deviation;
- the synthetic generator padded navigation with zero;
- train normalization mapped synthetic validity to approximately `-1e6`;
- AMP overflow propagated into the synthetic predictor.

Correction: set synthetic navigation channels to the train-only navigation
mean, which maps to neutral zero after normalization. Add a finite-loss guard
that raises `FloatingPointError` instead of writing a false successful run, and
test navigation neutrality explicitly. S0 is infrastructure evidence only, not
an accuracy result.

### Accepted GPU smoke S1

S0 was repeated from correction commit `07f7736` with the same seed, batch and
architecture. It completed without OOM or non-finite values:

- training elapsed: `90.779 s`;
- real validation SSL loss: `0.018911`;
- total train loss: `0.019014`;
- synthetic alignment loss: `0.001925`;
- synthetic inverse-TTC loss: `0.144244`;
- context/prediction/target effective rank: `4.193 / 9.972 / 12.216`;
- synthetic navigation conditioning: `train_mean_neutral_train_only`;
- real TTC labels used in SSL: false.

S1 establishes numerical viability for batch 12. It is not a TTC accuracy
result and its checkpoint will not be promoted. The main E0/E1/E2 runs will use
the subsequent provenance commit that places the physical cache hash, complete
resolved configuration, commit and run fingerprint in both checkpoint and
`metrics.json`.

### Rejected downstream provenance attempt D0/D1

Scratch and E0 were initially fine-tuned for 30 epochs, seed 7, batch 24 and LR
`3e-5`. Both used only train/validation. Diagnostic values were:

| Initializer | Validation MAE | MARE | RMSE |
| --- | ---: | ---: | ---: |
| scratch | `0.395920 s` | `12.8135%` | `0.499095 s` |
| E0 JEPA | `0.342204 s` | `9.8381%` | `0.498136 s` |

E0 was 13.6% lower in MAE, but both runs incorrectly received the same
`run_fingerprint`. Root cause: `checkpoint_provenance()` recorded path, role,
seed and epoch but not physical checkpoint SHA-256, while the supervised
fingerprint defaulted the missing hash to an empty string.

These two runs are rejected from the promotable matrix and will be repeated.
The correction hashes every pretrained checkpoint and includes batch size,
freeze state and requested train/validation/evaluation splits in the supervised
fingerprint. A regression assertion requires scratch and pretrained
fingerprints to differ.

### Accepted single-seed validation pilot P1

The pretraining matrix ran from commit `943f53f`; downstream runs with complete
checkpoint hashing ran from `810a935`. Fixed settings:

- seed 7 for SSL and downstream;
- cache SHA-256 `22d3ef...e88a`, physically train+validation only;
- event-tubelet transformer, dense transformer predictor, tubelet mask 0.45;
- horizons `[20, 60, 100, 240, 500] ms`;
- SSL: 8 epochs, batch 12, LR `3e-4`;
- downstream: full fine-tune, 30 epochs, batch 24, LR `3e-5`;
- selection: best real validation SSL checkpoint, then best validation TTC MAE;
- CPLA-high/final test: not present and not opened.

Generated signed result:

```text
artifacts/metrics/flowmimic_validation_pilot_seed7_summary.json
```

Summary code commit: `116a704`. Artifact SHA-256:
`5d8dbf26dc378601bb495dae2f86676565d25605bac5aa2a054cfc33dd0a93c3`.

| Variant | Synthetic alignment | Inverse-TTC | SSL val loss | Val MAE | MARE | RMSE | MAE vs E0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scratch | 0 | 0 | - | `0.389290 s` | `11.8730%` | `0.507569 s` | `-13.95%` |
| E0 JEPA | 0 | 0 | `0.012680` | `0.341637 s` | `9.8127%` | `0.497757 s` | reference |
| E1 alignment | 0.25 | 0 | `0.011522` | **`0.255227 s`** | **`8.3999%`** | **`0.332191 s`** | **`+25.29%`** |
| E2 alignment + inverse | 0.25 | 0.10 | `0.008622` | `0.325621 s` | `9.6676%` | `0.436310 s` | `+4.69%` |

Interpretation:

- clean JEPA E0 improves scratch by 12.24% MAE;
- physical FlowMimic alignment E1 improves E0 by 25.29% and scratch by 34.44%;
- inverse-TTC E2 achieves the best SSL loss but is 27.58% worse in TTC MAE than
  E1, showing that SSL loss is not a sufficient selection metric;
- the useful idea is synthetic physical future alignment, not the current
  inverse-TTC auxiliary head/weight;
- E1 costs `1050.4 s` of SSL versus `532.9 s` for E0, about 1.97x in this pilot;
  it does not increase inference cost because the synthetic branch is removed.

This is strong validation-pilot evidence, not a SOTA or final claim. It has one
SSL/downstream seed, CUDA is not bit-deterministic in this environment, and
both E0/E1 selected SSL epoch 8 at the pilot boundary. Promote only the E1
hypothesis to a full-schedule three-seed comparison against E0.

### Batch-1 model latency for selected E1

A signed synchronous benchmark was run from commit `c103dd9` with 50 warmups
and 300 measured iterations on the RTX 5070 Ti Laptop GPU. It loads the real E1
downstream checkpoint at epoch 22 and one real cache tensor `[1,21,90,160]`.

Artifact:

```text
artifacts/metrics/flowmimic_e1_seed7_batch1_model_latency.json
```

Artifact signature:
`b9e2ac55c69854603f1c418a21e760c7ff1475f71509777c64617c7403d31d28`.

Results:

- FP32 synchronous mean: `2.201 ms`;
- median: `2.096 ms`;
- p95 / p99: `2.779 / 3.411 ms`;
- maximum: `6.000 ms`;
- throughput derived from mean: `454.2 windows/s`;
- parameters: `2,884,417`;
- peak allocated device memory: `36,947,968 bytes`.

This is model-only latency and excludes HDF5/event reading and voxelization. It
cannot be compared directly with Garl-TTC's reported 13 ms without matching
hardware, precision, synchronization and preprocessing. It does establish that
the local `0.255 s` number is TTC error, not runtime: the selected neural model
itself runs in roughly 2.2 ms on this GPU.

## 2026 SOTA interpretation and next architecture

The selected E1 pilot is good local evidence: `0.2552 s` MAE and `8.3999%` MARE
on `CCRs-side-high` are materially better than the paired E0 and scratch
controls. They are not an official benchmark result. The current model is a
2.88M-parameter full-frame event-only tubelet transformer. Garl-TTC instead
uses object ROIs resized to 128x128, separate RGB/event ResNet-50 encoders,
explicit height-ratio TTC geometry and foreground-mask supervision during
training. Calling E1 “the SOTA architecture applied to our dataset” would
therefore be incorrect.

The primary [Garl-TTC/eAP paper](https://arxiv.org/html/2603.16303v1) reports
`10.60%` average RTE on three different EvTTC sequences without fine-tuning,
using RGB+events and object assistance. The numerical scale makes E1 an
encouraging diagnostic, but the metric aggregation, sequences, input modality,
assistance and hardware all differ. No ordering is valid until both methods run
on the same samples and protocol.

The most promising continuation is not a larger generic backbone. It is:

1. finish the E0/E1 stability gate;
2. retain physical future alignment;
3. add an object-centric causal branch with apparent-height/area expansion,
   foreground-boundary supervision available only during training and a
   differentiable height-ratio/looming head;
4. add ego-rotation compensation from causal IMU/navigation;
5. compare event-only and late RGB+event fusion as explicitly different
   operating points;
6. add a heteroscedastic or conformal head only after the deterministic
   robustness behavior is known.

This follows the useful inductive biases in Garl-TTC while preserving a fair
full-frame event-only branch. FlowMimic supplies synthetic physical motion but
is not itself a TTC benchmark
([FlowMimic](https://arxiv.org/abs/2607.18227)); SkyJEPA supports
spatiotemporal JEPA pretraining for driving but likewise does not establish
event-TTC accuracy ([SkyJEPA](https://arxiv.org/abs/2606.23444)).

## Data sufficiency

The current nine EvTTC starter sequences are sufficient for pipeline QA and
this local paired gate. They are not sufficient for a SOTA/generalization
claim: the accepted cache has seven train sequences and exactly one validation
sequence, while CPLA-high is already a reused diagnostic test. The eight local
eAP train sequences are useful for SSL and domain-shift pilots, but do not
reproduce the official eAP split, which contains 46 train and 12 test sequences
and roughly 174k annotated frames.

Priority data acquisition after the gate:

1. complete the official EvTTC evaluation sequences used by its published
   tables, especially the absent CCRs-2/CCRm families;
2. expand eAP from eight local train sequences toward the official 46-sequence
   train set and freeze its official 12-sequence test;
3. use a few DSEC sequences only as unlabeled SSL/domain-shift data;
4. reserve at least two or three complete, never-inspected sequences for model
   selection and a separate final holdout.

More independent scenarios are currently more valuable than increasing model
size.
