# FlowMimic physical-approach experiment

Updated: 2026-07-25.

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
- Full repository QA after the numerical guard: Ruff passing and 198 tests
  passing, including navigation neutrality.
- Cache v2 train+validation rebuild and exhaustive audit: passed.
- E0/E1/E2 validation results: pending; no result should be filled in manually.

## Continuation checklist

1. Run E0/E1/E2 without evaluating CPLA-high.
2. Append exact commands, commit hashes, artifact hashes and generated metrics
   paths below.

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

No training run has been completed yet. No metric in this section is a model
accuracy result.

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
