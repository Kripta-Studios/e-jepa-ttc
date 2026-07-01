# Full Starter Sealed Results

Generated on 2026-07-01 after adding the two pending CPLA HDF5 files.

This is the strongest local sealed protocol so far. The split was committed
before these final runs: previously inspected CCRs-1 sequences are train-only,
validation is `CCRs-side-high`, and the sealed test sequence is `CPLA-high`.

## Dataset

- Manifest: `data/manifests/evttc_full_starter_local.yaml`
- Split: `data/splits/evttc_full_starter_sealed.yaml`
- Index: `data/cache/evttc_full_starter_index.json` (ignored generated file)
- Cache: `artifacts/features/evttc_full_starter_voxel_160x90_b5_raw_meta.npz`
- Windows: 3972 total; train 3019, validation 475, test 478.
- Cache shape: `[3972, 12, 90, 160]`
- Mean events/window: 950477.998

Split:

- Train: `CCRs-1-low-100-overlap-100`,
  `CCRs-1-medium-100-overlap-100`,
  `CCRs-1-high-100-overlap-100`, `CCRs-side-low`,
  `CCRs-side-medium`, `CPLA-low`, `CPLA-medium`
- Validation: `CCRs-side-high`
- Sealed test: `CPLA-high`

## Token JEPA

The full-starter JEPA run uses a token transformer encoder and the current
default objective, `dense_temporal_token_motion_multihorizon`:

- patch-token transformer backbone instead of the earlier TinyCNN-only encoder;
- dense future token prediction instead of only global latent prediction;
- causal context-motion conditioning from event mass, temporal mass slope,
  centroid shift, and polarity balance;
- future horizons at 20, 60, 100, 240, and 500 ms;
- train events only for SSL pretraining and validation events only for SSL
  checkpoint selection;
- no TTC labels during pretraining.

Pretraining result:

| Run | Best Epoch | Best SSL Loss | Last Train Target Std | Last Val Target Std |
| --- | ---: | ---: | ---: | ---: |
| Token JEPA seed 7 | 56 | 0.003049 | 1.162 | 0.951 |

Leakage audit from the run:

- `uses_ttc_labels=false`
- `supervised_labels_reserved_for_finetune_only=true`
- `target_timestamps_are_after_context=true`
- `targets_cross_sequence_boundary=false`
- `targets_cross_split_boundary=false`
- `motion_conditioning_uses_context_only=true`

## Metrics

Lower MAE is better. Test is the sealed `CPLA-high` sequence and was not used
for model or hyperparameter selection in this protocol.

| Method | Train labels | Seeds | Validation MAE | Test MAE |
| --- | ---: | --- | ---: | ---: |
| Event-rate ridge | 100% | deterministic | 2.303 | 2.489 |
| TinyCNN scratch | 100% | 7 | 0.549 | 0.513 |
| Token transformer scratch | 100% | 7,13,21 | 0.702 +/- 0.052 | 0.844 +/- 0.008 |
| Token JEPA + fine-tune | 100% | 7,13,21 | 0.358 +/- 0.007 | 0.481 +/- 0.042 |
| Deep Token JEPA + fine-tune | 100% | 7 | 0.491 | 0.594 |
| Deep layer-aware Token JEPA + fine-tune | 100% | 7 | 0.472 | 0.505 |
| Large Token JEPA + fine-tune | 100% | 7 | 0.504 | 0.529 |
| Token transformer scratch | 5% | 7,13,21 | 1.226 +/- 0.031 | 1.382 +/- 0.044 |
| Token JEPA + fine-tune | 5% | 7,13,21 | 0.524 +/- 0.047 | 0.636 +/- 0.109 |
| Token transformer scratch | 10% | 7,13,21 | 1.178 +/- 0.056 | 1.327 +/- 0.104 |
| Token JEPA + fine-tune | 10% | 7,13,21 | 0.437 +/- 0.039 | 0.460 +/- 0.029 |

Percent improvements over matching scratch runs:

- 5% labels: validation MAE improves 57.3%; sealed-test MAE improves 53.9%.
- 10% labels: validation MAE improves 62.9%; sealed-test MAE improves 65.4%.
- 100% labels, same token backbone over three seeds: validation MAE improves
  49.0%; sealed-test MAE improves 43.0%.
- 100% labels versus TinyCNN scratch seed 7: three-seed Token JEPA validation
  MAE improves 34.7%; sealed-test MAE improves 6.2%. The best Token JEPA seed
  improves sealed-test MAE by 17.8%.
- 100% labels versus event-rate ridge: validation MAE improves 84.8%;
  sealed-test MAE improves 83.1%.

## Interpretation

The strongest local result is now `Token JEPA + fine-tune` with full labels:
`0.358 +/- 0.007 s` validation MAE and `0.481 +/- 0.042 s` sealed-test MAE over
three fine-tuning seeds. The best single seed reaches `0.350 s` validation MAE
and `0.422 s` sealed-test MAE. It beats the same token transformer trained from
scratch robustly, and the three-seed mean is slightly better than the supervised
TinyCNN full-label seed-7 baseline.

The low-label result is the clearest JEPA signal. With only 10% of train labels,
Token JEPA reaches `0.460 s` test MAE, which is better than the full-label
TinyCNN scratch baseline (`0.513 s`). With 5% labels it remains much better than
the matching scratch token model, but does not beat the full-label TinyCNN.

Deep self-supervision was implemented and tested as a SOTA-alignment ablation:

- `Deep Token JEPA`: supervised transformer layers 1 and 3 with the same dense
  predictor. SSL best loss `0.003523`; fine-tune test MAE `0.594 s`.
- `Deep layer-aware Token JEPA`: added a predictor layer-id embedding for layers
  1 and 3. SSL best loss `0.003933`; fine-tune test MAE `0.505 s`.

Both deep variants beat the scratch token transformer, and the layer-aware
variant roughly matches TinyCNN scratch, but neither beats the simpler final-layer
Token JEPA. For the current local dataset and model size, deep supervision is an
implemented negative ablation rather than a new best result.

Backbone scaling was also tested with `token-transformer-large`
(`embed_dim=256`, `depth=6`, `heads=8`). SSL best loss was `0.003237`, close to
but worse than the base token transformer's `0.003049`; fine-tune reached
`0.529 s` sealed-test MAE. This suggests the current full-starter data size and
objective do not yet benefit from a larger encoder without additional
regularization, data, or a stronger predictor.

This is not an official SOTA claim. The run is a local starter protocol, not a
published EvTTC leaderboard comparison. It is, however, the first local result
that is both sealed-protocol positive and directionally aligned with current
dense JEPA world-model practice.

## SOTA Position

Compared with current JEPA/world-model SOTA as of 2026-07-01:

- Aligned: latent prediction, EMA target encoder, future multi-horizon
  prediction, dense token loss, motion conditioning, no TTC-label leakage,
  low-label transfer evaluation.
- Still below SOTA: small local training scale, shallow token transformer, deep
  self-supervision currently negative in ablation, event-only input, no
  RGB/LiDAR/depth/box fusion, no action-conditioned planning or closed-loop
  evaluation, and no official benchmark replication.

Practical claim:

> On the local full-starter sealed EvTTC protocol, dense motion-conditioned
> token JEPA substantially improves label efficiency and gives the best local
> TTC MAE. It is a strong starter result, not yet a SOTA world-model result.
