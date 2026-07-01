# Available Starter Sealed Results

Generated on 2026-07-01 from the locally complete EvTTC starter subset.

This is not the final full-starter protocol. Google Drive quota blocked
`CPLA-medium/data.hdf5` and `CPLA-high/data.hdf5`; the committed full-starter
split remains in `data/splits/evttc_full_starter_sealed.yaml` for the moment
those HDF5 files become available.

## Dataset

- Manifest: `data/manifests/evttc_available_starter_local.yaml`
- Split: `data/splits/evttc_available_starter_sealed.yaml`
- Index: `data/cache/evttc_available_starter_index.json` (ignored generated file)
- Cache: `artifacts/features/evttc_available_starter_voxel_160x90_b5_raw_meta.npz`
- Windows: 3030 total; train 1957, validation 598, test 475.
- Cache shape: `[3030, 12, 90, 160]`
- Mean events/window: 837519.672

Split:

- Train: `CCRs-1-low-100-overlap-100`, `CCRs-1-medium-100-overlap-100`,
  `CCRs-1-high-100-overlap-100`, `CCRs-side-low`, `CCRs-side-medium`
- Validation: `CPLA-low`
- Sealed test: `CCRs-side-high`

## Dense Motion JEPA

The current default JEPA objective is `dense_temporal_token_motion_multihorizon`:

- predicts dense future encoder tokens, not only global pooled embeddings;
- conditions the predictor on causal context-motion proxies from the current
  event window: event mass, temporal mass slope, centroid shift, and polarity
  balance;
- predicts future targets at 20, 60, 100, 240, and 500 ms;
- uses train events only for pretraining and validation events only for SSL
  checkpoint selection;
- does not use TTC labels during pretraining.

Pretraining result:

| Run | Best Epoch | Best SSL Loss | Last Train Target Std | Last Val Target Std |
| --- | ---: | ---: | ---: | ---: |
| Dense motion JEPA seed 7 | 18 | 0.004487 | 1.219 | 0.725 |

## Metrics

Lower MAE is better. Test is the newly added `CCRs-side-high` sequence and was
not used for model or hyperparameter selection in this protocol.

| Method | Train labels | Seeds | Validation MAE | Test MAE |
| --- | ---: | --- | ---: | ---: |
| Event-rate ridge | 100% | deterministic | 3.103 | 2.406 |
| TinyCNN scratch | 100% | 7 | 1.171 | 0.519 |
| Dense motion JEPA + fine-tune | 100% | 7 | 1.223 | 0.945 |
| Dense motion JEPA frozen probe | 100% | 7 | 1.002 | 1.197 |
| TinyCNN scratch | 5% | 7,13,21 | 2.413 +/- 0.480 | 2.241 +/- 0.350 |
| Dense motion JEPA + fine-tune | 5% | 7,13,21 | 1.694 +/- 0.174 | 1.149 +/- 0.119 |
| TinyCNN scratch | 10% | 7,13,21 | 1.814 +/- 0.152 | 1.400 +/- 0.035 |
| Dense motion JEPA + fine-tune | 10% | 7,13,21 | 1.420 +/- 0.075 | 1.111 +/- 0.370 |

Percent improvements over matching scratch runs:

- 5% labels: validation MAE improves 29.8%; sealed-test MAE improves 48.7%.
- 10% labels: validation MAE improves 21.7%; sealed-test MAE improves 20.6%.
- 100% labels: dense JEPA does not improve full fine-tuning versus scratch on
  this split; scratch remains the best test MAE.

## Interpretation

Dense motion-conditioned JEPA is useful as a label-efficiency pretraining method:
it substantially improves low-label performance on a newly sealed test sequence.
It is not yet a better full-label predictor than supervised scratch training.

The most defensible current claim is therefore:

> On the locally available sealed starter protocol, dense temporal token JEPA
> with causal motion conditioning improves low-label TTC estimation, especially
> at 5% labels, but does not beat full-label supervised training.

## Reproduction

```powershell
$env:PYTHONPATH='src'
$env:OMP_NUM_THREADS='32'
.\.venv\Scripts\python.exe -m e_jepa_ttc data scan --root datasets\evttc --output data\manifests\evttc_available_starter_local.yaml
.\.venv\Scripts\python.exe -m e_jepa_ttc data validate --manifest data\manifests\evttc_available_starter_local.yaml
.\.venv\Scripts\python.exe -m e_jepa_ttc data index --manifest data\manifests\evttc_available_starter_local.yaml --output data\cache\evttc_available_starter_index.json --context-ms 100 --stride-ms 20 --horizons-ms 20 60 100 240 500
.\.venv\Scripts\python.exe -m e_jepa_ttc cache voxel --manifest data\manifests\evttc_available_starter_local.yaml --split data\splits\evttc_available_starter_sealed.yaml --index data\cache\evttc_available_starter_index.json --output artifacts\features\evttc_available_starter_voxel_160x90_b5_raw_meta.npz --width 160 --height 90 --bins 5 --no-normalize --metadata-channels
.\.venv\Scripts\python.exe -m e_jepa_ttc pretrain jepa --cache artifacts\features\evttc_available_starter_voxel_160x90_b5_raw_meta.npz --output-dir artifacts\runs\jepa_dense_motion_available_starter_seed7 --epochs 120 --batch-size 64 --learning-rate 0.0005 --seed 7 --device auto --pretrain-splits train --validation-splits validation --temporal-horizons-ms 20 60 100 240 500 --max-target-slop-ms 10 --variance-weight 1.0 --min-std 0.05
```

