# Local Results

Generated from local ignored artifacts under `artifacts/`.

These numbers are local smoke evidence, not a benchmark claim. The split contains only
three EvTTC `CCRs-1` speed sequences: train=`low-100`, validation=`medium-100`,
test=`high-100`. Lower MAE is better.

## Dataset And Caches

- Indexed windows: 1230 total; train 335, validation 418, test 477.
- Normalized voxel cache:
  `artifacts/features/evttc_voxel_160x90_b5.npz`, shape `[1230, 10, 90, 160]`,
  build time 228.428 s.
- Raw+metadata voxel cache:
  `artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz`, shape `[1230, 12, 90, 160]`,
  build time 227.292 s.
- Mean events/window: 816610.148.

## Results

| Method | Protocol | Validation MAE | Test MAE | Notes |
| --- | --- | ---: | ---: | --- |
| Constant mean TTC | `ttc.csv` rows | 2.197 (n=900) | 3.512 (n=1150) | Train split mean target. |
| Event-rate ridge | indexed windows | 2.338 (n=418) | 2.794 (n=477) | log count/rate features. |
| TinyCNN normalized voxel | indexed windows | 3.692 (n=418) | 3.364 (n=477) | seed 42. |
| TinyCNN raw+metadata | indexed windows | 2.946 +/- 0.713 | 3.283 +/- 0.271 | 5 seeds; best validation seed 7. |
| TinyCNN raw+metadata best-val | indexed windows | 1.877 | 2.886 | seed 7, best epoch 7. |
| Masked JEPA train-only + TinyCNN | indexed windows | 3.297 | 3.290 | Same-window masked objective; self-supervised on train split only. |
| Temporal JEPA train-only + TinyCNN | indexed windows | 1.518 | 3.183 | Multi-horizon future embedding objective; self-supervised on train only. |
| Temporal JEPA frozen probe | indexed windows | 1.916 | 2.911 | Only TTC head trained after JEPA pretraining. |
| JEPA all-splits + TinyCNN | indexed windows | 3.598 | 3.351 | Diagnostic only; uses validation/test events without labels. |
| Causal geometry calibrated | detection-assisted labeled frames | 0.439 (n=143) | 0.188 (n=96) | Uses current/past boxes only; calibration fit on train labels only. |
| Centered geometry diagnostic | labeled frames only | 0.680 (n=145) | 0.203 (n=98) | Non-causal centered derivative; not a valid claim. |

## Low-Label Results

These runs use the same train sequence but restrict supervised TTC labels. The temporal
JEPA encoder is pretrained on train events only, without TTC labels.

| Labels | Method | Seeds | Validation MAE | Test MAE | Notes |
| --- | --- | --- | ---: | ---: | --- |
| 5% (17 windows) | TinyCNN scratch | 7,13,21 | 2.909 +/- 0.743 | 3.107 +/- 0.277 | Random train-label subset per seed. |
| 5% (17 windows) | Temporal JEPA + fine-tune | 7,13,21 | 1.548 +/- 0.176 | 2.986 +/- 0.106 | Same label subsets; train-only SSL encoder. |
| 10% (34 windows) | TinyCNN scratch | 7 | 1.842 | 3.159 | Single-seed check. |
| 10% (34 windows) | Temporal JEPA + fine-tune | 7 | 1.813 | 3.007 | Single-seed check. |

## Partial Starter Exploratory

This protocol adds the downloaded `CCRs-side-low` HDF5+TTC sequence to train while
keeping the original validation/test sequences. It is useful for stress testing
domain shift, but it is not a sealed final protocol.

| Method | Train labels | Validation MAE | Test MAE | Notes |
| --- | ---: | ---: | ---: | --- |
| Event-rate ridge | 100% | 2.816 | 2.970 | Fit on CCRs-1-low + CCRs-side-low. |
| TinyCNN scratch | 100% | 3.047 | 2.798 | Raw+metadata partial-starter cache. |
| Temporal JEPA + fine-tune | 100% | 2.468 | 2.802 | Best validation of lr 3e-4/1e-4. |
| TinyCNN scratch | 5% | 1.299 | 3.091 | 38 labeled train windows. |
| Temporal JEPA + fine-tune | 5% | 1.522 | 2.939 | Best validation of lr 3e-4/1e-4. |
| Temporal JEPA diagnostic | 5% | 2.040 | 2.489 | Not validation-selected; included because test shift response is notable. |

## Anti-Leakage Audit

- The `causal_geometry_baseline.json` run reports `uses_future_bboxes=false`,
  `uses_future_events=false`, and `uses_validation_or_test_ttc_for_fit=false`.
- Its derivative at each labeled frame is fitted from that frame and earlier labeled
  frames only. The log-affine calibration uses train split labels only.
- It is detection-assisted, not event-only: it assumes an external detector or tracker
  provides current/past object boxes at inference.
- The older centered geometric baseline is retained only as a diagnostic and is marked
  non-causal because it uses future boxes inside the derivative window.
- The temporal JEPA run reports `uses_ttc_labels=false`; future event windows are used
  only as self-supervised targets and never cross sequence or split boundaries.
- Low-label subsets are sampled only from the train split. Validation is used for
  checkpoint selection; the mini test split has been inspected repeatedly and is
  therefore exploratory rather than a sealed final test.
- The partial starter runs add only `CCRs-side-low` to train. Remaining starter HDF5
  downloads were blocked by Google Drive/gdown access limits during this run.

## JEPA Diagnostics

| Pretrain scope | Best epoch | Best loss | Last train target std | Last validation target std | Downstream validation MAE | Downstream test MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| masked train only | 109 | 0.015 | 0.336 | 0.010 | 3.297 | 3.290 |
| temporal train only | 30 | 0.005 | 0.684 | 0.079 | 1.518 | 3.183 |
| masked train+validation+test | 158 | 0.011 | 0.700 | 0.098 | 3.598 | 3.351 |

## Conclusion

1. The strongest leakage-safe local result is the causal detection-assisted geometry
   model: validation MAE 0.439 s and test MAE 0.188 s on labeled frames.
   It is promising, but it is not an event-only model because it requires object boxes.
2. Temporal multi-horizon JEPA is the first positive self-supervised result: with
   only 5% train labels, validation MAE improves from 2.909 +/- 0.743 s to
   1.548 +/- 0.176 s across three seeds, and test mean improves modestly from
   3.107 +/- 0.277 s to 2.986 +/- 0.106 s.
3. With 100% labels, temporal JEPA improves validation MAE over the matching
   TinyCNN seed 7 run (1.518 s vs 1.877 s), but it does not beat the event-rate
   baseline on the repeatedly inspected high-speed mini test split.
4. On the full-label indexed event-window protocol, event-rate ridge remains the
   strongest robust held-out result among pure event-window models.
5. The CNN needs raw density information: normalized voxels underperform.
   Raw+metadata improves sharply and can beat event-rate on validation for one seed,
   but the five-seed mean remains behind event-rate on test and has high variance.
6. With one training sequence, there is still not enough evidence to claim learned
   visual event representations generalize across speeds. Adding only CCRs-side-low
   gives mixed results: JEPA improves partial full-label validation over scratch, but
   the partial low-label validation-selected model still does not beat scratch.
7. The next meaningful step is the full EvTTC starter subset with a fresh sealed test
   protocol; gdown retrieved one extra HDF5 but then hit Drive access limits.

## Reproduction

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m e_jepa_ttc cache voxel --manifest data/manifests/evttc_local.yaml --split data/splits/evttc_local.yaml --index data/cache/evttc_index.json --output artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz --width 160 --height 90 --bins 5 --no-normalize --metadata-channels
.\.venv\Scripts\python.exe -m e_jepa_ttc train tiny-cnn --cache artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz --output-dir artifacts/runs/tiny_cnn_voxel_160x90_b5_raw_meta_seed7 --epochs 80 --batch-size 96 --learning-rate 0.0003 --seed 7 --device auto
.\.venv\Scripts\python.exe -m e_jepa_ttc baseline causal-geometry --manifest data/manifests/evttc_local.yaml --split data/splits/evttc_local.yaml --output artifacts/metrics/causal_geometry_baseline.json --derivative-window 15
.\.venv\Scripts\python.exe -m e_jepa_ttc pretrain jepa --cache artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz --output-dir artifacts/runs/jepa_temporal_voxel_160x90_b5_raw_meta_train_seed7 --epochs 160 --batch-size 64 --learning-rate 0.0005 --seed 7 --device auto --pretrain-splits train --validation-splits validation --temporal-horizons-ms 20 60 100 240 500 --max-target-slop-ms 10 --variance-weight 1.0 --min-std 0.05
.\.venv\Scripts\python.exe -m e_jepa_ttc train tiny-cnn --cache artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz --output-dir artifacts/runs/tiny_cnn_voxel_160x90_b5_raw_meta_temporal_jepa_seed7 --epochs 80 --batch-size 96 --learning-rate 0.0003 --seed 7 --device auto --pretrained-encoder artifacts/runs/jepa_temporal_voxel_160x90_b5_raw_meta_train_seed7/jepa_encoder_best.pt
.\.venv\Scripts\python.exe scripts/write_local_results.py --output docs/local_results.md
```
