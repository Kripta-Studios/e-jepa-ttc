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
| JEPA train-only + TinyCNN | indexed windows | 3.297 | 3.290 | Self-supervised on train split only; best of lr 3e-4/1e-4. |
| JEPA all-splits + TinyCNN | indexed windows | 3.598 | 3.351 | Diagnostic only; uses validation/test events without labels. |
| Causal geometry calibrated | detection-assisted labeled frames | 0.439 (n=143) | 0.188 (n=96) | Uses current/past boxes only; calibration fit on train labels only. |
| Centered geometry diagnostic | labeled frames only | 0.680 (n=145) | 0.203 (n=98) | Non-causal centered derivative; not a valid claim. |

## Anti-Leakage Audit

- The `causal_geometry_baseline.json` run reports `uses_future_bboxes=false`,
  `uses_future_events=false`, and `uses_validation_or_test_ttc_for_fit=false`.
- Its derivative at each labeled frame is fitted from that frame and earlier labeled
  frames only. The log-affine calibration uses train split labels only.
- It is detection-assisted, not event-only: it assumes an external detector or tracker
  provides current/past object boxes at inference.
- The older centered geometric baseline is retained only as a diagnostic and is marked
  non-causal because it uses future boxes inside the derivative window.

## JEPA Diagnostics

| Pretrain scope | Best epoch | Best loss | Last train target std | Last validation target std | Downstream validation MAE | Downstream test MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train only | 109 | 0.015 | 0.336 | 0.010 | 3.297 | 3.290 |
| train+validation+test | 158 | 0.011 | 0.700 | 0.098 | 3.598 | 3.351 |

## Conclusion

1. The strongest leakage-safe local result is the causal detection-assisted geometry
   model: validation MAE 0.439 s and test MAE 0.188 s on labeled frames.
   It is promising, but it is not an event-only model because it requires object boxes.
2. On the indexed event-window protocol, the event-rate ridge baseline is the
   strongest robust result on the held-out high-speed sequence.
3. The CNN needs raw density information: normalized voxels underperform.
   Raw+metadata improves sharply and can beat event-rate on validation for one seed,
   but the five-seed mean remains behind event-rate on test and has high variance.
4. JEPA/self-supervised pretraining is implemented and runs on GPU, but this local
   train-only run does not improve downstream TTC. Even the all-splits diagnostic is
   worse than the non-pretrained TinyCNN seed 7, so the limiting factor is not just
   access to unlabeled validation/test event windows.
5. With one training sequence, there is still not enough evidence to claim learned
   visual event representations generalize across speeds. The next meaningful step is
   a larger EvTTC subset and stronger multi-horizon JEPA rather than this tiny run.

## Reproduction

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m e_jepa_ttc cache voxel --manifest data/manifests/evttc_local.yaml --split data/splits/evttc_local.yaml --index data/cache/evttc_index.json --output artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz --width 160 --height 90 --bins 5 --no-normalize --metadata-channels
.\.venv\Scripts\python.exe -m e_jepa_ttc train tiny-cnn --cache artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz --output-dir artifacts/runs/tiny_cnn_voxel_160x90_b5_raw_meta_seed7 --epochs 80 --batch-size 96 --learning-rate 0.0003 --seed 7 --device auto
.\.venv\Scripts\python.exe -m e_jepa_ttc baseline causal-geometry --manifest data/manifests/evttc_local.yaml --split data/splits/evttc_local.yaml --output artifacts/metrics/causal_geometry_baseline.json --derivative-window 15
.\.venv\Scripts\python.exe -m e_jepa_ttc pretrain jepa --cache artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz --output-dir artifacts/runs/jepa_voxel_160x90_b5_raw_meta_train_seed7 --epochs 160 --batch-size 128 --learning-rate 0.0005 --seed 7 --device auto --pretrain-splits train --validation-splits validation --variance-weight 1.0 --min-std 0.05
.\.venv\Scripts\python.exe -m e_jepa_ttc train tiny-cnn --cache artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz --output-dir artifacts/runs/tiny_cnn_voxel_160x90_b5_raw_meta_jepa_seed7 --epochs 80 --batch-size 96 --learning-rate 0.0003 --seed 7 --device auto --pretrained-encoder artifacts/runs/jepa_voxel_160x90_b5_raw_meta_train_seed7/jepa_encoder_best.pt
.\.venv\Scripts\python.exe scripts/write_local_results.py --output docs/local_results.md
```
