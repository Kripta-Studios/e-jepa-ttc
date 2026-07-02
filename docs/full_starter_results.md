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
- Navigation cache:
  `artifacts/features/evttc_full_starter_voxel_160x90_b5_raw_meta_nav.npz`
- Windows: 3972 total; train 3019, validation 475, test 478.
- Cache shape: `[3972, 12, 90, 160]`
- Navigation cache shape: `[3972, 21, 90, 160]`
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
- for runs created after 2026-07-02, navigation caches additionally enable
  predictor-level action conditioning from causal ego-motion/navigation
  features; older rows in the table used navigation as input channels but not
  as an explicit predictor action vector;
- runs created after 2026-07-02 can use `--dense-predictor transformer`,
  replacing the per-token MLP predictor with a horizon/action-conditioned
  transformer predictor over all dense tokens;
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

For a percentage-style reading, the current best
`Event tubelet JEPA + tubelet mask + transformer predictor` has mean absolute
relative error `8.19 +/- 0.53%` on validation and `6.42 +/- 0.45%` on the
sealed CPLA-high test over seeds 7/13/21.

| Method | Train labels | Seeds | Validation MAE | Test MAE |
| --- | ---: | --- | ---: | ---: |
| Event-rate ridge | 100% | deterministic | 2.303 | 2.489 |
| TinyCNN scratch | 100% | 7 | 0.549 | 0.513 |
| Token transformer scratch | 100% | 7,13,21 | 0.702 +/- 0.052 | 0.844 +/- 0.008 |
| Token JEPA + fine-tune | 100% | 7,13,21 | 0.358 +/- 0.007 | 0.481 +/- 0.042 |
| Token transformer + navigation scratch | 100% | 7,13,21 | 0.440 +/- 0.020 | 0.465 +/- 0.021 |
| Token JEPA + navigation fine-tune | 100% | 7,13,21 | 0.261 +/- 0.021 | 0.356 +/- 0.022 |
| Event tubelet JEPA + navigation fine-tune | 100% | 7,13,21 | 0.243 +/- 0.007 | 0.328 +/- 0.030 |
| Event tubelet JEPA + transformer predictor | 100% | 7,13,21 | 0.241 +/- 0.004 | 0.351 +/- 0.004 |
| Event tubelet JEPA + tubelet mask + transformer predictor | 100% | 7,13,21 | 0.231 +/- 0.018 | 0.312 +/- 0.044 |
| Deep Token JEPA + fine-tune | 100% | 7 | 0.491 | 0.594 |
| Deep layer-aware Token JEPA + fine-tune | 100% | 7 | 0.472 | 0.505 |
| Large Token JEPA + fine-tune | 100% | 7 | 0.504 | 0.529 |
| Token transformer scratch | 5% | 7,13,21 | 1.226 +/- 0.031 | 1.382 +/- 0.044 |
| Token JEPA + fine-tune | 5% | 7,13,21 | 0.524 +/- 0.047 | 0.636 +/- 0.109 |
| Token transformer scratch | 10% | 7,13,21 | 1.178 +/- 0.056 | 1.327 +/- 0.104 |
| Token JEPA + fine-tune | 10% | 7,13,21 | 0.437 +/- 0.039 | 0.460 +/- 0.029 |
| Token transformer + navigation scratch | 10% | 7,13,21 | 0.578 +/- 0.026 | 0.756 +/- 0.018 |
| Token JEPA + navigation fine-tune | 10% | 7,13,21 | 0.406 +/- 0.019 | 0.543 +/- 0.010 |

Percent improvements over matching scratch runs:

- 5% labels: validation MAE improves 57.3%; sealed-test MAE improves 53.9%.
- 10% labels: validation MAE improves 62.9%; sealed-test MAE improves 65.4%.
- 100% labels, same token backbone over three seeds: validation MAE improves
  49.0%; sealed-test MAE improves 43.0%.
- 100% labels, navigation token JEPA versus navigation token scratch over three
  seeds: validation MAE improves 40.6%; sealed-test MAE improves 23.3%.
- Navigation token JEPA versus event-only token JEPA over three seeds:
  validation MAE improves 27.1%; sealed-test MAE improves 25.9%.
- Navigation token JEPA versus event-only token scratch over three seeds:
  validation MAE improves 62.8%; sealed-test MAE improves 57.8%.
- 10% labels with navigation: navigation improves scratch sealed-test MAE by
  43.0%, and navigation JEPA improves over navigation scratch by 28.1%. However,
  event-only JEPA remains better than navigation JEPA at 10% labels on sealed
  test (`0.460 s` vs `0.543 s`).
- 100% labels versus TinyCNN scratch seed 7: three-seed Token JEPA validation
  MAE improves 34.7%; sealed-test MAE improves 6.2%. Three-seed navigation Token
  JEPA improves sealed-test MAE by 30.5%.
- 100% labels versus event-rate ridge: validation MAE improves 84.8%;
  sealed-test MAE improves 83.1%.

## Interpretation

The strongest robust local all-window result is now `Event tubelet JEPA +
tubelet mask + transformer predictor` with full labels:
`0.231 +/- 0.018 s` validation MAE and `0.312 +/- 0.044 s` CPLA-high test MAE
over three fine-tuning seeds. It improves test MAE by 4.9% versus the previous
event-tubelet navigation JEPA mean (`0.328 s`), by 12.4% versus navigation Token
JEPA (`0.356 s`), and by 32.9% versus navigation token scratch (`0.465 s`).

The previous best robust result was `Token JEPA + navigation fine-tune`:
`0.261 +/- 0.021 s` validation MAE and `0.356 +/- 0.022 s` sealed-test MAE over
three fine-tuning seeds.

The navigation channels are causal integrated-navigation features from the
current context window only: ego speed, velocity components, acceleration
components, yaw-rate, and a validity flag. They do not use TTC labels or future
target windows.

The low-label result is the clearest JEPA signal. With only 10% of train labels,
Token JEPA reaches `0.460 s` test MAE, which is better than the full-label
TinyCNN scratch baseline (`0.513 s`). With 5% labels it remains much better than
the matching scratch token model, but does not beat the full-label TinyCNN.
Navigation channels help the 10% scratch model strongly, but do not improve the
10% JEPA result on this split; the best low-label JEPA result remains event-only.

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

The first V-JEPA-like tubelet run changes the tokenization rather than only
scaling width/depth. It treats event bins as a polarity-by-time tensor and uses
3D tubelet patching before the transformer, while causal metadata/navigation
channels are added as auxiliary spatial patch context. SSL pretraining selected
epoch 12 with validation loss `0.001577`. Fine-tuning seeds 7/13/21 reached
sealed-test MAEs `0.323`, `0.301`, and `0.360 s`.

After adding explicit predictor-level action conditioning, a validation-only
ablation was run without evaluating the sealed test. The action-conditioned
tubelet JEPA pretrain selected epoch 14 with SSL validation loss `0.0017955`.
Fine-tuning seeds 7/13/21 with `--evaluation-splits train validation` reached
validation MAEs `0.236`, `0.247`, and `0.258 s`, for
`0.247 +/- 0.009 s`. The previous tubelet navigation JEPA validation mean was
`0.243 +/- 0.005 s`, so the action-conditioned predictor is 1.8% worse by
validation MAE. Because it did not beat validation, the sealed test was not run
for this ablation.

Adding train-only normalization for the 15-D action vector improved validation
but did not improve the sealed test. With LR `1e-4`, the frozen
action-normalized protocol reached validation MAEs `0.227`, `0.218`, and
`0.230 s` over seeds 7/13/21, or `0.225 +/- 0.005 s`. This is 7.42% better than
the previous tubelet navigation JEPA validation mean. After freezing that
protocol, the sealed test was evaluated once without retraining via
`train evaluate`; sealed-test MAEs were `0.370`, `0.421`, and `0.331 s`, or
`0.374 +/- 0.037 s`. This is 14.06% worse than the previous best sealed-test
mean `0.328 +/- 0.024 s`, so the best local result remains
`Event tubelet JEPA + navigation fine-tune`.

After this negative sealed result, a harder dev split was added to avoid using
`CPLA-high` for further selection. It moves `CPLA-medium` out of train into a
new `validation_pedestrian` split and keeps `CCRs-side-high` as
`validation_car`. On this split, seed 7 action-normalized JEPA reaches
`0.402 s` MAE on `validation_car` and `0.828 s` on `validation_pedestrian`;
scratch reaches `0.420 s` and `0.822 s`. The weighted multi-validation MAE is
`0.613 s` for JEPA versus `0.619 s` scratch, so the harder protocol exposes a
pedestrian-transfer failure that architecture changes alone have not solved.

## Transformer Dense Predictor

SOTA audit on 2026-07-02 found that V-JEPA 2.1 adds dense predictive loss across
all tokens and deep self-supervision, while V-JEPA 2-AC and LeWorldModel push
latent world models toward action-conditioned prediction. The repo already had
dense token losses and causal action vectors; the missing predictor-side piece
was token-token interaction. `--dense-predictor transformer` adds a transformer
predictor over dense tokens for each future horizon, conditioned by the same
train-normalized causal event/navigation action vector.

Dev multi-validation result, selected without evaluating sealed test:

| Method | Weighted multival MAE | validation_car MAE | validation_pedestrian MAE |
| --- | ---: | ---: | ---: |
| Scratch event-tubelet | 0.619 s | 0.420 s | 0.822 s |
| Action-normalized JEPA | 0.613 s | 0.402 s | 0.828 s |
| Transformer-predictor JEPA | 0.466 +/- 0.004 s | 0.343 +/- 0.038 s | 0.592 +/- 0.044 s |

This is a 24.0% weighted multival improvement over scratch and a 24.0%
improvement over the prior action-normalized JEPA on the same dev protocol.
Pedestrian validation improves from `0.828 s` to `0.592 s`.

Full-starter frozen check:

| Method | Validation MAE | Validation relative error | Test MAE | Test relative error |
| --- | ---: | ---: | ---: | ---: |
| Event tubelet JEPA + navigation fine-tune | 0.243 +/- 0.005 s | 7.51 +/- 0.51% | 0.328 +/- 0.024 s | 6.89 +/- 0.34% |
| Event tubelet JEPA + transformer predictor | 0.241 +/- 0.004 s | 8.28 +/- 0.16% | 0.351 +/- 0.004 s | 7.56 +/- 0.44% |

The transformer predictor is a real validation/protocol improvement on the
harder multi-domain split, but it does not beat the previous best sealed
CPLA-high result. The current best local sealed model remains the earlier
event-tubelet JEPA + navigation fine-tune. Do not use the frozen test result to
tune the next variant.

### All-Token Context Loss

V-JEPA 2.1 also emphasizes dense supervision over all context tokens, not only
future/masked targets. The repo now supports this with
`--context-token-weight`, which adds a same-window EMA target loss on all current
context tokens while preserving the future multi-horizon objective. It uses no
TTC labels and no future navigation.

Dev multi-validation ablation with `--dense-predictor transformer` and
`--context-token-weight 0.25`:

| Method | Weighted multival MAE | validation_car MAE | validation_pedestrian MAE |
| --- | ---: | ---: | ---: |
| Transformer-predictor JEPA | 0.466 +/- 0.004 s | 0.343 +/- 0.038 s | 0.592 +/- 0.044 s |
| All-token transformer JEPA, weight 0.25 | 0.604 +/- 0.052 s | 0.284 +/- 0.018 s | 0.932 +/- 0.111 s |

The all-token loss improves car validation but substantially hurts pedestrian
validation, so this ablation was stopped at validation and the sealed test was
not evaluated. The likely issue is over-emphasizing reconstruction/identity of
the current event context relative to cross-domain future dynamics. Future
variants should tune this only on multi-domain validation, e.g. smaller
context-token weights or a schedule, never on CPLA-high.

A smaller all-token weight was then selected on the same dev multi-validation
protocol, still without evaluating the sealed test:

| Method | Weighted multival MAE | validation_car MAE | validation_pedestrian MAE |
| --- | ---: | ---: | ---: |
| Transformer-predictor JEPA | 0.466 +/- 0.004 s | 0.343 +/- 0.038 s | 0.592 +/- 0.044 s |
| All-token transformer JEPA, weight 0.05 | 0.450 +/- 0.032 s | 0.325 +/- 0.071 s | 0.578 +/- 0.032 s |

This lower context weight improves dev validation, so it was tried on the
full-starter validation protocol. Fine-tuning the full-starter SSL-last
checkpoint over seeds 7/13/21 gave validation MAE
`0.241 +/- 0.006 s` and validation relative error `7.59 +/- 0.90%`.
Per-seed validation MAEs were `0.250`, `0.235`, and `0.239 s`.

This does not beat the selected full-starter transformer-predictor validation
MAE (`0.241 +/- 0.004 s`, numerically `0.240904 s` versus `0.241284 s`), so the
sealed CPLA-high test was not evaluated for the weight-0.05 all-token variant.
The best local sealed model remains event-tubelet JEPA + navigation fine-tune.

### Tubelet Masking

V-JEPA-style masking should operate over spatio-temporal tubelets, not only
over 2D image blocks. The pretraining path now supports `--mask-mode tubelet`,
which masks random `[time, y, x]` event-channel blocks while preserving
metadata/navigation channels. This keeps the causal ego-motion input available
and prevents artificial corruption of navigation.

Dev multi-validation, no sealed test:

| Method | Weighted multival MAE | validation_car MAE | validation_pedestrian MAE |
| --- | ---: | ---: | ---: |
| Transformer-predictor JEPA | 0.466 +/- 0.004 s | 0.343 +/- 0.038 s | 0.592 +/- 0.044 s |
| All-token transformer JEPA, weight 0.05 | 0.450 +/- 0.032 s | 0.325 +/- 0.071 s | 0.578 +/- 0.032 s |
| Tubelet-mask transformer JEPA, LR 3e-5 | 0.409 +/- 0.040 s | 0.242 +/- 0.041 s | 0.580 +/- 0.038 s |

Negative ablations on seed 7:

- tubelet mask plus all-token weight 0.05: weighted MAE `0.649 s`, pedestrian
  MAE `0.993 s`;
- tubelet mask, mask ratio 0.20, LR `1e-4`: pedestrian MAE `0.833 s`;
- tubelet mask, mask ratio 0.20, LR `3e-5`: weighted MAE about `0.479 s`.

The validation-selected variant is therefore: event-tubelet transformer,
transformer dense predictor, causal navigation/action conditioning, tubelet
mask ratio `0.45`, no all-token context loss, and supervised fine-tuning LR
`3e-5`.

Full-starter validation-only result after freezing that choice:

| Seed | Validation MAE | Validation relative error | Best epoch |
| ---: | ---: | ---: | ---: |
| 7 | 0.250 s | 8.17% | 7 |
| 13 | 0.208 s | 7.55% | 27 |
| 21 | 0.237 s | 8.86% | 29 |

Mean full-starter validation result: `0.231 +/- 0.018 s`, `8.19 +/- 0.53%`.
This improves validation MAE versus the previous transformer-predictor candidate
(`0.241 +/- 0.004 s`) and the earlier event-tubelet navigation JEPA
(`0.243 +/- 0.005 s`).

After freezing this validation-selected setup, the saved best checkpoints were
evaluated without retraining on the full starter protocol, including CPLA-high:

| Seed | CPLA-high test MAE | Test relative error | Best epoch |
| ---: | ---: | ---: | ---: |
| 7 | 0.365 s | 7.03% | 7 |
| 13 | 0.314 s | 5.94% | 27 |
| 21 | 0.257 s | 6.27% | 29 |

Mean CPLA-high test result: `0.312 +/- 0.044 s`, `6.42 +/- 0.45%`. This is the
best local all-window test result so far. It improves MAE by 4.9% versus the
previous event-tubelet navigation JEPA (`0.328 s`) and improves relative error
from `6.89%` to `6.42%`.

This still is not an official SOTA claim. CPLA-high has been inspected in prior
runs, and the official EvTTC comparison in Table V is a bbox/ROI-assisted
protocol over a broader sequence set. Treat this as the strongest local
full-starter result, not as a reproduced EvTTC leaderboard result.

## Detection-Assisted Reference

The missing official `bbox_segmentation` folders were recovered after
`gdown --folder` failed to resolve per-file public links. The workaround is
reproducible: write a `gdown --folder --json` listing, then use
`scripts/download_gdown_listing.py` to download the listed JSON files through
`drive.usercontent.google.com`.

Recovered bbox JSON counts:

- `CCRs-side-low`: 149
- `CCRs-side-medium`: 141
- `CCRs-side-high`: 108
- `CPLA-low`: 152
- `CPLA-medium`: 89
- `CPLA-high`: 87

After TTC alignment, the causal bbox geometry baseline gives:

| Method | Split | Labels | Predictions | MAE | RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| Causal bbox geometry | train bbox frames | 871 | 856 | 0.512 | 1.007 |
| Causal bbox geometry | validation bbox frames | 108 | 106 | 0.279 | 0.538 |
| Causal bbox geometry | CPLA-high bbox test | 83 | 81 | 0.157 | 0.331 |

This is detection-assisted and evaluated only on labeled frames, not on all 478
sealed test windows. It should not be compared as an event-only model result.
It does show that reproducing CMax/STRTTC-style SOTA fairly requires complete
bbox/segmentation assets and a benchmark-aligned frame protocol.

Added after the official-protocol audit: `baseline roi-events`, a causal
bbox/ROI event-feature ridge model. It scales RGB label boxes from `1920x1200`
into the event plane (`1280x720`) and uses only events in
`[timestamp - 100 ms, timestamp]`. It uses the current object box, so it is
detection-assisted, not all-window event-only.

Sealed full-starter result with fixed `context_ms=100` and `ridge_alpha=1.0`:

| Method | Split | Labels | Predictions | MAE | Mean relative error |
| --- | --- | ---: | ---: | ---: | ---: |
| ROI event ridge | train bbox frames | 871 | 871 | 0.659 s | 19.15% |
| ROI event ridge | validation bbox frames | 108 | 108 | 0.293 s | 13.63% |
| ROI event ridge | sealed CPLA-high bbox frames | 83 | 83 | 0.829 s | 47.12% |

Frozen ROI latent prober checkpoint evaluation was added after training the
prober only from train labels and selecting by validation. This command loads
the saved prober checkpoints and evaluates them without retraining. On
CPLA-high it matches 73/83 bbox labels to cached event windows:

| Method | Split | Seeds | Labels | Matched | MAE | Mean relative error |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ROI latent prober | validation bbox frames | 7,13,21 | 108 | 98 | 0.226 +/- 0.015 s | 10.78 +/- 0.53% |
| ROI latent prober | CPLA-high bbox test | 7,13,21 | 83 | 73 | 0.423 +/- 0.029 s | 23.41 +/- 2.23% |

This is a negative SOTA result: the frozen latent ROI prober improves strongly
over its train-only ridge prior on validation, but it is worse than the simple
causal bbox geometry reference on CPLA-high and worse than the current
all-window JEPA relative error. It remains useful because it gives a reproducible
SkyJEPA-style detection-assisted checkpoint evaluation path without changing
weights after validation.

Multi-domain validation without evaluating sealed test:

| Method | Split | Labels | MAE | Mean relative error |
| --- | --- | ---: | ---: | ---: |
| ROI event ridge | validation_car (`CCRs-side-high`) | 108 | 0.293 s | 14.63% |
| ROI event ridge | validation_pedestrian (`CPLA-medium`) | 85 | 1.342 s | 65.52% |

Interpretation: the simple ROI feature model is closer to the official
CMax/STRTTC input assumption than full-frame JEPA, but it is empirically much
weaker on pedestrian transfer. The best all-window JEPA remains the best local
sealed result (`6.42%` mean relative test error), while ROI event ridge is
useful mainly as protocol plumbing for bbox/ROI comparison.

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path
.\.venv\Scripts\e-jepa-ttc.exe baseline causal-geometry --manifest artifacts\metrics\evttc_scan_full_bbox.yaml --split data\splits\evttc_full_starter_sealed.yaml --output artifacts\metrics\causal_geometry_full_starter_full_bbox.json --derivative-window 15
.\.venv\Scripts\python.exe -m e_jepa_ttc baseline roi-events --manifest artifacts\metrics\evttc_scan_full_bbox.yaml --split data\splits\evttc_full_starter_sealed.yaml --output artifacts\metrics\roi_events_full_starter_full_bbox.json --context-ms 100 --ridge-alpha 1.0
.\.venv\Scripts\python.exe -m e_jepa_ttc baseline roi-events --manifest artifacts\metrics\evttc_scan_full_bbox.yaml --split data\splits\evttc_full_starter_dev_multival.yaml --output artifacts\metrics\roi_events_full_starter_dev_multival.json --context-ms 100 --ridge-alpha 1.0 --evaluation-splits validation_car validation_pedestrian
```

This is not an official SOTA claim. The run is a local starter protocol, not a
published EvTTC leaderboard comparison. It is, however, the first local result
that is both sealed-protocol positive and directionally aligned with current
dense JEPA world-model practice.

For official bbox/ROI comparison status, see
`docs/evttc_official_bbox_roi_protocol.md`. The short version is that EvTTC
Table V compares STRTTC, CMax, ETTCM, FAITH, AEB-Tracker, and Image FoE on a
bbox/ROI-assisted frame/event protocol over specific CCRs/CCRm/slider
sequences. The current all-window JEPA results and the causal bbox geometry
reference are not directly comparable to that table.

## SkyJEPA-Style Latent Probers

SkyJEPA (`https://arxiv.org/abs/2606.23444`) is highly relevant, but not as a
drop-in method. Its transferable idea is a two-stage world model: learn
action-conditioned latent dynamics first, then freeze the latent model and train
a lightweight physics-inspired prober that converts latents into interpretable
state. For EvTTC, the closest causal analog is:

- state/action history -> event tubelet context plus causal integrated
  navigation;
- latent dynamics -> dense multi-horizon JEPA prediction;
- physics prober -> bbox/ROI TTC head using current detection, causal event
  history, and train-only priors;
- no future boxes, future events, future navigation, or test labels for tuning.

This is now implemented as two local tools:

- `train latent-prober`: all-window frozen-latent residual TTC prober;
- `train roi-latent-prober`: detection-assisted frozen-latent bbox/ROI TTC
  prober with a train-only ridge physics prior;
- `train roi-rollout-prober`: detection-assisted TTC prober over frozen
  JEPA-predicted future latent token rollouts, with checkpoint-only evaluation
  via `train roi-rollout-prober-evaluate`.

The all-window latent prober was negative on dev validation. The ROI prober is
positive because the current bbox gives an object-local state estimate that is
closer to SkyJEPA's physical state input.

Dev multi-validation, no sealed test:

| Method | Weighted multival MAE | validation_car MAE | validation_pedestrian MAE |
| --- | ---: | ---: | ---: |
| ROI event ridge prior | 0.759 s | 0.293 s | 1.342 s |
| ROI latent prober, seeds 7/13/21 | 0.528 +/- 0.011 s | 0.340 +/- 0.003 s | 0.785 +/- 0.024 s |
| ROI rollout prober, seeds 7/13/21 | 0.613 +/- 0.017 s | 0.329 +/- 0.024 s | 1.000 +/- 0.055 s |

Full-starter validation used for model selection:

| Method | Validation MAE | Validation relative error |
| --- | ---: | ---: |
| ROI ridge prior inside latent prober | 0.344 s | 16.19% |
| ROI latent prober, seeds 7/13/21 | 0.226 +/- 0.015 s | 10.78 +/- 0.53% |
| ROI rollout prober, seeds 7/13/21 | 0.226 +/- 0.013 s | 11.15 +/- 0.79% |

That is a 34.3% validation MAE reduction versus its train-only ROI/ridge prior.
It also beats the earlier local ROI event ridge validation result (`0.293 s`),
but the comparison is not exact because the latent prober matches 98/108
validation bbox rows to cache windows.

After freezing the ROI prober checkpoints, a checkpoint-only evaluation was run
on the full protocol, including CPLA-high, with no retraining:

| Seed | CPLA-high matched frames | CPLA-high MAE | CPLA-high relative error |
| ---: | ---: | ---: | ---: |
| 7 | 73/83 | 0.383 s | 20.46% |
| 13 | 73/83 | 0.431 s | 23.91% |
| 21 | 73/83 | 0.454 s | 25.85% |

Mean CPLA-high ROI latent prober result: `0.423 +/- 0.029 s`,
`23.41 +/- 2.23%`. This is not competitive with the current all-window JEPA
test result (`6.42 +/- 0.45%`) or the simple causal bbox geometry reference
(`0.157 s` MAE on 81 valid frames), so the current ROI prober is protocol
plumbing and diagnostic evidence, not the SOTA path by itself.

The first predicted-rollout prober was also negative. It reconstructs the frozen
JEPA encoder and dense transformer predictor from
`jepa_event_tubelet_tubeletmask_transformerpred_nav_full_starter_seed7_30e`,
summarizes predicted future token rollouts at 20, 60, 100, 240, and 500 ms, and
trains the ROI head only on train labels. It slightly matches the ROI latent
prober on single-sequence validation MAE (`0.226 s`) but is worse in relative
error and much worse on the harder pedestrian validation. Therefore the
CPLA-high test was not evaluated for this rollout branch.

Conclusion: SkyJEPA is key for the next architecture direction, especially the
frozen latent prober and action-conditioned rollout framing. The current
predicted-rollout prober is still too weak because it summarizes rollouts into a
single MLP head. The next version should use a structured kinematic/TTC head
that consumes per-horizon latent state rather than flattening all horizons into
one feature vector.

## SOTA Position

Compared with current JEPA/world-model SOTA as of 2026-07-02:

- Aligned: latent prediction, EMA target encoder, future multi-horizon
  prediction, spatio-temporal tubelet masking, dense token loss, causal
  motion/action conditioning, token-attention dense predictor, optional
  all-token context loss, no TTC-label leakage, low-label transfer evaluation.
- Still below SOTA: small local training scale, shallow token transformer, deep
  self-supervision currently negative in ablation, all-token context loss
  currently negative on pedestrian validation, event plus ego-motion only, no
  RGB/LiDAR/depth/box fusion, no action-conditioned planning or closed-loop
  evaluation, and no official benchmark replication.

Practical claim:

> On the local full-starter EvTTC protocol, tubelet-masked dense token JEPA with
> causal integrated-navigation channels and a transformer dense predictor reaches
> `0.312 +/- 0.044 s` CPLA-high test MAE and is the best local all-window result.
> It is a strong starter result, not yet an official EvTTC SOTA result.

## External Dataset Triage

Markov Studios / Markov AI datasets were checked on 2026-07-02 and are not a
priority for this TTC task. `markov-ai/computer-use-large` is a large GUI
screen-recording dataset for desktop software agents, and
`markov-ai/gaming-500-hours` is action-annotated gameplay screen recording.
Both are useful conceptually for action-conditioned world models, but neither
provides event-camera streams, TTC labels, vehicle ego-motion, depth, or EvTTC
bbox/ROI labels. They should not be mixed into the EvTTC protocol.
