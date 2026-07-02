# E-JEPA-TTC: Dense Event-Token Joint-Embedding Predictive Pretraining for Time-to-Contact Estimation

Version: 2026-07-02

## Abstract

This thesis-style report describes E-JEPA-TTC, a self-supervised event-camera
representation learning system for Time-to-Contact / Time-to-Collision (TTC)
estimation on a local EvTTC full-starter protocol. The work replaces a TinyCNN
baseline with tokenized event transformers, then moves toward a V-JEPA-style
objective using dense temporal token prediction, spatio-temporal event tubelet
masking, causal ego-motion conditioning, and a transformer predictor. The best
local all-window model, an event-tubelet transformer pretrained with tubelet
masking and a dense transformer JEPA predictor, reaches `0.231 +/- 0.018 s`
validation MAE and `0.312 +/- 0.044 s` CPLA-high diagnostic test MAE over three
fine-tuning seeds. Its mean absolute relative error is `8.19 +/- 0.53%` on
validation and `6.42 +/- 0.45%` on CPLA-high.

The result is the strongest local all-window model in this repository, but it is
not an official EvTTC state-of-the-art claim. The official EvTTC comparison is a
bbox/ROI-assisted protocol over CCRs1, CCRs2, CCRm, and slider sequences, while
the current local protocol uses a smaller full-starter subset and CPLA-high has
already been inspected during development. Detection-assisted ROI probers and
JEPA-predicted rollout probers were implemented to align with recent SkyJEPA and
world-model practice, but their validation and diagnostic results are not
competitive with the all-window JEPA model or with the simple causal bbox
geometry reference. The main contribution is therefore a reproducible
engineering and experimental path toward dense event-token JEPA for TTC, with
clear anti-leakage discipline and explicit limits on claims.

## 1. Motivation

Event cameras provide asynchronous high-dynamic-range measurements that are
well-suited for collision-risk estimation under fast motion. TTC estimation is a
natural downstream task because it depends on temporal geometry, not only object
appearance. The EvTTC dataset introduced a modern event-camera benchmark for
TTC in driving scenarios, including ground-truth TTC and benchmark comparisons
against methods such as STRTTC, CMax, ETTCM, FAITH, AEB-Tracker, and Image FoE.

The central hypothesis of this project is that a predictive latent model can
learn useful event dynamics without TTC labels, and that the learned
representation can improve TTC regression compared with supervised scratch
models, especially when labels are limited. This follows the Joint-Embedding
Predictive Architecture (JEPA) line of work: learn by predicting future or
masked representations rather than reconstructing pixels or raw event tensors.

## 2. Related Work

I-JEPA introduced non-generative representation prediction for images: context
representations are trained to predict target representations in latent space.
V-JEPA moved this idea to video through masked spatio-temporal latent prediction.
V-JEPA 2 and V-JEPA 2.1 extended the recipe toward world modeling and dense
features. The V-JEPA 2.1 update is particularly relevant here because it
emphasizes dense predictive loss, deep self-supervision, spatially grounded
features, temporal consistency, and scaling.

LeWorldModel is a direct action-conditioned JEPA world-model reference. It
frames stable latent dynamics learning as next-embedding prediction plus a
distribution regularizer. SkyJEPA is also important because it uses frozen
latent dynamics and a physics-inspired prober for long-horizon robotic control.
Those works motivated the project changes after the first local JEPA models:
causal action/ego-motion conditioning, dense future-token prediction, frozen
latent probers, predicted latent rollout probers, and anti-collapse
regularization experiments.

Event-camera TTC research is less mature than RGB/video self-supervised
learning. EvTTC provides the most relevant local benchmark target. However, its
published benchmark protocol is not an all-window event-only regression task:
it is heavily tied to object bbox/ROI assumptions and evaluates frame/event
methods on specific sequence families. This distinction controls the claim made
in this report.

## 3. Dataset And Protocol

### 3.1 Local Full-Starter Data

The local full-starter protocol uses the recovered EvTTC sequences available
under `datasets/evttc`. The materialized navigation cache is:

`artifacts/features/evttc_full_starter_voxel_160x90_b5_raw_meta_nav.npz`

The cache contains 3972 windows with shape `[3972, 21, 90, 160]`:

- 10 event channels from 5 temporal bins and 2 polarities;
- 2 metadata channels;
- 9 causal integrated-navigation channels.

The split is sequence-level:

| Split | Sequences | Windows |
| --- | --- | ---: |
| Train | `CCRs-1-low-100-overlap-100`, `CCRs-1-medium-100-overlap-100`, `CCRs-1-high-100-overlap-100`, `CCRs-side-low`, `CCRs-side-medium`, `CPLA-low`, `CPLA-medium` | 3019 |
| Validation | `CCRs-side-high` | 475 |
| Diagnostic test | `CPLA-high` | 478 |

This split was frozen before the final tubelet-mask run, but CPLA-high had been
inspected in previous branches. Therefore CPLA-high is a diagnostic test for
local model comparison, not a fresh untouched benchmark test.

### 3.2 Development Multi-Validation Split

After CPLA-high had been opened, a harder development split was introduced to
avoid further tuning on it:

| Split | Sequences |
| --- | --- |
| Train | CCRs-1 plus `CCRs-side-low`, `CCRs-side-medium`, `CPLA-low` |
| `validation_car` | `CCRs-side-high` |
| `validation_pedestrian` | `CPLA-medium` |
| Test | `CPLA-high` |

This split is used for architecture selection and negative ablations. It is not
used to claim a new sealed CPLA-high result.

### 3.3 Official EvTTC Protocol Gap

The official EvTTC Table V benchmark reports CCRs1, CCRs2, CCRm, and slider
sequences. The current local workspace has CCRs1 and several starter side/CPLA
sequences, but not the complete CCRs2, CCRm, and slider set with all bbox/ROI
assets. Therefore this project cannot honestly claim official EvTTC SOTA yet.

The gap is now enforced by an automated coverage gate:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\e-jepa-ttc.exe data official-coverage --root datasets\evttc --output artifacts\metrics\evttc_official_table_v_coverage.json
```

On the local workspace, this checker scans 9 EvTTC sequences and finds 3/8
complete real-world official rows (`37.5%`) and 3/10 complete Table V rows
including slider (`30.0%`). Missing rows are CCRs2 low/medium/high, CCRm
low/medium, `Slider-750`, and `Slider-1000`. The exact checklist and command
are tracked in `docs/evttc_official_bbox_roi_protocol.md`.

## 4. Representations

Each event window is converted to a voxel grid at `160x90` resolution with 5
temporal bins. Positive and negative polarities are kept separate. Optional
metadata channels encode causal per-window metadata, and navigation channels
provide integrated ego-motion features from the current context window only.
These channels include speed, velocity components, acceleration components,
yaw-rate, and a validity flag.

All predictive and supervised models use only current and past context inputs.
Future event windows appear only as self-supervised target views during JEPA
pretraining; TTC labels are not used during self-supervised pretraining.

## 5. Model Families

### 5.1 Scratch Baselines

The initial baselines were:

- event-rate ridge;
- TinyCNN supervised regressor;
- token transformer supervised regressor from scratch;
- token transformer with navigation channels from scratch.

These provide a controlled comparison for JEPA pretraining.

### 5.2 Token JEPA

The first JEPA models used a token-transformer encoder with an online encoder,
an EMA target encoder, and a temporal latent prediction objective. Future
horizons were `20`, `60`, `100`, `240`, and `500` ms. Supervised fine-tuning was
performed after pretraining.

### 5.3 Event-Tubelet Transformer JEPA

The final local all-window model is closer to V-JEPA practice:

- event channels are interpreted as a polarity-by-time tensor;
- a 3D tubelet embedding tokenizes `[polarity, time, y, x]` structure;
- extra metadata and navigation channels are embedded as causal spatial
  auxiliary context;
- the predictor operates on dense tokens rather than only a global embedding;
- a transformer dense predictor replaces the per-token MLP predictor;
- tubelet masking masks event-channel spatio-temporal blocks while preserving
  metadata/navigation channels;
- causal action conditioning combines event-motion summaries and navigation
  ego-motion features;
- target encoder parameters are updated by EMA.

The final objective name recorded by the checkpoint is:

`tubeletmask_transformer_dense_temporal_token_action_multihorizon`

### 5.4 SkyJEPA-Style Probers

Three prober families were added:

1. `latent-prober`: all-window frozen encoder latent prober.
2. `roi-latent-prober`: detection-assisted bbox/ROI prober using frozen current
   JEPA latents plus causal ROI event features and a train-only ridge prior.
3. `roi-rollout-prober`: detection-assisted prober using frozen
   JEPA-predicted future latent token rollouts.

The ROI probers use TTC labels for the prober head only. Feature scaling and
ridge priors are fit on train labels only. Validation selects the prober
checkpoint. Checkpoint-only evaluation is available so test metrics can be
computed without retraining.

## 6. Anti-Leakage Controls

The following controls are implemented and recorded in metrics:

- sequence-level splits;
- no TTC labels in JEPA pretraining;
- future event targets are self-supervised only;
- target pairs do not cross sequence boundaries;
- target pairs do not cross split boundaries;
- navigation/action features are extracted from the current context only;
- action-feature normalization is estimated from train context windows only;
- ROI event features use `[timestamp - context_ms, timestamp]` only;
- no future bbox, future events, or future navigation in ROI probers;
- final checkpoint-only evaluators reload saved checkpoints and set
  `retrained_during_evaluation=false`.

## 7. Experimental Results

### 7.1 Main Full-Starter All-Window Results

Lower MAE is better.

| Method | Train labels | Seeds | Validation MAE | Diagnostic CPLA-high MAE |
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

The strongest all-window model reaches:

- validation MAE: `0.231477844 +/- 0.017632455 s`;
- validation relative error: `8.192429 +/- 0.533708%`;
- CPLA-high diagnostic MAE: `0.312034689 +/- 0.044063632 s`;
- CPLA-high diagnostic relative error: `6.416740 +/- 0.454934%`;
- CPLA-high diagnostic RMSE: `0.485851757 +/- 0.090484582 s`.

This is a `4.9%` MAE improvement over the previous event-tubelet navigation
JEPA mean (`0.328 s`) and a relative-error improvement from `6.89%` to `6.42%`.

### 7.2 Label-Efficiency Results

The strongest low-label evidence comes from event-only token JEPA:

| Method | Train labels | Validation MAE | Diagnostic CPLA-high MAE |
| --- | ---: | ---: | ---: |
| Token transformer scratch | 5% | 1.226 +/- 0.031 | 1.382 +/- 0.044 |
| Token JEPA + fine-tune | 5% | 0.524 +/- 0.047 | 0.636 +/- 0.109 |
| Token transformer scratch | 10% | 1.178 +/- 0.056 | 1.327 +/- 0.104 |
| Token JEPA + fine-tune | 10% | 0.437 +/- 0.039 | 0.460 +/- 0.029 |

At 10% labels, token JEPA improves diagnostic test MAE by `65.4%` versus the
matching scratch token transformer. It also beats the full-label TinyCNN seed-7
baseline on diagnostic CPLA-high MAE (`0.460 s` versus `0.513 s`).

### 7.3 Negative Architecture Ablations

Not every SOTA-inspired change helped:

| Ablation | Result |
| --- | --- |
| Deep token JEPA | Diagnostic CPLA-high MAE `0.594 s` |
| Deep layer-aware token JEPA | Diagnostic CPLA-high MAE `0.505 s` |
| Large token transformer JEPA | Diagnostic CPLA-high MAE `0.529 s` |
| All-token context loss, weight 0.25 | Better car validation, much worse pedestrian validation |
| All-token context loss, weight 0.05 | Improved dev validation, did not beat full-starter validation |
| Tubelet mask plus all-token weight 0.05 | Dev weighted MAE `0.649 s` |

The successful change was not simply larger scale or deeper supervision. The
best local result came from matching the event structure with tubelet masking
and preserving causal action/ego-motion conditioning.

### 7.4 Detection-Assisted Results

The recovered bbox labels enabled local detection-assisted references. These
are frame-label-only and not comparable to all-window metrics.

| Method | Split | Labels | Predictions | MAE | Relative error |
| --- | --- | ---: | ---: | ---: | ---: |
| Causal bbox geometry | validation bbox frames | 108 | 106 | 0.279 | - |
| Causal bbox geometry | CPLA-high bbox frames | 83 | 81 | 0.157 | - |
| ROI event ridge | validation bbox frames | 108 | 108 | 0.293 | 13.63% |
| ROI event ridge | CPLA-high bbox frames | 83 | 83 | 0.829 | 47.12% |
| ROI latent prober | validation bbox frames | 108 | 98 | 0.226 +/- 0.015 | 10.78 +/- 0.53% |
| ROI latent prober | CPLA-high bbox frames | 83 | 73 | 0.423 +/- 0.029 | 23.41 +/- 2.23% |

The causal bbox geometry reference is strong on CPLA-high bbox frames, showing
that current bounding-box scale contains powerful TTC information. However, it
is not an all-window event-only model and it does not reproduce the official
CMax/STRTTC benchmark table.

### 7.5 Predicted-Rollout Prober Results

The `roi-rollout-prober` was implemented to test the SkyJEPA idea more directly:
probe frozen predicted future latents, not only current context latents.

Full-starter validation:

| Method | Validation MAE | Validation relative error |
| --- | ---: | ---: |
| ROI latent prober | 0.226 +/- 0.015 | 10.78 +/- 0.53% |
| ROI rollout prober | 0.226 +/- 0.013 | 11.15 +/- 0.79% |

Harder dev multi-validation:

| Method | Weighted MAE | validation_car MAE | validation_pedestrian MAE |
| --- | ---: | ---: | ---: |
| ROI latent prober | 0.528 +/- 0.011 | 0.340 +/- 0.003 | 0.785 +/- 0.024 |
| ROI rollout prober | 0.613 +/- 0.017 | 0.329 +/- 0.024 | 1.000 +/- 0.055 |

The flat rollout prober is a negative result. It confirms that simply flattening
future latent summaries into an MLP is not sufficient. The next version should
preserve per-horizon structure and include a kinematic/TTC head.

A second validation-only rollout ablation added explicit latent dynamics
features: per-horizon deltas from the context summary, horizon-normalized
latent velocities, consecutive-horizon latent velocities, and compact
norm/cosine statistics. On seed 7, this `dynamics` feature mode reached
`0.248 s` validation MAE and `11.93%` relative error, worse than the flat seed-7
rollout prober (`0.210 s`, `10.17%`). Its best checkpoint occurred at epoch 2,
indicating immediate overfitting. CPLA-high was not evaluated for this branch.

## 8. Reproducibility

### 8.1 Environment

The final tests were run on Windows with:

- Python virtualenv `.venv`;
- PyTorch `2.11.0+cu128`;
- CUDA available;
- NVIDIA GeForce RTX 5070 Ti Laptop GPU;
- 32 GB RAM and 32 CPU threads available on the workstation.

### 8.2 Final Verification Commands

```powershell
.\.venv\Scripts\python.exe -m ruff check .
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m pytest
```

Final verification result:

- `ruff check .`: all checks passed;
- `pytest`: `50 passed`.

### 8.3 Main Result Artifacts

Main all-window summaries:

- `artifacts/metrics/event_tubelet_tubeletmask_transformerpred_nav_full_starter_last_lr3e5_eval_full_protocol_validation_summary.json`
- `artifacts/metrics/event_tubelet_tubeletmask_transformerpred_nav_full_starter_last_lr3e5_eval_full_protocol_test_summary.json`

ROI latent summaries:

- `artifacts/metrics/roi_latent_prober_event_tubelet_transformerpred_nav_full_starter_last_eval_full_protocol_validation_summary.json`
- `artifacts/metrics/roi_latent_prober_event_tubelet_transformerpred_nav_full_starter_last_eval_full_protocol_test_summary.json`

ROI rollout summaries:

- `artifacts/metrics/roi_rollout_prober_tubeletmask_full_starter_eval_validation_summary.json`
- `artifacts/metrics/roi_rollout_prober_tubeletmask_dev_multival_validation_car_summary.json`
- `artifacts/metrics/roi_rollout_prober_tubeletmask_dev_multival_validation_pedestrian_summary.json`

### 8.4 Representative Training Commands

Final JEPA pretraining used the event-tubelet transformer with tubelet masking
and transformer dense predictor:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m e_jepa_ttc pretrain jepa `
  --cache artifacts\features\evttc_full_starter_voxel_160x90_b5_raw_meta_nav.npz `
  --output-dir artifacts\runs\jepa_event_tubelet_tubeletmask_transformerpred_nav_full_starter_seed7_30e `
  --epochs 30 `
  --batch-size 64 `
  --learning-rate 5e-4 `
  --seed 7 `
  --device auto `
  --model event-tubelet-transformer `
  --pretrain-splits train `
  --validation-splits validation `
  --temporal-horizons-ms 20 60 100 240 500 `
  --max-target-slop-ms 10 `
  --mask-mode tubelet `
  --dense-predictor transformer
```

The final supervised checks fine-tuned validation-selected checkpoints with
seeds 7, 13, and 21. Exact run directories and metrics are recorded in
`docs/full_starter_results.md`.

## 9. Discussion

The evidence supports the following conclusions:

1. JEPA pretraining helps event-based TTC in this local protocol.
2. The strongest gains appear in label efficiency.
3. Tokenization and dense prediction matter more than scaling a small encoder.
4. Tubelet masking is the most successful V-JEPA-like modification tested.
5. Causal navigation/action conditioning is useful, but not always positive in
   low-label settings.
6. Detection-assisted ROI probers need more structure; flat latent probes and
   flat rollout probes are not enough.
7. Official EvTTC SOTA remains unproven: local asset coverage is only 37.5% of
   the real-world benchmark rows and 30.0% of complete Table V with slider, and
   the exact official baseline runtime protocol is not reproduced.

## 10. Limitations

The main limitations are:

- CPLA-high has been inspected, so it is diagnostic rather than pristine;
- official CCRs2, CCRm, and slider benchmark assets are missing locally, leaving
  3/8 real-world rows and 3/10 complete Table V rows covered;
- official STRTTC/CMax wrappers are not yet reproduced end-to-end;
- the event-tubelet backbone is still small compared with V-JEPA 2.1 scale;
- deep self-supervision and all-token context loss were negative at current
  data/model scale;
- no RGB, LiDAR, depth, or segmentation fusion is used during JEPA pretraining;
- no closed-loop planning or counterfactual action rollout is evaluated;
- ROI rollout probing currently flattens horizons instead of enforcing TTC
  dynamics structurally.

## 11. Future Work

The next technically valid steps are:

1. Download official CCRs2 and CCRm HDF5, TTC, and bbox/segmentation assets.
2. Add the slider sequences if reproducing complete EvTTC Table V.
3. Wrap or reimplement STRTTC and CMax under a deterministic CLI.
4. Build a horizon-structured TTC head over predicted latent rollouts.
5. Add VISReg/SIGReg-style regularization ablations tuned only on
   multi-domain validation.
6. Evaluate model calibration and latency.
7. Run one final official benchmark once all assets and baselines are present.

## 12. Claim Statement

The strongest defensible claim is:

> On the local EvTTC full-starter all-window protocol, event-tubelet
> transformer JEPA with tubelet masking, dense transformer future-token
> prediction, and causal integrated-navigation conditioning achieves the best
> local result in this repository: `0.231 +/- 0.018 s` validation MAE and
> `0.312 +/- 0.044 s` diagnostic CPLA-high MAE over three fine-tuning seeds.

The strongest forbidden claim is:

> This is official EvTTC SOTA.

That claim is not supported by the evidence in this workspace.

## References

1. EvTTC: An Event Camera Dataset for Time-to-Collision Estimation.
   https://arxiv.org/html/2412.05053v1
2. EvTTC Benchmark project page. https://nail-hnu.github.io/EvTTC/
3. I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding
   Predictive Architecture. https://arxiv.org/abs/2301.08243
4. V-JEPA: Revisiting Feature Prediction for Learning Visual Representations
   from Video. https://ai.meta.com/research/publications/revisiting-feature-prediction-for-learning-visual-representations-from-video/
5. V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning.
   https://arxiv.org/abs/2603.14482
6. LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from
   Pixels. https://arxiv.org/abs/2603.19312
7. SkyJEPA: Learning Long-Horizon World Models for Zero-Shot Sim-to-Real
   Control of Quadrotors. https://arxiv.org/abs/2606.23444
8. Event-Aided Time-to-Collision Estimation / STRTTC project.
   https://nail-hnu.github.io/EventAidedTTC/
