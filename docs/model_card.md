# Model Card

## Model

Best local model:

- name: Event tubelet JEPA + tubelet mask + transformer predictor;
- encoder: `event-tubelet-transformer`;
- pretraining objective:
  `tubeletmask_transformer_dense_temporal_token_action_multihorizon`;
- pretraining checkpoint:
  `artifacts/runs/jepa_event_tubelet_tubeletmask_transformerpred_nav_full_starter_seed7_30e/jepa_encoder_last.pt`;
- supervised fine-tune checkpoints:
  `artifacts/runs/event_tubelet_tubeletmask_transformerpred_nav_full_starter_last_lr3e5_seed{7,13,21}_30e`.

The artifact directories are local generated outputs and are not committed to
git by default.

## Intended Use

Research use for event-camera Time-to-Contact / Time-to-Collision estimation on
the local EvTTC full-starter protocol.

## Inputs

Materialized event voxel windows:

- cache:
  `artifacts/features/evttc_full_starter_voxel_160x90_b5_raw_meta_nav.npz`;
- shape: `[3972, 21, 90, 160]`;
- event channels: 5 temporal bins x 2 polarities;
- metadata channels: enabled;
- causal integrated-navigation channels: enabled.

## Outputs

The supervised model predicts log-TTC, converted to TTC seconds for metrics.

## Results

Over seeds 7/13/21:

| Split | MAE | Mean absolute relative error | RMSE |
| --- | ---: | ---: | ---: |
| validation | `0.231 +/- 0.018 s` | `8.19 +/- 0.53%` | `0.326 +/- 0.027 s` |
| diagnostic CPLA-high | `0.312 +/- 0.044 s` | `6.42 +/- 0.45%` | `0.486 +/- 0.090 s` |

## Claim Limits

This model is the best local all-window result in this repository. It is not an
official EvTTC SOTA model because:

- the official bbox/ROI benchmark sequence set is incomplete locally;
- CPLA-high has been inspected during development;
- STRTTC/CMax official-style wrappers have not been reproduced end-to-end.

## Safety And Leakage Controls

- no TTC labels are used during JEPA pretraining;
- future event windows are self-supervised targets only;
- target pairing does not cross sequence or split boundaries;
- action/ego-motion conditioning uses current context only;
- final summaries are aggregated over independent fine-tuning seeds.

## Paper

See [E-JEPA-TTC paper](e_jepa_ttc_paper.md).
