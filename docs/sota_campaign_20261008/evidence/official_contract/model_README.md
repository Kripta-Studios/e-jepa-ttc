---
library_name: pytorch
datasets:
- NAIL-HNU/eAP-dataset
- NAIL-HNU/GarlTTC-dataset
tags:
- ttc
- time-to-collision
- event-camera
- rgb-event
- multimodal
- autonomous-driving
- pytorch
---

# GarlTTC Model

This repository stages checkpoints for the GarlTTC release.

Dataset dependencies:

- `NAIL-HNU/eAP-dataset`
- `NAIL-HNU/GarlTTC-dataset`

Primary checkpoint:

- `paper_ours_full.pth`

Branch pretraining checkpoints:

- `paper_visual_only_lhr.pth`
- `paper_event_only_lhr.pth`

Reference metrics are stored in `metrics/paper_test12.json`. If a local release
verification run is available, its raw output can be staged as `metrics/test.json`.

Ablation checkpoints are included under `ablation/` for paper table reproduction.
Use the matching YAML files under `configs/ablation/`.
