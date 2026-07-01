# JEPA And World Models SOTA Alignment, 2026-07-01

This note summarizes the relevant JEPA/world-model state of the art checked on
2026-07-01 and maps it to the current `e-jepa-ttc` implementation. It is a
research alignment note, not a benchmark claim.

## SOTA Snapshot

1. JEPA core idea: learn by predicting target representations from context
   representations, not by reconstructing pixels. I-JEPA showed this can learn
   semantic image features without handcrafted augmentations by predicting target
   block embeddings from a context block.
   Source: https://arxiv.org/abs/2301.08243

2. V-JEPA moved the objective to video: masked spatio-temporal regions are
   predicted in latent space, with frozen evaluations and low-label transfer as
   key evidence. Meta emphasizes that large spatio-temporal masks make the task
   non-trivial and avoid wasting capacity on unpredictable pixel details.
   Sources:
   - https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/
   - https://ai.meta.com/research/publications/revisiting-feature-prediction-for-learning-visual-representations-from-video/

3. V-JEPA 2 is the main JEPA world-model reference before this date. It combines
   large-scale action-free video pretraining with a small amount of robot
   interaction post-training. The paper reports SOTA visual understanding and an
   action-conditioned latent world model, V-JEPA 2-AC, for zero-shot robot
   planning.
   Source: https://arxiv.org/html/2506.09985v1

4. V-JEPA 2.1 is the most relevant 2026 update. It adds dense predictive loss,
   deep self-supervision across intermediate layers, image/video tokenizers, and
   scaling. The target is dense, spatially grounded, temporally consistent
   features, not just global embeddings.
   Source: https://arxiv.org/html/2603.14482v2

5. Autonomous-driving JEPA work now exists. AD-L-JEPA applies JEPA to LiDAR BEV
   embeddings and reports label-efficiency and speed advantages versus masked
   occupancy/generative baselines. Drive-JEPA adapts V-JEPA-style video
   pretraining to end-to-end driving and combines it with multimodal trajectory
   distillation, reporting SOTA NAVSIM scores.
   Sources:
   - https://arxiv.org/html/2501.04969v1
   - https://arxiv.org/html/2601.22032

6. World models have split into several active tracks:
   - Latent predictive models for representation, planning, and control.
   - Generative interactive simulators, e.g. Genie/Genie 3.
   - Physical-AI platforms, e.g. NVIDIA Cosmos.
   - Driving-specific world models, e.g. Waymo World Model.
   Sources:
   - https://arxiv.org/html/2605.00080v1
   - https://arxiv.org/html/2510.16732v3
   - https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/
   - https://research.nvidia.com/publication/2025-01_cosmos-world-foundation-model-platform-physical-ai
   - https://waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation/

7. The broader world-model literature still treats evaluation as unsettled.
   Useful metrics now include downstream task performance, physical consistency,
   long-horizon temporal consistency, closed-loop control, and data efficiency.
   Pixel fidelity alone is not sufficient.
   Sources:
   - https://arxiv.org/html/2510.16732v3
   - https://arxiv.org/html/2502.10498v2

8. Event-camera SSL is still relatively early compared with RGB/video/LiDAR.
   Current event work emphasizes sparse asynchronous representations, dense
   event pretraining, low-latency perception, and TTC-specific datasets.
   Sources:
   - https://arxiv.org/html/2412.05053v3
   - https://arxiv.org/html/2505.07556v1
   - https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05986.pdf
   - https://nail-hnu.github.io/EventAidedTTC/

## How The Repo Applies It

The current project is aligned with the JEPA branch of world models, not the
large generative simulator branch.

What matches SOTA direction:

- Non-generative latent prediction: `pretrain jepa` predicts target encoder
  embeddings rather than pixels or event reconstructions.
- Action-free observational pretraining: the encoder is pretrained from event
  windows without TTC labels.
- Temporal future prediction: the current objective predicts future event-window
  embeddings at multiple horizons, which is closer to world modeling than the
  earlier same-window masked objective.
- EMA target encoder: the code uses an online encoder plus target encoder updated
  by EMA.
- Low-label and frozen evaluations: the repo reports 5%/10% label transfer and a
  frozen-encoder probe, which matches the V-JEPA evaluation style.
- Anti-leakage controls: future targets are self-supervised only, and pair
  construction forbids crossing sequence or split boundaries.

What is not yet SOTA:

- Backbone scale: the repo uses a small `TinyCNNEncoder`, not a ViT/transformer
  token backbone.
- Dense feature learning: the loss is on global latent vectors, not dense
  spatio-temporal patch tokens as in V-JEPA 2.1.
- Deep self-supervision: intermediate encoder layers are not supervised.
- Action conditioning: there is no ego-action, relative motion, optical-flow,
  bbox-motion, or control-conditioned predictor.
- Multi-modal fusion: the current event-only path does not use RGB, LiDAR,
  depth, boxes, or segmentation during JEPA pretraining.
- Closed-loop planning: the model estimates TTC; it does not simulate futures for
  planning, counterfactual rollout, or control.
- Benchmark maturity: current results are local and exploratory because the
  mini test split has been inspected repeatedly and the full starter dataset is
  not yet downloaded.

## Practical Interpretation

The implementation is a correct small-scale JEPA prototype for event-based TTC:
it is using latent prediction, future horizons, low-label evaluation, and
anti-leakage discipline. It should not be called a SOTA world model yet.

The strongest current evidence for JEPA in this repo is label efficiency:
with 5% train labels on the mini split, temporal JEPA improves validation MAE
from `2.909 +/- 0.743 s` to `1.548 +/- 0.176 s`, while test MAE improves modestly
from `3.107 +/- 0.277 s` to `2.986 +/- 0.106 s`. This is promising but not final.

## Next Alignment Steps

1. Finish the full EvTTC starter download and create a fresh sealed test split.
2. Replace global-only TinyCNN JEPA with dense spatio-temporal token prediction.
3. Add deep self-supervision from intermediate encoder stages.
4. Add motion/action conditioning suitable for TTC: ego speed if available,
   event-flow proxy, bbox scale derivative, or horizon-conditioned relative
   approach features.
5. Keep low-label and frozen-probe evaluations as first-class metrics.
6. Report final claims only on a new held-out test split that was not used during
   architecture or hyperparameter selection.

