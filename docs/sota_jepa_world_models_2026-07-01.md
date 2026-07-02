# JEPA And World Models SOTA Alignment, 2026-07-01

This note summarizes the relevant JEPA/world-model state of the art checked on
2026-07-01 and maps it to the current `e-jepa-ttc` implementation. It is a
research alignment note, not a benchmark claim.

Updated on 2026-07-02 with SkyJEPA (`arXiv:2606.23444v2`).

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

5. LeWorldModel is the cleanest 2026 JEPA world-model recipe for
   action-conditioned prediction. It predicts next latent embeddings conditioned
   on actions and uses a Gaussian-distribution regularizer to fight collapse,
   with control/planning evaluation rather than only representation evaluation.
   Sources:
   - https://arxiv.org/abs/2603.19312
   - https://le-wm.github.io/

6. SkyJEPA is the most relevant 2026 JEPA robotics/control paper for this
   project. It learns action-conditioned latent dynamics, maps frozen latent
   rollouts through a physics-inspired prober into interpretable state, studies
   long-horizon compounding error, and uses SIGReg-style anti-collapse
   regularization plus domain-randomized data coverage. It is a quadrotor
   control paper rather than an event-camera TTC benchmark, so it should guide
   architecture and diagnostics, not be used as a direct numeric comparison.
   Source: https://arxiv.org/abs/2606.23444

7. Autonomous-driving JEPA work now exists. AD-L-JEPA applies JEPA to LiDAR BEV
   embeddings and reports label-efficiency and speed advantages versus masked
   occupancy/generative baselines. Drive-JEPA adapts V-JEPA-style video
   pretraining to end-to-end driving and combines it with multimodal trajectory
   distillation, reporting SOTA NAVSIM scores.
   Sources:
   - https://arxiv.org/html/2501.04969v1
   - https://arxiv.org/html/2601.22032

8. World models have split into several active tracks:
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

9. The broader world-model literature still treats evaluation as unsettled.
   Useful metrics now include downstream task performance, physical consistency,
   long-horizon temporal consistency, closed-loop control, and data efficiency.
   Pixel fidelity alone is not sufficient.
   Sources:
   - https://arxiv.org/html/2510.16732v3
   - https://arxiv.org/html/2502.10498v2

10. Event-camera SSL is still relatively early compared with RGB/video/LiDAR.
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

What is now closer to SOTA after the full-starter pass:

- The repo includes a token-transformer backbone for `pretrain jepa` and
  supervised TTC via `--model token-transformer`.
- The repo now also includes `--model event-tubelet-transformer`, a V-JEPA-like
  tubelet tokenizer that embeds the polarity-by-time event bins with 3D
  tubelet patches and adds metadata/navigation channels as causal auxiliary
  context.
- The default temporal objective is dense token prediction with causal
  motion-conditioning, not only pooled global prediction.
- Deep token supervision is implemented for selected transformer layers, with
  optional predictor layer-id conditioning.
- Causal integrated-navigation channels provide ego-motion conditioning from the
  current context window.
- The JEPA predictor now has an action-conditioned path: when a cache includes
  navigation channels, the dense temporal predictor receives 6 event-motion
  context features plus 9 causal navigation/ego-motion features. The objective
  is recorded as `dense_temporal_token_action_multihorizon` for those runs.
- The repo now has SkyJEPA-style frozen latent probers for both all-window TTC
  and detection-assisted bbox/ROI TTC. The ROI prober has checkpoint-only
  evaluation so validation-selected probers can be evaluated without retraining.
- The full-starter protocol reports sealed validation/test results and
  low-label transfer for the token JEPA model.

What is still not SOTA:

- Backbone scale: the token transformer is small and local; it is not a
  large-scale ViT trained on broad video data.
- Dense feature learning: the loss is now dense over spatial tokens, but not
  full V-JEPA 2.1 scale or tokenizer depth.
- Deep self-supervision: intermediate encoder layers can be supervised, but the
  current two-layer ablations underperform final-layer-only Token JEPA.
- Action conditioning: the predictor now uses event-derived causal motion
  proxies plus integrated-navigation ego-action features when present, but there
  is still no counterfactual action rollout, planner-conditioned objective, or
  closed-loop control.
- Multi-modal fusion: the current event plus ego-motion path does not use RGB,
  LiDAR, depth, boxes, or segmentation during JEPA pretraining.
- Closed-loop planning: the model estimates TTC; it does not simulate futures for
  planning, counterfactual rollout, or control.
- Latent rollout probing: the current ROI latent prober maps current frozen
  context latents to TTC. SkyJEPA's stronger pattern probes predicted
  multi-step latent rollouts through a structured physical head.
- Benchmark maturity: current results are local and exploratory because the
  mini test split has been inspected repeatedly and the full starter dataset is
  not yet downloaded.

## Practical Interpretation

The implementation is now a stronger small-scale JEPA prototype for event-based
TTC: it is using latent prediction, future horizons, dense token loss,
event-motion conditioning, optional deep token supervision, low-label
evaluation, and anti-leakage discipline. It should not be called a SOTA world
model yet.

The strongest current evidence for JEPA in this repo is the full-starter sealed
run. With 100% labels and event-only input, token JEPA improves sealed-test MAE
from `0.844 +/- 0.008 s` to `0.481 +/- 0.042 s` over three fine-tuning seeds.
With causal integrated-navigation channels, token JEPA improves further to
`0.356 +/- 0.022 s`. With 10% labels and event-only input, token JEPA improves
sealed-test MAE from `1.327 +/- 0.104 s` to `0.460 +/- 0.029 s`.

The current best all-window result in the local full-starter protocol is the
event-tubelet transformer with tubelet masking and transformer dense predictor.
Using the same frozen SSL protocol and fine-tuning seeds 7/13/21, it reaches
`0.231 +/- 0.018 s` validation MAE and `0.312 +/- 0.044 s` CPLA-high test MAE.
This improves MAE by 4.9% versus the previous event-tubelet navigation JEPA
mean (`0.328 s`).

Deep-supervision ablations did not improve this result: the plain deep variant
reached `0.594 s` sealed-test MAE, and the layer-aware deep variant reached
`0.505 s`. These are useful negative results because they show that simply
adding intermediate-layer prediction is not enough at the current model/data
scale.

A larger token transformer was also tested. It reached `0.529 s` sealed-test MAE,
behind the base Token JEPA result. The navigation result suggests that stronger
conditioning is currently more valuable than parameter scaling alone.

The official EvTTC benchmark evaluates bbox/ROI-assisted methods such as STRTTC,
CMax, ETTCM, FAITH, AEB-Tracker, and Image FoE. The recovered local starter
`bbox_segmentation` folders now allow a causal bbox-geometry reference with
train-only calibration: validation reaches `0.279 s` MAE on 106 valid labeled
frames, and sealed `CPLA-high` reaches `0.157 s` MAE on 81 valid labeled test
frames. This is not comparable to the all-window JEPA protocol, but it confirms
that a real SOTA claim requires the official frame/ROI benchmark rather than an
event-only all-window proxy.

## Next Alignment Steps

1. Move from current-context probing to predicted multi-step latent rollout
   probing for TTC, following the SkyJEPA prober pattern.
2. Complete the official bbox/ROI benchmark adapter and download missing CCRs2
   and CCRm assets before any CMax/STRTTC comparison claim.
3. Add a collapse-regularization ablation, e.g. SIGReg-style distribution
   matching or a stronger covariance/variance regularizer, while tuning only on
   validation.
4. Add multi-modal ablations using RGB, depth, boxes, or segmentation without
   leaking TTC labels into SSL.
5. Reproduce or reimplement the Event-Aided TTC/EvTTC reference protocol for
   direct benchmark comparison.
6. Keep low-label and frozen-probe evaluations as first-class metrics.
7. Report final claims only on a new held-out test split that was not used during
   architecture or hyperparameter selection.

## 2026-07-02 Implementation Update

The repo now implements the LeWorldModel/V-JEPA-2-AC-aligned action path in
`src/e_jepa_ttc/training/jepa.py`. It does not use future events, future
navigation, or TTC labels during SSL. Checkpoints and `metrics.json` include:

- `action_conditioning`
- `uses_navigation_action_conditioning`
- `action_feature_dim`
- `action_feature_names`
- `leakage_audit.uses_future_navigation=false`
- `leakage_audit.action_conditioning_uses_context_only=true`

This was the action-conditioning implementation milestone. Subsequent
validation-selected tubelet-mask runs are recorded in
`docs/full_starter_results.md`; those remain local full-starter results, not an
official EvTTC benchmark claim.

## 2026-07-02 SkyJEPA Update

SkyJEPA changes the recommended next step. The current event-tubelet JEPA already
has causal action/ego-motion conditioning and dense token prediction, but the
latest ROI latent prober result shows that probing only current frozen context
latents is insufficient:

- full-starter validation ROI latent prober: `0.226 +/- 0.015 s`,
  `10.78 +/- 0.53%`;
- CPLA-high checkpoint-only ROI latent prober: `0.423 +/- 0.029 s`,
  `23.41 +/- 2.23%`;
- current best all-window JEPA: `0.312 +/- 0.044 s`,
  `6.42 +/- 0.45%`.

The actionable SkyJEPA-aligned path is therefore:

1. predict future latent token rollouts with the JEPA predictor;
2. train a structured TTC prober on those predicted future latents and causal
   bbox/ROI state, using train labels only and validation for selection;
3. add long-horizon error/compounding diagnostics by horizon;
4. add a SIGReg/VISReg-style anti-collapse regularization ablation;
5. only then run the official bbox/ROI protocol on the complete EvTTC sequence
   set.
