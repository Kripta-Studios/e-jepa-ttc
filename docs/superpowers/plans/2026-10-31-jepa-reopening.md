# JEPA reopening: gated execution plan

> **Closed on 2026-10-31 by user decision.** The three-seed P512/H256
> fixed-teacher pilot and terminal evaluation completed; the primary benefit
> gate failed. Do not execute the remaining rollout, scaling or integration
> tasks. See [the closure report](../../jepa_closure_20261031/README.md) and
> [ADR-0045](../../decisions/ADR-0045-close-jepa-fixed-teacher-pilot.md).
> The task list and execution update below are historical planning records,
> not the current runtime state. The actual pilot used label-free static/future
> targets, not the entire proposed supervised A–E matrix. Inputs finished
> uploading, and the research queue and pilot completed. The bounded closure
> analysis and verified archive supersede the earlier waiting statuses.

> **For agentic workers:** Use superpowers:executing-plans. Record evidence before advancing a gate.

**Goal:** Determine whether predictive training can improve PHASE-TTC representations and prepare a reproducible, bounded experiment on the supplied RunPod volume.

**Architecture:** Preserve the published model and historical evidence. Audit V8 first, then qualify a frozen teacher with causal temporal probes, then compare trainable event students under matched supervision and compute. Autoregressive training is conditional on one-step prediction demonstrating value.

**Tech Stack:** Existing Python/PyTorch pipelines, JSON evidence, SSH, sequence-grouped evaluation.

**Spec:** User-approved research stages in the conversation of 2026-10-31; AGENTS.md scientific integrity requirements.

## Global constraints

- Do not modify historical artifacts, the published model, or the running detector campaign.
- Separate historical diagnostic evidence from independent confirmation.
- Preserve sequence grouping and teacher/normalizer ancestry; TRAIN40-trained producers cannot create an independent historical fold.
- Keep future observations as targets only. Context crops and conditioning must be causal.
- Record seed, commit, data/split hashes, environment and actual compute.
- Do not start large GPU training before prerequisites pass; do not share the GPU with the running campaign implicitly.

## Review focus

- Stale aggregate vs rerun OOF: retain both identities and dates.
- Healthy spatial variance masking constant outputs across examples: measure both axes.
- TTC clipping concealing tiny predicted expansion: record raw ratio and inverse TTC.
- Present files with absent events/timestamps/calibration: validate required modalities, not just directory names.
- Teacher exposure or shuffled targets with leaked correspondence: check ancestors and track/sequence membership.

## Task 1: Historical V8 audit

- [x] Verify checkpoint and OOF identities, exact snapshot and nominal/effective label fractions.
- [x] Replay pretrained encoder on real cached inputs on CPU; compare sample-wise foreground/token/ratio variance with scratch.
- [ ] Check decoder round trip, clipping, trainable masks and transfer hashes.
- [x] Report what is and is not recoverable: downstream weights were not retained.

Deliverable: diagnostic JSON under `artifacts/jepa_council_20261031/`, with commands and source identities. Replay is a diagnostic, not a new transfer result.

## Task 2: Pod readiness

- [x] Inventory persistent storage, GPU workload, repository identity, data modalities, checkpoints and software without reading credentials.
- [x] Check per-sequence modality counts, representative schemas, required manifests and missing historical caches.
- [ ] Save machine-readable inventory and a readiness decision by stage.
- [x] Transfer the bounded historical replay bundle; verify hashes before CPU execution in an isolated directory.

Deliverable: `artifacts/jepa_council_20261031/pod_inventory.json` and isolated `/workspace/jepa_research_20261031/` if staging is needed.

## Task 3: Teacher qualification (conditional)

- [ ] Establish a sequence-held-out diagnostic split and explicit exposed-data status.
- [ ] Compare current producers and a fixed video teacher using the same causal ROI/history and linear plus small temporal probes.
- [ ] Recover expansion, sign and TTC on held-out sequences; compare persistence and simple extrapolation.
- [ ] Stop if data alignment, ancestry or dynamic information is inadequate.

## Task 4: Event-student experiment (conditional)

- [ ] A: supervised reference; B: A plus current-frame distillation; C: B plus one-step future prediction; C-shuffle: matched shuffled future targets.
- [ ] Ensure gradients update the student; evaluate after discarding the auxiliary predictor.
- [ ] Only if C exceeds B and shuffle: compare direct multihorizon and short autoregressive rollouts with identical targets and compute accounting.
- [ ] Replicate representation training across three seeds, bootstrap by sequence, preserve negative/short-TTC/tail performance and coverage.
- [ ] Test integration against equal-capacity ZERO-event and RGB controls only after these gates pass.

No numerical gain or novelty is assumed. Exact new training configurations must be fixed after the diagnostic/data gates, before inspecting their results.

## Execution update: 2026-10-31

The user explicitly authorized uploading inputs immediately and beginning when the GPU becomes available. Two resumable uploads now stage 53,082,864,750 bytes (8,368 files) in `/workspace/jepa_research_20261031/data_inputs`. They preserve original source manifests and validate SHA256 per transferred chunk. This includes matched P/H/V event and RGB inputs (53,890 / 18,422 / 16,432 queries), original P-only producers and historical probe features. The current TRAIN40-exposed producers are staged separately.

The pod originally had all 136 RGB tar shards and 40 matching label tables, plus matching media metadata, but none of its 40 referenced event HDF5 files. The 53 GB cache bundle is sufficient for current-window probes; it is not a validated future-rollout dataset. No 449 GB raw-data migration was launched.

Historical replay has completed on CPU both locally and remotely: nine checkpoints verify; D2/D3/D4 representations have negligible sample-dependent foreground variation, and their saved downstream predictions saturate at +/-60 s. The decoder round trip passes. Lost downstream weights and training trajectories prevent a complete causal diagnosis of failed fine-tuning.

Official V-JEPA 2.1 distilled ViT-B is downloaded and strictly loaded on CPU. It is a screening candidate, not the ViT-G used by RoboJEPA. Code commit, checkpoint hashes, independent-image preprocessing and fixed spatial projection are recorded in `artifacts/jepa_council_20261031/teacher_setup_result.json` and related receipts.

The remote diagnostic queue has 29 explicit tasks: historical CPU probes, GPU preflight, fixed-teacher feature extraction, current and P-only producer extraction, and sequence-disjoint probes. It protects the existing detector controller by PID plus process start identity, then requires six idle GPU observations. Each task also waits for its required upload receipts. The queue does not automatically train a student or run future rollouts. Exact H/V cohorts are fixed before inference: T3 only, 64 evenly spaced tokens per sequence. All comparisons retain the historically exposed-data limitation.

Remote status: `/workspace/jepa_research_20261031/research_queue_status.json`.
Local upload logs: `artifacts/jepa_council_20261031/upload_progress.jsonl` and `upload_event_progress.jsonl`.
The first historical CPU task has already completed on the pod (18,422 H and 16,432 V rows); the queue is waiting for the existing GPU campaign to finish.
