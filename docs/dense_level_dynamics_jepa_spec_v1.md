# Dense Level–Dynamics JEPA v1 — architecture commitment specification

Status: proposed commitment boundary, 2026-08-02. This document is an
implementation contract, not evidence that an objective arm works.

## 1. Objective and settled architecture

Build a label-free, high-resolution pretraining path whose online encoder is exactly
the downstream `EJEPATubeletLHR` encoder and whose transfer checkpoint loads into
that model without key remapping or silent shape filtering. The representation keeps
`[B,T,P,D]` semantics throughout the encoder and separates two learned projections:

- `level`: absolute event structure, apparent scale and appearance;
- `dynamics`: temporal change, looming and expansion.

The four preregistered objective arms share the same encoder, predictor capacity,
rows, sampler order, split, optimizer/update budget and seeds:

1. `level` predicts absolute future level patches.
2. `level+temporal_residual` retains the level loss and additionally predicts the
   future-minus-reference residual in the dynamics projection.
3. `level+dynamics_nce` retains the level loss and uses the dynamics prediction to
   identify the correct future state among temporally valid candidates from the same
   track.
4. `level+dynamics_nce+residual_visreg` adds VISReg only to
   `z_dynamic - masked_mean_time(z_dynamic)`.

The online representation module contains the shared high-resolution encoder plus
level and dynamics heads. The target representation is an exact deep copy at
construction, has `requires_grad=False`, is kept in `eval()` even when the wrapper is
in train mode, and is updated by cosine-scheduled EMA. Target patches are detached.
The production path has no end-to-end target gradients.

The predictor is a small aligned-patch transformer. For every valid target patch it
receives only online context tokens, a target-position embedding and a combined
Fourier/learned Delta-t embedding. It never receives future target content. Spatial
mixing remains in the shared high-resolution encoder; predictor attention is
factorized over time for each aligned patch to remain inside the 12 GiB VRAM budget.
There is no action input.

The initial resource contract is numerical, not merely “small”: encoder input is at
most `B=2, T=5, P=240` for the 320x192/p16 pilot, with at most `H=3`; level and
dynamics projection dimensions are 96; the predictor has dimension 96, two layers,
four heads and MLP ratio 2; horizons are processed one at a time and patches in
chunks of at most 60 target queries. Any config exceeding those values must fail
preflight unless it supplies and passes a separately versioned memory profile below
10.5 GiB peak allocated VRAM, leaving margin inside the 12 GiB device. Gradient
accumulation changes effective batch size but never the resident microbatch bound.

Level supervision is dense cosine prediction on valid target patches. Dynamics NCE
uses masked-pooled dynamics tokens for identification, while the underlying head
remains dense. Positives match the requested future horizon. Negatives must share the
same sequence and track, must not be the positive, and must lie outside a configurable
temporal exclusion window. Multiple positives and distance weighting are represented
by an explicit positive-weight matrix; an exclusion mask is forbidden from removing
all positive mass. If a track lacks a valid negative, that anchor is reported and
masked rather than substituted with a cross-track negative.

The temporal-residual target is defined only on the target path. The frozen target
representation encodes both the complete reference window and the aligned complete
future window; its target dynamics head produces `d_ref_target` and
`d_future_target`. On the intersection of their validity masks:

```text
r_target = stop_gradient(layer_norm(d_future_target) - layer_norm(d_ref_target))
```

Cosine loss L2-normalizes `r_target` and its prediction only at the loss boundary;
the raw residual norm is retained as a detached diagnostic. An online reference is
never used to construct `r_target`, and no gradient may enter either target state.

VISReg uses controlled random projections of temporally centred dynamics only.
Temperature, exclusion window, projection count and effective VIS weight are
configurable; `0.12` and `0.04` are pilot hypotheses. Level embeddings are never
passed to residual VISReg. Target-target MSE may be logged as a detached diagnostic
but is not part of the optimized loss.

## 2. Files, ownership and dependency order

Complex implementation lane (Terra), serial first:

- `src/e_jepa_ttc/models/highres_factorized.py`: expose an encoder-only transfer
  contract and explicit post-merge grid/coordinates without changing downstream
  tensor semantics.
- `src/e_jepa_ttc/models/dense_level_dynamics_jepa.py`: online/EMA target
  representations, dual heads and factorized horizon/position predictor.
- `src/e_jepa_ttc/losses/level_dynamics_jepa.py`: dense level, temporal residual,
  within-track NCE and dynamics-only residual VISReg losses.
- `src/e_jepa_ttc/training/eap_highres_jepa.py`: label-free dataset consumption,
  objective assembly, EMA, deterministic resume, health gates and compact artifacts.
- `scripts/pretrain_eap_tubelet_jepa.py`: replace the current guard with the bounded
  high-resolution trainer entry point.
- focused model/loss/training tests owned by the Terra lane.

Routine implementation lane (Luna), only after the Terra interfaces are accepted:

- `src/e_jepa_ttc/data/matched_eap_subset.py` and
  `scripts/build_matched_eap_subset.py`: canonical, signed, nested 256/512/1024/2048
  row manifest with a label-free selection allowlist.
- `src/e_jepa_ttc/evaluation/level_dynamics_probes.py` and a probe CLI: frozen,
  split-safe probes and shortcut diagnostics.
- objective/model/train/experiment YAML files and a matched-manifest schema.
- bounded manifest, CLI, config, provenance and probe tests.
- `STATUS.md`, `PLAN.md`, `RESULTS_INVALIDATION.md`, methodology/model-card and
  reproducibility updates after actual evidence exists.

Shared files and dependency chains are serial. Workers own only the file sets named
in their five-part implementation prompts and must preserve concurrent/user edits.
The primary session inspects every diff and reruns every verification command.

Later phases are separate commitments, not part of the first core patch:

- matched supervised-from-random and JEPA initialization runner;
- reproduced event-only Garl training on the exact same manifest;
- paired macro-by-sequence comparison and bootstrap;
- frozen EvTTC label-free predict manifests and separate scoring;
- RGB-E only after event-only gates pass.

## 3. Interfaces and provenance contracts

The representation contract is:

```text
DenseLevelDynamicsOutput
  level_tokens       [B,T,P,D_level]
  dynamics_tokens    [B,T,P,D_dynamic]
  valid_patch_mask   [B,T,P]
  geometry           PatchGeometry
  diagnostics        mapping[str, Tensor]
```

The predictor contract returns `[B,H,P,D_head]` for level and dynamics, plus the
valid target-patch mask. `horizon_delta_t_s` is a floating tensor `[B,H]`; integer
horizon IDs alone are insufficient. Patch positions are derived from the encoder
geometry and never from boxes or masks.

`HighResFeatures` exposes the encoded grid height/width and normalized post-merge
patch coordinates matching the actual `P` axis. `PatchGeometry` continues to record
source/padding geometry, but callers must not infer post-merge positions from its
pre-merge grid. Tests cover odd grids and verify one coordinate per emitted patch.

The pretraining checkpoint contains at minimum:

```text
artifact_type=dense_level_dynamics_jepa_checkpoint_v1
online_encoder_state_dict
online_level_head_state_dict
online_dynamics_head_state_dict
target_representation_state_dict
predictor_state_dict
optimizer_state_dict
scheduler_state_dict
rng_state
epoch/update counters
objective arm and resolved config
matched manifest hash, dataset hashes, split hash and sampler-order hash
selection rule and health-gate state
explicit per-label-family provenance booleans
```

`online_encoder_state_dict` uses an exact backbone-only key contract. Its allowlist is
`patch_embed.*`, `spatial.*`, `merge.*` when present, `temporal.*` and
`final_norm.*`. TTC/collision heads, query tokens and query attention are excluded
unless a future SSL objective explicitly trains them. The checkpoint records the
structural encoder config (`in_channels`, embedding/patch/window sizes, heads,
spatial/temporal depth and mixer, merge policy). The supervised loader requires exact
structural-config equality and exact expected backbone key/shape equality; it reports
all keys and fails closed on missing, extra or mismatched backbone state. Predictor,
target representation, SSL projection heads and untrained pooling state are not
downstream inference state.

The matched-subset manifest is canonical JSON with its own SHA-256 over content
excluding only the signature field. It freezes source sequence IDs, track IDs, sample
tokens/row IDs, endpoint timestamps/IDs, role, nested selection order, event-only
modality, distinct SSL/downstream input policies, raw-event resize and official ROI
rules, temporal interval, calibration/focal policy, signed TTC scoring convention,
seed set, optimizer/update budget, sampler-order hash, source dataset hashes and
resolved config hashes.

The subset builder reads only `GarlTTC_dataset/data/train.parquet`, using Parquet
column projection at read time. It must never open `annotations/train.parquet`, a
`labels_path`, EvTTC, or unprojected rows. The projection allowlist contains only
identity, sequence, public track, sample token, timestamp, endpoint IDs/timestamps,
event path and event-window fields. TTC, depth/3-D, category, boxes and masks are not
read and are rejected if supplied by another source. Mutation tests prove that
changing prohibited columns in an otherwise identical fixture cannot change selected
row IDs, order or manifest hashes.

Selection builds deterministic temporal track blocks rather than isolated rows. An
eligible block has at least four chronologically ordered rows from one sequence/track,
with separations sufficient for the configured positive horizon, exclusion window
and at least two valid negative candidates. Blocks round-robin by role and sequence;
tracks are ordered by a stable hash and rows by timestamp. Every objective arm receives
the identical block-aware batch order and candidate mask, including arms that do not
use NCE. The 256/512/1024/2048 stages are nested prefixes of whole blocks; a stage may
slightly exceed its nominal row count to avoid splitting a block, and records both
nominal and actual counts. Eligibility/exclusion counts by sequence and track are
reported to expose any long-track bias. Real pilots require at least 80% valid NCE
anchor coverage and at least two valid negatives per included NCE anchor; otherwise
the NCE arms fail the mechanistic gate without training.

The pretrainer accepts the signed manifest and `eap_root` only; it has no
`garlttc_root` or EvTTC argument. Downstream training joins labels separately by
frozen row/sample token after selection is over.

The manifest freezes two distinct preprocessing policies. `ssl_input_policy` is
`full_frame_event_only_320x192_from_raw` and uses no object conditioning.
`downstream_input_policy` is the official Garl shared square object ROI resized to
128x128, with event-only time volumes and the same object box/ROI for our random,
level-only, promoted JEPA and reproduced Garl controls. Boxes are therefore used only
after SSL for the explicitly declared object-conditioned downstream/inference
protocol (`uses_boxes_at_inference=true`); they never enter SSL-Pure sampling,
targets, batching or loss. Rows sharing a timestamp but referring to different tracks
must yield track-specific downstream ROI tensors, preventing identical full-frame
inputs with conflicting object TTC targets. SSL-Object-Conditioned remains a separate
future ablation.

Frozen probes fit only after the encoder checkpoint is fixed. Probe train/validation
roles are sequence-disjoint and inherited from the manifest. Outputs include
expansion/log-height-ratio, TTC/log-TTC, event count/rate, sequence ID, valid track ID,
timestamp/horizon, effective rank, duplication and variance. Level/scale retention is
reported separately. Probe labels never enter SSL loss, batching, EMA, checkpoint
health selection or target construction. Promotion may use the preregistered eAP
development probe rule, never EvTTC.

## 4. Scientific, safety and resource constraints

- SSL-Pure may read raw eAP events, timestamps, sequence boundaries and valid track
  identity only. It may not read TTC, depth/3-D, categories, boxes, masks, RGB or any
  EvTTC asset. Boxes are reserved for a separately named SSL-Object-Conditioned
  ablation that is not part of the first pilot.
- Benchmark-10 remains sealed. EvTTC cannot participate in architecture, checkpoint,
  arm or hyperparameter selection.
- The four arms are preregistered before pilots. No R2 regularization, CMI or HSIC is
  optimized in production. R2 is a detached frozen diagnostic only.
- The same signed rows, split, event modality, sampler order, seeds, optimizer,
  number of updates, validation information and metric implementation are used by all
  arms and later by supervised/random and Garl controls where applicable.
- The 256-row pilot runs first. A stage can grow to 512, 1024 and at most 2048 only
  after finite training, tiny-overfit, validation macro MiD, dynamics/level/shortcut
  probes, collapse and resource gates pass. Full seeds 7/13/23 and RGB-E/full eAP are
  forbidden before pilot promotion.
- The governing confirmation seeds are `[7,13,23]`, following the explicit current
  program and `PLAN.md`; the stale `[7,13,21]` line in `RESULTS_INVALIDATION.md` must
  be corrected before a signed comparison manifest is committed. Each pilot arm uses
  the same active seed set (initially seed 7 only); if confirmation is authorized,
  every retained comparison row is rerun on all three confirmation seeds.
- Promotion requires at least 2% relative validation `paper_MiD_overall` improvement
  over the matched level-only or random control, or a favourable paired confidence
  interval; no material failure-rate regression; improvement in at least two of three
  seeds once three are run; no collapse; no material level/shortcut regression; and
  runtime overhead no greater than 25% unless explicitly retained accuracy-first.
- Results are macro by sequence and global. Paired uncertainty resamples sequences
  and, where support permits, tracks; windows are not treated as independent.
- The pipeline is raw/on-demand and batch bounded. It may write compact manifests,
  metrics, predictions and checkpoints, but no dense event cache. A run preflights
  free disk and estimated VRAM/RAM, caps workers for 32 GiB RAM, uses BF16 where safe,
  closes HDF5 handles and records peak VRAM, RAM, throughput and p50/p95 latency.
- Official Garl release, Garl annotations and eAP sources are operationally
  read-only. No submission or external write occurs without explicit authorization.
- Claim levels stay separate: matched local subset, frozen zero-shot EvTTC and official
  eAP/CodaBench. A local subset result is never described as SOTA.

If no arm passes, the deliverable is a reproducible negative result containing every
arm, gate outcome, exact blocker and the cheapest next falsifiable experiment. No
result may imply that Garl-TTC was beaten without modality-matched evidence.

## 5. Verification and staged acceptance evidence

Core mechanistic acceptance requires tests for exact target initialization, no target
gradients, EMA arithmetic/schedule, eval-only target behaviour, `[B,T,P,D]` shape and
padding preservation, structured/causal masks, changed predictions under changed
Delta-t, intended gradients into both heads, correct within-track NCE positive labels,
positive-preserving exclusion masks, finite/deterministic/differentiable VISReg,
constant-shortcut improvement, no catastrophic frame-varying-nuisance regression,
tiny-batch overfit, deterministic resume equivalence, inference-state exclusion and
absence of label reachability in SSL-Pure.

It also requires post-merge coordinate/shape tests, exact backbone key/config transfer
tests that reject pooling/task state and partial shape matches, track-block coverage
tests, and an assertion that every NCE arm has the predeclared valid-anchor/negative
coverage before the first optimizer step.

Manifest/probe acceptance requires canonical mutation/hash rejection, nested subset
prefixes, disjoint roles, exact row/order reuse, explicit modalities and budget,
prohibited-field rejection, frozen encoder parameters, sequence-disjoint probe fitting
and metrics regenerated from stored predictions rather than handwritten values.

Before real pilots, the primary session runs the focused unit/integration tests, the
semantic shortcut audit, high-resolution integration smoke, tiny overfit,
deterministic resume and label-leakage suites. At every coherent boundary it also runs
Ruff check/format, Pyright, Pytest and `git diff --check` in proportion to the change.

The pilot orchestrator must emit a decision JSON for every arm and stage containing
config/dataset/split/sampler/checkpoint hashes, per-seed metrics, macro-by-sequence
metrics, probe metrics, paired deltas, gate booleans, failure reasons and resource
profiles. Promotion is computed, not edited by hand. Negative and interrupted runs
remain visible and are added to `RESULTS_INVALIDATION.md` when not claim-eligible.

After implementation and primary verification, a fresh Sol reviewer must inspect the
actual diff and evidence. The final program may be reported complete only after a
fresh `ship` verdict, or after a defensible reproducible negative/blocker outcome that
satisfies the goal's alternative definition of done.

## Current gap table at the commitment boundary

| Requirement | Current evidence | State before implementation |
|---|---|---|
| Compatible high-resolution JEPA pretrainer | Entry point exits with an explicit guard | missing |
| Shared dense downstream encoder | `EJEPATubeletLHR.forward_features()` returns `[B,T,P,D]` | operational |
| Dense level/dynamics heads and four arms | No production module/config | missing |
| Exact EMA target for this encoder | Legacy EMA tests exist for another encoder | indirect only |
| Label-free matched subset | Current sampler reads category and TTC bucket | rejected for SSL-Pure |
| Frozen semantic probes on real matched rows | Synthetic shortcut audit only | missing |
| Matched random/JEPA/Garl comparison | Existing screen selects its own capped rows | missing |
| Real label-free EvTTC manifest | Runner guard exists; real manifest absent | externally/data blocked |
| Official eAP/CodaBench | Six train sequences and test authorization/assets absent | externally blocked |
| Quality suite | Green at prior commit per `STATUS.md` | must be rerun on current HEAD |
