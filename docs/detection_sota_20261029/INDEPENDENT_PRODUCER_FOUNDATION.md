# Independent E/R producer foundation

## Decision

The first executable learned leaf is **E_A5**, initialized from seed 7 with no
historical learned checkpoint. The common contract now also represents R_A5 as
a separate fresh RGB leaf. Its exact rotation-0 raw-RGB population metadata is
frozen, but neither its first physical batch nor training is admitted. This is
a dependency decision, not an authorization to train. All 24 physical event-H5 files now
have full-hash and bounds admission. The admitted population has 52,950
history-valid queries and excludes 97 prefix queries. No optimizer update has
been executed.

The frozen DINO cache is an auxiliary target parent, not a learned PHASE parent
and not an inference input.  Its producer-role view is joined by unique
`sample_token`, exposes only `float16 [2,6,32,32]` relational targets, and does
not expose TTC values or source pixels to the fit.

## Historical source map

The reference source commit is
`2ad7a05797799caf1bcdc9202c307d244f9fe31d`.

| Concern | Frozen source |
|---|---|
| E_A5 runner and resume state | `operational/train40_system/models.py` |
| E_A5 protocol | `artifacts/train40_system_20261005/TRAINING_PROTOCOL.json` (`39fac190...`) |
| R_A5 runner | `operational/final_refit/producers.py` |
| R_A5 recipe | `configs/rgb_port/producers.json` (`9ffb221e...`) |
| Batch schema | `src/e_jepa_ttc/data/object_event_v4.py` |
| Producer model | `src/e_jepa_ttc/models/causal_scale_ttc.py` |
| Objective | `src/e_jepa_ttc/training/causal_scale_eap.py` and `src/e_jepa_ttc/losses/causal_scale_ttc.py` |

Historical commands, recorded as references rather than launch commands:

```text
python -m operational.train40_system.models --output <run> --arm a5
python -m operational.final_refit.producers --source <descriptor> --teacher-manifest <manifest> --config configs/rgb_port/producers.json --run <run> --recipe <recipe> --fit-id R_A5 --device cuda
```

Both leaves use the exact historical `CausalScaleTTC` source class: hidden 64,
geometry 128, residual depth 2, resize-conv foreground decoder, full-resolution
dimension 24, causal-left smoothing 0.15, and legacy transport for A5.  R_A5
changes modality and input channels to RGB/3 and explicitly enables deterministic
transport sampling.  This does not claim that the engineering-default smoke
configuration was the sealed production recipe.

The fixed optimization recipe is seed 7, 18 epochs, batch 32, AdamW at `3e-4`,
weight decay `1e-4`, cosine scheduling to `3e-5`, gradient clipping at 1.0,
BF16, three foreground-warmup epochs, and last-epoch selection.  The historical
runner contains no additional online augmentation.  The new independent update
count is `18 * ceil(producer_role_queries / 32)` and must be derived from the
exact rotation-role receipt, not copied from the historical 49,932-update
TRAIN40 run. For rotation 0, the history-valid population is 35,048: the
historical no-drop loop therefore uses 1,096 updates per epoch, including a
final batch of 8, and 19,728 updates across 18 epochs. The earlier prospective
19,710-update/drop-last plan is preserved only as **NOADMITTED** and is not the
scientific execution contract.

## Inputs and supervision

The raw independent history is 31 distinct half-open 20 ms observation packets
covering 620 ms.  Their 31 endpoints span 600 ms and define 30 transitions.
This is not the historical producer tensor. `CausalScaleTTC` accepts a genuine
ObjectEventV4 T3 tensor from the frozen raw-derived INPUT cache; therefore no
“last three packets” rule is assumed. The cache is joined by exact sample token,
and its common ROI, clocks, channels and delta values are verified independently.

The historical A5 objective is supervised: TTC/log-ratio, risk thresholds and
training-only bbox geometry are combined with DINO relational distillation.
It must not be called SSL.  Raw GT ROIs may be used only for training preparation
or diagnostics and are absent at inference.  A future representation-only SSL
mode is explicitly fail-closed against TTC, collision, bbox and raw GT ROI
fields.

Inference-visible fields are sensor tensors, measured delta time and observable
motion.  TTC, collision labels, bbox geometry, raw GT ROI and teacher relation
targets are training-only.

## Implemented fitter and current gates

`operational/independent_refit/producer_contract.py` binds the exact fold,
raw-history V4, H5 admission lineage, teacher cache, producer-role receipt and
historical source bytes. `producer_inputs.py` exposes only the 35,048
history-valid rotation-0 producer tokens through the native T3/C12 cache, with
an LRU of at most two INPUT shards. `producer_train.py` is the real supervised
E_A5 fitter: fresh seed-7 initialization, fixed last-epoch selection, durable
optimizer/scheduler/sampler/RNG state and transactional curves. It has not been
launched.

The integrated E/R source physical gate passed on a real E_A5 batch of 32. INPUT and selected
teacher payloads were hash-checked, tokens and all training tensors were finite,
and the exact shapes included event `[32,3,12,128,128]` and DINO relations
`[32,2,6,32,32]`. The receipt is
`output/independent_phase/e_a5_rotation0_source_admission_v3/FIRST_BATCH_RECEIPT.json`
(file SHA-256 `e821b305...`). Peak observed process RSS remained below 1 GiB.

The engineering-only gate then passed for the preselected token
`2cyv0Oedzg_000004_19318000000`: fresh E_A5 seed 7, native T3/C12 input,
full FP32 CPU forward, historical supervised loss and backward. Every model
head, diagnostic, loss component and all 59 trainable gradients were finite;
the gradient L2 norm was `11.7692`. Peak RSS was 848,097,280 bytes under a
3 GiB guard. No optimizer was constructed, no update occurred, and CUDA was
hidden. This is not evidence of BF16/GPU equivalence or model quality. The
receipt is
`output/independent_phase/e_a5_rotation0_cpu_engineering_smoke_v2/RECEIPT.json`.
Its integrated pre-execution source bytes are retained under
`output/independent_phase/producer_er_source_snapshot_v2` (manifest SHA-256
`f6c168f3...`).

The preceding v2 source-admission and v1 CPU-smoke receipts remain valid
observations of their runs, but their exact pre-R_A5 glue bytes were not
retained and cannot be reconstructed. They are historical evidence, not a
reproducible source freeze for the current common tree.

For R_A5, `producer_rgb_inputs.py` freezes the 35,048 history-valid raw-RGB
rows and 68 excluded rows against the independent population, P/H/V manifests,
public file inventory and nominal teacher role. The current descriptor is
`output/independent_phase/r_a5_rotation0_rgb_source_v2/SOURCE.json` (file
SHA-256 `88a4b4fc...`). Its status is metadata-only: no tar member was decoded,
no first batch was admitted and no optimizer update occurred.

E_A5 uses the historical row-local robust per-window event normalization. It
has no learned population normalizer to refit, so a nonexistent normalizer
receipt is not a gate. Any future component that actually learns normalization
statistics must fit them from its producer-role population and bind its own
receipt. Remaining launch authority is limited to explicit GPU allocation and
the root run controller. E_A5 has passed its source and CPU engineering gates
on the immutable integrated E/R snapshot; R_A5 remains closed at its metadata
gate.
