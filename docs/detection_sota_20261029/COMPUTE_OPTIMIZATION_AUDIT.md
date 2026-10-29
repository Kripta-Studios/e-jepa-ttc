# Portable detector compute-optimization audit

Status: CPU/source audit only. No GPU experiment, runtime patch, campaign change, or speed claim
was made.

## Evidence boundary

The inspected producer is the frozen R0-8192 source under
`output/detection_campaign/rgb_screen_v5_seed7_v1/automatic_runs_v1/r0_8192/code`.
The retained py-spy profiles contain 1,171 training samples and 1,451 evaluation samples
(`923b8386...` and `51383dce...`). They sample Python stacks, include sampling errors, and do not
measure GPU kernel duration. Counts below therefore identify host-side activity, not elapsed-time
speedups.

The training profile contains 12 samples through `portable_attention.py`, 12 through recursive
`require_finite`, and 6 at the grouped Hungarian assignment site. The evaluation profile contains
116 samples through `portable_attention.py` and 49 through `require_finite`. The executed portable
attention source is `e3a5ca88...`.

The model input is fixed at 640 x 384. The forward creates four feature-level shape tensors,
zero masks for the added level, level-start offsets, valid ratios and positional encodings on every
call. The portable attention implementation converts CUDA `spatial_shapes` with `.tolist()` and
performs data-dependent finite/equality checks using `.item()`/Python branching. The outer runner
then recursively performs another finite reduction and host decision for every output tensor.

The apparent NumPy inverse in `PositionEmbedding` is not on the executed MonoDGP forward path:
that call is commented out and the model consumes `depth_pos_embed` from `depth_predictor`.
`decode_eap()` does invert `event_from_ego` once per decoded frame, so a calibration cache is valid
but lower priority. Hungarian assignments cannot be cached: their costs depend on current model
outputs, current targets and decoder layer/group.

## Prioritized candidates

### 1. Hoist the static attention layout and consolidate its validations

Build an immutable layout descriptor once per fixed feature-shape signature containing the Python
`[(H, W), ...]`, split sizes, level-start tensor and expected level count. Pass it explicitly through
the 2D/3D transformers and portable attention. Validate the freshly produced feature shapes once
per model forward, then reuse the descriptor in every encoder/decoder attention call. Keep finite
validation, but reduce all sampling-location flags into one CUDA boolean per forward and synchronize
once; on failure, rerun the existing per-call checks to retain the actionable error path.

This removes repeated `.tolist()`, `torch.equal` host decisions and `.item()` synchronizations while
leaving `grid_sample`, weights, operation order and gradients unchanged. It must fail closed on a
different resolution, level count, layout, dtype or device.

Admission: CPU layout unit tests; four-frame frozen-checkpoint forward parity for every nested output
including velocity with maximum error zero; training-mode isolated forward/backward comparison on
identical tensors; invalid shape, nonfinite sampling and wrong level-start negative fixtures; CUDA
profiler evidence showing fewer host synchronizations. No AP claim is needed for a bit-identical
change.

### 2. Fuse recursive output finite checks into one synchronization

Flatten the nested output structure deterministically, compute one `isfinite().all()` flag per tensor
on device, stack/reduce those flags and make one host decision. Only if it fails should the current
recursive validator locate and report the offending tensor. Apply the same pattern separately to
model outputs and loss dictionaries; do not remove either gate.

The evaluation py-spy profile has 49 samples in this validator, making it the clearest small,
exact host-side candidate after attention. Risk is diagnostic degradation or accidentally omitting a
nested tensor, controlled by retaining the old traversal as the failure slow path and testing nested
dict/list/tuple structures, empty structures, NaN and infinity.

Admission: exact pass/fail equivalence against the historical validator over generated nested
structures; exact error path/name on the slow path; frozen forward parity; profiler count of CUDA
synchronizations. This does not cache tensor values and cannot hide a later nonfinite result.

The isolated implementation is `detection/finite_validation.py`. It is not integrated into a
campaign. A future copied producer must invoke it at the same gates as the historical validator:
model outputs before loss construction, and losses before backward, clipping or optimizer update.
Its fused success path still evaluates every tensor leaf. If any per-device reduction fails, it
reruns the historical traversal and reports the same first named offender; consolidation therefore
does not mask or postpone NaNs/infinities.

### 3. Cache fixed-shape masks, ratios and parameter-free positional scaffolding

For the admitted 640 x 384 geometry, cache all-zero masks, `spatial_shapes`, `level_start_index`,
all-one valid ratios, and parameter-free sine positional tensors by a strict key containing batch
size, feature shapes, device, dtype and producer version. The last partial evaluation batch needs a
separate key or eager fallback. Do not cache learned projections, region/depth outputs or anything
derived from RGB pixels.

The cached tensors have no gradient dependency on images or trainable parameters. Nevertheless,
the patch must retain a per-forward feature-shape assertion before reuse. The main risks are stale
device/dtype tensors, overlooking a padded/nonzero mask, and memory growth across shapes; use a
small fixed-capacity cache and reject rather than generalize unknown geometry.

Admission: compare masks/ratios/position tensors exactly to eager construction for train batch four,
evaluation batch four and tail batches; full nested-output parity and backward gradient parity;
bounded cache-entry/RAM receipt; negative padded-mask fixture proving it does not reuse the zero-mask
entry.

### 4. Compile only a static tensor region before considering CUDA Graphs

Start with a new experimental producer that compiles the backbone plus tensor-only detector forward
using `torch.compile(..., dynamic=False, mode="max-autotune-no-cudagraphs")`. Keep target preparation,
SciPy matching, loss assembly, finite host gates, decoding and metrics eager. Use
`TORCH_LOGS=graph_breaks,recompiles,guards` and record cold compilation separately from steady state.

Full training capture is not presently viable: grouped Hungarian matching transfers changing costs
to CPU and calls SciPy, targets have variable cardinality, and current attention/validation contains
host synchronization. Evaluation CUDA Graph capture can be reconsidered only after candidates 1-2,
with fixed batch-four static buffers and eager fallback for the final short batch. Raw CUDA Graphs
require fixed shapes, control flow and memory addresses and prohibit CPU/GPU synchronization inside
capture. `torch.compile` can also retain additional workspace and CUDA-graph memory.

Admission: eager-versus-compiled nested outputs, decoded predictions and velocity; training loss and
gradient comparisons over a frozen fixture; compile time, graph breaks, recompiles, peak VRAM and
steady-state latency after warm-up. Any numerical difference makes this a new recipe rather than an
engineering-equivalent backend.

### 5. Evaluate TF32, then BF16, only as numerical recipe changes

The campaign explicitly uses `torch.set_float32_matmul_precision("highest")`. A preregistered
`"high"` candidate may enable TF32/bfloat16-based internal CUDA matmul while retaining float32 output
dtype; cuDNN convolution TF32 is controlled separately. This is not exact and needs detector metrics,
loss/gradient stability and several seeds. BF16 autocast is a second, separate experiment with a
complete allow/deny list for sensitive geometry, depth, softmax, matching costs and losses. It must
also remeasure host RAM and VRAM because the earlier BF16 profile was confounded by memory pressure.

Admission: frozen-input error distributions for every output head and decoded geometry, finite/loss
gates, short-run loss trajectory, peak RAM/VRAM, steady-state latency, and final multi-seed metric
gate. Do not combine precision, compilation and CUDA Graphs in the same first experiment.

## Explicit non-candidates

- Do not cache Hungarian assignments or matching costs; both change with weights, inputs and layer.
- Do not preload all prepared float images. At 28,054 x 3 x 384 x 640 x float32, image tensors alone
  require about 82.7 GB before targets/caches; even resized uint8 images are about 20.7 GB.
- Do not optimize the inactive NumPy inverse in `PositionEmbedding` as though it were a current
  bottleneck. A small exact cache for `decode_eap()` calibration inverses is acceptable only after
  profiling shows decode CPU time matters.
- Do not remove validation to make capture possible. Move or consolidate it outside the captured
  tensor region and prove equivalent failure behavior.

## Recommended order

Implement candidates 1 and 2 together in a copied portable variant because they remove graph-hostile
synchronizations while preserving semantics. Measure before adding candidate 3. Only if GPU traces
then show launch/host overhead should candidate 4 proceed. Candidate 5 remains a scientific recipe
experiment, not a transparent acceleration.

Official API constraints used in this audit:

- https://docs.pytorch.org/docs/stable/notes/cuda.html
- https://docs.pytorch.org/docs/stable/generated/torch.compile
- https://docs.pytorch.org/docs/main/generated/torch.set_float32_matmul_precision.html
