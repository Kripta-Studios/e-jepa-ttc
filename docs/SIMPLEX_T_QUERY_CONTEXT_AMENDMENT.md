# SIMPLEX-T query-conditioned context amendment — pre-fit

Authority: the user explicitly states that original histories are unavailable,
requests an executable alternative, and confirms effective replay handoff.
No scientific fit has run and no scientific model/config freeze exists.

The candidate alternative is retrospective sensor context conditioned on the
current query's supplied detection ROI. It does not reconstruct Garl/eAP IDs,
use target-filtered prior query membership, or assert verified same-object
trajectories. Prior windows must come from raw event timestamps on a fixed
input-only schedule; their crop depends on the current supplied ROI. Consequently
their availability is at least the current ROI availability, not their historical
sensor endpoint. They are NOT predictions that an online tracker emitted then.

This preserves the comparison of additional past sensor information versus
current-only correction, unchanged expert selection and smoothing, but changes
the object-history estimand. Results must use a separate query-context namespace
and cannot satisfy a claim of original SIMPLEX-T same-track history equivalence.
The original source-absence diagnosis is retained, not overwritten.

Before this alternative can be frozen: audit the existing producer crop and
clock implementations, demonstrate current-query replay, establish strictly
past event access and current-ROI dependency timestamps, implement controls and
input-only membership tests, and bind the resulting cache/schema. Preserve the
registered head recipe, finite fit budget, group exclusions and all closed roles.
No expert refit or teacher change is authorized. Do not train from a speculative
adapter or claim that this design note is a validated executable alternative.

First execution step: the already selected64 TRAIN queries are replayed through
all12 matching A5/C2F/PAIR families using historical frozen input tensors. Export
actual differences and latent values; a failed replay is evidence, not permission
to tune tolerances or silently alter preprocessing. This diagnostic has zero
optimizer updates and is independent of the missing trajectory source.

Operational note: user-confirmed replay handoff supersedes the old absent-slot
status. Stage70 cache worker29120 remained visible at confirmation. Do not kill,
modify or resume it; keep a single inference process and bounded host memory.

## Completed replay audit, before scientific freeze

The192 raw windows for all64 selected TRAIN queries reproduce the historical
input tensors bit for bit. The original and recovered training caches also
agree bit for bit on these64 rows. The initial single-query t0 discrepancy
was a1us error in the probe: the historical shifted_precontext_window helper
uses max(requested shift, actual window duration). The corrected probe reuses
that helper; original failed evidence is retained.

Historical A5 FP32 extraction depends numerically on cuDNN TF32 and on the
query's producer-filtered batch size AND its position. Reproducing that shape
and position yields exact phase and128-D token equality on all64 queries.
The diagnostic uses repeated identical inputs to reproduce shape/position;
these padding copies are not additional scientific observations.

Historical A5/C2F point outputs were produced by the BF16 training evaluator,
with deterministic algorithms and its own validation batch positions. Restoring
that documented runtime reproduces the64 A5 and64 C2F point outputs exactly
in their source FP32 representation. Remaining float64 CSV differences are
at most7.11e-15 seconds, not new numerical model differences. PAIR likewise
reproduces all64 outputs using the signed FP32 feature payload and original
GPU1024-row inference shape/position. Together the checks cover36 checkpoints
and all12 producer families. No fitted weights or tolerances were changed.

HISTORICAL_REPLAY_QA_PASSED.json binds this component-wise proof to file hashes.
It does not authorize mixing BF16 scalar/variance fields into a new FP32 cache.
The future query-context cache must consistently use one declared FP32 route
for all current and prior fields; historical RISK17 replay remains a separate
comparator. Production cache integration, actual timing-bound CPU resume,
remaining pre-fit QA and scientific freeze still precede every head fit.

## Input-only context index

The D0 metadata index now covers all8192 original queries and assigns each to
the appropriate nested family independently for each of the three outer folds.
Raw stream bounds permit H8 for8182 queries and H16 for8171. Other queries keep
only complete supported windows, with explicit missing slots. These counts are
input availability, not predictive results or evidence of same-object tracking.

Content deduplication binds raw-source digest, frozen preprocessing digest,
producer family, three shifted intervals, exact supplied crop, anchor and current
ROI availability. Query names do not force duplicate feature rows. Distinct
producer families and distinct availability dependencies cannot alias. Tests
verify label perturbation invariance, cold starts and dependency isolation.
Neither the index nor its deduplicated keys are an extracted feature cache.

## Coherent FP32 extractor integration

The reusable extractor now produces PHASE17,128 A5 pair tokens, original expert
TTC and known-support flags from a single FP32 route. Its scalar definitions
reuse the historical evaluator: current voxel count/rate channels, final
transport-flow diagnostic, minimum phase/support guard margin and log variance.
The PHASE17 ordering is assembled by the existing Stage61 feature function.

On all64 signed current-query inputs across12 families, this implementation
reproduces the earlier same-runtime FP32 expert predictions and133-D PAIR
inputs exactly. The hashed receipt is in
artifacts/simplex_t/T0/coherent_fp32_extractor_receipt/QA.json.
This validates extraction integration, not the unbuilt full temporal cache;
the earlier historical BF16 parity proof remains separate. All operations here
use frozen weights and zero optimizer updates. Four contract tests reject
wrong precision, wrong window count, nonpositive time deltas and training mode.

## Temporal cache execution started

The production extractor reads the actual past raw event windows for each
query-conditioned crop. It uses a fixed16-slot chronological batch per query,
FP32 with TF32 disabled, and exports only supported slots. This layout is
distinct from the historical evaluator and is bound in the cache identity;
current and prior observations share the same route. Frozen experts are not
rerun during head updates.

Each completed query/family block is fsynced and hashed before its receipt is
published. Restart verifies receipts and skips completed blocks. Unreceipted
files are retained and trigger recovery review, never silently overwritten.
The initial4-block run and subsequent resumed run use the same source identity.
The content audit checks exact observation indices, phase/point consistency,
chronology and non-backdated ROI availability without reading targets.

Execution remains partial. The required D0 cache has24576 query/family blocks.
No completion or scientific fit is implied by an intermediate block receipt.
The runner currently has one Pyright diagnostic on NumPy keyword unpacking at
savez_compressed; runtime output and its schema are separately checked. This
typing issue remains on the pre-freeze QA list.

The fold compiler requires all8192 query/family receipts for its outer fold
before allocating feature arrays. It verifies each block's hash, exact indices,
float32 shapes, finite values, integer times and current ROI availability, then
builds memory-mappable arrays and publishes a completion manifest only after
every consumed observation is covered. A real partial-cache invocation rejected
the first missing receipt and created no compiled output. Seven unit tests cover
schema, identity, timing, dtype, nonfinite and forbidden-target rejection.
