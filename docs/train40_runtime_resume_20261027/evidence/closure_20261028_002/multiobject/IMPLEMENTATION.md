# Multiobject integration — current implementation

New modules: `operational/train40_runtime/batched_context.py`,
`operational/train40_runtime/multiobject.py` and
`scripts/benchmark_train40_multiobject.py`. Frozen source files and weights are
not edited. This is an additive, experimental runtime. GPU pilots below passed
for one/two regions; the twenty-region pilot failed its fixed parity gate.

The service accepts one raw event packet, one shared causal RGB frame history,
and up to twenty distinct object IDs with separate event/RGB regions. It ingests
the packet once, queries each event region, crops original RGB frames per region,
runs grouped producers with group size four for events and native RGB history
length for RGB, then runs a batched padded-eight context. E/R concurrent streams
wait on input production; tickets and guards are checked before publishing TTC.
The final host transfer contains all object predictions. Object birth/removal
does not reuse another object's context. Missing RGB produces the frozen event
fallback. The service does not implement detection or tracking.

The event backend is injected and follows the existing IntegralEventHistory
initialize/ingest/query_many interface. TemporalPrefixEventHistory uses the same
interface. Backend outputs are borrowed and consumed before another query.
Raw frames and their clocks must remain immutable through update completion.

`update` measures from available CPU RAW packet plus raw RGB frames to every
host TTC, including current ROI work and all transfers/models. Startup graph
priming uses synthetic zero inputs. The pilot rebuilds history per registered
query outside the clock and loads raw frames outside the clock. It is not a
continuous stream. Artificial region zero is the original annotated region;
the other nineteen are deterministic distinct translations, not real tracks.

The pilot records each timed update before oracle evaluation. It then compares
against the original event encoder, original PIL crop and original complete
model predictor for each declared region. The primary region is also checked
against the preserved prediction. Every actual consumed event/RGB tensor is
hashed after timing; repetitions must have identical hashes before the final
repetition's full tensor comparison is applied to them. ROI, exact event-window
microseconds, frame indices, sensor clocks and availability are separately
bound and compared. All three output TTC components use atol=rtol=1e-5.

CPU validation: 27 tests pass, Ruff passes and Pyright reports zero errors.
Tests cover 1/2/5/20 batched contexts versus independent contexts, weight
preservation, object identity, missing RGB, births/removals, gaps/rollback,
ROI application, two-frame native startup, causal clocks and reference receipts.

Suggested initial GPU command (must reserve GPU and freeze runtime sources):

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' scripts/benchmark_train40_multiobject.py --candidate D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE --manifest artifacts/final_train40_transfer_20261027/QUERY_MANIFEST.json --reference artifacts/final_train40_transfer_20261027/final_train40/own --output artifacts/train40_runtime_resume_20261027/multiobject/pilot_1_2_v1 --counts 1 2 --indices 0 --repeats 2 --parallel
```

Then extend to counts 5/10/20 and indices 0/200/553, with new output directories.
`--backend prefix` explicitly selects the temporal-prefix experiment. No run may
be admitted if another agent edits a source bound in original_snapshot.json.
The first source inventory included all runtime modules. The later runner
excludes only explicitly named, unimported experimental modules; their names
are recorded and a later import invalidates admission. Enabled cache modules
are imported before this decision and are included. Parent owns
GPU scheduling, final inspection, publication and any sustained paced run.

## GPU pilots executed after reservation

All pilots used prefix/sort, event endpoint reuse disabled and RGB endpoint reuse
enabled (`--parallel --no-event-reuse --backend prefix`, except the declared
serial control). They include actual consumed-input hashes, independent tensor
checks and detailed history receipts. All preservation checks found 610 files
intact. No twenty-track or sustained-rate claim follows from these pilots.

1. `pilot_1_2_v1`: counts 1/2, indices 0/137, two repetitions. FAILED.
   One object passed all cases. The first two-object update at index 0 had
   PHASE error up to 0.119083881 s and E_CTX error up to 0.258887291 s.
   The second repetition, with identical consumed input hashes, passed.
   RGB passed. This was far larger than ordinary rounding drift.
2. `pilot_2_only_parallel_v1`: two objects alone in a fresh process, index 0,
   four repetitions. COMPLETE.
3. `pilot_1_2_serial_v1`: counts 1/2, index 0, three repetitions. COMPLETE.
   This control omitted index 137, so it is not an otherwise identical causal
   experiment against the initial failure.
4. The producers lane changed graph capture to an explicit per-entry stream
   and retained that stream. The previous code captured both E/R graphs on the
   implicit global capture stream, a plausible source of shared library scratch
   workspace during concurrent replay. The old producer source is preserved
   by the producers lane with its matching original SHA.
5. `pilot_1_2_dedicated_stream_v1`: counts 1/2, indices 0/137, five repetitions.
   COMPLETE: 20/20 updates pass all three TTC outputs, original primary parity,
   tensor checks, history receipts and preservation. One-object median 24.82975
   ms, mean 29.74366 ms, range 23.1521–75.0675 ms (first cold update retained).
   Two-object median 32.99125 ms, mean 33.38843 ms, range 31.8877–38.8414 ms.
   The large discrepancy did not recur in this probe; that is not a proof of
   correctness for every graph lifetime or shape transition.
6. `pilot_20_dedicated_stream_v1`: twenty distinct translated regions, index 553,
   three repetitions. FAILED parity. Complete times: 413.0838, 389.7983 and
   374.7580 ms (mean 392.5467 ms). Host preprocessing spans were 175.4503,
   159.9605 and 144.8758 ms; these are host spans rather than isolated GPU stages.
   All twenty input tensor comparisons and history receipts passed, as did the
   preserved primary prediction. E_CTX passed every region. R_CTX exceeded
   tolerance for ROI 04/07/08/13/16 in all repetitions: maximum absolute error
   0.000785827637 s, maximum relative error 2.03449797e-5. PHASE exceeded tolerance
   on ROI 13 (absolute error 0.000101566315 s versus allowed 0.0000824322033 s).
   Maximum PHASE error across all regions was 0.000110626221 s. This smaller,
   repeatable batch-sensitive discrepancy remains unadmitted; tolerances were
   not changed.

The exact 33 source files bound by the post-fix pilots were archived under
`source_snapshot_dedicated_stream_v1/` as `.txt` files with their SHA-256 manifest.
Each current source was required to match the executed binding before archival.
GPU was explicitly released after the twenty-region pilot.

## Twenty-region chunked RGB and prefix query graph pilot

`pilot_20_chunk1_querygraphs_v1` is COMPLETE: 20 distinct artificial regions,
index 553, three complete updates. The run uses E endpoint reuse OFF, R reuse
ON with one object's RGB history per producer call, prefix queries captured
at startup and exact top-k normalization. FP32 and TF32 OFF remain unchanged.
The twenty-region input/original-predictor checks, primary frozen prediction,
all receipts and coverage pass. All 607 candidate/reference/source files are
intact. This does not validate twenty annotated tracks or sustained operation.

Complete latencies are 379.1087, 386.4516 and 385.5828 ms: mean 383.714367 ms,
p95 386.36472 ms, p99 386.434224 ms (linear interpolation, only three samples).
None meets 20 ms. Event host dispatch spans are 37.2418, 33.5845 and 41.314 ms;
RGB preprocessing spans are 127.5622, 129.777 and 128.733 ms. All preprocessing
is ready at 129.034, 138.5779 and 136.9045 ms after the update boundary. These
host intervals overlap; they must not be summed as independent stages.

Maximum TTC absolute differences in the first repeat are PHASE 5.912781e-5 s,
E_CTX 3.337860e-6 s and R_CTX 1.678467e-4 s; all satisfy the unchanged
`1e-5 + 1e-5 * abs(reference)` threshold. RGB object chunking resolves the
previous batch-dependent failures for this case. Query graphs and chunking
changed together, so this pilot is not an isolated speed comparison for either.

Prefix buffers report 3,402,602,592 bytes. Query capture changes the process
allocator by 121,634,816 reserved bytes and 1,024 allocated bytes. Reserved
private graph workspace is therefore material despite the small live delta.
The exact 30 executed source files were checked against binding SHA-256 values
and archived in `source_snapshot_chunk1_querygraphs_v1/` before further edits.

The optional shared-PIL path is not the default. Its CPU comparison on real RAW
frames and twenty translated ROIs is in `shared_pil_cpu_v1.json`: both paths
are bit-exact. Index 0 ROI-first median 36.10445 ms versus shared-PIL 46.54475 ms;
index 553 ROI-first 137.7785 ms versus shared-PIL 118.6429 ms. Benefits depend on
ROI size. The old crop already sliced the ROI before creating a PIL image.

## Exact RGB feature cache integration

`--rgb-feature-cache` enables the separately tested bounded `RGBBatchCache`.
Every timed update fingerprints shared immutable RAW frames once, binds current
ROI, frames, native clocks, model/source/runtime identities, chunk shape and
object slot. A complete lot must hit; a partial hit recomputes the original lot.
Hits skip RGB pixels, transfer and producers, but rebuild metadata including
current query availability and recompute all event/context/TTC outputs. Guarded
producer features alone enter the cache; targets, context and TTC never do.

The runner clears the cache before each registered case, retaining its first
cold miss and subsequent identical-input hits. This measures the mechanism's
cost and correctness; its hit rate is not representative of a moving scene.
No hit claims to have consumed an RGB pixel tensor. Outside the update clock,
actual cached features are hashed; admission requires their full cache key and
both feature hashes to match a preceding miss whose actual pixel hashes match
the independent original crop. Missing provenance, changed keys, features,
events or original pixels fail the gate. Every TTC output is still compared
against the original predictor. Hashing RAW for cache lookup remains timed;
post-update diagnostic hashes and original-model verification are excluded.

CPU validation after integration: 48 own tests pass; Ruff and Pyright pass.
Tests include a later query with unchanged RAW proving new context clocks,
whole-lot recomputation after one ROI changes, RGB absence/invalidation,
prepared-tensor input alignment, and falsified cache-provenance rejection.
The service rejects a cache whose chunk identity differs from its producer
configuration before constructing any runtime resources.
An independent read-only review caught an admission gap: a cache run with only
misses could previously pass. Before any GPU cache run, admission was tightened
to require at least two repetitions, an initial stored miss and an actual hit
without pixel work in every RGB-present case, and at least one hit overall.
Six tests exercise the all-miss, failed-store and false-hit failure modes.
`PreparedRGBBatch` also accepts a flattened resident pixel tensor with aligned
metadata, preparing a future uint8/pinned transport integration. That separate
path was not yet enabled at this stage. GPU cache parity and timing were then
measured under an exclusive reservation as recorded below.

## Cache pilot: correct reuse, insufficient complete-update latency

`pilot_20_exact_cache_v1` completed all five index-553 updates using the same
twenty-region/chunk-one/prefix-query-graph protocol. All three TTC outputs,
input provenance, detailed history receipts and coverage pass. All 611 bound
files were intact at final verification. The first miss took 408.6445 ms. The
four hits took 176.7052, 184.2034, 203.0099 and 194.0419 ms: mean 189.4901 ms,
p95 201.6647 ms and p99 202.74086 ms. These are identical-input replay hits,
not a measured streaming hit rate. `cache_breakdown.json` regenerates separate
MISS/HIT statistics from the recorded case; the mixed mean is not used as the
hit latency. No update meets 20 ms.

Each update hashes 48,384,000 RAW bytes inside its timer. Fingerprinting takes
22.3652–23.0006 ms, already exceeding a 20 ms budget before inference. A future
immutable-source identity must count the cost of sealing each newly arriving
frame and cannot merely move that work before an undocumented timer boundary.

`source_snapshot_exact_cache_v1/MANIFEST.json` preserves 33 matching source
files. The unrelated, unimported `rgb_pil_exact.py` changed after the successful
final preservation check and is listed as unavailable with its expected SHA;
no current file was substituted for the executed binding. The next runner now
binds the actual imported runtime module closure after explicitly loading
enabled optional paths. Any new runtime import later invalidates admission,
including a file created after the initial inventory.

## Current priority: complete 1/2-object updates, uint8 GPU gate pending

The parent relayed the user's revised order: reach 50 Hz / under 20 ms for one
object, then two; twenty remains a later objective. No further twenty-object
GPU experiment is scheduled without a new instruction.

`--rgb-uint8` now connects the separately tested transport primitive to the
complete service. A CPU worker packs each unique resized ROI/frame once as
uint8. The owner stream uploads these bytes, applies the exact NumPy FP32 LUT,
and assembles the native repeated frames plus preallocated FP32 deltas.
Object metadata remains aligned; `--rgb-object-chunk-size 1` retains the
admitted per-object RGB producer batch. Shared PIL images remain optional;
the default stays ROI-first because the CPU comparison found a small-ROI loss.

Normal updates release the borrowed transport buffers after all consumers
have joined. Diagnostic updates keep a valid lease until the next update or
close, allowing actual GPU pixel/delta hashes outside the update clock without
reading released aliases. Both modes keep all actual inference work inside
the clock. A failed event ingestion discards a completed but unassembled pack;
a later model failure synchronizes consumers and releases the assembled lease.
Tests verify recovery and prevent a stale lease from being consumed again.

The runner checks actual FP32 delta hashes against native original metadata,
in addition to event tensors, RGB pixels and TTC. Transport allocation and
byte receipts are recorded. The optional injected producer API validates the
effective device and every resident parameter/buffer; implicit CUDA ordinals
cannot silently mix GPUs. Injected producers with feature caching are rejected
until an explicit backend identity/ownership contract exists.

Current own validation: 60 CPU tests pass, Ruff passes and Pyright reports no
errors. Independent read-only review found no other lease, stream, cleanup or
provenance defect. The uint8 integration has not yet received a complete GPU
parity/latency gate; its helper's CPU tests are not a substitute.

The next paired commands use counts `1 2`, indices `0 137 553`, repeats `5`,
`--parallel --no-event-reuse --backend prefix --event-query-graphs
--event-normalization topk --rgb-object-chunk-size 1`, first without and then
with `--rgb-uint8`, each in a fresh output directory. Start without feature
caching so cold RGB work is included; any later cache comparison must retain
MISS/HIT statistics and the original TTC/provenance gates.
