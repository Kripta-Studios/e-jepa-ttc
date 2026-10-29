# Temporal prefixes — CPU implementation checkpoint

Status: implemented; GPU slot pending. No final TTC admission and no 50 Hz claim.

Owned source files:
- operational/train40_runtime/temporal_prefix.py
- scripts/benchmark_train40_prefix.py
- tests/unit/test_train40_runtime_temporal_prefix.py

## Algebra and boundedness

A ring of 121 spatial prefix images stores counts and timestamp sums for all 5 ms boundaries in the retained 600 ms window. Query samples 25 quarter boundaries rather than gathering 120 independent buckets. Prefixes reset every 600 ms and one previous-epoch total is retained. The query crosses at most one epoch boundary.

For a 25 ms interval, let Cl/Ml and Cr/Mr be the endpoint prefix count/moment, let Cp/Mp be the preceding full epoch total, let cross indicate that the right endpoint is in the next epoch, and l be the left endpoint time relative to its own epoch:

    count = Cr - Cl + cross * Cp
    moment = Mr - Ml + cross * (Mp + 600000 * Cr) - l * count

The resulting moment is relative to the quarter start and is fed into the original float64 bin interpolation. New epoch boundary prefixes are zero. Slot indices use the absolute number of ingested buckets, independently of physical ring slots.

max_epoch_events = 30 * max_packet_events must fit int32. Timestamp moments remain below max_epoch_events * 600000, independently of session duration. All validation occurs before mutating the packet state. Important implementation trap caught by a regression: torch.mul(int32, 600000, out=int64) does not widen the multiplication; copy to int64 before multiplying.

Normalization defaults to the original full float64 sort. An explicit normalization='topk' option uses ExactTopKNormalizer from the percentile worker; it preserves the original rank/interpolation formula and only changes order-statistic selection.

## Memory (retained device buffers)

Native 1280x720, ROI128, max_packet_events2000000:
- 1 object/batch1, sort: 2991453304 bytes
- 1 object/batch1, topk: 2976511192 bytes
- 20 objects/batch4, sort: 3462329752 bytes
- 20 objects/batch4, topk: 3402561304 bytes

These exclude allocator caching and disclosed bounded eager temporaries. The biggest take temporary is 25 * 2 * roi_batch_size * roi_size^2 * 8 bytes. Host raw staging and optional pinned memory are reported separately. Outputs are borrowed, and the history requires a single owner stream.

## Verification performed

15 CPU tests passed; 4 CUDA tests skipped until an exclusive GPU slot. Tests compare bit-exact against both the previous bucket ring and canonical event encoder, across arbitrary phases, large absolute timestamps, all 20 ms phases through two epoch transitions, dense repeated pixels, 20 distinct ROIs/partial batches, empty inputs, reset/gap/rollback/capacity failures, exact temporal boundary events, and both normalization modes. Retained-buffer accounting is checked exactly.

Ruff and Pyright passed. Small CPU benchmark completed for 1 and20 regions with exact bucket/prefix equality. These fixtures use canvas32x24 and ROI8; their times are not sensor-scale evidence.

## GPU commands when the coordinator grants the slot

    ../e-jepa-ttc/.venv/Scripts/python.exe scripts/benchmark_train40_prefix.py --manifest artifacts/final_train40_transfer_20261027/QUERY_MANIFEST.json --index 553 --objects 1 20 --steps 5 --warmup 2 --output artifacts/train40_runtime_resume_20261027/prefix/gpu_sort_v1

Repeat with --normalization topk and a fresh output directory. The runner holds raw packets in CPU RAM, initializes outside the clock, then measures each complete new packet plus all requested ROI tensors. It excludes models, final TTC and disk replay. Additional ROIs are explicit synthetic capacity regions, not twenty annotated tracks. Backends run sequentially to prevent GPU contention and excess memory.

The benchmark source hashes are checked at completion. Root must keep these sources fixed during GPU measurements. No original candidate file, prediction, scientific module or existing integral.py was changed.

## GPU result, exclusive slot

Four CUDA tests passed (sort/topk x synchronous/pinned transfers). Both native sensor-scale benchmarks completed with source hashes unchanged. Every update and all requested ROI tensors matched the old ring bit-for-bit; the final primary ROI matched the canonical encoder exactly. Twenty capacity regions produced twenty distinct final tensors.

Five measured updates after two warmups; these are small-sample percentiles, not a sustained-service tail estimate:

| Paired run | Objects | Bucket mean/p95 ms | Prefix mean/p95 ms |
|---|---:|---:|---:|
| sort | 1 | 10.653 / 11.062 | 9.583 / 9.909 |
| sort | 20 | 79.418 / 80.934 | 59.375 / 65.306 |
| topk | 1 | 9.162 / 9.735 | 9.831 / 10.365 |
| topk | 20 | 81.479 / 84.226 | 53.913 / 59.928 |

The topk1-object route was slower in that pair. All timings include validation, ingesting the new RAW CPU packet, transfers, ROI projection and event encoding, ending with device tensors available. They exclude models/TTC, disk replay and initial history. No 20ms or50Hz claim follows. Paired backend order was fixed (bucket beforeprefix); repeat with alternating order and more samples before treating the improvement as stable.

Results: gpu_sort_v1/summary.json and gpu_topk_v1/summary.json. Matching source bytes are preserved in gpu_source_snapshot_v1 with SHA-256.

Next design (not implemented at this checkpoint): capture the full twenty-ROI query in one CUDA Graph after separating CPU bounds preparation from device execution. Use static [objects,129] x/y bounds and a dynamic first-bucket scalar, keep all epoch metadata dependent on that scalar, and prime only synthetic zero inputs at service startup. Ingest remains outside the query graph. All per-update host packing and transfers must remain in the clock.

## Query graph result, exclusive GPU slot

Query capture is implemented behind graphs=True and explicit prime_query_graphs(counts). Priming must run before reset/initialize/ingest, otherwise it raises. All priming uses zero startup prefix planes and synthetic zero ROI metadata. Each query copies its current canonical projection bounds and first-bucket phase inside the timed call, then replays one graph across all workspace chunks. Ingestion remains outside this graph and inside the full representation clock.

Eight CUDA tests passed: eager/graph, sort/topk, synchronous/pinned transfers, shapes1/5/20 in reordered calls, dynamic ROI bounds and epoch transitions. Eighteen CPU tests passed. Full native dense553 benchmark gpu_query_graph_v1 completed with source integrity and bit-exact tensor parity for all7updates x1/20ROI:

| Objects | Bucket mean/p95 ms | Prefix + topk + query graph mean/p95 ms | Graph route max ms |
|---:|---:|---:|---:|
| 1 | 7.841 /8.183 | 6.465 /6.803 | 6.854 |
| 20 | 73.736 /74.042 | 30.623 /30.897 | 30.960 |

Startup synthetic capture was41.2ms and64.5ms respectively and excluded explicitly. Scope remains RAWCPU packet to all event representations, excluding models/TTC. Five measured updates cannot establish sustained tail latency.

Important memory caveat: this first graph receipt measured only torch.cuda.memory_allocated delta, which does not describe reserved graph scratch. The code now reports both allocated/reserved before/after/deltas with a process-allocator scope; that metadata-only patch was CPU checked after this measurement. The first graph source snapshot is preserved exactly by recorded SHA under gpu_query_graph_source_snapshot_v1, with the recovery procedure disclosed in its manifest.

A private integer-only selector prototype is in fused_selector_prototype.py (outside runtime). Twenty-one CPU cases match quarter_count/quarter_moment exactly for1/4/20ROI and7epoch phases. It is not compiled or GPU measured yet. A fused implementation could remove address and gather intermediates, but potentially duplicates temporal endpoint loads; measure before admission.

## Compiled integer selector result

New opt-in subclass CompiledTemporalPrefixEventHistory compiles only integer spatial rectangles and temporal differences, writing preallocated quarter count/moment tensors. Float64 interpolation and normalization remain independent unchanged operators. Legacy gather/address/quarter-work tensors are actually released after base construction; initial allocation and retained storage are distinct.

12 CPU tests and1 explicit CUDA Inductor test passed. The CUDA test covers graph shapes1/5/20, changing epoch phase and exact output equality. A regression caught an initial Dynamo recompilation caused by mismatched inference_mode between synthetic startup and real queries; the corrected route enforces the same mode, and a counting-backend test confirms no new compilations on first real calls. The private read-only code review found no correctness blocker.

Native dense553 compiled_gpu_v1 completed, all seven updates for1/20ROI bit-exact against prefix graph, final primary canonical max error0, and all source hashes unchanged. It preserves matching source bytes automatically under sources/.

| Objects | Prefix graph mean/p95 ms | Compiled selector graph mean/p95 ms |
|---:|---:|---:|
| 1 | 6.337 /6.549 | 6.849 /7.401 |
| 20 | 32.686 /33.933 | 24.655 /25.810 |

The one-object compiled route is slower in this pair. Twenty-object retained buffers decrease from3402602592 to3269433440 bytes (133169152 bytes removed). Process peak allocated changes3472415744 to3419983360 bytes; peak reserved remains roughly3.56GB due compiler/allocator pools. Startup compilation/capture cost4.522s for1 and2.677s for20, excluded explicitly and based only on zero synthetic startup buffers.

Generated Inductor code still uses three kernels per chunk:25-boundary rectangles, previous-epoch rectangles, then quarter differences. Private direct_selector_prototype.py exposes the24-quarter output domain directly;21 CPU cases are exact, but that route has no GPU evidence. It may eliminate intermediate boundary arrays while duplicating endpoint reads.

Latest parent steering: prioritize complete1- and2-object updates below20ms/50Hz before20objects. Keep ordinary prefix+querygraph as the evidenced faster one-object representation. Future direct-selector comparison should target1/2; no additional20-object GPU work is scheduled from this lane. Production sources remain stable for the parent's checkpoint.
