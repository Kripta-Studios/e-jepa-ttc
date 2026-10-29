# Immutable RAW identity and exact integer RGB crop

Date: 2026-10-28. Current priority: complete 50 Hz updates for **one and two objects**; twenty objects remain a later capacity target. This handoff contains CPU validation only. No CUDA execution, model-output admission, complete TTC latency or sustained-rate result was produced here.

## Delivered modules

`operational/train40_runtime/rgb_raw_ring.py` seals each arriving RGB frame into owned immutable bytes. `freeze_rgb_frame` copies and hashes once; its factory-only token contains source revision, frame ID, timestamp, shape and content identity. `token.array` returns a readonly view backed by immutable bytes. Unlike a readonly NumPy owner array, that storage cannot be made writable merely by toggling its WRITEABLE flag. `validate_frozen_rgb_frame` and resident lookups validate provenance without rehashing the payload. This is an ownership contract, not isolation from hostile native code.

`operational/train40_runtime/rgb_frozen_identity.py::bind_frozen_rgb_history` creates the existing `RGBHistoryIdentity` schema from those tokens. It checks IDs, shared source, causal timestamps, T2/T3 groups, chronological anchors and the 650 ms bound. Its SHA covers only a descriptor bounded at 16 KiB; `hashed_bytes=0` explicitly means no RAW payload was hashed during that query. Query availability is checked on every call but excluded from the producer-input digest. Producers must use `token.array` as their pixels: binding a token identity while cropping unrelated original arrays is invalid. The existing feature-cache core and wrapper have not been switched to this new binding by this task.

`operational/train40_runtime/rgb_pil_exact.py` provides exact integer coefficient generation, separable Torch resize kernels, `FrozenRGBHistory` and `ExactPILRGBRuntime`. Current ROI coefficients and offsets are prepared inside `prepare`. The kernels use 22-bit fixed coefficients, int32 multiply/reduce with checked bounds, and uint8 rounding/clipping after each axis. The final float32 normalization is an exact 256-entry NumPy division LUT. There are no handwritten CUDA kernels. Compilation is explicit with `compile_kernels=True`; `fullgraph=True` and `dynamic=False` are used for the two axis functions. CPU tests exercised the eager compiler backend, not CUDA Inductor.

The arithmetic specification was checked against [Pillow 12.3.0 Resample.c](https://github.com/python-pillow/Pillow/blob/12.3.0/src/libImaging/Resample.c) and the repository's original crop oracle. The frozen adapter clips integer bounds before resize. Partially outside ROIs therefore clip; fully outside/empty crops are rejected. Introducing black border padding would change that contract. Padded coefficient entries have zero weights and valid source indexes; they are only an implementation detail.

## ABI for service integration

```python
source = RGBSourceIdentity(source_name, revision_sha256)
# Run once per new arrival; this copy/hash is part of sensor-to-TTC work.
token = freeze_rgb_frame(raw_array, frame_id=frame_id,
                         timestamp_us=timestamp_us, source=source)
# Retain immutable tokens for the causal historical window, including both phases.
history = FrozenRGBHistory(query_us, tokens_by_id, triplets)
ring = ResidentRGBFrameRing(device="cuda", frame_shape=(1200, 1920, 3), capacity=16)
runtime = ExactPILRGBRuntime(ring, max_objects=2, max_history=8,
                            frame_chunk_size=1, compile_kernels=True)
ticket = runtime.prepare(current_objects, history)
# ticket.pixels: FP32 [present_objects * history, T, 3, 128, 128]
# ticket.delta: FP32 [present_objects * history, T-1]
# ticket.metadata: lists aligned with ALL objects; absent RGB is []
# ticket.present_indices, receipt, ready_event are also available.
run_original_producers_and_current_context(ticket)
ticket.release()
```

This is an integration sketch, not an executed GPU recipe. Device assembly and release use the owner thread/stream. Other consumer streams wait for `ready_event`, and the owner must join them before release. The next call waits before rewriting pinned tables or reusing RAW slots. The externally supplied ring remains owned by the caller. Startup allocations and lazy compilation must be reported separately or included in the call where they occur. No ROI-dependent result is prepared using future tracker data.

For the immediate one/two-object objective, immutable identity and GPU cropping are separate choices. Fast identity can remove repeated RAW hashing without moving complete frames to GPU. A new 1200x1920 RAW frame costs 6,912,000 upload bytes, while a one-object seven-frame set of resized uint8 crops is only 344,064 bytes. Measure the tradeoff against existing CPU/PIL and uint8 transport rather than assuming the GPU crop is superior at low object counts.

## Why the resident history needs sixteen slots

The native selector walks backward in 100 ms increments from the latest frame, while the observed RGB sensor spacing is approximately 50 ms. Consecutive arrivals can therefore alternate seven even/odd frame IDs with no overlap. Ten is the maximum size of one native query, not the capacity required to retain both phases across queries.

The ring now defaults to **sixteen historical slots**, permits at most thirty-two subject to the byte limit, and still admits at most ten requested frames per bind. A CPU replay uses the actual `sensor_triplets` selector with 20 Hz arrivals and 50 Hz queries. Across 69 queries, capacity sixteen produces two initial loads of seven frames, forty-one queries with no uploads, and twenty-six queries with one new upload. Capacity ten reproduces four/five-frame reloads on those same arrivals. This is a deterministic transport-count test, not a GPU bandwidth measurement.

At 1200x1920x3, sixteen resident frames occupy 110,592,000 bytes. CUDA staging has a separate equally sized pinned pool. The default 128 MiB limit applies to each RAW pool; receipts report both. The sensor-side token cache must also be bounded and retain both phases. First-fill work remains a cold-start cost; only after causal warmup does the replay demonstrate one new frame per changed history.

The crop runtime's workspaces are sized by at most ten current frames, independently of the larger resident history. Use `max_objects=1` or `2` for the current target. It reports fixed host/device table and workspace bytes. A conservative logical eager-intermediate bound is enforced per shape; allocator overhead, compiler workspace and actual CUDA peak memory still require measurement. Larger frame chunks may improve launch overhead but must retain the byte guard and be measured before adoption.

## CPU evidence

The combined command below passes **225 tests**: 157 exact crop/runtime tests, 38 ring tests and 30 immutable-identity tests. Optional fullgraph eager compilation is enabled. Cases include one/two/twenty objects, 1200x1920 frames, odd/small/singleton dimensions, fractional/clipped/outside ROIs, all normalization bytes, current ROI changes, padded chunks, changing arrivals, source reset, clock rollback, capacity/byte bounds and lease/thread misuse. Inputs and normalization match the original pixel oracle exactly. Ruff, formatting and Pyright also pass.

```powershell
$env:TRAIN40_TEST_CPU_COMPILE='1'
& '../e-jepa-ttc/.venv/Scripts/python.exe' -m pytest tests/unit/test_train40_runtime_rgb_pil_exact.py tests/unit/test_train40_runtime_rgb_raw_ring.py tests/unit/test_train40_runtime_rgb_frozen_identity.py -q
```

`probe_identity_cpu.py` and `identity_cpu_v1.json` separately measure seven synthetic 1200x1920 frames. Ten interleaved identity checks measured 40.894 ms mean for hashing 48,384,000 RAW bytes and 0.162 ms for the immutable descriptor. Copying/sealing each newly arrived 6,912,000-byte frame measured 5.869 ms mean across seven frames. Source generation was excluded, and seal cost was reported separately rather than hidden. The host was not exclusive and other CPU tests were running; these numbers neither isolate production tail latency nor establish a complete-update speedup. They demonstrate the scale and placement of the work that must be measured in service integration. Reproduce into a new output location; the probe refuses to overwrite its result.

## Remaining gates

1. Integrate immutable token ownership at the sensor reader and use token-backed pixels for cache misses. Keep file/source revision validation before reusing a token; immutable old bytes do not prove an external file has not changed. Add an explicit frozen-identity route to the wrapper rather than silently trusting a caller-supplied digest for unrelated pixels.
2. Measure one/two-object complete updates with fast identity and existing CPU crop first. Count arrival sealing inside the last-observation-to-TTC boundary.
3. Under an exclusive GPU lease, validate eager and compiled integer kernels bit for bit, then producer/final TTC parity at the existing tolerances. Include changing ROI sizes and both frame phases; verify cold compiler behavior and all shape buckets.
4. Compare RAW resident GPU cropping with unique uint8 crop transport under identical source histories. Report frame uploads, table transfers, startup, real CUDA memory peak and complete-update p50/p95/p99/max plus backlog over sustained 50 Hz input.

The earlier uint8 runtime also gained `discard(packed)` for recovery when another update stage fails before assembly. It requires the latest unassembled owner packet and performs no CUDA work; 85 tests of that runtime pass. The historical CPU benchmark source remains preserved separately. No previously measured prediction or frozen weight was changed.
