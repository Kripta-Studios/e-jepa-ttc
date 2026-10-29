# Exact uint8 RGB transport primitive

Date: 2026-10-28. Status: CPU implementation and parity validation. No GPU execution, TTC prediction admission or sustained-frequency claim is included in this work.

Files:

- `operational/train40_runtime/rgb_uint8.py`
- `tests/unit/test_train40_runtime_rgb_uint8.py`
- `scripts/benchmark_train40_rgb_uint8.py`

## Behavior

`RGBUInt8Runtime` crops each distinct object/frame pair once with the original PIL bilinear arithmetic. The default shares one PIL image per source frame across ROI tasks; `shared_images=False` keeps the existing ROI-first crop adapter. It packs uint8 CHW crops into bounded host buffers, copies those bytes to the selected device, casts byte values into preallocated int32 indices, gathers a 256-entry exact float32 normalization table, and assembles the original object-major history/triplet ordering with a second gather. Tensor workspaces and outputs are preallocated; the implementation does not enable CUDA Graphs or compilation.

The LUT is generated using NumPy float32 division by 255. It avoids reciprocal multiplication differences and is byte-identical to the original normalization for all 256 possible byte inputs. No producer weights, model inputs, ROI semantics, original predictions or frozen files are changed.

`prepare(objects, history)` returns `PreparedRGBUInt8`:

- `pixels` / `sensor`: contiguous FP32 `[present_objects * history, T, 3, 128, 128]`.
- `delta`: contiguous FP32 `[present_objects * history, T-1]`.
- `metadata`: one list per original object, including empty lists for absent RGB.
- `present_indices`: indexes back into that original object order.
- `receipt`: pixel/transfer byte counts, host-stage timing and source/crop counts.
- `ready_event`: CUDA readiness event, or `None` on CPU.
- `release()` / `assert_valid()`: explicit ownership checks for borrowed buffers.

To overlap CPU crops with event work, submit `pack(objects, history)` to the preprocessing worker, then call `assemble(packed)` on the runtime owner's thread/stream. Only one packet or prepared lease can be outstanding. This primitive does not cache crops or feature outputs between updates.

## Lifetime and integration

The output tensors alias reusable storage. Release them only after all producers finish consuming them. Consumers on another CUDA stream wait for `ready_event`; before release, the owning stream must wait for those consumer streams. Release records a completion fence. The next CPU pack waits on that fence before changing pinned host storage; this also protects the subsequent reuse of device outputs. `close()` joins owner-stream work and invalidates any remaining lease. CPU worker tasks are joined before their shared PIL images are closed, including error paths.

The service adapter can pass `ticket.metadata` and `ticket.pixels` to its prepared-input path. Passing `ticket.delta` as well avoids rebuilding a GPU delta tensor from metadata. Context and final TTC computation remain current-update work. The source arrays must be read-only and remain unchanged until pack finishes; neither NumPy's read-only flag nor this primitive is protection against mutation through an external alias.

## Bounds and byte accounting

Configured maxima are 20 objects, eight observations, ten distinct frames and T2/T3. Raw history is limited to 128 MiB before constructing PIL images. Each ROI/frame crop is 49,152 bytes. Worker count is bounded at twenty; PIL temporary images and task metadata are bounded host allocations, not claimed to be allocation-free. The public allocation receipt reports logical fixed tensor storage, excluding allocator overhead and PIL workspaces.

At maximum default capacities, fixed host tensors total 9,835,520 bytes and compute-device tensors total 182,851,584 bytes. On CPU those compute buffers are also CPU resident; `device_buffer_bytes` describes their role, not a claim of GPU use. Resizing capacities to a known workload reduces these reservations.

For 20 objects, five T3 observations and seven unique source frames:

| Input representation | Bytes |
|---|---:|
| Unique resized uint8 crop payload | 6,881,280 |
| Mapping indexes | 2,400 |
| Delta values | 800 |
| Candidate update H2D total | 6,884,480 |
| Original expanded FP32 pixel payload | 58,982,400 |
| Original delta values | 800 |

The 1,024-byte LUT transfers once during startup. CPU runs report `h2d_bytes=0` and separately report the proposed CUDA byte count. No transfer-speed inference follows from these byte counts. The CPU crop cost remains, and complete TTC latency includes the event representation and all producers/context.

## Validation and remaining gates

The focused test suite passes **83 CPU tests**, including exact zero-error comparison for 48 combinations of 1/2/20 objects, G1 through G8 and T2/T3. Independent references use original `crop_uint8` and current `_normalized_crop`. Additional cases cover all 256 normalization values, borders/fractional/identity/one-pixel ROIs, both PIL modes, multiple workers, metadata/order, missing RGB, changing raw frames, fixed storage pointers, capacity limits and lease/thread misuse. A CPU sentinel rejects accidental CUDA API access. Ruff, formatting and Pyright pass.

CPU benchmark evidence is produced separately by `benchmark_train40_rgb_uint8.py`; source snapshots before/after prevent interpreting a run while executable sources change. Synthetic source generation and persistent-buffer startup are explicitly outside its RGB-preparation measurements. These are neither real-dataset model predictions nor last-observation-to-TTC latency results.

Next GPU admission must check exact input pixels/deltas first, unchanged producer output and final prediction gates second, and only then compare complete twenty-object update timing under equal inputs and scheduling. Keep R chunk shape/order unchanged; producer batch-size parity is a separate issue. Measure CUDA memory peak, transfers, cold startup, frame-arrival bursts, ROI changes, all percentiles and sustained backlog. A successful preprocessing microbenchmark does not establish <20 ms or 50 Hz.
