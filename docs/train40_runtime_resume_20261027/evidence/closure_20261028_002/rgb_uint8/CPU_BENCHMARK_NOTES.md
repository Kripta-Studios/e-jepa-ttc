# Unique uint8 RGB preprocessing: CPU evidence

Generated from JSON receipts by `render_notes.py`. No GPU was executed. No models, targets, candidate weights or original predictions were accessed.

The matrix contains 48 exact cases: 1/2/20 objects, 1 through 8 RGB observations and T2/T3 inputs. Two further runs contribute six exact cases. Sensor pixels, delta times, metadata, object ordering and the byte/allocation contracts all pass. An independent comparison also checks the original crop_uint8 against the ROI-first normalized adapter.

## CPU preparation timing only

Both timed paths use the same synthetic inputs, ROI-first PIL crop, one crop worker and one Torch thread. The baseline normalizes and expands triplets on the host; the candidate uses unique uint8 crops and its CPU implementation of LUT/gather. Raw generation, startup allocation and parity comparison are excluded. Device completion and lease release are included. The matrix's single timing sample per case is for smoke evidence, so it is omitted below.

| RAW size | Objects | Repetitions | New mean ms | New p95 ms | New p99 ms | Reference mean ms |
|---|---:|---:|---:|---:|---:|---:|
| 180x240 | 1 | 5 | 3.638 | 3.696 | 3.703 | 3.327 |
| 180x240 | 2 | 5 | 8.009 | 8.330 | 8.394 | 8.729 |
| 180x240 | 20 | 5 | 58.076 | 59.188 | 59.219 | 83.315 |
| 1200x1920 | 1 | 3 | 70.423 | 85.921 | 87.599 | 49.504 |
| 1200x1920 | 2 | 3 | 119.447 | 120.193 | 120.247 | 120.588 |
| 1200x1920 | 20 | 3 | 3329.663 | 4926.052 | 5056.590 | 1178.286 |

The 1200x1920 workload uses broad synthetic ROIs approximately 82% of frame width and 90% of frame height. It is not the actual pilot's ROI population. Its 20-object candidate crop stages took 5005.291, 3421.393, 1427.833 ms. The negative result and large variability are retained. The run does not determine their cause and must not be extrapolated to the real pilot. Three or five samples do not validate production tail latency. All these are RGB-preparation microbenchmarks, not complete TTC latency or sustained-rate measurements.

Whole-host/CPU exclusivity against other agents' work was not imposed. Matching inputs and options inside this harness does not establish equal host contention between samples or causally attribute the observed jitter. A controlled follow-up must reserve the host as well as the GPU before making performance claims.

## Bytes and memory

For 20 objects/G5/T3 there are 140 distinct crops and 300 expanded crops. The unique uint8 payload is 6,881,280 bytes; proposed H2D with mapping/deltas is 6,884,480 bytes. The previous expanded FP32 sensor is 58,982,400 bytes. Actual H2D is **zero in these CPU runs**.

The configured 20/G5/seven-frame case reserves 6,884,480 host-buffer bytes and 120,918,144 logical device-buffer bytes; device='cpu' means the latter are also CPU allocations here. Allocator overhead, Pillow objects and transient bounded crop results are not included in those tensor-storage totals. Startup allocation is recorded separately in each case.

## Reproduction

Use the existing environment and a **new** output path; the harness refuses to overwrite earlier evidence. Full executed options and source hashes are in each JSON. The command below regenerates this note, not the benchmarks:

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' artifacts/train40_runtime_resume_20261027/rgb_uint8/render_notes.py
```

Benchmark entry point: `scripts/benchmark_train40_rgb_uint8.py --help`. Use `--roi-first-images --crop-workers 1 --measure-reference` for the same crop options as the controlled comparison. The default shared-PIL option is also covered by the 48-case matrix. CUDA execution remains pending a separate authorized trial.

## Evidence hashes

- `cpu_matrix_v1/summary.json`: `fe3ad2099dfcb0c2f9c97b0bc99dbd6f601ec1b3604ca478df680f8a85c5573a`; 48 cases, sources unchanged during execution.
- `cpu_roi_first_v1/summary.json`: `7e87b50f54d076bf394e30ac2f61c6578519c6f7697947d984604d6913e66252`; 3 cases, sources unchanged during execution.
- `cpu_1200x1920_roi_first_v1/summary.json`: `a94703cfbf5162e510ca851cc1e039f359656b44f74052eeb4e313fe7d779de4`; 3 cases, sources unchanged during execution.

Executed benchmark source SHA-256: `258b29b1a1cd5b2be169d33386c1dd64befb85d7532e15ed78f59cfa301322d4`.

## Source snapshots

`source_snapshot_receipt.json` records 18 verified run/source bindings across 6 unique content-addressed text snapshots. Only bytes matching the executed SHA were copied. Snapshotting did not rerun any benchmark or modify executed source files.

Sources already changed or missing when snapshots were captured: `operational/train40_runtime/multiobject.py`. Their current bytes were not copied as executed evidence; the receipt retains the executed and observed identities separately.
