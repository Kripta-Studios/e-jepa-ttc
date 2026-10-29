# Object Event v4.31: box-conditioned causal audit

v4.31 is a diagnostic implementation, not a new TTC arm and not a claim of
performance.  It is **TTC-label-free but train-box-conditioned**: train boxes
briefly define the common object ROI while materializing the event tensor, then are
discarded.  The auditor never opens TTC, sign, bucket, target, annotations,
development/test, EvTTC, RGB, or a v4.30 cache/mixed OOF archive.  No v4.31 GPU run
or real-data materialization had been performed when the handoff was written.

## Diagnostic result (2026-08-09)

The 512-row train-only diagnostic completed on an RTX 5070 Ti Laptop GPU after
strict cache preflight.  It is non-selectable and non-authoritative because stage 2
was not supplied and the run manifest records a dirty implementation worktree at
base commit `f232ff2`.  No development, test, EvTTC, RGB or TTC labels were opened.
The emitted summary reports artifact identity
`2d1d68ca96fbab453f6145270826423ff4593dbc85f5120208ac63b970462461`;
the serialized `summary.json` file SHA256 is
`6434f23f145dcd7901a73849c868f58d60298b8f2bdbdc665479c2bced0046d2`.

The operator is stable but not physically equivariant.  Joint JS median/p95 are
`.00482/.05262` and displacement p95 is `.18040`, all inside their diagnostic
stability limits.  In contrast, median analytic zoom Pearson is `.29172` (required
`.95`), slope `.00852` (required `.8–1.2`), sign accuracy `.59082` (required `.95`),
oddness median/p95 `1/1` (required `.2/.5`), translation leakage p95 `.28859`
(required `.2`), and real-swap coverage `.00391` (required `.25`).  Therefore
`all_gates_pass=false`, `evidence_complete=false`, `selectable=false`, and the only
valid status is `not_issued_diagnostic`.  This falsifies promotion of the frozen
v4.30 operator; it does not measure TTC accuracy and cannot support a SOTA claim.

## Why this audit exists

The authoritative v4.30 summary SHA is
`9722202A4D33F6B5D1B933EEDA1F9143E13E4E2FD64B21356E93783AFAA1C689` and its
frozen OOF gates failed.  Its post-hoc, non-preregistered saved-NPZ check found
forward/swap correlation `+.53338`, zero flips and 95.8% coverage.  This suggests a
failure in object-local correspondence or later supervised readout, but is not by
itself a causal conclusion.  v4.31 separates representation instability,
object-local operator failure, supervised readout collapse, and TTC magnitude or
calibration failure using frozen gates.  SPAE may motivate a later compact
channel-structured bottleneck only if this audit identifies that failure; an
Event-Aided/INTACT-style option remains conditional future work, not an implemented
arm.  No SOTA statement is warranted.

## Data and fixed representation

The sole raw source is passed as `--source-parquet` or
`E_JEPA_TTC_SOURCE_PARQUET`, and must have SHA
`03dd3022db4b5f43bb10244fc8778476d74351e764f73a90c8566af949c17fd6` and exactly
the locked nine-column projection.  The relative `events_path` is resolved beneath
`event_root` from the YAML, `--event-root`, or `E_JEPA_TTC_EVENT_ROOT` (default
`E:/eAP_dataset`); it is never allowed to escape that root.  `event_windows_us` is
the real `[2,2]` pair for t1/t2 and t0 is a same-duration causal pre-context shifted
by at least 100 ms.  When the recorded window is slightly longer because of clock
jitter, the shift grows to prevent overlap.  `EventH5Reader` keeps HDF5 handles by path, bounds with `ms_to_idx`, and
searches the bounded timestamp slice rather than reading a complete stream.

Each interval uses `[start,end)` filtering before the historical v4.30 common-square
ROI voxelizer.  It retains its union t1/t2 ROI, 0.25 margin, minimum edge 8, t1 box
proxy at t0, event-pixel difference 5, five bins per polarity, robust normalization,
and log count/rate scalar channels.  Cache tensors are `[N,3,12,128,128]` float16;
the analyzer applies the historical area resize 128→64 only to channels 0:10 and
copies channels 10:12 as constants.  Spatial controls likewise leave scalar channels
unchanged.  The physical convention is t2 object expansion → negative `log_eta`;
zoom analytic expectations therefore use `-amount`.  Offset grids are explicit,
deterministic meshgrids at the v4.30 scales.

Materialization streams row by row to `open_memmap` files.  `rows.jsonl` has only a
row hash, sequence, pool and delta; boxes and raw paths are transient.  The manifest
records source/projection/representation/path metadata and hashes, including the
rows hash.  JSON schemas and preflight reject shape/dtype/count/hash/marker/row
identity/pool/quota/delta violations.

## Locked train-only sample ladder

Adaptation sequences are `5ilM1PX2vz`, `2cyv0Oedzg`, `qGsgzl4Q8B`,
`6h5yRW2LGc`; audit sequences are `OBneIVg4Cw`, `WbCh1DRerJ`, `mHGFBekt7X`,
`OYgB6RGWcq`, `t79dBxj1WS`.  Salted identity SHA ranking and an actual pairwise
100 ms per-track gap are the only selection rule.  Diagnostic is the exact
per-sequence prefix of full: adaptation 64×4 and audit 52+51×4 (512); full is
512×4 and 410×3+409×2 (4096).  There is no event-rate, motion, TTC or proxy filter.

## Runtime, controls and gates

The diagnostic runtime is wired to the frozen v4.30/v4.8 configuration and intended
three frozen seeds 7/13/23.  It adapts only `local_projection` for ten epochs on the
adaptation rows, with locked shuffle and EMA epochs 8–10; audit rows are held out.
Audit forwards, spectrum hooks, and all controls are chunked at batch ≤8.  The
temporary `local_projection` hook is removed in `finally` and yields projected maps
for per-row FFT radial bands, effective rank and cross-seed high-band CV.  Raw
channels 0:10 spectrum is descriptive only.  Stability is pairwise per row,
seed-pair and scale before balanced aggregation; no pixel count can dominate it.

Controls include identity, zoom ±.02/±.04 with inverse, translations, rotations,
zero-event, real swap `[t0,t2,t1]`, and reverse `[t2,t1,t0]` report.  Swap coverage
uses `abs(base)>=.005`; every seed and every sequence must satisfy the frozen swap
correlation, flip and coverage thresholds.  Missing/nonfinite data fails closed.
The decision priority is invalid/incomplete, representation instability,
object-local correspondence failure, supervised objective/readout collapse, then
TTC magnitude/calibration.  Diagnostic always reports
`not_issued_diagnostic`, even with real metrics.

Stage 2 is deliberately independent supervised provenance: its sole sanitizer may
open three mixed source NPZs (`7=`, `13=`, `23=`), copies only
`oof_row_index`, `log_eta`, `endpoint_swap_log_eta`, `unknown`, verifies each
0..2047 evidence set and emits seed-specific NPZs.  It is not length-aligned to the
512/4096 audit rows.  Full mode requires the sanitized three-seed directory.

## Safe execution and handoff

`AtomicDirectory` writes an exact artifact/config/source marker, stages beside the
target, quarantines a verified prior target only after staging, restores it if
promotion fails, and only removes generated sibling paths after ownership checks.

```powershell
# Prints tokenized commands and writes only run logs/manifests; no data or GPU use.
uv run --no-sync python scripts/run_object_event_v4_31_pipeline.py diagnostic `
  --source-parquet D:\data\train.parquet --event-root D:\data\eap `
  --cache C:\cache\v431 --output-dir C:\audit\v431 --dry-run

# Thin PowerShell equivalent.
.\scripts\run_object_event_v4_31_operator_audit.ps1 -Cache C:\cache\v431 `
  -OutputDir C:\audit\v431 -DryRun
```

The Python runner records UTF-8 stage logs, `run_manifest.json`,
`pipeline_summary.json`, and a SHA sidecar under `--log-root` (default
`artifacts/logs`).  It uses argv arrays, records Python/Torch/CUDA/GPU and Git facts,
propagates infrastructure failures as exit 2, and preserves scientific negative
stage exit 0.  On real runs it preflights an existing cache before sanitizer/analyzer
output mutation; an explicitly requested first build necessarily materializes its
new atomic cache before it can be preflighted.

Next agent checklist: keep v4.31 full closed, commit the Windows/provenance fixes,
and rerun only if a clean-commit diagnostic is explicitly required.  The next model
must replace the frozen local matcher rather than add another TTC readout on top of
it.  Event-only, RGB-only and RGB-E work must share the same sequence splits,
object-ROI contract and height-ratio target; RGB-E should use late fusion with
event-sparsity/exposure reliability gates, following the eAP ablation rather than
early channel concatenation.  Never promote this diagnostic or relax its gates.
