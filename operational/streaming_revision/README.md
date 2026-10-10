# Streaming revision

Experimental inference around the frozen TRAIN40 producers and three H8 heads.
Frozen modules, checkpoints, splits and official test12 predictions remain unchanged.

## Execution

Run from the V12 root with `src` and the root in `PYTHONPATH`, using the working
Python environment. Large inputs and outputs remain on E:.

```powershell
python -m operational.streaming_revision.benchmark --help
python -m operational.streaming_revision.evaluate --help
python -m operational.streaming_revision.distill --help
python -m operational.streaming_revision.packet_profile --help
python -m operational.streaming_revision.external_stream --help
python -m operational.streaming_revision.report --root <results> --document <report.md>
```

The benchmark requires a TRAIN40 campaign, raw eAP root, student checkpoint, new output
directory and GPU accounting directory. It reserves budget before loading CUDA, limits
wall time, archives its sources and journals every query. Do not run it alongside the
frozen test campaign or another Python GPU job.

`--live-input --roi-first` enables lossless packet compression and early spatial cropping
for both H8 and Garl. Disk reads are excluded for both; packet ingestion is included.
`--garl-native-input` adds native full-sensor input construction without early cropping.
`--garl-graph` also applies CUDA graph capture to Garl. No Garl variant uses reprojection.
Native tensor parity is checked on every selected query.

`h8_warp_graph_fixed` retains eight observations and all three heads, uses approximate
history/voxel reuse, two-observation producer batches and masked head padding. Compilation
is recorded and must be handled before deployment; improved median is not a tail bound.

`isolated --prepare-pid <pid> --receipt <json> --timeout 270 -- <benchmark arguments>`
temporarily suspends only the identified campaign CPU preparation tree and resumes it
on child exit or timeout. Do not use stale PIDs or suspend GPU training/inference jobs.

## Runtime contract

`PacketRing.push` owns integer event columns and explicit contiguous coverage, including
empty packets. Gaps and rollback require reset. Compact coordinates and relative
timestamps are lossless; reads restore public integer dtypes. Memory overflow clears
state and raises an error instead of silently returning incomplete history.
Timestamp compaction without temporary int64 buffers is enabled by default.
`packet_span_us=50000` optionally splits incoming packets; the default zero retains
incoming boundaries because smaller blocks have not improved ingestion consistently.

`IncrementalPreparer` takes an event-window reader and optional `roi_reader`.
`StreamRuntime.predict` takes a `Query` with sequence, object ID, sensor timestamp,
metadata availability, three endpoint windows, ROI, measured delta, pixel offset and
valid history mask. It returns TTC, the three seed predictions, reuse counts and timings.
Sequence/object changes and rollback invalidate caches.

Approximate reuse has time, ROI overlap, scale and age gates. Reprojection samples
original voxels once; warped results never become cache sources. It cannot recover
events outside an old ROI or exactly update counts and normalization. Cached observations
retain original timestamps and metadata availability.

H4/H2 truncate H8 weights. The A5 student removes C2F/PAIR inference but has different
accuracy on DEV32 and FCWD. These paths are experiments, not replacements for test12.

## Evidence

See `docs/sota_evidence_20261010/STREAMING_OPTIMIZATION.md` and the corresponding
`streaming_revision_20261010` results on E:. CSV/JSON tables are regenerated from
per-query journals. Failures and partial runs are retained.

Pilots use 12 queries from two teacher-seen TRAIN40 sequences. They establish engineering
comparisons, not external generalization or SOTA. Full DEV32/FCWD replay evaluates history
truncation and distillation, not voxel reprojection. External raw-event validation and
additional latency repetitions remain necessary.

`external_stream` evaluates all 630 frozen FCWD queries chronologically from raw HDF5
on CPU. It journals predictions before joining labels, checks the exact-reference
baseline against frozen predictions, and records MiD/FR by sequence. It does not
measure GPU latency or train/select a policy using these labels.
