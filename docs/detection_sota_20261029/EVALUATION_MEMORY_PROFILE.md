# Detection evaluation memory profile

## Decision

The original monolithic JSON serialization was unsafe for the planned 23,958-frame
selection run on this host. The new cold-start v5 producer uses byte-equivalent
streaming serialization. The 4,096-frame profile and conservative full-selection
admission below passed independent review. The complete 23,958-frame memory use
remains a projection until measured during the campaign.

This is a resource conclusion only. No score produced by this probe is a scientific result.

## What was measured

`scripts/profile_detection_evaluation_memory.py` deep-cloned the published 256-frame pilot
`selection.json` prediction and ground-truth structures into unique
`resource_only_synthetic_*` tokens. For each size, a fresh CPU process:

1. loaded the 7,543,598-byte pilot artifact;
2. built detector-independent v2 GT objects from the cloned pilot GT;
3. ran the real `evaluate_detections_v2()` scorer;
4. materialized the same `metrics_v2`, `predictions`, `ground_truth_v2`, and `decode_counts`
   payload shape used by the campaign evaluator; and
5. ran the same `json.dumps(..., indent=2, allow_nan=False)` plus UTF-8 file write pattern.

It did not open images, HDF5, or a GPU. Each worker had a 1,000,000,000-byte RSS gate. Windows
private bytes are recorded separately because the runtime begins with a large virtual/private
reservation; RSS is the physical-memory safety gate.

Source SHA-256:
`7a547a22222d4425cdd4faeaaf871f69210925adde94f69c629b48e94aceb896`.

| Synthetic frames | Predictions | Peak RSS | Indented JSON |
|---:|---:|---:|---:|
| 256 | 12,800 | 161,599,488 B | 14,602,402 B |
| 512 | 25,600 | 267,071,488 B | 29,216,649 B |
| 1,024 | 51,200 | 479,641,600 B | 58,486,303 B |

All three bounded workers passed. Their source data, per-phase samples, timings, process IDs,
payload hashes, and virtual/private measurements are in their individual receipts.

## 23,958-frame projection

The largest observed 512-to-1,024 marginal RSS slope was 415,176 bytes per frame. Extending that
slope from the measured 1,024 endpoint gives 10,001,287,984 bytes (9.31 GiB). A 1.5x allowance on
that marginal projection is about 13.97 GiB.

The stricter total-bytes-per-frame envelope gives 15,123,439,584 bytes (14.08 GiB), or 21.13 GiB
with the receipt's 1.5x safety factor. The serialized payload alone projects to approximately
1.37 GB, and `Path.write_text(json.dumps(...))` transiently co-resides with the live Python object
graph.

Even the least-conservative measured-slope projection is too close to or above the supplied
operational memory readings (about 10.7 GiB free before the campaign and materially less during
the active run). The safety projections exceed them. Allocator effects and a different full-data
class/GT distribution can make scaling worse, so these figures are not capacity promises.

## Byte-exact streaming serialization control

A separate fresh 1,024-frame worker replaced only `json.dumps(...) + Path.write_text(...)` with
`json.dump(..., stream)` followed by the same newline. The 58,486,303 output bytes had the exact
same SHA-256 in both modes:
`63578b3eb4f4de0cc9cf00fda7a5c188c2d204a30ebce68eacc4842531950c2a`.

| 1,024-frame mode | Process peak RSS | Serialization time |
|---|---:|---:|
| Current dumps then write | 479,641,600 B | 17.59 s |
| Incremental `json.dump` | 161,931,264 B | 25.76 s |

Incremental serialization removed 317,710,336 bytes (66.24%) from the observed peak, at a cost of
about 8.17 seconds in this resource probe. Its serialization phase remained essentially flat:
161,890,304 bytes before, 161,865,728 after, and 161,931,264 peak.

This establishes a byte-neutral minimal remedy for the dominant serialization transient. It does
not make the evaluator bounded: the scorer peaks were 80,637,952, 107,913,216, and 162,418,688
bytes at 256, 512, and 1,024 frames. Their maximum observed marginal slope projects 2,605,898,784
bytes at 23,958 frames, or 3,908,848,176 bytes with 1.5x margin, before the loaded detector,
dataset/cache, inference tensors, and allocator effects. A larger post-fix admission profile is
therefore still required.

## Why the active path scales this way

The inspected frozen path retains all frame predictions and v2 ground truth, constructs global
ranked predictions and per-class/per-threshold PR curves plus diagnostic lists, converts every GT
object to another JSON-compatible mapping, then materializes an indented JSON string before the
write. The 50-query detector produces up to 1,197,900 prediction rows at 23,958 frames.

## Initial remediation proposal and gate

The initial proposal recorded the following options before full-selection evaluation:

- persist inference in validated shards instead of retaining all raw output in one mapping;
- replace Python-dict PR records with compact typed arrays or a bounded external-sort/reduction;
- apply the proven byte-exact incremental JSON write before any larger admission run;
- stream or shard predictions and GT rather than retaining one complete payload in memory;
- keep failed/intermediate receipts and hashes so resume cannot mix populations; and
- rerun a scaling profile to at least 4,096 frames, then require the conservative projected peak
  plus concurrent training/runtime memory to remain below an explicit host budget.

The global score must remain mathematically identical; chunking frames must not turn AP into an
average of per-chunk AP values.

The adopted remedy is incremental JSON writing, followed by the 4,096-frame
measurement and admission below. Sharding and compact scorer arrays remain
possible later optimizations; they are not claimed as implemented. The failed
historical R0 remains frozen, and the v5 cohort starts from fresh weights.

## Reproduction

```powershell
..\phase-ttc-main-integration\.venv\Scripts\python.exe `
  scripts/profile_detection_evaluation_memory.py
```

Normative aggregate:
`output/evaluation_memory/evaluation_memory_scaling_v1.json`.

The serialization control is recorded in
`output/evaluation_memory/serialization_comparison_1024_v1.json` and its two worker receipts.

## 4,096-frame admission and full-selection budget

The streaming-only admission was repeated at 4,096 frames under an owned-process-tree watchdog.
The accepted run started with 11,419,246,592 bytes of physical RAM available, reached
492,060,672 bytes across its launcher/interpreter tree, and never reduced host availability below
8,724,439,040 bytes. The worker's own sampled peak was 487,043,072 bytes. An earlier attempt that
monitored only the venv launcher is retained and explicitly marked invalid for watchdog admission.

A permitted 4,096-frame legacy serialization control then ran under the same 2 GB tree limit. It
peaked at 1,770,307,584 bytes across the owned tree. Streaming and legacy modes both wrote exactly
234,328,274 bytes with SHA-256
`5520656625962a8769aede91282d422faa5e3aa702b64f3f77b30e7ed00065d0`, establishing full payload
and metric parity at 4,096 without running the unsafe legacy route at 23,958.

The conservative full-selection evaluator projection is:

```text
ceil(487,043,072 * 23,958 / 4,096 * 1.5) = 4,273,160,859 bytes
```

This worker figure excludes the resident trainer/model/cache. Windows reported a
`PeakWorkingSet64` of 2,950,291,456 bytes for the real trainer process. The admission rounds that
to a separate 3 GiB allowance, including the bounded sample/label caches. The cold-start combined
budget is therefore 7,494,386,331 bytes. Windows private/pagefile commitment is recorded only as a
diagnostic and is not confused with physical RAM.

When the trainer is already resident, full evaluation requires at least 5,883,773,595 bytes of
physical RAM available before start: projected evaluator RSS plus a 1.5 GiB host floor. When
budgeting before loading the trainer, compare the combined 7,494,386,331-byte budget with physical
RAM available. Continue to abort only the owned evaluator if either its declared budget or the
1.5 GiB host floor is crossed.

The two scopes must not share a misleading `observed_min` field. The cold pre-migration gate uses
10,464,067,584 bytes observed before the legacy control and requires 9,104,999,067 bytes including
the safety margin. The trainer-resident gate uses the lower 8,215,842,816-byte runtime minimum and
requires 5,883,773,595 bytes including the same margin. Their post-margin headrooms are
1,359,068,517 and 2,332,069,221 bytes, respectively.

The normative decision receipt is
`output/evaluation_memory/full_selection_streaming_admission_v1.json`. The first review correctly
rejected a profile that mislabeled prestart memory as the runtime minimum. After separating the two
scopes, independent review closed `SHIP` and recomputed projection hash
`907a44de6313c241c9eed9d4b58f900a4166790937224bfe5a737a81d8debd69`.

The final evidence-bound profile is
`output/evaluation_memory/streaming_writer_profile_v1.json`, SHA-256
`a892e08d6048332a0f2928a43131ee7d7b3f3375d8d81e52f8562fff1f7f31d4`. The migrator pins that
exact digest and revalidates all seven referenced evidence files. This admission does not itself
execute or authorize a checkpoint migration.

The retained `worker_512_failed_private_gate_v1.json` records the initial, deliberately preserved
failure caused by treating Windows private reservation as the physical-memory gate. The retained
`orchestrator_failed_projection_v1.json` records the subsequent aggregation defect; neither
failure invalidates the three successful worker receipts.
