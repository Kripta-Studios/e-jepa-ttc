# A40 input-pipeline optimization — 2026-10-30

The initial cloud producer spent a mean 3.1995 seconds preparing each batch and
0.2626 seconds in the forward/backward/update region over its first 74 updates.
This is startup evidence, not a completed campaign benchmark. The original run
and its update-256 checkpoint are retained under `cloud_runs/r0_seed7` on the Pod.

## Measured cause and intervention

The frozen reader retained only two open TAR handles. Every miss reopened the
archive and enumerated all member names to reject duplicates. Random sampling
across 78 train shards repeatedly paid for that enumeration over network storage.
The prepared-sample cache retained eight samples, despite an unchanged D4096
population being visited for eight effective passes.

A CPU probe compared two versus 128 handles over the same 48 training indices,
then repeated those indices. On the repeated pass, raw read plus preparation
took 13.5592 seconds with two handles and 1.5744 seconds with 128. TAR index time
fell from 11.7187 seconds to 0.000157 seconds. PNG hashes and prepared RGB tensors
matched exactly. Cache ordering and filesystem warming mean these numbers must
not be presented as a full-campaign speedup or a cold-storage benchmark.

The separately materialized producer changes only input residency:

- At most 128 open TAR handles per dataset; duplicate-name checking remains.
- At most 4096 training samples retained, with normalized FP32 images on CUDA.
- A 13 GiB image-byte budget with LRU eviction; no unbounded growth.
- At most eight prepared evaluation samples, on CPU, with the same
  annotation-empty preparation used by the frozen producer.
- Raw images, original annotations, calibration and target tensors retain their
  existing values. Targets stay on CPU and are copied through the original batch
  transfer function. Only consumed images acquire resident GPU copies.
- The sampler, RNG ownership, batch size, resolution, precision, augmentation,
  model, optimizer, matching, losses and finite checks remain unchanged.

One prepared image is 2,949,120 bytes. D4096 image residency is 12,079,595,520
bytes, before allocator overhead. Raw CPU images and target tensors are about
15.4 GB for this population, plus metadata and the runtime. These are size-based
estimates; a complete production population must be monitored for actual peaks.

The profile is recorded separately in preregistration as
`phase_ttc_exact_input_residency_v1`; it does not masquerade as the original
eight-element cache setting. The original producer remains unchanged. The
materializer records the parent runner hash, modified runner hash and added
cache-module hash. The runner snapshots the new module into each experiment.

## Real GPU admission

On 64 real training samples, the following comparisons passed:

- PNG SHA-256, raw RGB arrays and every prepared sample field: exact.
- CPU-stacked versus GPU-resident model batches: exact.
- Fixed-state train-mode forward outputs and individual losses: exact.
- Label-free evaluation preparation: exact; no supervised targets present.
- Maximum gradient difference: 7.29e-8, compared with 7.07e-8 between two
  baseline repetitions. This is within the measured eager-CUDA repeat envelope;
  gradients are not claimed bitwise deterministic.

Over 32 measured updates using 64 resident samples, median complete update time
was 0.2490 seconds and input handling was 0.00135 seconds. This is a warm-cache
microbenchmark, not an estimate covering first-pass loading, checkpoint writes
or full evaluations. A separate single-worker, 128-update probe reserved an
extra 12 GiB of VRAM to represent the full residency budget and measured
0.2406 seconds per update; peak Torch allocation was 17.64 GB in that probe.

## Concurrency and remaining decisions

One versus two independent producers is measured with identical cache fixtures
and the extra VRAM reservation. The controller permits two jobs only if aggregate
throughput improves by more than 10 percent and the container has at least 44 GB
of RAM. Each job has its own portable source, output, RNG and checkpoints.
Configuration, checkpoint and full-evaluation population checks remain active.

The measured single-worker rate was 4.135 updates/s. Two simultaneous workers
achieved 4.052 updates/s in aggregate (0.980x); their median individual update
times were 0.493 and 0.495 seconds. The admitted production setting is therefore
**one worker**, since two showed no throughput gain and consume more memory.

This is a warm-cache comparison, with synthetic GPU memory padding; it does
not measure two complete D4096 populations in CPU memory or their cold I/O.
The approximately two-percent rate difference does not establish a statistically
significant advantage for one worker. It shows no observed aggregate benefit
from two workers in this probe. Overlapping cold reads or evaluation with training
remains a separate measurement question.

Cold optimized repetitions are recorded separately from the original partial
run. Its checkpoint is not rewritten to bypass strict runtime provenance.

Pinned-memory transfers and ordered prefetch are candidates if remaining
uncached input latency is material after the index fix. Filling all memory is
not an acceptance criterion; complete updates per second and bounded resources
are the criteria. TF32/BF16 would change numerical recipes and require separate
accuracy experiments. Full CUDA Graph capture currently conflicts with CPU
Hungarian matching, variable targets and host-side finite checks. Neither is
silently introduced by this optimization.

Official API references used: [PyTorch 2.11 data loading](https://docs.pytorch.org/docs/2.11/data.html)
and [CUDA semantics](https://docs.pytorch.org/docs/2.11/notes/cuda.html).

Local evidence and the deployment receipt are collected in the adjacent
`evidence/runpod_input_optimization_20261030` directory after admission.

## Deployment

The optimized producer passed a real 16-update smoke run, including the
unscheduled-stop evaluation and checkpoint transaction. The autonomous queue
then started R0 seed 7 under `cloud_runs_input_v1`; it contains R0 and R2 for
seeds 7, 13 and 23. The original partial run was stopped only after this admission,
and its update-256 checkpoint remains preserved with its original hash.

The activation and hardware-admission receipts are included with the evidence.
`production_startup_snapshot.json` records a point-in-time observation, not a
live status or a completed result. Startup includes cold archive indexing and
image loading, so the warm-cache probe must not be used as a full-run ETA.
Container memory usage includes reclaimable file cache; it must be read alongside
`memory.stat`, rather than equated to process memory.

At update 215, the most recent 64 updates averaged 0.1357 seconds of input
preparation and 0.2343 seconds of forward/backward/update work, equivalent to
162.1 updates/minute inside those measured regions. This excludes evaluation,
checkpoint writes and earlier indexing startup. The cache held 860 unique images
(2.536 GB); all were first visits, so this observation does not depend on repeat
image-cache hits. The process RSS was approximately 6.07 GB and device usage was
7601 MiB at this snapshot. It is not a final memory peak or campaign ETA.

Validation before deployment: 14 targeted tests passed, Ruff passed, and Pyright
reported zero errors for the added implementation and queue files. Input and
forward parity on the actual GPU is documented above; no accuracy improvement
is claimed from this operational change.
