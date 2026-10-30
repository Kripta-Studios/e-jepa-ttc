# A40 evaluation overlap — 2026-10-30

The input-residency producer reached update 4096 and entered the scheduled
23,958-frame evaluation. Training was already approximately 256 updates/minute;
evaluation still executed CPU image loading, preparation, single-frame GPU
inference, decoding and ground-truth construction sequentially.

## Measurement

Attaching py-spy to the existing container process was unavailable because the
container lacks SYS_PTRACE. No container permissions were changed. A separate
fixed-checkpoint evaluation probe used 128 real selection frames, spaced across
the existing 1024-frame selection panel. Both readers were warmed before timing.
The live producer was temporarily suspended only during the paired GPU probe,
with an independent 240-second resume watchdog and unconditional resume cleanup.

The probe ran baseline, candidate, candidate, baseline. The first baseline used
cProfile and is excluded from speed comparisons. Every completed evaluation
payload compared exactly, excluding only its two inference timing fields. This
comparison includes predictions and velocity, GT, denominator hashes, ignored
objects, PR curves, v2 metrics and the separate legacy metrics.

| Mode | Full panel wall time | GPU forward region |
|---|---:|---:|
| Baseline, unprofiled | 10.7971 s | 4.1623 s |
| CPU overlap, first repetition | 7.1725 s | 4.7845 s |
| CPU overlap, second repetition | 7.2193 s | 4.8737 s |

The candidate's mean wall time is 7.1959 seconds: **1.5004x throughput**, or
approximately **33.35% less time** on this warmed panel. Forward itself did not
accelerate; overlapping CPU preparation reduced the total elapsed time despite
some increased contention. This is not a full-population or full-campaign
speedup. The profiled baseline took 15.9235 seconds, demonstrating why its
instrumented wall time must not be used as the speedup denominator.

The profile identified PNG decode, duplicated image preparation for legacy GT,
and JSON serialization alongside forward computation. The scorer remains
unchanged. A separate one-CPU-thread probe passed the same payload comparison
and gave 1.5127x versus its own baseline, close to the four-thread result. This
does not establish an additional benefit, so deployment retains four threads.
The first thread-probe launch failed because its filename shadowed Python's
`profile` module; it resumed the original process and the corrected rerun is
retained separately from that failure.

## Implementation and bounded resources

`prefetch_evaluation_rows` uses the existing tested `BoundedOrderedPrefetch` with
one CPU worker and at most two speculative frames. The worker exclusively owns
cache reads and computes the original annotation-empty model preparation,
independent GT, and legacy GT. It returns rows in the original order. Model
inference, decoding, scoring and checkpoint selection remain in the main thread.
Batch size remains one; precision, weights, classes, filters and matching are
unchanged. No approximation, BF16, TF32 or CUDA Graph is introduced.

The worker rejects GPU-resident inputs and any nonempty training labels in model
preparation. It joins before the caller regains cache ownership. Errors propagate
without skipping frames. Successful evaluation consumes every frame and its
member hashes before a checkpoint can be written.

Progress artifacts live separately in `evaluation_progress/`. They report stage
(inference, scoring, serialization, complete), processed/total frames, elapsed
time, forward time and CPU preparation totals. CPU totals overlap with GPU time
and must not be added to elapsed time. Progress is refreshed every 256 frames.

## Admission and rollout

The new producer is separately frozen under
`output/cloud_evaluation_prefetch_v1/producer`; the active R0 producer and its
checkpoints remain unchanged. The new controller waits for the original R0 to
complete and verifies both full evaluations and checkpoint hashes. It then runs
the five pending R0/R2 replicas, recording that R0 seed 7 came from the original
queue. A stop-after-current marker prevents duplicate launches by the old queue.
The old controller's expected stop exception is retained in its log.

Deployment requires exact panel parity, more than 1.1x measured throughput,
hash-bound producer/module/evidence identities and a real 16-update training
smoke including stop evaluation and durable checkpoint. No checkpoint provenance
is rewritten and no optimizer state is discarded from the active run.

Validation: 27 focused tests passed, including bounded prefetch contracts and an
AST comparison preserving model, calibration, decoder, geometry and scorer calls.
Ruff and Pyright checks cover the added implementation, materializer and controller.
Raw panel outputs remain on the Pod; small receipts, profile and hashes are stored
in `evidence/runpod_evaluation_optimization_20261030/`.

Profiler reference: [py-spy documentation](https://github.com/benfred/py-spy).

## Applying the revision to the active R0

The user subsequently requested immediate application to the active R0. Its
selected checkpoint at update 4096 contains complete optimizer, scheduler,
sampler and RNG state. Its member ledger already covers all 23,958 full-selection
tokens (28,054 known hashes overall), so re-evaluation retains every consumed-data
identity check.

`migrate_runpod_evaluation_checkpoint.py` admits exactly the previously profiled
parent/child producer hashes and two added prefetch modules. It checks scientific
runtime fields and source snapshots and writes a new checkpoint copy with only
the two runtime-provenance fields changed. A recursive fingerprint covers every
other scalar and tensor byte. The original checkpoint is preserved. This is an
explicit, recorded migration, not a relaxation of the strict checkpoint loader.

The strict new loader successfully resumed update 4096 to 4097, completed its
1024-frame stop evaluation and saved a checkpoint. That one-update smoke is
diagnostic; production resumes from the preserved update-4096 state. The original
controllers and producer were then stopped, and `resume_runpod_r0_prefetch.py`
started the pending full evaluation with the admitted prefetch implementation.
No completed training update was discarded. The unfinished full evaluation has
to be repeated because the original evaluator did not save reusable partial
predictions; the original run directory remains intact.

The controller completes that full evaluation before resuming training at update
4097. It binds the inherited update-4096 evaluation to the checkpoint SHA and
attaches the verified artifact to the resumed run before full-population gates.
The resumed summary retains its start update, checkpoint ancestry and cumulative
training progress; operational time spent in the interrupted evaluation and the
external replacement evaluation must be accounted for separately from the
runner's training-time field. After R0 finishes, the five remaining runs continue.

Initial production observation: 1536/23,958 frames completed in 80.05 seconds,
including preparation, before scoring and serialization. This is progress evidence,
not a completed-evaluation speed comparison. Six additional migration tests pass;
the combined migration/prefetch contract suite contains 33 tests.
