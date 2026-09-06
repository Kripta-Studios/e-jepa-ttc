# Concurrent resource scheduling — 2026-09-07

The user explicitly authorized SIMPLEX-T work while Stage70 remains active and
asked for independent implementation work during long cache/training operations.
This supersedes waiting solely because Stage70 is live. It does not authorize
changing expert weights, numerical recipes, roles, holdouts, scientific gates or
the separate expert-scaling proposal's training budget.

The original coordination flag records an earlier exclusive handoff. The resumed
D0 invocation is now **user-authorized concurrent inference**, not an exclusive
GPU measurement. Stage70 is neither modified nor stopped. All timings from this
invocation must be identified as concurrent-resource observations.

The unchanged replay runner resumes completed content receipts and requests the
remaining 16,384 D0 blocks in one supervised invocation. It retains FP32 batch16,
four CPU threads/two interop, existing model/code hashes, and per-query persistence.
Head fits remain forbidden before actual scientific freeze and complete QA.

At admission: GPU free9,056 MiB, host available15,394,668 KiB, C free120,954,093,568
bytes. Carry forward the prior20GB reservation for other work and2GB for replay
outputs. Preserve40GB free after reservations,8GiB host available and4GiB replay
process-tree RSS. Reassess at resource/implementation boundaries; pause on actual
resource failure, never reduce the recipe or kill Stage70.

Invocation evidence: `artifacts/simplex_t/T0/CONCURRENT_RESOURCE_AUTHORIZATION_2026_09_07.json`.
Replay process at first live check: executable PID26984, launcher40124, local
start2026-09-07 00:27:45. The owned lease is `T1/CURRENT_REPLAY.lock`; do not steal
or clear it. Check the same process/session before any resume or replacement.
