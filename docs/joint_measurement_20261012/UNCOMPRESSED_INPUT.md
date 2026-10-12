# Exact uncompressed input transport

The primary matrix remains four seed-7 fits of 12,000 updates. This work changes
only the proposed storage transport of the already admitted P inputs. At this
document's registration, the original primary runner is still active, and no
uncompressed runtime has been installed. P/H/V roles, sampler, losses, weights,
endpoints and prospective scientific gates remain unchanged.

## Evidence motivating the diagnostic

The retained CPU profile found substantial warm-loader time in ZIP decompression,
archive opening and array parsing. The earlier NVMe-versus-USB test failed its
prospective speed gate and did not justify a cache migration.

A different, bounded test compared the original compressed rows with identical
uncompressed arrays: 128 P queries from four packs, eight alternating paired
32-row reads. Every returned field matched exactly, including repeated queries.
The ratio of summed uncompressed read time to original read time was
0.06451605640294696, passing the preregistered 0.5 threshold and all eight paired
0.75 thresholds. This measures warm CPU reads on a small cohort under concurrent
training load. It does not establish a full-training speedup. It used zero
optimizer updates and did not initialize CUDA.

Evidence: `artifacts/joint_measurement_20261012/diagnostics/UNCOMPRESSED_ROW_PROBE/`.
The original negative storage probe remains retained.

## Complete cache and integrity

`operational.joint_measurement.mapped_data` prepares all 53,890 P queries in
256-row packs under the admitted cache root's `sensor_uncompressed/P` directory.
Original compressed packs remain intact. Each new array is reread and compared
exactly before its pack receipt is published. Receipts bind original pack
receipts, cohort identity, source code, array sizes, shapes, dtypes and SHA-256.
The complete cache is published only after all packs finish. Resource checks
reserve the remaining output size and the existing minimum free disk space.

The optional reader hashes every array during process admission outside GPU
scopes, retains those identities across scope closes, and checks filesystem
stamps before using admitted arrays. It bounds open memory maps to eight packs.
Returned row arrays own their memory, so eviction cannot invalidate a batch.
Native collation, metadata and duplicate sampler draws are inherited unchanged.
This is not a guarantee against adversarial same-size, same-timestamp mutation;
the cache is private to this campaign and is immutable after admission.

CPU tests cover exact fields, duplicate reads, eviction, corruption, changed
source ancestry, cohort/token alignment, and rejection of incomplete admission.

## Required CUDA comparison before installation

`operational.joint_measurement.profile_mapped --register-only` freezes the
diagnostic before its updates. For each primary architecture it compares the
original loader for 10 updates with the mapped loader for 5 updates, full
checkpoint restoration, and 5 further updates. These are 80 real-P diagnostic
updates, accounted separately from the primary endpoints. Together with the
earlier 80-update prefetch test, this uses 160 of the 200-update real-P input
preflight allowance. Neither diagnostic supplies a candidate model.

Admission requires bit-exact weights, optimizer, scheduler, sampler, all RNG
states, sampled IDs and loss cursor. The measured mean update-time ratio must
be at most 0.9 in every architecture and at most 0.8 averaged across the four
architectures. Trajectory order alternates across architectures. Inputs are
admitted before GPU timing; timed steps include reading, H2D, backward,
clipping, optimizer, scheduler and update journals. The full cache preparation
and byte-admission cost must be reported separately.

The diagnostic reuses the existing `profile_prefetch.trajectory` implementation
with mode `mapped`; no prefetch thread is used. Its legacy checkpoint experiment
name prefix is preserved, while registration hash and separate directory bind
this distinct diagnostic. All negative outcomes must remain in the bundle.

## Controlled transition and provenance

Only after the new cache is complete may the finite boundary controller stop
the original runner for this diagnostic. It must acquire the exclusive GPU
lock after a settled scope, verify journal equals durable checkpoint, require
post-scope runner evidence, and verify the checkpoint SHA-256. It refuses a
different worktree, command or recycled process identity.

If the diagnostic passes, `operational.joint_measurement.mapped_run` reuses the
existing queue with an explicit process-local storage adapter. It refuses a
second live primary writer. It does not rewrite legacy checkpoint identities:
the unchanged scientific source identity is supplemented by
`runtime/MAPPED_INPUT.json` and per-fit `RUNTIME_SEGMENTS/*/{START,END}.json`,
which bind adapter source, diagnostic receipts, cache identity and checkpoint
receipts at every scope. The original inherited dataset collation and all
scientific trainer code remain unchanged. Runtime byte admission has a separate
CPU cost receipt. Inference still uses the original sealed evaluator.

If any gate fails, keep the diagnostic evidence and resume the original queue.
Do not lower a speed threshold after observing its result. The result bundle
includes uncompressed-cache JSON provenance, not the large sensor arrays.

Actual state is authoritative in the diagnostic `COMPLETE.json`, transition
`STOP_RESULT.json` and runtime registration. Their absence means those stages
have not completed. No scientific improvement or SOTA claim follows from an
input-storage speedup.
