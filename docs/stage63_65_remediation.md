# Stage 63–65 review remediation

The attempt at training commit `a9c2d6e8689e24c4eed4acac7aa9b6eede7f8a84`
is invalidated for scientific use. Its files remain in
`STAGE63_65_RUN_20260904`, unchanged. The final observed training progress was
outer0/S64-STATE-L1 update 506; no outer evaluation files were found during the
process audit. No campaign Python process was alive at the audit. The cause of
its termination has not been established.

`CAMPAIGN_INTEGRITY_HOLD.json` blocks the master, Stage 64 and Stage 65 entrypoints.
The previous readiness, smoke and lock cannot authorize a new training attempt.

## Review coverage ledger

| Finding | Work completed | Still required before release |
|---|---|---|
| 1 Temporal bins | Single floor-edge convention; boundary tests; all 8192 tokens rebuilt; final uint16 SHA `3c5ca311cf60d12ac7d0a243b16a79d98ecba6f2e2ae587753acb7550cd3ae0f` | Final-code QA receipt |
| 2 A5 uncertainty | All three fields recovered from one FP32 forward of each of 12 frozen producers; original torch2.11/global32/TF32 policy; exact phase/support replay on all rows and exact outer-final triplets | Final-code QA receipt; old BF16 uncertainty remains historical evidence, not a training input |
| 3 Nested provenance | Split/token/config exclusions and uninitialized ancestors verified; existing frozen DINO weight hash and all8192 teacher-cache rows audited; all36 Stage65 producers pass; exact aggregate reconstruction through signed same-producer point binding and historical CSV serialization | Final-code QA receipt |
| 4 Cache integrity | Complete file SHA/size/layout inventory, token order/uniqueness/universe checks, mutation tests; lock binds raw, A5, supervision, source configs/checkpoints and schedules | Final lock |
| 5 CUDA RNG | Load resume checkpoint on CPU; explicit CPU RNG tensors; real CUDA 10 vs5+5 exact full-checkpoint test passes | Bind test evidence to final code/input/config QA manifest |
| 6 Resume states | CPU/CUDA RNG, normalizers, schedule/init/update validation; lost freeze marker recovery; verified previous checkpoint recovery; all12 endpoints checked before evaluation; completed result outputs bound; frozen Stage65 fits never refitted | Final-code QA receipt |
| 7 Authorization | Shared direct-CLI/master validation of branch, clean commit, runtime, complete inputs, QA and smoke; replicas require all seed7 gates; failed replicas cannot enable rescue | Archive hold, freeze code, run final QA, emit new lock |
| 8 Resources | Persistent 12h/24h/shared48h/1h budgets; in-loop checks; partial cap checkpoints; GPU sensors/I/O telemetry; warmed smoke ETA; bounded count conversion | Run final train-only smoke before scientific training |
| 9 Failure closure | Independent CLI writer locks and categorized failure closure; no blanket Stage65 exception downgrade; old campaign results preserved; partial/corrupt endpoints can be indexed for a blocked delivery | Final-code QA receipt |
| 10 Access evidence | Append/fsync journal; restart and torn-journal tests; policy assessment explicitly non-exhaustive; whole-process access remains unknown | Preserve all access attempts in final bundle |
| 11 Metrics | Explicit failure flags rejected; canonical8192/nine-sequence/three-fold universe and all pairing fields verified before gates | Final-code QA receipt |
| 12 Bundle | Includes router NPZ fits, code/diff, XML/logs, remediation and invalidated attempt; recomputes checkpoint hashes and verifies every closed-ZIP member | Execute final packaging after campaign endpoint |

## CUDA determinism amendment

The real-model CUDA resume test first failed exact equality. Enabling strict
deterministic algorithms then identified the adaptive-average-pool CUDA backward
as unsupported. Since the model fixes its canvas at 64x64, its last stem map is
always 8x8. The specified adaptive pooling from 8x8 to 4x4 is the same operator as
non-overlapping fixed 2x2 average pooling. The implementation uses the latter.
A test proves exact forward and gradient equality on the prescribed shape.
CUDNN deterministic mode and deterministic algorithms are enabled; TF32 remains
disabled. With that implementation, the real CUDA model passed 10 updates versus
5+resume+5 with exact model, optimizer, RNG, schedule, normalization and loss
history equality. Recreating the missing freeze manifest did not alter checkpoint
contents or perform additional updates. These are synthetic QA updates, not a new
Stage 64 scientific training run.

## A5 diagnostic interpretation

Historical checkpoints declare BF16 training/evaluation; the Stage 61 feature
builder executes FP32. The paired diagnostic reduces the historical uncertainty
discrepancy substantially under BF16 but does not reproduce all values exactly.
FP32 also differs from some stored dense phases beyond the 1e-6 replay tolerance.
Neither precision differences nor matching checkpoint hashes are accepted as a
complete reconciliation. The previous hybrid train-state policy remains
scientifically unaccepted. Diagnostic JSONs live under
`artifacts/stage63_65_remediation/` and contain tokens, input hashes and values.

The subsequent original-batch diagnostic under the historical interpreter
(Python 3.11 / PyTorch 2.11.0+cu128) exactly reproduced all three dense state
components on one 32-token batch per outer fold when FP32 and CUDNN TF32 were
enabled. The same batches on PyTorch 2.10 did not reproduce the dense values.
That diagnostic was subsequently extended to all twelve producers and all their
rows in `coherent_a5_state_v2_attempt3`. Phase and support reproduce the stored
dense cache exactly for every producer. All three outer-final fields also match
exactly. Historical BF16 log-variance is not claimed to be reproduced exactly;
it is no longer combined with dense FP32 fields. The accepted state triplet is
the single-forward recovered FP32 output. No A5 weights were updated.

The new budget helper persists an immutable conservative wall deadline. Replica
13 and 23 share one 48-hour budget; downtime counts rather than renewing a cap.
Stage64 checks before each update and saves a non-final partial checkpoint on
timeout. Cache/hash loops, inference batches and bootstrap loops also check
deadlines and margins. Stage65 checks its shared CPU deadline through fitting
and evaluation. The smoke reports post-warmup timing variation; it is an ETA,
not a scientific confidence interval.

## Physical Stage63 endpoint and engineering interruptions

The corrected physical audit completed with `X3_DATA_READY`, all8192 tokens
supported and all three outer-train support checks passed. The frozen A5 replay
score is 162.2045478952051 MiD, exactly equal to the preserved reference; the
maximum phase difference is 4.17e-17. This is baseline verification, not Stage64
training or a new outer-model result.

Two engineering attempts hit the mandatory 20% free-RAM margin. The first saved
7488 count rows; the next finished8192 rows but stopped in whole-map uint32→uint16
conversion. Both failure records are preserved. Conversion now releases each
64-row mapping, with an exact/overflow/interruption test. The complete uint32
source and interrupted uint16 temporary remain preserved. No raw temporal
convention changed during these conversion repairs, and no raw windows needed
to be rasterized again after the8192-row checkpoint. The final audit records
its source-file hashes and verifies that they did not change while running.

`elapsed_seconds` in feasibility describes the final invocation, not all prior
attempts. The persisted Stage63 budget and transition/access journals retain the
full wall-clock history. No budget was renewed.

## Historical QA baseline

An archive of the exact pre-review `a9c2d6e` tracked source and metadata was
executed inside this worktree under the same Python3.11 environment. Archive
inspection rejected raw-data or checkpoint file extensions. No previous
worktree was modified. Baseline:1620 tests,1569 passed,14 skipped,37 failed.
The37 failed test IDs exactly match the corrected-environment suite's existing
failures:23 E-Clock schema contracts,7 V8 freeze failures caused by missing
baseline metadata,4 unavailable historical artifacts,2 grouped-protocol pin
checks and1 historical teacher-manifest pin check. They remain failures, not
passes. Final acceptance requires reproducing this classification and proving
that no new historical-suite failure appeared.

An intermediate QA run forced UTF-8 child output without changing the parent's
default subprocess decoder; three tests then lost stderr to decoding errors.
The final QA environment sets both `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8`.
That failed intermediate run is retained rather than relabelled.

## Local execution environment

The original worktree `.venv` cannot import its torch extension. The isolated
`.venv-stage63` uses Python3.11.15 and reads existing dependency packages from
the reference repository's environment, without modifying it. Its ASCII `.pth`
uses an escaped Unicode path; no packages or pretrained weights were downloaded.
Stage63 raw integer-cache construction used Python3.14; coherent A5 inference,
real CUDA resume QA and the prospective adapter campaign use torch2.11.0+cu128.
Final QA captures package versions and GPU/driver information.

Finalization order: clean implementation commit → prepare QA input receipt →
complete final QA and baseline comparison → accept QA receipt → real train-only
smoke → TRAINING_LOCK. The master runner receives explicit paths through
`scripts/RUN_STAGE63_65.ps1`; scientific continuation remains forbidden until
this order has completed.
