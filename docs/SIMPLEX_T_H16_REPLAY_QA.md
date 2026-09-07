# Independent H16 replay QA

This is a technical parity check, not a scientific fit or permission to evaluate
protected data. It preserves the original signed64 cohort and its twelve frozen
producer families. No query is selected using the observed layout discrepancies.

The executable entry point is `scripts/run_simplex_t_h16_replay_qa.py`. It accepts
`--local-paths`, `--preprocessing-manifest`, `--output`, and the explicit
`--other-reserved-bytes` outstanding reservation. Output must be a separate
directory under `artifacts/simplex_t/T0`; production cache files are read-only.

Before inference, the runner requires every reference receipt and payload, checks
their hashes, and validates observation identities and times against the pinned
deduplicated index. It then acquires `T1/CURRENT_REPLAY.lock`. It never clears an
existing lock or launches a competing replay. The original user-authorized
coexistence with Stage70 does not permit two own GPU inference processes.

The adapter uses the historical preprocessing and checkpoints, Torch version,
GPU model, FP32, H16 layout, TF32 disabled and 4/2 CPU threads. Original QA permits
outer-dev producers only with the acknowledged OLD8192 index. Expanded inference
still defaults to inner producers only. Neither path retrains any producer.

Memory admission requires 8 GiB available host RAM and at most 4 GiB process-tree
RSS. Disk admission leaves at least 40,000,000,000 bytes after outstanding outputs
and a 64 MiB QA reservation. These are absolute quantities, not free percentages.

Each recomputed block is saved independently before its exact-comparison verdict.
All seven arrays must have the same keys, dtype, shape and element values. There
is no adjustable tolerance. A mismatch is preserved and stops execution. Completed
receipts and payloads are rechecked during resume, without loading their producer
again. Orphan payloads are preserved and require inspection, not automatic deletion.

On completion, `QA.json` records the bound contract, all per-query verdicts and
observed inference times. Passing this check alone does not admit scientific fits:
full loader integration, freeze, resource admission and campaign gates still apply.

## Current evidence

The initial real CLI admission check stopped with
`WAITING_H16_REFERENCE_COHORT` before loading a GPU model. It created no QA output
directory. The prior metadata check had 32 of 64 references available. This is not
a parity result. The original production replay remains the dependency for the
remaining references; no numerical amendment or replacement cache was made.

Unit/integration tests use simulated producers and temporary NPZ files to test
leases, exact comparison, resumed rows, preserved failures, and changed payloads.
They are not substitutes for the pending real same-layout GPU execution.

## Evidence consumption

After real completion, a separately pinned component-evidence profile may add
`h16_replay` with the companion-relative `root` and actual `report_sha256` of
`QA.json`. The verifier then reads all 64 independent NPZ outputs and compares
their arrays to the separately validated production references. It also checks
the executable inventory, historical numerical source hashes, acknowledged
interfaces, preprocessing hash and recorded runtime. Missing or changed output
is a failure even if a JSON status claims success.

The existing profile intentionally has no H16 success reference yet. Its report
continues to list production H16 numerical parity as not covered. Supplying this
new evidence does not remove the separate requirements for complete source
integration, expanded time authority, repository QA and scientific freeze.
