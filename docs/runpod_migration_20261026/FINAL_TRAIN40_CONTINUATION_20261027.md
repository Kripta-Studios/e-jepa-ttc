# Pod2: final TRAIN40 continuation, 2026-10-27

Verified at Unix time 1793115448 (15:37:28 UTC): all five final fits are
training concurrently. This is execution evidence, not a completed model or
official evaluation.

| Stage | Durable updates | Frozen endpoint |
| --- | ---: | ---: |
| PAIR_E_MATCHED | 7,300 | 11,264 |
| E_CTX_MATCHED | 6,900 | 12,044 |
| R_CTX | 7,000 | 12,044 |
| PHASE_ONLY_EVENT_S7_TRAIN40 | 6,200 | 12,044 |
| PAIR_R | 200 | 11,264 |

All receipts reported complete checkpoint state, `IN_PROGRESS`, and zero
recovery upper bound. These are TRAIN40 fits, not validation measurements.
The final PAIR_R cache contains 88,744 rows across 40 groups. The OOF feature
preparation and fold components had completed before these fits began.

## Durable locations and continuation

- Outputs: `/workspace/ejepa_final_runs`.
- Runtime and queue: `/workspace/ejepa_final_operations`.
- Queue PID at verification: 8906.
- Worker PIDs: PAIR_E 8773, E_CTX 8774, R_CTX 8775, PHASE_ONLY 8824,
  PAIR_R 8907. Always verify process creation time and exact command before
  operating on a PID; these numbers are historical evidence only.
- Queue status: `FINAL_QUEUE_STATUS.json`; durable worker registry:
  `FINAL_QUEUE_WORKERS.json` in the runtime directory.
- Checkpoints remain every 100 updates. Per-invocation limit also remains 100;
  full-endpoint invocation was reviewed but was not deployed.
- Authorized deadline: **2026-10-28 00:00 UTC / 01:00 Europe/Madrid**.
- Manual stop marker: `/workspace/ejepa_final_runs/STOP_REQUEST`.

The controller was replaced while all four existing training workers continued.
The fifth worker, PAIR_R, then started and persisted its first checkpoints.
New controller recovery checks PID, creation time, exact stage command, and
process inventory; duplicate live workers fail closed. Scientific errors are
not silently retried. Resource checks and the 2 GiB memory reserves remain.
The new bootstrap catches resource pauses raised by either the outer check or
the selected stage's pre-fit check. Existing workers retain their loaded
bootstrap version until their next process restart.

After all five endpoints, the queue seals deployment dependencies and exports
the PHASE_ONLY head with the existing parity checks. Its explicit terminal
state is `TRAINING_EXPORTED_TEST12_PENDING`, not project completion. The export
is the head, not the complete sensor-to-TTC pipeline. Official submission is
not authorized and is not implemented in this queue.

## Deployment and validation

Deployment record and latest measured status are saved locally under
`artifacts/runpod_event_migration_20261026/final_20261027/` and the deployment
record is also stored in the runtime directory on the persistent volume.

Deployed bootstrap SHA256:
`18a6b4ff25e4a0e5e9e2afc0e9c84ecd875d40f5d1cde441e80cb7288be3dd32`.

Deployed queue SHA256:
`c52b037504331307c06db41905ddc400ace7294aa4c5161d214777abd6fb05d0`.

The agent subsequently formatted/annotated the local queue file, producing
`dbdafa37a7abb5a6a48a909a29510fa5dae64b31662919d3cffb37c41caedd5a`;
that cosmetic revision was not redeployed just to restart the controller.

The cache, path-composition and phase bridge unit suites passed (15 tests).
Separate targeted queue checks passed for orphan adoption, duplicate rejection,
PID reuse, five head slots, and distinguishing pending Test12 from completion.

At verification, GPU memory was 3,313 MiB and cgroup memory headroom was
8,394,350,592 bytes. The instantaneous GPU utilization was 7%; this is not
a sustained throughput measurement or an ETA.

## Test12 and remaining data work

Public metadata and the media inventory were verified. E: is unavailable.
Test12 requires staged reconstruction and exact input/source hash checks before
inference; the existing validator expects all raw inputs together. Downloading
the full set into the current free space is not an admitted solution.
The minimal public media inventory is approximately 161.38 GB. See
`TEST12_RECOVERY_PLAN.json` and `TEST12_EXACT_INVENTORY.json` in the local
artifact directory. No hidden labels are needed or requested.

The RGB cache under `/opt/ejepa_final_rgb` is ephemeral; source archives and
verified bindings are retained. Training outputs are on `/workspace`.
Stopping the training processes does not stop Runpod rental billing.
