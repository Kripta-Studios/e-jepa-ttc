# EVENT Pod2 recovery controller

This is prepared code, not evidence that a remote controller is running. The
Pod2 operator deploys it; the RGB Pod1 controller and all scientific sources
remain unchanged.

The controller covers six producers (`E_A5`, `E_C2F`, rotations 2–4) and three
`EVENT_DOWNSTREAM` workers. It preserves the exact command lines already admitted
for those processes, including the separate R2/E_A5 Windows path compatibility
wrapper. It does not execute technical probes, change recipes, remove STOP files,
or stop Pod billing.

## Deployment

1. Finish the current EVENT runtime admission and start the nine intended workers.
   Confirm they have their correct assignments and input/runtime bindings.
2. Disable the previous shell retry controllers without terminating their Python
   workers. Record the controller paths, verified absent PIDs, and all nine task
   IDs in `ejepa_event_legacy_controller_exclusion_v1` with `status=PASS`,
   `controllers_disabled=true`, `shell_supervisors_running=false`, and
   `active_legacy_pids=[]`. This receipt must describe an actual check.
3. Transfer the two new scripts into one operations directory:
   `prepare_runpod_event_watchdog_v1.py` and
   `runpod_event_resource_watchdog_v1.py`.
4. Prepare a JSON specification with schema
   `ejepa_event_watchdog_preparation_spec_v1`. Required top-level fields are
   `bundle_root`, `run_root`, `required_environment`, `status_path`, `lock_path`,
   `authority`, and `tasks`. The environment must explicitly include the actual
   `PYTHONPATH` shared by the workers. Do not copy secrets into this mapping.
5. `authority` supplies four existing paths: `pod_budget_path`,
   `downstream_budget_path`, `migration_binding_path`, and
   `legacy_controller_exclusion_path`. Each task supplies `id`, `rotation`, `fit`,
   `entrypoint`, `log`, and `runtime_files` (a list of existing source paths).
   Producers must include their bootstrap/wrapper and
   `scripts/runpod_event_scientific_v1.py` plus `event_runner.py`,
   `event_handoff.py`, and `event_resources.py` under
   `operational/runpod_migration`. Downstream tasks must include their bootstrap,
   `scripts/runpod_event_queue_v1.py`, and `operational/runpod_migration/event_queue.py`.
6. Run the preparation command under the admitted environment. It discovers
   existing workers from `/proc`, rejects duplicate owners, checks their inherited
   environment, hashes the authorities/runtime, and validates the resulting config.
   It starts and stops no process. It refuses to overwrite an existing config.

```sh
python prepare_runpod_event_watchdog_v1.py --spec EVENT_WATCHDOG_SPEC.json --output EVENT_WATCHDOG_CONFIG.json
python runpod_event_resource_watchdog_v1.py --config EVENT_WATCHDOG_CONFIG.json --validate-config
python -u runpod_event_resource_watchdog_v1.py --config EVENT_WATCHDOG_CONFIG.json
```

Use the actual admitted absolute Python executable and script paths. Starting the
last command enables recovery: it is not a read-only validation command. Retain
the controller PID, log, source SHA, config SHA, and initial status showing all
nine adopted identities. A single filesystem lock excludes duplicate controllers.

## Recovery evidence

An automatic restart requires a fresh resource-pause receipt with `manual=false`,
a checkpoint and journal agreeing on completed/durable updates and no pending
update, the expected worker identity, no STOP request, and unexpired authorization.
The controller waits 60 seconds between retries, allows at most 12 retries per
task, and continues monitoring other tasks when one becomes blocked. It does not
convert arbitrary nonzero exit codes into permission to restart.

A producer completes only with a sealed scientific endpoint and its EVENT cloud
admission. Downstream completion additionally checks both head endpoints, its
admission and canonical result receipt, and the declared artifact sizes/hashes.
Progress counters alone never establish completion.

The existing downstream pause format did not include `manual`. That absence must
remain blocked until the queue operational runtime records the actual STOP/resource
cause. Pauses before any durable head boundary, or during raw-only work without a
fresh durable paused head, are conservatively blocked by this version. Do not
fabricate a pause receipt to bypass that restriction.

The controller reuses the Pod2 budget and shared downstream budget, including their
hash link and common deadline. It does not create a second $24 allowance and does
not stop billing; the user remains responsible for stopping the Pod.

## Local validation

`tests/unit/test_runpod_event_resource_watchdog.py` and
`tests/unit/test_prepare_runpod_event_watchdog.py`: 13 tests passed. Ruff passed;
Pyright reported zero errors for both scripts. These checks do not certify remote
deployment or a real Pod2 pause/restart cycle.

## Pod2 scheduling update — 2026-10-27

The migrated Pod `lreiivpf18vxu7` now uses `EVENT_WATCHDOG_CONFIG_V25.json`
and `EVENT_WATCHDOG_STATUS_V25.json`, with the existing V12 watchdog source.
The operational entrypoint is
`resume_20261027/runpod_event_relocated_parallel_20261027.py`. Its hash-bound
`runpod_event_context_overlap_v1.py` overlay starts a separate E_CTX worker as
soon as that rotation's refiner RAW receipt is complete and verified. The main
process continues generating target-free OOF inputs. R2, R3 and R4 use this rule.

The worker extracts the unchanged normalizer/cache/fit block from the sealed
scientific pipeline, checks its source SHA-256, and consumes only the `refiner`
role. Its separate interpreter isolates training RNG from RAW inference. An
OS-owned `EARLY_E_CTX.lock` excludes duplicate head writers. The main pipeline
waits for an admitted head endpoint and release of that lock before its original
head/cache continuation. Scientific groups, losses, seeds, exposure, and the
100-update checkpoint interval are unchanged.

Automatic resource pressure inside these stages waits for the original guards
to pass; it does not lower the 2 GiB reserve. Manual stops, budget deadlines and
integrity failures still propagate. The child also enforces the same budget and
global stop through its original queue guard, including between 100-update fits.
Inspect `EARLY_E_CTX_STATUS.json` and `EARLY_E_CTX.log` per rotation in addition
to the main watchdog. A failed auxiliary worker stops its main continuation;
it is not silently retrained. For an explicitly authorized manual recovery of a
partial head, preserve/archive its manual pause status before relaunching, verify
the checkpoint journal, and clear only the corresponding authorized stop request.
Do not discard checkpoint, cache, normalizer, or RAW receipts.

Deployment evidence is under
`artifacts/runpod_event_migration_20261026/resume_20261027/`:

- `OVERLAP_VERIFIED_STATUS.json`: R2 E_CTX complete at 2,479 updates while OOF
  generation continued; R3 E_CTX automatically started and reached 510 updates
  while its OOF generation also advanced; R4 continued preparing refiner inputs.
- `COMPLETED_HEAD_REPLAY.json`: real completed-head recovery returned the same
  2,479 updates and identical hashes for weights, checkpoint receipt, endpoint,
  normalizer and training cache. No additional optimizer updates occurred.
- `tests/unit/test_event_context_overlap.py`: six tests passed, covering exact
  training-block preservation, refiner-only access, source tampering, resource
  retry and propagation of manual stops/deadlines/integrity failures. Ruff passed
  for the overlay and these tests.

The existing deadline remains **2026-10-27 14:09 UTC / 15:09 Madrid**. This update
does not authorize additional rental time or stop provider billing.

### Explicit six-hour extension (subsequent user instruction)

The user subsequently authorized: `amplia ese límite otras 6h`. The effective
deadline is now **2026-10-27 20:09 UTC / 21:09 Madrid**. Current supervision uses
`EVENT_WATCHDOG_CONFIG_V26.json`, `EVENT_WATCHDOG_STATUS_V26.json`, V13 watchdog
source, and `runpod_event_extended_parallel_20261027.py`.

`resume_20261027/DEADLINE_EXTENSION_6H.json` records this additive authorization
and hashes the historical budget and assignment receipts. Those original files
remain unchanged. The extension overlay retains their integrity checks and
extends only their operational expiration. Each restarted worker records its
effective deadline in `DEADLINE_EXTENSION_ACK_R{2,3,4}.json` under the run root.
Five focused tests cover the six-hour bound, expiry, tampering, and preservation
of other validation checks. This extends execution time; it does not recharge
the Runpod account or stop provider billing automatically.

### RAW cache pressure repair — 2026-10-27 14:28 Madrid

All three RAW generators stopped making progress at approximately 14:06 while
remaining alive in the resource guard. The cgroup had about 1.7 GiB headroom,
below the unchanged 2 GiB training reserve; approximately 43 GiB was file cache.
The old cache controller covered producer/teacher NPZ files, not the RAW H5 files
currently being read. Scoped read-only `POSIX_FADV_DONTNEED` on verified RAW inputs
restored about 10 GiB headroom and all three generators resumed without restart.

The replacement entrypoint is
`resume_20261027/runpod_raw_cache_scope_20261027.py`, using the same controller
lock, receipt, and stop-file arguments. It preserves the original sealed-NPZ
validation and adds the 40 RAW H5 files bound to the verified manifest/coverage.
It checks file identity before and after advice, opens only read-only descriptors,
and changes no input contents. Below 4 GiB headroom it advises RAW pages until
8 GiB is available, without changing training guards or global host cache settings.
Two focused tests passed. Live receipt PID 5478 and new fragment timestamps on all
three rotations verified deployment. Evidence: `RAW_CACHE_RECOVERY_VERIFIED.json`
under the local migration artifacts; remote `RAW_CACHE_CONTROLLER_DEPLOYMENT.json`
records the exact command. Use this entrypoint instead of the older V3 bootstrap
when restoring the cache controller after a future Pod restart.

### Second explicit deadline extension — 01:00 Madrid, 28 October

The user authorized execution until **2026-10-28 01:00 Europe/Madrid**, equivalent
to **2026-10-28 00:00 UTC**, and reported a further **10 USD top-up**. This is a
user-reported deposit, not a verified provider balance. The additive receipt
`resume_20261027/DEADLINE_EXTENSION_0100.json` binds the prior six-hour extension
and its unchanged historical budget/assignment hashes. No budget counters are
reset. Current configuration/status are V27, watchdog source V14, and worker
entrypoint `runpod_event_extended_0100_20261027.py`. The previous extension and
its code remain preserved. Six tests verify the exact Madrid/UTC cutoff, expiry,
original deadline identity, and rejection of receipt/parent/authority tampering.
