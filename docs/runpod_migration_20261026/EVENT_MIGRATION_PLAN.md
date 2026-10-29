# Event migration plan — second Runpod

Status: **prepared for deployment and staging; scientific execution is not yet authorized**.

This plan is event-only. It does not pause the active Windows `E_A5/R2` fit, rent a Pod,
start a deadline, upload secrets, or execute optimizer updates.

## Scope and ownership

- Rotations: `R2`, `R3`, `R4`.
- Producers: `E_A5`, then `E_C2F` for each rotation.
- Downstream coverage: `PAIR_E`, `E_CTX`, event OOF feature production and the event inputs
  required by the final OOF stitch.
- The RGB branch is owned by the separate RGB execution plan. Raw RGB archives are excluded.
- Windows remains owner of active `E_A5/R2` until the cloud input preflight, zero-update parity
  check and technical resume proof pass. Only then may a durable 100-update boundary be captured,
  Windows ownership excluded and the cloud assignment activated.

## Audited inventory

Authority: `artifacts/runpod_migration_20261026/INVENTORY_FULL_V2.json`, SHA-256
`7fa2bfb48ae9860364b311519445bcdec0179e8105315c337e774344fa398793`.

| Set | Files | Size | Initial action |
|---|---:|---:|---|
| Prepared event producer shards | 5,548 | 40.980 GiB | Transfer first; stage on container-local disk |
| Teacher and event metadata closure | 3,038 | about 1.5 GiB | Transfer with producer shards |
| Event producer closure total | 8,586 | about 42.4 GiB | Required before producer preflight |
| Raw event streams | 40 HDF5 plus `train.parquet` | 418.164 GiB | Deferred until producers are progressing/complete |
| Raw RGB TRAIN archives | 136 | 118.360 GiB | Excluded |

Every event-producer file already has a SHA-256 bound by the existing cache receipts or source
manifest. The 40 raw event HDF5 files are bound by `RAW_PLAN.json`. The migration does not replace
those bindings with file names or sizes alone.

Generated preparation artifacts are under the latest
`artifacts/runpod_event_migration_20261026_v*/` directory. The matching
`PREPARATION_RECEIPT.json` and `*.tar.json` files are the checksum authority; do not mix versions.

- `EVENT_CONTROL.tar` — small code/control archive without bulk event inputs;
- `BUNDLE_MANIFEST.json` — exact union of control files and producer data;
- `EVENT_PRODUCER_DATA_MANIFEST.json` — resumable 8,586-file producer transfer;
- `EVENT_RAW_DEFERRED_MANIFEST.json` — 40-sequence raw-event transfer, not part of phase one;
- `EVENT_BUDGET_PENDING_DEPLOYMENT.json` — separate USD 24 authorization with null deployment and
  deadline timestamps;
- `E_A5_R2_CHECKPOINT_OBSERVATION.json` — coherent durable-state record; its latest version is
  authoritative only when it binds the explicit local pause receipt.
- `E_A5_R2_HANDOFF.tar` — authoritative paused handoff at update 6,328, checkpoint SHA-256
  `8f7cb270ed02b5b1a8d8d123330008884594d733349eb174bf34417d60764f26`; cloud resume remains
  blocked until the additive cross-hardware execution binding is admitted.

## Measured evidence and limits

The first Pod's RGB baseline measured about 190 updates/min, mean GPU utilization about 18.6%,
median 5%, process CPU about 47% of one core and 10.5 MB of logical reads per update. Kernel
wait-channel sampling found 95/200 samples in `request_wait_answer` and 12/200 in
`folio_wait_bit_common`. This supports local staging for immutable caches, but it is **RGB evidence,
not an event throughput claim**. Event throughput must be measured independently after parity.

The event checkpoint can restore model, optimizer, scheduler, sampler and RNG state. Exact
cross-hardware numerical continuity is not claimed: Windows and Linux/CUDA kernels can diverge even
with deterministic algorithms. The migration must record the immutable origin checkpoint and an
additive execution binding; it must not relabel a Windows identity as if it were created on Linux.

## Pod configuration and budget gate

Requested host:

- one RTX 4090 (24 GiB VRAM);
- preferably at least 64 GiB RAM; the deployed host's authoritative cgroup limit is
  60,999,999,488 bytes and is admitted only with a hard 2 GiB reserve;
- at least 12 vCPU;
- 100 GB container disk;
- 600 GB standard network volume, provisional until actual requirements are verified;
- PyTorch image compatible with Python 3.11 and CUDA 12.8.

The selected listing displayed 83 GB RAM, 16 vCPU and USD 0.904/hour including container disk. At
that displayed rate, 24 hours of compute is USD 21.696. Standard network storage under 1 TB is
currently advertised at USD 0.07/GB/month, approximately USD 1.40 for 600 GB over one day, giving
about USD 23.10 total before any dashboard-specific adjustments. Deployment is forbidden if the
dashboard estimate exceeds the separate USD 24 cap. Pricing and availability must be rechecked at
the actual click; they are not contractual repository constants.

The deployment timestamp and 24-hour deadline are written only after the Pod is actually rented.

## Transfer and local staging

After SSH host-key verification, substitute the real host and port. Never paste credentials into a
receipt or shell history.

```powershell
$key = Join-Path $env:USERPROFILE '.ssh\id_ed25519'
$prep = 'artifacts/runpod_event_migration_20261026_v4'
$control = Get-Content "$prep/EVENT_CONTROL.tar.json" -Raw | ConvertFrom-Json

scp -P <PORT> -i $key "$prep/EVENT_CONTROL.tar" root@<HOST>:/workspace/EVENT_CONTROL.tar.partial
ssh -p <PORT> -i $key -o BatchMode=yes -o StrictHostKeyChecking=yes root@<HOST> `
  "set -eu; echo $($control.sha256)'  /workspace/EVENT_CONTROL.tar.partial' | sha256sum -c -; mkdir -p /workspace/ejepa_event_bundle; tar -xf /workspace/EVENT_CONTROL.tar.partial -C /workspace/ejepa_event_bundle"

& 'C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\EVOCON_JEPA_Codex_Handoff\e-jepa-ttc\.venv\Scripts\python.exe' `
  scripts/runpod_event_transfer_v1.py `
  --manifest "$prep/EVENT_PRODUCER_DATA_MANIFEST.json" `
  --host <HOST> --port <PORT> --identity-file $key `
  --remote-root /workspace/ejepa_event_bundle --execute
```

The uploader uses one authenticated SSH stream. Each file is written to a `.partial`, resumes from
the last durable offset, is SHA-256 verified remotely, atomically published, made read-only and
given a transfer seal. A restart skips only files carrying a matching seal; it does not accept a
same-sized unverified file.

The 42.4 GiB transfer cannot be promised in ten minutes. Ten minutes requires over 72 MiB/s of
sustained end-to-end throughput, excluding verification and local staging. Measure the first GiB
before estimating completion.

After durable network transfer, create the Python environment and immutable cache on container-local
storage, retaining at least 12 GiB free:

```bash
export UV_CACHE_DIR=/tmp/ejepa_uv_cache
export UV_PYTHON_INSTALL_DIR=/opt/ejepa_python
bash /workspace/ejepa_event_bundle/workspace/e-jepa-ttc-v18-joint-measurement/scripts/runpod_bootstrap_local_v2.sh \
  /workspace/ejepa_event_bundle

export PYTHONPATH=/opt/ejepa_event_bundle/workspace/e-jepa-ttc-v18-joint-measurement/src:\
/opt/ejepa_event_bundle/workspace/e-jepa-ttc-v18-joint-measurement
/opt/ejepa_env/bin/python -m operational.runpod_migration.local_cache stage \
  --logical-bundle /workspace/ejepa_event_bundle \
  --stage /opt/ejepa_event_bundle \
  --reserve-bytes 12884901888
```

The environment belongs on `/opt`, not the network volume. Scientific checkpoints and journals stay
on `/workspace` and remain durable. The local cache is disposable and must never be the only copy.

Raw HDF5 transfer is phase two. Do not build or upload a monolithic 418 GiB tar. Transfer the 40
manifest entries independently and, for downstream raw-history work, stage one sequence at a time
after removing the no-longer-needed local producer cache. This keeps the 100 GB container disk below
its real capacity instead of assuming advertised capacity equals usable free space.

## Preflight and execution gates

Before any scientific update:

1. Record actual `nvidia-smi`, CPU quota/affinity, `/proc/meminfo`, cgroup memory limit and `df` for
   `/`, `/opt` and `/workspace`.
2. Use the real cgroup limit rather than the marketplace RAM label. Fail if aggregate usage leaves
   less than 2 GiB cgroup headroom, logical CPU availability is below 12, local free disk is below
   the staging requirement plus 12 GiB, or the network volume is below the remaining transfer plus
   outputs.
3. Reproduce `EVENT_INPUT_REFERENCE_R2.json`, `R3.json` and `R4.json` with zero optimizer updates.
4. Run an eight-update technical checkpoint/resume proof. This proof is not a scientific endpoint.
5. Capture the latest Windows `E_A5/R2` immutable 100-update checkpoint only after cloud readiness.
6. Publish and verify local exclusion before cloud ownership. Do not run both copies from the same
   checkpoint.
7. Preserve seed 7, BF16, effective batch 32, scheduler, objective and checkpoint cadence 100.
8. The requested producer schedule is six concurrent workers:
   `E_A5/R2`, `E_C2F/R2`, `E_A5/R3`, `E_C2F/R3`, `E_A5/R4`, `E_C2F/R4`.
   `E_A5/R2` alone imports the 6,328-update handoff; the other five start from their registered
   fresh seed-7 initialization. Launches must be staggered until all six have constructed their
   model/input caches, then admitted as one set only if aggregate measured VRAM, cgroup RAM and CPU
   retain their configured reserves. Do not silently substitute fewer workers for the requested
   concurrency; fail closed and report if one 4090/83 GB host cannot safely hold all six.
9. Download and verify each durable 100-update snapshot on Windows. Retain at least the latest two
   plus every complete endpoint.
10. Stop only through the user's explicit Pod action; Python deadline expiry must pause and report,
    not claim that billing stopped.

## Remaining implementation gate

The control/data transfer path is implemented. Scientific execution remains blocked until the
cross-hardware execution-binding adapter and event-specific ownership/snapshot receipts pass their
zero-update and technical tests. This block is deliberate: directly copying the current checkpoint
into the generic cloud runner would fail its identity contract or, worse, require misrepresenting
the Windows origin.

After all six producer admissions are complete, start a separate event-only downstream queue for
`PAIR_E`, then `E_CTX`, then event OOF feature generation. That queue must consume only verified
event endpoint admissions and must not wait for, launch, or rewrite `PAIR_R`, `R_CTX` or RGB OOF
work. The final cross-modality stitch remains gated on both independently verified branches.
