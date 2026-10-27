# Network-volume technical preparation, v3

The replacement Pod mounts `/workspace` as MooseFS (`fuse`, source
`mfs#euro-3.runpod.net:9421`), not the previous global object volume. The actual
mount root is `/networkvolumes/fub2i3urhl`. Its cgroup hierarchy is v1/hybrid,
with a 60,999,999,488-byte memory limit. The declared pilot allowance is the
panel allocation of 46,000,000,000 bytes; the guard keeps a further 2 GiB reserve.

External, separately hashed files (never inserted into the frozen tar):

- `scripts/runpod_resources_v3.py`
- `scripts/runpod_technical_overlay_v3.py`
- `scripts/runpod_zero_update_preflight_v3.py`

Place all three in `/workspace/ejepa_operations`. The technical overlay explicitly
replaces the old resource adapter before its binding is installed. Its migration
record includes both adapter SHA-256 values and the original manifest/authority
hashes. The original scientific files and bundle manifest remain byte-identical.

The resource adapter supports both cgroup versions, limits each visible ancestor,
and honors v1 `hierarchical_memory_limit` even when an ancestor is not mounted.
The Pod allowance applies at the container scope. It includes all charged cache
in usage and records `memory.stat`; no cache is speculatively deducted, and the
old fixed 23 GB usage ceiling is removed. Missing/corrupt controllers or counters
deny execution. Deadlines and the 2 GiB reserve remain enforced.

Storage is bounded by the user-confirmed **50,000,000,000-byte allocation**, minus
logical bytes measured with `du -sb /workspace`. The multi-petabyte shared `df`
capacity is never treated as this Pod's quota. A 10 GB free reserve plus 1 GiB
checkpoint reservation remains required. This technical policy measures volume
usage synchronously; its cost is not a benchmark of an optimized training guard.

MooseFS is admitted by its identified filesystem/source plus a functional probe
of file fsync, rename, directory fsync, readback and exclusive locking. Unknown
FUSE and `fuse.g` are rejected. The v3 probe passed on the deployed Pod; this does
not prove survival of power loss or a later Pod stop. Scientific execution stays
disabled. Both `producer` and `finish-rotation` commands are rejected by this
entrypoint.

For a direct zero-update preflight after transfer and installation:

```bash
export PYTHONPATH=/workspace/ejepa_bundle/workspace/e-jepa-ttc-v18-joint-measurement/src:/workspace/ejepa_bundle/workspace/e-jepa-ttc-v18-joint-measurement
export PYTHONDONTWRITEBYTECODE=1
/workspace/ejepa_env/bin/python /workspace/ejepa_operations/runpod_technical_overlay_v3.py preflight \
  --bundle /workspace/ejepa_bundle --run /workspace/ejepa_technical/r4_rgb_v3 \
  --rotation 4 --fit R_A5 --budget-bytes 46000000000
```

The optional orchestration script waits for the *existing* transfer process,
identified by PID and `/proc/PID/stat` start ticks. It then checks the frozen
manifest SHA-256 `8326ce1d265430369815d520680fe5f9fd70b29e49501e303a8979e1bf86b298`
and runs the command above exactly once, with a bounded timeout. It never creates
an assignment or invokes a probe/producer. Status and logs are written under
`/workspace/ejepa_operations/`. A transfer failure leads to manifest/file-integrity
failure, not training. A PASS is accepted only with `optimizer_updates=0`.

The pilot tar does not contain `origin_git_commit`. This does not block the RGB
preflight or producer resume probe: those routes use original source hashes and
do not call `operational.final_refit.training._git_commit`. The latter belongs to
later head-training stages. The local repository base observed during review was
`771c96a23784e1f3955cec21ba1f4d9a063a9145`; this alone is not a statement that the
dirty frozen execution code equals that commit. No git provenance is fabricated.

Local validation: 35 migration/overlay unit tests passed, Ruff passed, and Pyright
for the Linux scripts reported zero errors. The new Pod's functional filesystem
probe and cgroup resource guard also passed, with zero optimizer updates.

The complete remote preflight finished with exit 0 at **2026-10-26 11:52:15 UTC**.
All 7,093 transferred entries passed manifest verification. R4/R_A5 loaded its
54,615-query producer population, retained its 30,726-update scientific endpoint,
and matched the four-query local CPU reference bit-for-bit. This is a sampled
input parity test, not all-query output parity or model evaluation. Runtime:
Python 3.11.15, torch 2.11.0+cu128, torchvision 0.26.0+cu128, CUDA 12.8, one RTX
4090, native hardware with BF16 support. All 87 installed packages passed
`uv pip check`. No forward/backward training or optimizer update was executed.

Migration binding SHA-256:
`4df2e19515c296cf1fdcabc05cc8c99de2c73aed6a5e006f77ebd6795e279361`.
Remote receipts were copied to `artifacts/runpod_migration_20261026/`:
`PREFLIGHT_R4_R_A5.json`, `MIGRATION_BINDING.json`,
`ZERO_UPDATE_PREFLIGHT_STATUS.json`, `ZERO_UPDATE_PREFLIGHT.log`,
`ENVIRONMENT_V3.json`, `ENVIRONMENT_V3.lock.txt`, and `RESOURCE_FS_PROBE_V3.json`.

At the preflight resource check, charged cgroup usage was 21.34 GB, including
20.75 GB cache, with 24.66 GB remaining under the 46 GB allowance. Volume logical
usage was 18.23 GB, leaving 31.77 GB under the confirmed 50 GB allocation. These
are measurements at that check, not guarantees for a future workload.

Still required before scientific continuation: an explicitly budgeted technical
resume probe, its matching GPU forward/backward and state-restoration evidence,
local/cloud task ownership exclusion, and a separately admitted scientific
entrypoint/assignment. The current v3 entrypoint deliberately remains technical
only. The pilot transfer does not contain the inputs for complete end-to-end
rotations, PAIR/context heads or test12 inference.
