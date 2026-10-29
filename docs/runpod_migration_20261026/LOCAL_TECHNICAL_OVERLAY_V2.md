# Technical pilot on local POSIX storage

The deployed `/workspace` is a global object volume mounted with FUSE.g. The
observed chmod/chown/utime failures and documented absence of atomic rename/locks
make it unsuitable for the current checkpoint protocol. The interrupted partial
transfer there is not a usable bundle. No scientific training is admitted on it.

Two operational files are transferred separately from the immutable pilot tar:

- `scripts/runpod_bootstrap_local_v2.sh`
- `scripts/runpod_technical_overlay_v2.py`

The original 10,569,820,160-byte tar and its manifest remain unchanged. The new
overlay records its own SHA-256 in a new technical migration binding. Original
scientific sources, configuration bytes, seed, precision and recipe remain bound
to the original manifest. Run names must be new, not reused from `/workspace`.

After the parent has completed transfer into `/opt/ejepa_bundle`, place the two
small files in `/opt/ejepa_operations/` and run:

```bash
bash /opt/ejepa_operations/runpod_bootstrap_local_v2.sh /opt/ejepa_bundle
export PYTHONPATH=/opt/ejepa_bundle/workspace/e-jepa-ttc-v18-joint-measurement/src:/opt/ejepa_bundle/workspace/e-jepa-ttc-v18-joint-measurement
/opt/ejepa_env/bin/python /opt/ejepa_operations/runpod_technical_overlay_v2.py preflight \
  --bundle /opt/ejepa_bundle --run /opt/ejepa_technical/r4_rgb_v2 \
  --rotation 4 --fit R_A5 --budget-bytes 31000000000
```

Bootstrap installs Python 3.11.15, torch 2.11.0+cu128, torchvision 0.26.0+cu128,
and the package versions in the frozen manifest. The Runpod template's torch
2.9.1 is not the scientific runtime. No editable install modifies bundle files.
The installer cache and interpreter live on container storage. Installation needs
free disk in addition to the roughly 10.6 GB inputs; inspect real container free
space before and after installation. The resource guard retains a 10 GB reserve
plus a 1 GiB checkpoint reservation. Bootstrap never deletes data automatically.

Preflight does zero optimizer updates. It verifies all manifest bytes, validates
the historical authority and recipe, compares a deterministic CPU batch with the
Windows reference, checks exact runtime versions and one BF16 CUDA device, and
checks memory/disk. The overlay additionally rejects FUSE filesystems and tests
local fsync, rename, directory fsync, readback and exclusive locking. This is not
a crash or power-loss durability proof. The container disk can be lost when the
Pod stops, so `scientific_execution_allowed` remains false even after PASS.

The optional `resume-probe` command still requires an explicit assignment with
the new migration SHA, a positive USD budget, future execution deadline and at
least eight authorized technical updates. It compares four uninterrupted updates
against two plus restored two. It is a same-process reconstruction test, not an
abrupt-process/power-loss test, and does not admit a scientific endpoint.
`producer` and `finish-rotation` are rejected before importing training code.

The 31,000,000,000-byte memory allowance is a conservative declared technical
budget, not a claim about physical Pod RAM. The guard also reads actual cgroup
limits; host MemAvailable alone is never used. Its unchanged 23 GB cgroup-current
ceiling includes charged file cache and can pause a cache-heavy run. If that
happens, inspect `memory.stat` (anon/file/inactive_file), current/maximum and actual
allocation before changing the policy. The host's 1 TiB and FUSE's virtual 1 PiB
are not usable allocation measurements.

Scientific continuation requires a suitable volume disk/network volume and a
separate binding for the durable run. Background copying local checkpoints into
the global volume does not reproduce the atomic checkpoint protocol and is not
implemented or claimed here. No local queue tasks are reassigned by this overlay.

Local validation: 21 unit tests passed (13 migration tests plus 8 overlay tests),
Ruff lint/format passed. Linux filesystem and CUDA probes must run on the Pod;
the local Windows checks do not certify those paths.
