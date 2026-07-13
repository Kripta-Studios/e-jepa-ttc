# Recovery rerun plan

## Entry condition

Do not start a promotable run until the recovery changes are committed and
`git status --porcelain` is empty. `scripts/run_recovery_multiseed.ps1` enforces
that rule. `-Smoke` is allowed on a dirty tree but every resulting artifact must
be registered as `smoke_only`.

Before training:

```powershell
$env:PYTHONPATH = (Resolve-Path src)
.\.venv\Scripts\python.exe scripts\validate_artifact_registry.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests scripts
Get-FileHash data\manifests\evttc_full_starter_local.yaml -Algorithm SHA256
Get-FileHash data\splits\evttc_full_starter_sealed.yaml -Algorithm SHA256
```

Copy the clean commit and current hashes into each registry record; do not reuse
the historical hashes after any data or protocol change.

## Claim-eligible end-to-end rerun

Run the fixed configuration in
`configs/experiment/recovery_full_starter_multiseed.yaml` with paired seeds
7, 13, and 21:

```powershell
.\scripts\run_recovery_multiseed.ps1
```

This produces three independent SSL initializations and, for every SSL seed,
three downstream seeds (nine downstream runs total). It evaluates only
train/validation and initializes downstream training from each SSL
validation-selected `best` checkpoint. The runner appends a complete registry
record after every run. Record both SSL `best` and `last`; use only `best` for
the primary downstream result.

A 1x1 execution is available only through `-Smoke`; it is not claim-eligible
and the runner records it as `smoke_only`.

## Verifiable time estimate

On the local RTX 5070 Ti Laptop GPU, the matching historical configuration took
`1,343.407499 s` for 30 SSL epochs. The three historical 80-epoch downstream
runs took `292.269081`, `312.281032`, and `254.896432 s` (median
`292.269081 s`). Therefore:

- paired 3 SSL + 3 downstream: approximately `4,907.0 s` (`81.8 min`);
- full 3 SSL + 9 downstream: approximately `6,660.6 s` (`111.0 min`);
- existing cache construction took `830.668003 s` and need not repeat if its
  hash and split mapping still match.

These are measured extrapolations, not promised runtimes. Run the GPU jobs
sequentially: 12 GB VRAM is not enough to assume three concurrent trainers are
safe. CPU-only manifest validation and registry hashing can run in parallel.

## Test discipline

1. Freeze configuration, seeds, checkpoint selection, and aggregation using
   validation only.
2. Hash the frozen artifacts and append their validation records.
3. Evaluate all frozen checkpoints on CPLA-high in one diagnostic batch.
4. Aggregate once with `--claim-level diagnostic --split-protocol
   data/splits/evttc_full_starter_sealed.yaml`.
5. Never use that result for another architecture, hyperparameter, or checkpoint
   decision.

No command in this plan can produce a final claim from the currently available
nine sequences. A final run requires a newly acquired, never-inspected test set
and a new split protocol whose metadata explicitly permits `final`.
