# Recovery status

Updated: 2026-07-25.

## Current state

- Code gates and checkpoint provenance fixes: implemented and hardened after audit.
- Split status: `reused_test_diagnostic`.
- Local data inspected: 9 EvTTC starter sequences plus 8 local eAP training sequences,
  approximately 117 GB across 1,181 files.
- Full navigation cache: present, but physical `x.npy` has 3,494 samples while its sidecar
  declares shape 3,972 and `window_count=3494`; format v1 is diagnostic only.
- Historical claimed family: inventoried in `artifacts/registry.jsonl`.
- Physical registry audit: failed with 116 incidences (85 missing references, 26 hash
  mismatches, 5 missing hashes) across 45 records.
- Historical end-to-end seeds: SSL `{7}`; downstream `{7,13,21}`.
- Post-fix promotable metrics: none.
- Final test: unavailable; CPLA-high is diagnostic only.
- ONNX Export: implemented, requires reproducible real-smoke validation.
- Streaming Inference: implemented, requires reproducible real-smoke validation.
- Robustness: placeholder evaluation only. Phase C and D remain incomplete.
- QA after hardening: 191 tests pass; Ruff passes; Pyright/mypy is not installed.
- Scientific hardening was committed and pushed as `416b498` on
  `scientific-recovery-v3-hardening`.
- Physics-constrained FlowMimic implementation is complete; Ruff and 198 tests
  pass. The train+validation-only cache v2 rebuild and exhaustive audit pass;
  E0/E1/E2 validation runs are pending.
- The first train+validation cache build was rejected because a dormant
  `exclude_splits` parameter left 478 test windows in the physical NPZ. The
  filtering path and its regression test are now corrected; no model was
  trained on the rejected cache.
- Accepted cache: 3,494 samples (3,019 train, 475 validation, zero test), SHA-256
  `22d3ef27018925aae62825f0a7f51d1420ae93cacf59aeb18b04758f5a35e88a`.
- Exhaustive cache audit: `passed` at commit `80ff992`, artifact SHA-256
  `02f3f633b13f413c4bf6b49176c3e70d373af52d63ac1602e119402af3a819c2`.
- First FlowMimic GPU smoke S0 was rejected before downstream evaluation: zero
  synthetic navigation combined with a near-constant validity channel caused
  AMP overflow and NaN synthetic loss. Navigation is now neutralized from the
  train-only mean and non-finite JEPA loss aborts explicitly.
- Corrected GPU smoke S1 passed at batch 12 with finite total, synthetic
  alignment and inverse-TTC losses. It is a numerical gate only; E0/E1/E2 TTC
  accuracy remains unmeasured.
- First scratch/E0 downstream attempt was rejected from the promotable matrix:
  the checkpoint SHA was absent from provenance, causing identical run
  fingerprints. Diagnostic MAE was 0.3959 s scratch vs 0.3422 s E0. Physical
  checkpoint hashing and fingerprint coverage are now corrected; both runs
  must be repeated.

The complete audit, SOTA comparison, and MVA/FlowMimic architecture proposal are in
`docs/scientific_audit_2026-07-25.md`.
The active experiment handoff is `docs/flowmimic_experiment_2026-07-25.md`.

> **Integrity Note**: Commit `f96bc35` prepares the real Master Smoke orchestrator. It is not evidence that the Master Smoke passed.

## Core Progress
- Long recovery run: not started because the worktree is dirty.

## Ready for clean baseline commit

The repository has a claim gate, a registry validator, an immutable rerun
configuration, a clean-tree runner, explicit best/last handling, and a measured
runtime estimate. After review and commit, execute the validation-only
multi-seed plan. Do not open CPLA-high until all choices are frozen.

## Verification commands

```powershell
.\.venv\Scripts\python.exe scripts\validate_artifact_registry.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests scripts
```
