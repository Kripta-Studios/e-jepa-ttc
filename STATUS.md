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

The complete audit, SOTA comparison, and MVA/FlowMimic architecture proposal are in
`docs/scientific_audit_2026-07-25.md`.

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
