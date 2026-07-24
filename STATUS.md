# Recovery status

Updated: 2026-07-13.

## Current state

- Code gates and checkpoint provenance fixes: implemented.
- Split status: `reused_test_diagnostic`.
- Local source data: 9 sequences, 1,121 files, 58,137,313,248 bytes.
- Full navigation cache: present, 2,402,953,055 bytes.
- Historical claimed family: inventoried in `artifacts/registry.jsonl`.
- Historical end-to-end seeds: SSL `{7}`; downstream `{7,13,21}`.
- Post-fix promotable metrics: none.
- Final test: unavailable; CPLA-high is diagnostic only.
- ONNX Export: implemented, requires reproducible real-smoke validation.
- Streaming Inference: implemented, requires reproducible real-smoke validation.
- Robustness: placeholder evaluation only. Phase C and D remain incomplete.

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
