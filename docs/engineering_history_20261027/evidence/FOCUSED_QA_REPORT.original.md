# Publication verification, 2026-10-27

The selected 128-file CPU test run covered 873 tests: 864 passed, eight failed due to stale fixtures, and one optional technical replay was skipped. The five corrected test files then passed all 22 targeted tests, resolving all eight failures. This combines evidence for 872 passing tests and one skip; it is not a claim that the whole suite was rerun after editing.

Only test fixtures changed. The export fixture now supplies the full required RGB source closure; the handoff fixture fixes its clock; V8/V9 RGB tests construct small exact-schema identity chains independently of local artifacts. Production validators and frozen scientific bytes remain unchanged. No scientific inference or training ran.

Global Ruff still reports 495 issues in 33 files; formatting: 144 files would be reformatted, 2624 files already formatted. Residual diagnostics remain unsuppressed in `ruff_final.json` and `RUFF_RESIDUAL_BY_FILE.json`. The five edited tests pass Ruff lint and formatting. Hash-bound executed sources were deliberately not mass-formatted.

Pyright passes with the actual environment (`--venvpath ../e-jepa-ttc`), zero errors/warnings in the five files selected by the existing configuration. Default local .venv resolution instead reports twelve torch.Tensor attribute errors. This does not establish type coverage of every new operational module. All 80 PowerShell scripts parse without errors.

The real V12 event reuse replay and the multi-process integration checkpoint fault suite were excluded to avoid repeated scientific execution and concurrent Torch memory pressure. Exact selected paths and commands are saved alongside this report, together with both JUnit files and original failures.
