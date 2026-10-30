# Execution ledger — PHASE_TTC_Ruta_SOTA_2026-10-30

Source: supplied ZIP, chapters 03–15; baseline commit
`30b28b5c14eb9ed60067ed2f1dc262d7b60f2654`.
Branch: `research/evttc-transfer-v2`. Existing linked worktree reused.

## Scope and dependency decisions

Execute T00–T07 sequentially. Open only the intervention justified by G0–G3.
The supplied chapter 09 explicitly says not to execute every conditional branch.
No official submission or message to organizers is authorized.

- T00 identifies resources; T01 produces identity and clock evidence for T03/T04.
- T02 records genealogy uncertainty; T03 must not turn unknown relations into independence.
- T04 supplies target eligibility independently of predictions; T05 shares that population.
- T06 traces inference without labels; T07 consumes diagnostics and records a gate decision.
- Ruling: use the existing linked worktree and a new branch, preserving all pre-existing
  untracked files. No extra checkout is needed.
- Ruling: retain this durable ledger and receipts after review so incomplete scientific
  gates remain recoverable; no claim of training or SOTA follows from software tests.

## T00

- ZIP extracted without overwriting existing project files.
- Original `.venv` fails during site initialization decoding its UTF-8 editable `.pth`
  as cp1252, including with `-X utf8`.
- Created `.venv-transfer` with `uv sync --locked --no-editable --python 3.11`.
- Initial regression collection failed: cached torch install lacked its package initializer.
  Re-downloaded torch with `uv --no-cache sync --locked --no-editable --python 3.11
  --reinstall-package torch`. CUDA forward/backward and sealed candidate parity passed.
- Current-source verifier FAIL retained in `original_run_verification.json`.
  `local_transport.py` differs by the formatting-only change in commit `820e6a6f`.
  An isolated snapshot from the hash-verified preserved archive passes the existing
  verifier for all 434 rows; AST equality recorded. No historical artifact edited.
- Ruling: verify exact historical bytes in a separate source snapshot; run diagnostics
  with current code and explicitly record the formatting difference. Cost if wrong:
  no byte-identical current-runtime claim; prediction parity is separately tested.

## T01 complete

- `test_evttc_transfer_identity.py`: missing implementation observed, then 3 tests pass.
- Real CLI: 8 unique correspondences, 365 frames, all semantic fields equal to reference.
- Commit `0f091880`; subsequent Ruff formatting is semantic-preserving.

## T02/T03 implemented; scientific status unresolved

- Nine deployed components and two normalizers hash-verified against portable seal.
- Genealogy preserves unknown parents/teacher and selection scopes.
- Inventory: 32 recordings read, 8 excluded, 24 `development_exposed`; acquisition
  relationships not certified. Validator rejects aliases/acquisition role leaks.
- Policy and missing-identity guard tests observed failing, then passed after fixes.
- Ruling: do not assign fit/selection/independent holdout while acquisition relationships
  remain unresolved. Cost: no learned pilot yet; this avoids inventing independence.

## T04/T05 diagnostic implementation complete

- Target/scoring tests first failed on absent modules, then passed: exact endpoints,
  interpolation, duplicate/nonmonotone clocks, missing/nonfinite/zero/signed labels,
  gaps, no extrapolation, macro/micro and official-claim rejection.
- Real population: 434 requested, 354 compatible local targets. Eleven road queries
  remain outside support, and 69 Slider queries lack a compatible verified target.
- Six saved baselines scored; five paired comparisons regenerate using
  `scripts/report_evttc_transfer.py`. Source ledgers retained and clock identity checked.
- Ruling: scores remain diagnostic_partial; local target/scorer equivalence is unresolved.
  Cost: these numbers cannot establish official rank or choose the learned candidate.

## T06 diagnosis

- Eight new integration cases cover physical endpoint, cadence, recession/static,
  parallax, clock shift, future observations and passive trace parity.
- Existing sensor tests cover future frames/events/boxes, short prefixes, observed
  silence and missing recording support.
- Forty real queries: indices 0,1,10,last in all ten sequences; untraced/traced
  predictions equal exactly. First compact receipt: `A00_compact/report.json`.
- First trace attempt retained full intermediate feature maps (~918 MB). Added a
  failing bounded-serialization test, summarized large tensors by shape/hash/statistics,
  and reran all forty queries. Scalar/vector observations remain available.
- First attempt metadata captured source files at run end during formatting edits;
  the compact rerun bound only three source hashes before/after execution. Independent
  review found that insufficient. The reviewed rerun binds the entire local Python
  scientific graph, orchestration and candidate components before/after execution.
  Earlier attempts remain preserved but are not authoritative provenance evidence.

## External data search

- Official Event-Aided TTC page exposes Slider-500, boxes, calibration and GT.
  Downloaded ~466 MB bag plus small annotations; hashes retained. No target values
  inspected and no model inference performed on it.
- Existing reader rejects topics `/dvs/events` and `/aps/image_raw`, differing from
  the frozen competition contract. Connection metadata inspected using existing
  low-level reader helpers. License/acquisition independence remain unresolved.
- Ruling: preserve Slider-500 as a candidate source; do not adapt the frozen reader or
  declare it independent from names alone. Cost: a separate adapter/provenance task remains.

## T07 decision

No learned intervention opened. See `PILOT_DECISION.md`. G4/G5 and representation
experiments remain conditional on eligible development and a resolved target contract.

## Verification attempts

- Focused original regression passed, initially with one resource-gated skip.
- Sealed CUDA parity test then executed with candidate path and passed.
- Full suite attempt stopped at five failures caused by subprocesses selecting the
  broken original `.venv`: `test_cache_requires_lock`, both ONNX provenance tests,
  `test_audit_cache_fails_test_split`, and
  `test_audit_cache_reads_current_builder_metadata_without_float16_overflow`.
- Recheck with `UV_PROJECT_ENVIRONMENT` bound to `.venv-transfer`: 8 passed in 26.19 s.
  Full-suite rerun uses that environment; final outcome recorded below when complete.
- Pyright initially selected `.venv` from project config despite `--pythonpath`.
  A local derived config changes only environment/search paths and includes the new
  audit files; type checking passes. No repository-wide type-check claim is made.

## Task state

Audit tooling and first exposed diagnostic implemented; scientific G1/G2/G3 remain
limited as detailed above. G4/G5 conditional and unopened. No push or submission.

## Independent review and one correction pass

- A read-only reviewer identified three Important issues: declarative lineage could
  become clear without verified evidence; status-only receipts and unknown exposure
  could open training; diagnostics did not bind all preparation sources and candidate
  components to the protocol. Accepted all three findings.
- Added regression tests, observed six failures before fixes, then verified corrected
  behavior. Evidence references now resolve inside the candidate and match SHA-256;
  empty ancestry scopes need explicit verified absence evidence. Unknown selections
  and recursive ancestors remain unresolved.
- Protocol now validates receipt schemas, identity/frame details, manifest/inventory
  hashes, component bindings and known exposure vocabulary. Missing or declarative
  evidence cannot authorize training.
- Diagnosis now matches the actual candidate against the lineage binding, verifies
  it again after execution, and snapshots all Python sources under `src/e_jepa_ttc`
  and `operational`, plus the audit and transfer entry points, before and after.
- Reviewer also reported a Minor zero-coverage report crash. Regraded Important:
  coverage accounting is a scientific acceptance requirement. Report renders absent
  metrics as `N/A`; end-to-end fixtures cover absent and zero-valued targets with
  missing predictions, retaining all six method rows.
- Reviewer declined judgments on official target equivalence, acquisition independence,
  unknown ancestor cleanliness, JEPA benefit and real-world causes inferred from
  physical fixtures. Accepted those limits; none is represented as established.
- Reviewed inputs and receipts use new paths `inputs_reviewed`, `D01_reviewed`,
  `D02_reviewed`, `A00_reviewed`; previous evidence is preserved.
- Final focused review tests: 44 passed in 11.21 s. Ruff passes and targeted Pyright
  reports zero errors/warnings. No second review was requested.
- Reviewed diagnosis completed with exact parity on all 40 queries. Authoritative
  trace receipt: `A00_reviewed/report.json`; candidate and full source snapshot
  verified unchanged before/after execution. Original RUN_FREEZE, results ZIP and
  prediction ledger hashes also rechecked unchanged.

## Final verification outcome

- Whole repository: 7,254 passed, 46 skipped, 17 subtests passed, one failed in
  1,369.56 s. Failure: existing
  `test_result_timeout_keeps_arena_leased_until_terminal_consumption`, where the
  second `future.result(timeout=2)` exhausted its wall-clock budget. The suite ran
  concurrently with the scientific diagnosis. No audit code occurs in that traceback.
- Repeated the complete shared-pool file after diagnosis finished: 10 passed in
  21.14 s, including the failed test, with no code changes. This is consistent with
  timing sensitivity under load; it does not prove the full suite is unconditionally
  green. Preserve both logs, and treat the two-second test budget as a remaining
  reliability concern rather than altering the sealed runtime for this task.
- The full suite was collected before the final review additions. Those changes
  have their own final 44-test run. The earlier broader targeted run passed 115 tests.
- Regenerated six baseline reports and five paired comparisons using the final
  report implementation (`E01_reviewed`); no metric changes.
- Canonical logs: `full_suite_environment_fixed.log`, `shared_pool_recheck.log`,
  `review_final_tests.log`, `pyright_reviewed.log` under `output/evttc_transfer_v2`.
