# Final independent review and decisions

A fresh reviewer (`submission_final_review`, GPT-6 Astra) reviewed the new
submission implementation against the final-submission plan, read-only. It found
0 Critical, 4 Important and 2 Minor issues. No second reviewer was dispatched.

The four Important findings were addressed in one correction pass:

1. The archive omitted root `README.md`, required by package metadata. The
   installation-file inventory now includes it. The staged repository built and
   installed successfully with the documented dependency group.
2. `dev` was incorrectly described as an extra. The command now uses
   `uv sync --group dev --no-editable`. The non-editable form also avoids an
   observed Python 3.11/Windows editable `.pth` decoding failure in accented paths.
   Research commands use `uv run --no-sync` and the documented source path.
3. Model and evidence inputs were individually valid but could be mismatched
   during packaging. The packager now binds model identity across replay,
   evaluation and final checks; training inputs, target inputs and historical
   files must match their generating receipts, and the checked commit must match.
4. The replay's source inventory omitted executable dependencies. The live runner
   and its sealed outputs were retained. A supplemental audit compares all 2,054
   pre-existing tracked Python files against the pre-run commit; 17 differ only
   in CRLF/LF checkout encoding. Exact current file hashes are bound at packaging.
   The README now states that this is a post-run retained-checkout audit, not
   continuous monitoring or a complete pre-run dependency freeze.

Regression tests first failed for the absent release-binding/install-inventory
guards, then passed after implementation. They reject substituted model bytes,
targets, training inputs, executable dependency bytes and commit identity.
The full-suite outcome and static checks are recorded in `FINAL_CHECKS.json`.

The two Minor findings remain documented:

- Root `START_HERE.md` is a copy of the repository guide, so its phrase "this
  document's directory" for `REPORT.md` applies to the repository copy. The
  report is at `repository/docs/evttc_submission_20261031/REPORT.md` inside the ZIP.
- The final fit's bound configuration has seeds, source/input hashes, checkpoints
  and durations, but lacks its own exact training start/end timestamps and full
  environment receipt. Those historical times have not been invented. The replay
  has a full environment and timing receipt; training provenance is narrower.

Decisions and limits retained by the implementer:

- Keep the inherited fixed fusion recipe and exclude the actual benchmark
  recordings from this fit. Historical selection exposure remains; this cannot
  establish independent generalization.
- Apply negative interpolation only in evaluation views. An unsupported endpoint
  negative raises rather than inventing an organizer extrapolation rule.
- Describe right-camera 16 mm identification as a calibration-FoV inference;
  give Slider pixel intrinsics without inventing nominal lens millimetres.
- Retain the sealed runner and supplement its source evidence, accepting that
  post-run Git comparison is weaker than a full dependency freeze before execution.

The reviewer declined to certify pending runtime results, full tests, report and
ZIP bytes; the implementer subsequently checks these through the receipts and
archive verification. Official target alignment/scores and physical calibration
remain outside what these software checks establish. No raw sensor dataset is
redistributed; derived numeric features and small public target tables accompany
the research evidence. Dataset ownership or licensing is not asserted by this
package. Publication and leaderboard acceptance remain with the organizers.
