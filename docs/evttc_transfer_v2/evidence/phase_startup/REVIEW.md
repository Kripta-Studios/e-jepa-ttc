# Independent startup-router review, 2026-10-31

Reviewer: fresh agent `startup_router_final_review`, read-only review, no additional
reviewers. Scope: all new startup modules, training/refit/validation/robustness/
inference/audit/report/packaging scripts, focused tests, plan and generated evidence.

Verdict: ready as an additional research candidate after final verification;
previous robust release remains default. Critical:0. Important:0. Minor:2.
The reviewer independently loaded both bundle variants and verified exact
eager/TorchScript outputs on2677rows each, all finite. This is tensor-interface
verification, not a claim of bitwise end-to-end sensor replay.

1. Calibration/inference dispersion differed when geometric TTC exceeded60s.
   Concrete trigger: neural[1,100,200], raw geometry1000 gives runtime MAD99.5s
   versus calibration49.5s. Fixed by applying the same float32/clipped geometry
   convention before inference dispersion. RED->GREEN regression retained in
   `output/phase_startup_interval_red.log`; focused final suite covers it. No
   checkpoint or point prediction changed; bundle rebuilt with new implementation
   binding. Pre-review bundle, source snapshot and report retained separately.
2. Narrative metrics in report generator were literals although tables were
   generated. Fixed: trial, dropout, public-score, population and experiment
   counts now derive from receipts/arrays. Table explicitly labels pooled-row
   metrics versus macro-record selection.

Packaging observation: source inventory uses `git ls-files`; source must be
committed before archiving. Enforced by execution order, with ZIP source inventory
and SHA verification at delivery.

Declined-to-judge rulings: independent/SOTA/safety claims remain unsupported and
are not made; candidate extrinsic remains unverified physically; official750
target alignment remains unavailable; JEPA benefit cannot be inferred; fresh
sensor replay is measured separately from exact tensor export. Full-suite and
archive completion are coordinator release gates, not reviewer findings.

No second review requested. Both minor fixes were handled in the one final fix pass.
