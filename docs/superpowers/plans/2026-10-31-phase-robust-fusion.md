# PHASE robust fusion implementation plan

> Execute inline with executing-plans; one fresh final review. User explicitly
> authorizes completing the remaining investigation and stronger model locally.

Goal: deliver a trained and auditable candidate that improves the measured weak
regimes without hiding regressions, or preserve the strongest supported baseline
if a preregistered intervention fails. No promise of a universally best model.

Spec: PHASE_TTC_Ruta_SOTA_2026-10-30 chapters05–10,14,15 and the user continuation.
Base73a74f4d; existing isolated research/evttc-transfer-v2 worktree retained.

## Design and constraints

Slider500 is now exposed development. All32 road recordings and FCWD historical
evaluations are exposed; reserve refers ONLY to fitting/design in this iteration.
Do not manufacture acquisition independence or official equivalence. Original
candidate and all previous delivered weights/results remain immutable.

First diagnose physical expansion using causal bbox histories <=650ms and actual
timestamps. Compare inverse-scale linear motion fits with native neural producers,
contexts and F0/F1. Labels join after feature construction. Geometry cannot consume
GT depth, TTC, identity or a future box. Missing inputs remain explicit.

If geometry carries usable complementary signal, train a bounded mixture of
neural and geometric TTC using inference-available quality. A matched-capacity
control is required to distinguish quality information from more parameters.
If it does not, investigate the identified producer before changing its weights.
Direct encoder/JEPA work is conditional on an isolated representation deficit,
not on poor final scores alone. New variables and wider history must be documented
as an input-contract change rather than a pure original-head refit.

Fixed scientific priorities: coverage never silently falls; report both macro
relative error and MAE, TTC<1s/>4s and source-domain protection. Favor a candidate
that improves BOTH road metrics and exposed Slider over F0; no universal practical
deployment margin is invented. Freeze selected model before current-iteration
confirmation, keep negative outcomes, do not retune after reserve results.

## Tasks

- [x] 1. Physical diagnostic: causal inverse-scale fit, quality features, synthetic
  endpoint/noise/occlusion/future/clock tests; road and Slider stage decomposition.
  Files: operational/phase_robust/geometry.py, data.py; tests/unit/test_phase_robust_geometry.py;
  scripts/diagnose_phase_robust.py. Record conclusions before selecting architecture.
- [x] 2. Complete available external contracts/ancestry: public Slider GT metadata
  versus local sensor identities, existing FCWD inputs and history; no equivalence
  asserted on names. Preserve authoritative gaps and freeze prospective comparison.
- [x] 3. Implement the smallest justified trained correction, capacity/information
  control and deterministic source fallback. Fixed small search; grouped fitting
  with train-only normalizers/calibrators; seeds7/13/21. Save every attempt.
- [x] 4. Select by declared development criteria, fit final candidate, freeze it,
  then perform reserved current-iteration evaluation with historical exposure stated.
- [x] 5. Raw sensor and bbox robustness, acceleration/startup/negative fixtures,
  uncertainty calibration with separate fit/calibration groups, runtime/cache parity,
  strict reload/export; package immutable receipts and runnable inference.
- [x] 6. Fresh independent review, required fixes, focused/static/full checks once
  warranted, generated report and local delivery; update docs/progress.md.

## Ledger / rulings

- Using planning, systematic diagnosis, TDD and inline execution. User has already
  opted into autonomous implementation; no repeated design/permission question.
- Ruling: no cloud/publishing; preserve current worktree and past research artifacts.
- Ruling: FCWD is historically exposed (existing 20261008/13 evaluation evidence),
  not a new clean dataset. Cost: additional evaluation has a narrower claim.
- Broad repository baseline: previous 7297passed/46skipped +17subtests; new step
  starts with focused tests rather than needlessly repeating an unchanged full run.
- Task 1 done: robust inverse-scale geometry improves Slider but is unreliable on
  long road TTC; retain a learned quality-conditioned convex mixture. Unit math
  tests cover current endpoint, future exclusion, sign, outlier and acceleration.
- Task 2 done: public Slider 750/1000 bbox arrays exactly match local query CSVs;
  exact frame and <=2us sensor-clock target join passes for 51/18queries. Public
  target scoring is local, not an official benchmark revalidation. FCWD full
  fusion remains unavailable due missing audited spatial mapping.
- Ruling: quality 23->16->1 gate and native 3->80->1 control both 401params;
  both outputs use geometry. This isolates gate-quality inputs, not availability
  of geometry. Six development families, CCRm confirmation, CPNAO calibration;
  all historically exposed, with no claim that exposure was undone.
- Task 3/4 done:36CVfits+3finalfits, seeds 7/13/21, fixed 200updates. Select quality
  by preregistered road macro RAE+MAE and Slider 500 RAE+MAE gates. Frozen before
  CCRm/Slider 750/1000 evaluation. All experiments and negative control retained.
- Ruling: honor requested all-data final training with a supplementary fixed
  recipe refit on 2677queries/27recordings after confirmation. No hyperparameter
  or architecture change;24eligible roads+3Slider, original 8overlap recordings
  remain excluded. Its scores are resubstitution only and calibration labels are
  now fitted. Keep evaluated weights as default; deliver both immutable variants.
- Task 5 done:144raw perturbation queries,144bbox perturbation queries,88744exact
  source-fallback predictions, exact TorchScript exports and lossless sensor-cache
  parity. Conservative fallback retains serious Slider 750 startup failures.
- Review: one fresh reviewer found two Important packaging/adapter gaps, no
  Critical or additional scientific defects. Both fixed with contract/tamper
  tests: Slider-only competition CLI; separate native-road CLI reproduces cached
  7.188720703125s exactly; outer variant/export/calibration binding prevents swaps.
- Ruling: no encoder/JEPA training added. Evidence supports useful physical inputs
  and quality fusion; it does not isolate an encoder representation deficit.

- Task 6 verification:7309passed/46skipped+17subtests;14final focused tests pass;
  Ruff and Pyright pass. Preserve the existing research branch/worktree and all
  prior artifacts; no merge or remote publication was requested. Final archive
  is verified member-by-member before delivery.
