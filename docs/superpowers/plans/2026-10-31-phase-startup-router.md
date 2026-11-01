# PHASE startup router implementation plan

> Execute inline with executing-plans, TDD and one fresh final review. User
> explicitly requests continued autonomous work; no repeated approval questions.

Goal: address demonstrated startup/sign failures without masking coverage or
claiming fresh independence for exposed data. Preserve commit70efabae artifacts.

Spec: remaining work in the user's status discussion and original ZIP chapters
04–07,10–12,15. Existing isolated research/evttc-transfer-v2 worktree retained.

Diagnosis: Slider750 F0 starts11.97,-42.33,26.34s while the event context is
4.75,7.28,6.05s. RGB saturates/sign-flips. The previous gate cannot select event
context directly, forbids opposite-sign geometry, and requires3boxes/50ms before
geometry can help. These are demonstrated structural restrictions, not proof
that an encoder change is required. Public-GT MAPE differs from the old aggregate;
no score-alignment offset or clipping will be fitted to force agreement.

Architecture: a separate versioned four-expert router over F0, RGB context, event
context and causal geometry. Permit a two-observation physical estimate with
>=10ms separation; retain count/span/quantization quality. A single box does not
create geometry. No supplied bbox history preserves exact F0. Gate inputs use
native magnitudes, finite masks, sign agreements and geometric quality, so joint
sign reversal changes prediction sign exactly. No forced positive TTC.

Controls before fitting: Q=quality router without augmentation; QA=same router
with neural-expert failure augmentation; NAcap=matched parameter count without
20 geometric-quality features, same augmentation. Shared13 gate inputs, quality
33->18->4 and control13->38->4 both688parameters. All experts are present in all
outputs. Geometry is never augmented into an oracle. Fixed300full-batch AdamW
updates,lr.01,wd.01,clipping1; same family-balanced MAE/relative loss as prior.
Augmentation:25%rows, one randomly chosen native expert multiplied by a signed
factor from{-3,-1,.25,4}, leaving labels unchanged; use50%clean+50%augmented loss.
Train-only normalization. Seeds7,13,21, average every seed.

All three Slider recordings are now exposed development. Retain CCRm outside
current fit/selection and CPNAO for descriptive calibration. These are historical
exposure controls, NOT new independent confirmation. Development folds hold out
entire families, including all Slider together. Select lowest road macro RAE
among variants improving both road macro RAE/MAE over F0 and improving Slider
macro RAE/MAE over previous delivered quality; show comparison with prior OOF on
identical road rows and geometry-only coverage. Keep all attempts.

- [x] 1. Implement/test versioned short-history geometry and sign-equivariant
  router; quantify source startup failures and audit old-score provenance.
- [x] 2. Build sensor-only features, bind exposed target joins, run fixed controls
  and3seeds; select/freeze before current reserve scoring. Save every endpoint.
- [x] 3. Evaluate/calibrate, raw/bbox robustness, source fallback, streaming
  startup and exact export; fit an explicitly descriptive all-eligible-data model.
- [x] 4. Audit remaining calibration/latency gaps with actual available evidence;
  do not invent an authoritative extrinsic or clean external acquisition.
- [x] 5. Fresh final review, focused/static/regression checks, regenerated report,
  portable immutable delivery and progress ledger. No remote publish requested.

Ledger: original weights, data, historical reports and ZIP remain immutable.

2026-10-31 execution ledger:

- Seconds-coordinate Q/QA/NAcap x 6 families x 3 seeds completed, plus three
  selected development fits. QA fails promotion versus prior road OOF:
  15.2286%/.541610s versus12.8336%/.422359s. Preserved training/ without alteration.
- Ruling: execute one physically motivated coordinate ablation, identical budget
  and controls, mixing inverse TTC rather than seconds. Signed cancellation
  explicitly falls back to F0; zero-TTC experts are unavailable in inverse space.
  Cost: more exposed-development selection, explicitly not clean confirmation.
- Inverse QA road12.6513%/.404882s; family-paired95% bootstrap delta to prior
  [-1.5176,+.8001]pp across only five families. CCRm RAE15.1910% improves but
  MAE1.035545s worsens from1.019652s. No superiority claim.
- All-data refit three seeds,2677eligible rows/27recordings: road10.9488%,
  Slider19.7372%; descriptive resubstitution only, no validated intervals.
  Total117research fits including failed seconds control; encoders unchanged.
- Raw24recording x 6condition stress run complete; all three native expert
  outputs cached for future reuse. Candidate clean13.9528% and dropout70%
  14.9805%, worse than previous release. Previous release remains default.
- Ruling: bbox-dropout revealed stale-history neural rerouting even when geometry
  is rejected. Apply the existing150ms physical freshness limit to the entire
  runtime router, exactF0 fallback. RED->GREEN regression. Zero stale rows in
  training/evaluation; cached2677row outputs unchanged bitwise, no weight edits.
  Retain failed validation/; corrected validation_v2/ dropout28.2881%=F0.
  Stress test is now exposed to implementation selection.
- Source fallback exact88744eAP rows without history; TorchScript exact2677rows
  per variant. Native CLI replay differs0.00000095s; Slider8query fresh CUDA
  replay max0.00354445s versus cached prior sensor outputs (precision contract
  differs: old competition extraction inherited TF32, new CLI disables TF32).
  Do not describe end-to-end sensor replay as bitwise equal.
- PDF audit: original submitted750predictions on publicGT282.4982% vsPDF132%;
  1000matches156.8813%. Formula/counts explicit, aligned evaluatorGT unavailable.
  Candidate extrinsic synthetic parallax measured, no production projection edit.
- Full-suite first launch omitted UV_PROJECT_ENVIRONMENT and encountered three
  early failures; interrupted before detailed traces, aborted log retained.
  Relaunched complete suite with
  explicit .venv-transfer for all child commands, result pending.
- Ruff/Pyright pass at this checkpoint. Portable bundle built with named variant,
  implementation, upstream/export/calibration hashes. Final independent review,
  full-suite completion, final receipt, source commit and verified archive pending.
- Final reviewer found0Critical/0Important/2Minor. Both minors fixed: uncertainty
  now uses the calibration float32/clipped geometry convention (RED->GREEN), and
  narrative metrics regenerate from receipts. Final focused11tests pass. Original
  pre-review model/source/report retained; fresh final bundle rebuilt and both
  native/Slider CLI replays completed against its binding.
- Ruling: keep the existing research branch locally, without merge or publish.
  User requested autonomous local continuation; an integration-choice menu would
  not advance the authorized training/delivery task. No shared branch changed.
- Final release gate passed: full suite7320passed/46skipped plus17subtests;
  final focused11passed; Ruff and Pyright clean. Both variants export exactly on
  2677rows, bind all original upstream components and completed final native/Slider
  replays. `FINAL_CHECKS.json` seals results, source-reproduction proof and runtime.
  Source/evidence committed before packaging; the archive command rehashes every
  member and checks CRC, recording its final commit/hash in `DELIVERY.json`.
