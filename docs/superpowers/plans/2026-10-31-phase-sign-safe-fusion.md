# PHASE sign-safe fusion implementation plan

> Execute inline using executing-plans, test-driven-development and a fresh final
> review. User explicitly authorizes continuing research and training without
> repeated approval. Existing isolated research/evttc-transfer-v2 worktree retained.

Goal: test whether avoiding signed inverse-rate cancellation and learning a bounded
correction to causal geometry improves transfer, with explicit eAP preservation.
SOTA is an objective, not an assumed or guaranteed outcome.

Spec: user's 2026-10-31 continuation and the preceding measured diagnosis. The
frozen submission remains immutable. New package operational/phase_safe_fusion,
tests/unit/test_phase_safe_fusion.py, scripts/train_phase_safe_fusion.py; outputs
under output/phase_safe_fusion. Other uncommitted JEPA closure work is untouched.

## Protocol fixed before new fitting

- Input: audited submission training population: 2608 queries, 25 recordings,
  24 road recordings plus Slider-500. Exclude all original eight overlap road
  recordings and Slider-750/1000. All data have historical exposure; no pristine
  test claim. No benchmark targets enter model fitting or candidate selection.
- Eight leave-family-out folds; acquisition groups remain disjoint. Seeds 7,13,21.
  Normalization and target loss scale fit only on training rows. Fixed 300 AdamW
  updates, lr .01, weight decay .01, gradient clip1, family-balanced relative plus
  absolute error. Average all seeds; no best seed selection.
- Controls: original inverse QA retrained on identical folds; geometry with F0
  fallback; LG learned same-sign log-magnitude fusion; LA same with existing
  expert-failure augmentation; GR geometry anchored bounded residual plus LA
  fallback when geometry unavailable. GR correction bounded to factor [0.5,2].
- Log fusion chooses the sign with larger learned total mass, then mixes ONLY
  experts of that sign. Exact sign-mass ties use first available expert's sign.
  This prevents opposite-sign rate cancellation; it does not force positive TTC.
  No history, stale history, or nonfinite quality preserves original F0 exactly.
- Candidate promotion requires road macro-record MAE and RAE no worse than
  retrained inverse QA OOF and Slider-500 MAE/RAE no worse than that control.
  Among passing candidates select minimum overall macro-record RAE; otherwise
  explicitly reject promotion. No tuning after scoring benchmark outputs.
- eAP: first audit active-history availability independently. Missing-history
  identity is checked but never claimed as active-router source validation.
- CPU full-batch cached features initially; measure runtime before GPU/CUDA graphs.
  No expensive sensor regeneration unless required by a distinct experiment.

## Tasks and verification

- [ ] 1. RED/GREEN tests: inverse cancellation fixture bounded under sign-safe
  fusion; sign reversal, permutation-independent non-tie sign, exact fallback,
  invalid experts, geometry residual bounds, finite gradients, TorchScript parity.
- [ ] 2. Train fixed controls/folds/seeds; persist configs, source/input hashes,
  fit group lists, checkpoints, losses, predictions, timings, selection receipt.
  Verify exclusion and train/validation group separation before every fit.
- [ ] 3. Evaluate only frozen selected candidate on exposed benchmark and source
  conditions; preserve coverage, negative-processing protocol and missing labels.
  If no candidate passes, retain negative result and diagnose development failure.
- [ ] 4. Based on development evidence, preregister any further augmentation or
  geometry experiment separately. Include rejected controls in reports. Review,
  run relevant regression/static checks and commit only this campaign's files.

## Review focus

Sign cancellation vs sign selection, all-invalid inputs, invalid geometry with
nonfinite quality, train-only statistics, data grouping, accidental benchmark
selection, source fallback vs active source performance, reproducible frozen
selection before exposed benchmark scoring.

## Ledger

- Existing git worktree verified; HEAD ad875421. Unrelated JEPA closure files
  already present and retained. No active local training process observed.
- Independent read-only audits delegated: active eAP history and evaluator
  target/clock compatibility. They do not modify candidate code or selection.
- Task1: seven physical/model tests passed after observed missing-module RED;
  candidate selection test also RED->GREEN. Initial campaign_v1 stopped after
  six fits because immutable save_npz cannot overwrite a changing progress file.
  Retain that failed run; distinct fold/control snapshots tested RED->GREEN.
  Restart unchanged scientific recipe in campaign_v1b; no score-based change.
- campaign_v1b completed96fits in220.6s. All new controls rejected. Road macro
  RAE/MAE IQ17.7489%/.5434s; LG18.6083%/.5962; LA18.1358%/.5674;
  GR24.8917%/.7715. Slider500 IQ20.8519%/.3177s; GR19.4199%/.3800.
  No new model promoted or benchmark evaluated. Frozen complete result retained.
- Fresh review: individual fusion contracts pass, but averaging signed seeds
  can cancel magnitudes and perturbs bitwise fallback. Hard sign decision can
  have zero gradient for a sole wrong-sign expert. These limitations are retained
  in v1; v2 preregisters explicit training supervision and median ensemble.

## Preregistered v2 after v1 development/review, before v2 fitting

Two controls SG and SGA use the same log SafeFusion, unchanged initial gate,
three seeds,300updates, optimizer, folds and exclusions. Both add 0.5 times the
expected per-expert supervised cost under gate probabilities to the original
prediction objective. Expert cost is relative+scaled absolute error capped at10;
only training targets enter it. This supplies gradients even when the hard sign
choice admits a single wrong-sign expert. The existing25%neural failure
augmentation remains. SGA additionally samples one of four per-query geometric
channel perturbations: last2boxes,last4boxes,2pixel jitter,50%dropout. Native
predictions remain unchanged; this is NOT augmented raw sensor inference.

Median of three signed seed outputs prevents seed cancellation; explicit
post-ensemble history mask returns exactF0. Persist every seed OOF output and
dispersion. IQ baseline is the existing v1b arithmetic ensemble as deployed.
Selection uses the same four road/Slider macro MAE/RAE inequalities and full
coverage, then minimum all-record macroRAE. If selected, refit all2608 and freeze
before active-source and exposedbenchmark evaluation. Reject promotion if
active-source TRAIN40 RAE orMAE worsens versus current adapted model (22.19564%,
1.17704749s) on identical rows. These source diagnostics are not held-out;
Test12 has deployment features but no local target labels. No claim of source
test accuracy from mere prediction drift. No benchmark target selects v2.

## v2 execution result and next decision

- All48v2fold/seedfits completed in176.2s. SG road macroRAE17.6366% improves
  slightly versusIQ17.7489%, but macroMAE.5627s worsens versus.5434s. SG
  Slider50022.4804%/.4239s; SGA road21.7806%/.6720s andSlider20.9875%/.3842s.
  Both rejected by the preregistered gate; no finalfit or benchmark scoring.
- Fresh review confirms v2 median/fallback and supervised-gradient fixes; tests
  cover those failure modes. Type-only predict helper generalized to nn.Module
  after execution; archived executed sources preserve exact training provenance.
- Active eAPsource audit currentmodel22.1956387%/1.17704749s versus
  F027.3657379%/1.41602981s on88744TRAIN40 rows. Original Test12deployment
  features recovered for6762logicalqueries; no targets or accuracyclaim.
- Evaluator source mapping corrected low/medium fromofficialdownloadlinks with
  unchanged RGB-proven offsets. Restores11targets; all434publictargets available,
  nineof10originalper-sequencescores reproducePDF. Slider750stillunresolved.
- Next hypothesis supplies NEW event-motion information at startup. See
  2026-10-31-phase-causal-event-expansion.md. Do not extend rejectedrouter sweeps
  solely to find a favorable exposedbenchmark score.
