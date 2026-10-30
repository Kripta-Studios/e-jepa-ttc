# PHASE-ADAPT final refit implementation plan

> Execute inline with superpowers:executing-plans; user explicitly requests continuous execution through a trained final model.

**Goal:** Train and deliver a verified final PHASE-ADAPT F1 model from all compatible local training data.
**Spec:** User continuation on 2026-10-30 plus package chapter 07. The user now authorizes actual refitting despite unresolved independent-evaluation gates.
**Architecture:** Freeze eight upstream models and normalizers; initialize the 1,955-parameter PhaseOnlyFinal head from the sealed candidate. Reuse verified TRAIN40 feature cache and extract all eligible EvTTC-32 annotated sensor anchors with the existing causal source adapter. Local labels train a declared exposed model; no independent-test or official-score claim.
**Stack:** Existing Python 3.11, PyTorch CUDA, NumPy, HDF5, TorchScript.

## Constraints and choices made before fitting

- Preserve sealed candidate, historical receipts, original audit policy and exclusions.
- A separate explicit `all_compatible_local_refit` role authorizes training, not clean transfer certification.
- Use every supported annotated anchor across all 32 source recordings, not the historical 16-anchor subset. Competition extracts are not added again; Slider has no verified compatible supervised adapter/target contract.
- Read CSV timestamp column 1 and TTC column 4; bounded linear interpolation, maximum gap 20 ms, no extrapolation. Train only within the head's signed phase domain, with all excluded rows counted.
- TRAIN40 cache retains its original targets and lineage, verified against archived receipt. Public final-test caches remain excluded.
- F1 retains original architecture, normalizers and fusion loss. Equal domain mass, then equal recording mass within each domain; recording balance is not a claim of independent acquisitions.
- AdamW LR 1e-4, weight decay 1e-4, batch128, clip1.0; complete epochs up to30 and at most2000 updates. Seeds7,13,21. Final artifact is seed7 by declaration, no best-seed or test selection. Last checkpoint, no early stopping on exposed metrics.
- Report baseline and trained resubstitution metrics, coverage and loss curves. Do not present training-fit changes as generalization.
- Export the head to TorchScript; ship upstream components and a strict runtime loader; validate saved/reloaded predictions and raw-sensor inference.

## Task 1: all-data population and immutable feature cache

- Write tests for target boundaries, duplicate identity, scope, domain/group weighting, and complete-anchor enumeration; observe failure.
- Add optional all-anchor selection to existing builder with unchanged default.
- Implement separate refit population/cache CLI; bind sources, input data identities, original candidate, source-cache receipts. Resume per-sequence shards only with matching binding.
- Enumerate, extract and verify every eligible row; preserve reason-coded missingness. Commit after focused tests.

## Task 2: train, serialize, export and replay

- Write tests for real parameter updates, deterministic fixed-budget training, immutable upstream weights, save/load parity and tamper rejection.
- Train fixed recipe on all accepted rows for seeds7/13/21; save optimizer/history/provenance/checkpoints.
- Package seed7 plus upstream weights/normalizers, model card, reproduction commands and metrics. Verify CPU export and raw input runtime.
- Commit code after tests; review once with independent reviewer, fix material findings with regression tests.

## Review focus

- Cache resume cannot mix candidate/source/target identities.
- Label fields cannot enter upstream inference; omitted rows remain visible.
- Every accepted row is visited each complete epoch; domain/recording mass has intended totals.
- Final model selection is fixed before scores; no checkpoint silently substituted.
- Deployment package verifies exact parent graph and new head, preserving original seal separately.

## Execution ledger

- Start: branch research/evttc-transfer-v2 at65edf4ee; existing linked worktree. User opts out of stopping for research gates. These remain limits on claims, not on an explicitly exposed final refit.
- Pre-flight interfaces: source sensor adapter ->340D head inputs; archived TRAIN40 cache ->same schema. Both require original parent graph. New targets are joined only after sensor feature extraction.
