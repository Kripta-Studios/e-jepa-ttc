# A8.0 grouped dual-transport implementation plan

## 1. Grouped dataset views

- Add failing tests for one-pass sequence partitioning, exact token hashes, disjoint/exhaustive views, and shard-group preservation.
- Implement indexed read-only views in `src/e_jepa_ttc/data/scientific_recovery_v5.py`.
- Run the new unit module and Ruff.

## 2. Canonical runner grouped mode

- Add failing runner tests for signed protocol validation, fold/hash mismatch, train-only split construction, and teacher exclusion from dev.
- Extend `scripts/train_causal_scale_eap_screen.py` with a train-only grouped branch; leave the historical public-validation branch unchanged.
- Emit grouped-dev artifact/status names, `dev_predictions.csv`, fold provenance, and closed validation/test contracts.
- Run runner, selection, resume, teacher, and cache unit tests.

## 3. A8.0 frozen configs

- Add failing tests for causal dual-stream invariants, parent checkpoint SHA, fixed r/temperature/residual, seed semantics, row counts, and `num_workers=0`.
- Add an A8.0 freezer that derives three fold configs from the existing A7 causal dual-stream config plus the frozen grouped protocol.
- Freeze configs only from a clean committed implementation SHA, then commit their signed manifest.

## 4. Geometry and causality audits

- Add failing tests for optimizer exclusion, exact primary-encoder state hashes, fixed-probe output equality, and dual-stream prefix invariance through transport/residual/fusion outputs.
- Implement a signed geometry-freeze audit and extend the prefix audit without opening a checkpoint during training.
- Run audits only after each completed fold.

## 5. Fold aggregation and smoke gate

- Add manual synthetic fold-aggregation tests, including nonfinite/incomplete rejection.
- Implement a signed A8.0 grouped aggregator with first-stage/strong gates and explicit A6 comparator status.
- Run a tiny CPU grouped-view/model/loss smoke before CUDA.

## 6. Sequential experiments

- Check CUDA/VRAM and confirm no training process is active.
- Run fold 0, then 1, then 2 with seed 7 and `num_workers=0`; monitor logs/progress only.
- After each fold validate exit code, summary, predictions, checkpoint, effective hashes, finite metrics, causality, and geometry freeze.
- Aggregate and decide `PASS`, `FAIL`, or `INCONCLUSIVE` from the preregistered grouped-dev gates.
