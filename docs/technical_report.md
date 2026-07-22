# E-JEPA-TTC Technical Report: Diagnostic Matrix Recovery

## Overview

This report documents the recovery and validation of the E-JEPA-TTC system. A systematic defect in the sparse voxel normalization algorithm (cache format v1) invalidated all prior baseline and JEPA runs. A revised normalization strategy (`cache_format_version = 2`) was successfully implemented, and a multi-seed matrix execution (Phases 5-9) was orchestrated to rebuild the scientific foundation.

## Experimental Protocol

- **Dataset**: `evttc_full_starter_sealed.yaml` (diagnostic CPLA-high test sequence)
- **Feature Schema**: `voxel_160x90_b5_raw_meta_nav_recovery_v2`
- **Pretraining**: 3 independent seeds (7, 13, 21) using `event-tubelet-transformer` with the `tubelet` mask strategy, trained for 30 epochs each.
- **Downstream Fine-tuning**: For each of the 3 pretraining seeds, downstream fine-tuning was performed using 3 independent downstream seeds (7, 13, 21) for 80 epochs each, resulting in 9 independent candidate checkpoints.
- **Selection Criterion**: Validation Mean Absolute Error (MAE).
- **Execution Platform**: NVIDIA GeForce RTX 5070 Ti Laptop GPU.

## Results on Diagnostic Split (CPLA-high)

All 9 checkpoints were evaluated on the fixed `CPLA-high` test split without re-fitting. The diagnostic aggregate metrics demonstrate the model's predictive capacity:

- **Mean Absolute Error (MAE)**: 
  - Mean: 0.532 s
  - Min: 0.403 s
  - Max: 0.682 s
  - Std: 0.080 s
- **Median Absolute Error**:
  - Mean: 0.295 s
  - Min: 0.222 s
- **RMSE**:
  - Mean: 0.877 s
  - Min: 0.694 s
- **Mean Absolute Relative Error**:
  - Mean: 11.95%
  - Min: 9.30%

The most performant individual seed pair was Pretrain Seed `13` with Downstream Seed `7`, achieving an MAE of `0.403 s` and an RMSE of `0.694 s`.

## Validation and Artifact Audit

- Cache loader now rigidly enforces `cache_format_version >= 2` to prevent silent normalization degradation.
- All pre-fix `recovery_v1` artifacts in the registry have been formally marked as `invalid_pre_fix`.
- All post-fix artifacts have been registered successfully as `official_candidate` with `claim_level: development` pending the arrival of a strictly uninspected final test sequence.

## ONNX Validation and Export Status

The repository infrastructure provides an ONNX exporter for `ObjectCentricEventJEPA` (`export-object-ttc`). However, the recovery matrix execution focused on the standalone `EventTubeletTransformer` backbone fine-tuning via `tiny-cnn`. ONNX export and runtime verification for the fine-tuned downstream branches remain as future work to complete Phase 13 and Phase 14 once the exact production architecture is frozen.
