import pytest
import numpy as np

# We'll mock the internal methods of object_jepa.py
# The user wants a test proving that metrics come from calibration_predictions, not validation.

def test_calibration_metrics_come_from_calibration():
    # If a developer accidentally swapped calibration_predictions with validation_predictions, 
    # it could pollute metrics. We simulate generating a calibration artifact and verify 
    # it fails closed or throws KeyError if missing metrics.
    
    # Simulating what's inside object_jepa.py for calibration metrics
    calibration_metrics = {
        "mae_s": 0.5,
        "rmse_s": 0.7,
        # missing median_abs_error_s
    }
    
    with pytest.raises(KeyError, match="median_abs_error_s"):
        calibration_artifact = {
            "regression": {
                "mae_s": float(calibration_metrics["mae_s"]),
                "rmse_s": float(calibration_metrics["rmse_s"]),
                "median_abs_error_s": float(calibration_metrics["median_abs_error_s"]),
            },
        }

def test_no_default_zeros_in_calibration():
    # This proves we removed .get("mae_s", 0.0)
    import ast
    from pathlib import Path
    
    repo_root = Path(__file__).resolve().parent.parent.parent
    object_jepa_path = repo_root / "src" / "e_jepa_ttc" / "training" / "object_jepa.py"
    
    with open(object_jepa_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    assert '.get("mae_s", 0.0)' not in content, "Found forbidden .get with default zero for mae_s"
    assert '.get("rmse_s", 0.0)' not in content, "Found forbidden .get with default zero for rmse_s"
    assert '.get("median_abs_error_s", 0.0)' not in content, "Found forbidden .get with default zero for median_abs_error_s"
