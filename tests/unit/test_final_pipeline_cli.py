import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_evttc_final_pipeline.py"


def test_final_pipeline_help_does_not_require_editable_install() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "fit-holdout" in result.stdout
    assert "evaluate-holdout" in result.stdout
    assert "select-final" in result.stdout


def test_diagnostic_test_requires_matching_selection_manifest(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"frozen weights")
    common = [
        sys.executable,
        str(SCRIPT),
        "evaluate-holdout",
        "--checkpoint",
        str(checkpoint),
        "--cache-manifest",
        str(tmp_path / "cache-manifest.json"),
        "--output-dir",
        str(tmp_path / "metrics"),
        "--splits",
        "test",
        "--allow-diagnostic-test",
    ]
    missing = subprocess.run(
        common,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing.returncode != 0
    assert "--selection-manifest" in missing.stderr

    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "artifact_type": "evttc_final_profile_selection_v1",
                "diagnostic_test_used_for_selection": False,
                "benchmark10_opened": False,
                "selected_checkpoint_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    mismatch = subprocess.run(
        [*common, "--selection-manifest", str(selection)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert mismatch.returncode != 0
    assert "Checkpoint hash" in mismatch.stderr
