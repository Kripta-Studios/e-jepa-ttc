"""Read-only Stage66 nested producer audit; never calls a historical fitter."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from run_scientific_recovery_v9_stage65 import _validate_nested_sources


def main() -> int:
    """Revalidate physical ancestors, preserving an actionable failure receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-inputs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    local = json.loads(args.local_inputs.read_text(encoding="utf-8-sig"))
    destination = args.output_root / "NESTED_ANCESTRY_AUDIT.json"
    if destination.exists():
        raise FileExistsError(destination)
    try:
        result = _validate_nested_sources(
            Path(local["reference_worktree"]) / "artifacts/scientific_recovery_v8/results/router",
            Path(local["stage61_worktree"])
            / "artifacts/scientific_recovery_v9_stage61_stage62/stage61",
            Path(local["stage63_worktree"])
            / "artifacts/stage63_65_remediation/FROZEN_TEACHER_PROVENANCE.json",
        )
    except Exception as error:
        result = {
            "status": "INTEGRITY_BLOCKED",
            "operation": "revalidate_exact_nested_producer_chain",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "scientific_fits": 0,
            "scientific_updates": 0,
        }
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps({"status": result["status"], "receipt": str(destination)}))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
