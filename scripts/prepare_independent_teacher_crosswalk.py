"""Build token-exact TRAIN24 to frozen teacher crosswalk."""

from __future__ import annotations

import json
import os
import shutil
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path


def main() -> int:
    """Run the CPU-only write-once crosswalk builder."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from operational.independent_refit.teacher_crosswalk import (
        build_teacher_crosswalk,
        verify_teacher_crosswalk,
    )

    handoff = Path(__file__).resolve().parents[2]
    historical = handoff / "e-jepa-ttc-v12-efficient-context/artifacts/train40_system_20261005"
    fold = Path("output/independent_phase/train24_population_v2/FOLD_CONTRACT.json")
    output = Path("output/independent_phase/teacher_crosswalk_v3")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.building.{uuid.uuid4().hex}"
    try:
        receipt = build_teacher_crosswalk(fold, historical / "teacher", historical, staging)
        verify_teacher_crosswalk(staging / "RECEIPT.json", fold)
        os.replace(staging, output)
    except Exception as error:
        if staging.exists():
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            failed = output.parent / f"{output.name}.failed.{stamp}.{uuid.uuid4().hex[:8]}"
            shutil.move(str(staging), str(failed))
            (failed / "FAILED.json").write_text(
                json.dumps({"status": "FAILED_PRESERVED", "error": str(error)}, indent=2) + "\n",
                encoding="utf-8",
            )
        raise
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
