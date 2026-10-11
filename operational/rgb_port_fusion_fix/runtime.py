"""Apply the admitted fusion amendment to training and evaluation entrypoints."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from unittest.mock import patch

from e_jepa_ttc.rgb_port.fusion import DualClockFusion
from operational.rgb_port.accounting import atomic_write_json, read_json_shared, sha256_file

from .contracts import NAME, validate
from .fusion import deterministic_forward

TARGETS = {
    "operational.rgb_port.train_heads",
    "operational.rgb_port.predict_heads",
    "operational.rgb_port.transfer",
    "operational.rgb_port.profile",
    "operational.rgb_port.cost_routes",
}


def main() -> None:
    """Validate code and record the runtime binding before executing the original CLI."""
    run = Path(sys.argv[1]).resolve(strict=True)
    target = sys.argv[2]
    args = sys.argv[3:]
    if target not in TARGETS:
        raise ValueError("Unsupported fusion runtime target")
    validate(run)
    if target.endswith("train_heads"):
        fit_id = args[args.index("--fit-id") + 1]
        if fit_id not in {"F_TRUE", "F_ZERO"}:
            raise ValueError("Fusion amendment only admits the two fusion fits")
        receipt = {
            "schema": "rgb_port_fusion_runtime_v1",
            "fit_id": fit_id,
            "freeze_sha256": sha256_file(run / NAME),
            "head_admission_sha256": sha256_file(run / "admissions" / f"{fit_id}.json"),
            "deterministic_algorithms": True,
            "parameterization_changed": False,
        }
        path = run / "fits" / fit_id / "FUSION_RUNTIME.json"
        if path.exists() and read_json_shared(path) != receipt:
            raise ValueError("Fusion runtime binding changed")
        atomic_write_json(path, receipt)
    sys.argv = [target, *args]
    with patch.object(DualClockFusion, "forward", deterministic_forward):
        runpy.run_module(target, run_name="__main__")


if __name__ == "__main__":
    main()
