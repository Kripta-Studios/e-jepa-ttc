"""Run the existing OOF executor with an explicitly admitted metadata repair."""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from operational.final_refit.recipe import canonical_sha256, sha256
from operational.oof_rgb_metadata_fix import repaired_rgb_metadata
from operational.oof_runtime_binding import load_admission
from operational.rgb_port.accounting import atomic_write_json, read_json_shared


def main() -> int:
    """Verify additive repair evidence before entering the original executor."""
    run = Path(sys.argv[sys.argv.index("--run") + 1]).resolve()
    load_admission(run)
    path = run / "RGB_METADATA_REPAIR_ADMISSION.json"
    admission = read_json_shared(path)
    if admission.get("status") != "PASS" or canonical_sha256(
        {k: v for k, v in admission.items() if k != "payload_sha256"}
    ) != admission.get("payload_sha256"):
        raise ValueError("Unsealed RGB metadata repair admission")
    required = {
        str(Path(__file__).resolve()),
        str(Path(__file__).with_name("oof_rgb_metadata_fix.py").resolve()),
        str(Path(__file__).resolve().parents[1] / "scripts/run_rgb_fixed_oof_stage.ps1"),
    }
    if set(admission["source_sha256"]) != required:
        raise ValueError("Incomplete RGB metadata repair source bindings")
    for category in ("source_sha256", "evidence_sha256"):
        if not admission.get(category):
            raise ValueError("Missing RGB metadata repair evidence")
        for filename, expected in admission[category].items():
            if sha256(Path(filename)) != expected:
                raise ValueError(f"RGB metadata repair binding changed: {filename}")
    if sha256(run / "SCIENTIFIC_CONFIG.json") != admission["scientific_config_sha256"]:
        raise ValueError("RGB metadata repair scientific authority changed")
    atomic_write_json(
        run / f"RGB_METADATA_REPAIR_SESSION_{os.getpid()}.json",
        {
            "pid": os.getpid(),
            "started_utc": datetime.now(UTC).isoformat(),
            "admission_sha256": sha256(path),
            "arguments": sys.argv[1:],
            "scope": "exact RGB observation metadata equality only",
        },
    )
    from operational.scientific_oof_isolated import main as original_main

    with repaired_rgb_metadata():
        return original_main()


if __name__ == "__main__":
    raise SystemExit(main())
