"""Verify, extract once, and replay a V17 bundle using only its own source and artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from verify_v17_bundle import verify


def main(archive: Path) -> None:
    archive = archive.resolve()
    verified = verify(archive)
    destination = archive.parent / ("V17_VERIFY_" + verified["sha256"][:12])
    destination.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)  # verify() already rejects unsafe or duplicate paths.
    source = destination / "source"
    campaign = destination / "campaign"
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=os.pathsep.join((str(source / "src"), str(source))),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    commands = [
        [sys.executable, "scripts/verify_v17_exports.py", str(campaign / "exports")],
        [sys.executable, "scripts/replay_v17_bundle_metrics.py", str(campaign)],
    ]
    for index, command in enumerate(commands):
        with (destination / f"REPLAY_{index}.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                command,
                cwd=source,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    exports = json.loads((campaign / "PORTABLE_EXPORT_VERIFICATION.json").read_text())
    metrics = json.loads((campaign / "BUNDLE_METRIC_REPLAY.json").read_text())
    result = {
        "status": "PASS",
        "utc": datetime.now(UTC).isoformat(),
        "bundle_sha256": verified["sha256"],
        "extracted_to": str(destination),
        "source_used": str(source),
        "commands": commands,
        "models_replayed": len(exports["models"]),
        "prediction_rows_including_reused_controls": metrics["rows_including_reused_controls"],
        "external_parent_files_opened": exports["external_checkpoints_or_caches_opened"],
        "metric_replay_performs_new_inference": metrics["new_inference"],
        "scope": "independent extraction on the same host/runtime; not a clean-machine install",
    }
    archive.with_suffix(archive.suffix + ".independent.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path)
    main(parser.parse_args().zip)
