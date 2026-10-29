"""Create a prospective detector development split from public TRAIN metadata.

Reads only Parquet metadata columns.  It neither loads pixels/checkpoints nor
opens test labels.  Geometry and GPU gates must independently pass before G2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.detection.protocol import build_sequence_split  # noqa: E402


def main() -> int:
    """Write an explicit metadata-only partition, failing on non-train rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-table", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-count", type=int, default=24)
    parser.add_argument("--selection-count", type=int, default=8)
    parser.add_argument("--calibration-count", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a new version, do not overwrite a frozen split")
    try:
        import pyarrow.parquet as pq
    except ImportError:
        parser.error("Install the isolated G0 audit environment with pyarrow")
    try:
        table = pq.read_table(args.media_table, columns=["sequence_id", "split"])
        rows = table.to_pylist()
        if not rows or {row["split"] for row in rows} != {"train"}:
            raise ValueError("Every manifest row must explicitly belong to public train")
        if any(not isinstance(row["sequence_id"], str) or not row["sequence_id"] for row in rows):
            raise ValueError("Invalid sequence identifier in manifest")
        split = build_sequence_split(
            sorted({row["sequence_id"] for row in rows}),
            source_split="train",
            dataset_revision=args.revision,
            media_sha256=hashlib.sha256(args.media_table.read_bytes()).hexdigest(),
            seed=args.seed,
            train_count=args.train_count,
            selection_count=args.selection_count,
            calibration_count=args.calibration_count,
        )
        output = split.to_dict()
        output["source_frame_count"] = len(rows)
        output["code_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        output["implementation_sha256"] = {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in [
                "scripts/prepare_detection_protocol.py",
                "src/e_jepa_ttc/detection/protocol.py",
            ]
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": split.status, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
