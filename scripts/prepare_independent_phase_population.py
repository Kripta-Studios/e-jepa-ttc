"""Prepare the write-once label-free TRAIN24 independent refit population."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse population builder arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", type=Path, default=Path("data/splits/eap_detection_candidate_v1.json")
    )
    parser.add_argument(
        "--garl-data", type=Path, required=True
    )
    parser.add_argument(
        "--garl-annotations",
        type=Path,
        required=True,
    )
    parser.add_argument("--eap-data", type=Path, required=True)
    parser.add_argument(
        "--teacher-weights",
        type=Path,
        default=Path.home()
        / ".cache/huggingface/hub/models--facebook--dinov3-convnext-large-pretrain-lvd1689m"
        / "snapshots/e959efa74c867491dcfe3ec3e4f97382e39025b3/model.safetensors",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/independent_phase/train24_population_v2")
    )
    return parser.parse_args()


def main() -> int:
    """Build atomically, preserving any failed attempt and never overwriting."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from operational.independent_refit import build_independent_population

    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.building.{uuid.uuid4().hex}"
    try:
        result = build_independent_population(
            split_path=args.split,
            garl_data_path=args.garl_data,
            garl_annotations_path=args.garl_annotations,
            eap_data_path=args.eap_data,
            teacher_weights_path=args.teacher_weights,
            output_dir=staging,
        )
        os.replace(staging, output)
    except Exception as error:
        if staging.exists():
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            failed = output.parent / f"{output.name}.failed.{stamp}.{uuid.uuid4().hex[:8]}"
            shutil.move(str(staging), str(failed))
            (failed / "FAILED.json").write_text(
                json.dumps(
                    {
                        "status": "FAILED_PRESERVED",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"failed_attempt={failed}", file=sys.stderr)
        raise
    print(json.dumps({**result, "output_dir": str(output)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
