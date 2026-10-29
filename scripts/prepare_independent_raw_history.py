"""Build independent TRAIN24 raw-history clock metadata."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path


def main() -> int:
    """Run the write-once CPU metadata builder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold-contract",
        type=Path,
        default=Path("output/independent_phase/train24_population_v2/FOLD_CONTRACT.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/independent_phase/raw_history_index_v4"),
    )
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from operational.independent_refit.raw_history import (
        build_raw_history_index,
        verify_raw_history_index,
    )

    output = args.output_dir.resolve()
    publication_root = (Path.cwd() / "output" / "independent_phase").resolve()
    if output.parent != publication_root:
        raise ValueError(f"Output must be a direct child of {publication_root}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    publication_root.mkdir(parents=True, exist_ok=True)
    staging = publication_root / f".{output.name}.building.{uuid.uuid4().hex}"
    try:
        build_raw_history_index(args.fold_contract, staging)
        result = verify_raw_history_index(staging / "RECEIPT.json")
        # Both resolved paths were checked inside publication_root before the atomic move.
        if (
            staging.parent.resolve() != publication_root
            or output.parent.resolve() != publication_root
        ):
            raise RuntimeError("Atomic publication paths escaped independent_phase")
        os.replace(staging, output)
    except Exception as error:
        if staging.exists():
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            failed = publication_root / f"{output.name}.failed.{stamp}.{uuid.uuid4().hex[:8]}"
            if failed.parent.resolve() != publication_root:
                raise RuntimeError("Failure preservation path escaped independent_phase") from error
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
