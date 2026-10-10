"""Package validated GarlTTC predictions in the exact CodaBench ZIP layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd

from operational.sota_evidence.submission import build_submission


def package_predictions(
    tokens: list[str], frame: pd.DataFrame, checkpoint: Path, output: Path
) -> dict:
    """Validate real prediction rows and create exactly one root submission.json.

    Input must be a prediction table, not a sample-submission template. Model
    provenance and validation receipts remain outside the upload ZIP.
    """
    if output.suffix.lower() != ".zip":
        raise ValueError("CodaBench output must be a ZIP")
    if output.exists():
        raise FileExistsError(output)
    with checkpoint.open("rb") as stream:
        checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    payload = build_submission(tokens, frame, checkpoint_sha256=checkpoint_sha256)
    data = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("submission.json", data)
    with zipfile.ZipFile(output) as archive:
        if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
            raise ValueError("invalid ZIP member structure or CRC")
        if archive.read("submission.json") != data:
            raise ValueError("serialized submission differs")
    with output.open("rb") as stream:
        zip_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "status": "PACKAGED_NOT_SUBMITTED",
        "sample_count": len(tokens),
        "checkpoint_sha256": checkpoint_sha256,
        "zip_sha256": zip_sha256,
        "members": ["submission.json"],
    }


def main() -> None:
    """Create a local upload ZIP from complete official-test predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = package_predictions(
        pd.read_parquet(args.inputs).sample_token.tolist(),
        pd.read_csv(args.predictions),
        args.checkpoint,
        args.output,
    )
    receipt_path = args.output.with_suffix(".receipt.json")
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
