"""Replay included V18 predictions and portable weights from an independent extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .bundle import MANIFEST, safe_name
from .verify_predictions import recalculate


def file_hash(path: Path) -> str:
    """Stream file bytes without depending on a campaign's external source paths."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_extraction(root: Path) -> dict:
    """Verify the extracted bytes and reject files escaping the extraction root."""
    root = root.resolve()
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if manifest["schema"] != "v18_essential_bundle_v1":
        raise ValueError("Unknown bundle schema")
    expected = manifest["files"]
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(expected) | {MANIFEST}:
        raise ValueError("Extracted file set differs from the bundle manifest; run Python with -B")
    for name, receipt in expected.items():
        path = root / safe_name(name)
        if not path.resolve().is_relative_to(root):
            raise ValueError("Extracted file escapes the bundle root")
        if {"bytes": path.stat().st_size, "sha256": file_hash(path)} != receipt:
            raise ValueError(f"Extracted file differs: {name}")
    return {"status": "PASS", "payload_members": len(expected)}


def replay_predictions(folder: Path) -> dict:
    """Recalculate explicit scalar fields using NumPy, without the production scorer."""
    receipt = json.loads((folder / "COMPLETE.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(folder / "predictions.parquet")
    if (
        receipt["status"] != "COMPLETE"
        or file_hash(folder / "predictions.parquet") != receipt["sha256"]
        or len(frame) != receipt["rows"]
        or frame.sample_token.duplicated().any()
    ):
        raise ValueError("Incomplete or altered prediction population")
    original = json.loads((folder / "METRICS.json").read_text(encoding="utf-8"))
    expected = recalculate(frame)
    differences = {}
    for key, value in expected.items():
        found = original
        for part in key.split("."):
            found = found[part]
        if value is None or found is None:
            if value is not None or found is not None:
                raise ValueError(f"Defined metric status changed: {key}")
            differences[key] = None
        elif isinstance(value, (int, bool)):
            if value != found:
                raise ValueError(f"Metric count/flag changed: {key}")
            differences[key] = 0.0
        else:
            if not np.isclose(value, found, atol=1e-9, rtol=1e-10):
                raise ValueError(f"Metric scalar changed: {key}")
            differences[key] = abs(float(value) - float(found))
    return {"status": "PASS", "rows": len(frame), "absolute_differences": differences}


def replay_export(folder: Path) -> dict:
    """Load actual portable bytes on CPU and replay the included label-free example."""
    receipt = json.loads((folder / "COMPLETE.json").read_text(encoding="utf-8"))
    if receipt["status"] != "COMPLETE":
        raise ValueError("Portable export is incomplete")
    for name, digest in receipt["files"].items():
        if file_hash(folder / safe_name(name)) != digest:
            raise ValueError("Portable artifact differs from its receipt")
    binding = json.loads((folder / "BINDING.json").read_text(encoding="utf-8"))
    with (folder / "model.torchscript.pt").open("rb") as stream:
        model = torch.jit.load(stream, map_location="cpu").eval()
    with np.load(folder / "example_input.npz", allow_pickle=False) as archive:
        inputs = (torch.from_numpy(archive["events"]), torch.from_numpy(archive["delta_t_s"]))
    expected = json.loads((folder / "example_output.json").read_text(encoding="utf-8"))[
        "signed_ttc_seconds"
    ]
    with torch.inference_mode():
        output = model(*inputs).numpy()
    if not np.isfinite(output).all() or not np.isfinite(expected).all():
        raise ValueError("Portable example contains nonfinite predictions")
    np.testing.assert_allclose(
        output, expected, atol=binding["atol_seconds"], rtol=binding["rtol"], equal_nan=False
    )
    return {
        "status": "PASS",
        "maxabs_seconds": float(np.max(np.abs(output - np.asarray(expected)))),
        "input_example_count": len(output),
        "rtol": binding["rtol"],
        "atol_seconds": binding["atol_seconds"],
        "bit_exact": bool(np.array_equal(output.astype(np.float64), np.asarray(expected))),
    }


def replay(root: Path) -> dict:
    """Check included evidence only; never reopen training datasets or run an optimizer."""
    integrity = verify_extraction(root)
    torch.set_num_threads(1)
    campaign = root / "campaign"
    predictions = {
        path.parent.relative_to(campaign)
        .as_posix()
        .removeprefix("predictions/"): replay_predictions(path.parent)
        for path in sorted(campaign.glob("**/predictions/*/*/COMPLETE.json"))
    }
    exports = {
        path.parent.relative_to(campaign).as_posix().removeprefix("exports/"): replay_export(
            path.parent
        )
        for path in sorted(campaign.glob("**/exports/*/COMPLETE.json"))
    }
    return {
        "integrity": integrity,
        "prediction_tables": predictions,
        "portable_exports": exports,
        "rows_recalculated": sum(result["rows"] for result in predictions.values()),
        "scientific_replay_performed": bool(predictions or exports),
        "scope": (
            "Included prediction scalars and portable example outputs; not raw producer replay"
        ),
        "optimizer_updates": 0,
        "torch_version": str(torch.__version__),
        "SOTA_confirmation": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.root.resolve()):
        raise ValueError("Save independent replay output outside the original extraction")
    result = replay(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
