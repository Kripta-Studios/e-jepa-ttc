"""Verify portable inference examples using no parent checkpoint or external cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from operational.range_transfer.common import read, sha, write
from operational.representation_audit.export_weights import load_export


def verify(folder: Path) -> dict:
    def reject_external_cache(event: str, args: tuple) -> None:
        if event == "open" and isinstance(args[0], (str, bytes)):
            name = str(args[0]).replace("\\", "/").lower()
            if (
                "/artifacts/rgb_port_20261008/" in name
                or "/artifacts/range_readout_20261012/" in name
            ):
                raise PermissionError("Portable replay attempted an external cache/checkpoint")

    sys.addaudithook(reject_external_cache)
    torch.set_num_threads(1)
    result = {}
    for directory in sorted(folder.iterdir()):
        if not directory.is_dir():
            continue
        receipt = read(directory / "RECEIPT.json")
        for filename, key in (
            ("WEIGHTS.pt", "weights_sha256"),
            ("EXAMPLE_INPUT.npz", "example_input_sha256"),
            ("EXAMPLE_OUTPUT.npz", "example_output_sha256"),
        ):
            if sha(directory / filename) != receipt[key]:
                raise ValueError("Portable artifact changed")
        metadata = read(directory / "MODEL_METADATA.json")
        model = load_export(directory / "WEIGHTS.pt")
        with np.load(directory / "EXAMPLE_INPUT.npz", allow_pickle=False) as arc:
            inputs = {key: torch.from_numpy(arc[key].copy()) for key in arc.files}
        if any(key.startswith(("target_", "dino_")) for key in inputs):
            raise ValueError("A target leaked into inference example")
        with torch.inference_mode():
            actual = (
                model(*(inputs[key] for key in metadata["input_order"]))
                if metadata["input_style"] == "positional"
                else model(inputs)
            )
        with np.load(directory / "EXAMPLE_OUTPUT.npz", allow_pickle=False) as arc:
            checks = {
                key: np.array_equal(
                    (actual[key] if isinstance(actual, dict) else getattr(actual, key)).numpy(),
                    arc[key],
                )
                for key in arc.files
            }
        if not all(checks.values()):
            raise AssertionError(f"Portable replay failed: {directory.name}")
        result[directory.name] = {"exact": True, "outputs": checks}
    write(
        folder.parent / "PORTABLE_EXPORT_VERIFICATION.json",
        {
            "status": "PASS",
            "models": result,
            "external_checkpoints_or_caches_opened": False,
            "audit_hook_blocks_original_V13_V15_artifact_roots": True,
            "scope": "CPU prepared-input inference; no optimizer updates",
        },
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    print(f"{len(verify(parser.parse_args().folder))} portable models verified")
