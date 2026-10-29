"""Replay a portable fusion head using NumPy/PyTorch only, without project model imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def replay(directory: Path) -> dict:
    """Check packaged hashes, outputs and exact missing-RGB fallback in a fresh process."""
    torch.set_num_threads(1)
    receipt = json.loads((directory / "COMPLETE.json").read_text(encoding="utf-8"))
    if receipt["status"] != "PASS":
        raise ValueError("Export is incomplete")
    for name, expected in receipt["files"].items():
        path = (directory / name).resolve()
        if path.parent != directory.resolve() or checksum(path) != expected:
            raise ValueError("Portable export file identity changed")
    binding = json.loads((directory / "BINDING.json").read_text(encoding="utf-8"))
    outputs = json.loads((directory / "example_outputs.json").read_text(encoding="utf-8"))[
        "examples"
    ]
    with (directory / "model.torchscript.pt").open("rb") as stream:
        model = torch.jit.load(stream, map_location="cpu").eval()
    errors, coverage = [], set()
    with (
        np.load(directory / "example_inputs.npz", allow_pickle=False) as archive,
        torch.inference_mode(),
    ):
        for index, expected in enumerate(outputs):
            inputs = [
                torch.from_numpy(archive[f"example_{index:03d}_{name}"])
                for name in binding["input_names"]
            ]
            actual = model(*inputs)
            if actual.keys() != expected.keys():
                raise ValueError("Output schema changed")
            for name, value in actual.items():
                reference = torch.tensor(expected[name], dtype=value.dtype)
                torch.testing.assert_close(
                    value, reference, rtol=binding["rtol"], atol=binding["atol"]
                )
            available = inputs[-1]
            coverage.add("all" if available.all() else "none" if not available.any() else "mixed")
            if not torch.equal(actual["ttc"][~available], inputs[3][~available]):
                raise ValueError("Portable event fallback is not bit exact")
            errors.append(float((actual["ttc"] - torch.tensor(expected["ttc"])).abs().max()))
    if coverage != {"all", "none", "mixed"}:
        raise ValueError("Missing dynamic availability replay case")
    project_imports = [
        name for name in sys.modules if name.startswith(("e_jepa_ttc", "operational."))
    ]
    if project_imports or torch.cuda.is_initialized():
        raise ValueError("Replay must be CPU and independent of project Python models")
    result = {
        "status": "PASS",
        "examples": len(errors),
        "maximum_ttc_absolute_error": max(errors),
        "fallback_bit_exact": True,
        "project_python_modules_loaded": project_imports,
        "cuda_initialized": False,
        "optimizer_updates": 0,
        "complete_receipt_sha256": checksum(directory / "COMPLETE.json"),
        "replay_source_sha256": checksum(Path(__file__)),
        "scope": "Portable head and embedded normalizers; not sensor-to-TTC pipeline",
    }
    pending = directory / "REPLAY.pending.json"
    pending.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pending.replace(directory / "REPLAY.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(replay(parser.parse_args().directory))
