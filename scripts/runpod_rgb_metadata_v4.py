"""Preserve RGB history values while making duplicate metadata comparison scalar."""

from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def metadata_values(value: Any) -> Any:
    """Convert NumPy containers to equal-valued Python metadata, never model tensors."""
    if isinstance(value, np.ndarray):
        return metadata_values(value.tolist())
    if isinstance(value, np.generic):
        return metadata_values(value.item())
    if isinstance(value, Mapping):
        return {key: metadata_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [metadata_values(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite RGB history metadata is not admissible")
    return value


def install_history_adapter(module: Any) -> None:
    """Adapt only AllTrainRGBSource.history; row/targets/decode remain unchanged."""
    source = module.AllTrainRGBSource
    if getattr(source.history, "_ejepa_metadata_v4", False):
        raise ValueError("RGB history adapter was already installed")
    original = source.history

    def history(self: Any, index: int) -> Any:
        return metadata_values(original(self, index))

    history._ejepa_metadata_v4 = True  # type: ignore[attr-defined]
    source.history = history


def main() -> None:
    ops = Path(__file__).resolve().parent
    sys.path.insert(0, str(ops))
    import runpod_rgb_recovery_v3 as v3

    from operational.runpod_migration import runtime

    expected = "b42cd136eb08541d4d5676403258bc29684d03fb419e25fd0836625d67bead17"
    if v3.digest(Path(v3.__file__)) != expected:
        raise ValueError("Frozen RGB recovery V3 changed")
    if "operational.final_refit.rgb" in sys.modules:
        raise ValueError("RGB metadata adapter must precede scientific imports")
    run = Path(sys.argv[sys.argv.index("--run") + 1]).resolve(strict=True)
    binding = {
        "schema": "ejepa_rgb_history_metadata_adapter_v4",
        "status": "PASS",
        "source_sha256": v3.digest(Path(__file__)),
        "recovery_v3_sha256": expected,
        "migration_binding_sha256": v3.digest(run / "MIGRATION_BINDING.json"),
        "scope": "AllTrainRGBSource.history metadata containers only",
        "numpy_array_values_preserved": True,
        "model_tensors_modified": False,
        "targets_modified": False,
        "duplicate_conflict_check_retained": True,
        "scientific_sources_rewritten": False,
    }
    target = run / "RGB_HISTORY_METADATA_ADAPTER_V4.json"
    if target.exists() and json.loads(target.read_text()) != binding:
        raise ValueError("RGB metadata adapter binding changed")
    if not target.exists():
        temporary = target.with_name(f".{target.name}.{os.getpid()}.pending")
        with temporary.open("w") as stream:
            json.dump(binding, stream, sort_keys=True, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    original_exec = runtime.ProjectLoader.exec_module

    def exec_module(self: Any, module: Any) -> None:
        original_exec(self, module)
        if module.__name__ == "operational.final_refit.rgb":
            install_history_adapter(module)

    runtime.ProjectLoader.exec_module = exec_module
    try:
        v3.main()
    finally:
        runtime.ProjectLoader.exec_module = original_exec


if __name__ == "__main__":
    main()
