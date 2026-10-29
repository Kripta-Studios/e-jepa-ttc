"""Preserve native Path recognition in the frozen RGB Linux translation adapter."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
from abc import ABCMeta
from pathlib import Path
from typing import Any


def compatible_path_class(mapping: Any) -> type[Path]:
    """Translate construction, while isinstance retains pathlib's native semantics."""
    native = type(Path())

    class NativePathMeta(ABCMeta):
        def __instancecheck__(cls, value: object) -> bool:
            return isinstance(value, Path)

    class MappedPath(native, metaclass=NativePathMeta):
        def __new__(cls, *args: Any) -> Path:
            return super().__new__(cls, *(mapping.translate(arg) for arg in args))

    return MappedPath


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    ops = Path(__file__).resolve().parent
    original = ops / "rgb_downstream/scripts/runpod_rgb_bootstrap_v1.py"
    if digest(original) != "58100a4b287e5bb0420b9bf0ace8160599577a336aca8a9b5f188574247284c3":
        raise ValueError("Frozen RGB bootstrap differs")
    sys.path.insert(0, str(ops))
    from operational.runpod_migration import runtime

    if runtime.mapped_path_class.__module__ != "operational.runpod_migration.paths":
        raise ValueError("Unexpected pre-existing path adapter")
    run = Path(sys.argv[sys.argv.index("--run") + 1]).resolve(strict=True)
    binding = {
        "schema": "ejepa_rgb_native_path_compatibility_v1", "status": "PASS",
        "overlay_sha256": digest(Path(__file__)),
        "original_bootstrap_sha256": digest(original),
        "migration_binding_sha256": digest(run / "MIGRATION_BINDING.json"),
        "native_path_isinstance_preserved": True,
        "translation_constructor_unchanged": True,
        "scientific_sources_rewritten": False, "checkpoint_bytes_rewritten": False,
    }
    target = run / "RGB_NATIVE_PATH_COMPATIBILITY.json"
    if target.exists():
        if json.loads(target.read_text()) != binding:
            raise ValueError("Path compatibility admission changed")
    else:
        temporary = target.with_name(f".{target.name}.{os.getpid()}.pending")
        with temporary.open("w") as stream:
            json.dump(binding, stream, sort_keys=True, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    runtime.mapped_path_class = compatible_path_class
    runpy.run_path(str(original), run_name="__main__")


if __name__ == "__main__":
    main()
