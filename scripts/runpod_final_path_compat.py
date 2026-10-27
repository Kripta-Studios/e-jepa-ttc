"""Compose verified final-input trees without changing historical paths or bytes."""

from __future__ import annotations

import os
from abc import ABCMeta
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _normalized(path: Path) -> Path:
    """Normalize relative joins lexically, before requiring missing parents to exist."""
    return Path(os.path.normpath(os.fspath(path)))


def route(path: Path, roots: Sequence[Path]) -> Path:
    """Select the first existing copy of one logical file in registered input roots.

    Content verification remains the responsibility of the unchanged scientific
    consumers. Root order is explicit and stable; paths outside these trees are
    untouched, including writable final outputs and historical event runs.
    """
    normalized = _normalized(path)
    relative = None
    for root in roots:
        try:
            relative = normalized.relative_to(root)
            break
        except ValueError:
            continue
    if relative is None:
        # A joined relative path must not escape a registered immutable input root.
        if ".." in path.parts and any(path.is_relative_to(root) for root in roots):
            raise ValueError("Final input traversal escaped its registered root")
        return normalized
    for root in roots:
        candidate = root / relative
        if candidate.is_symlink():
            raise ValueError(f"Final input routing encountered a symlink: {candidate}")
        if candidate.exists():
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(root.resolve(strict=True)):
                raise ValueError("Final input routing escaped its registered root")
            # Preserve file-first lookup when a directory exists in several trees.
            # Its future child joins will run through this same routing function.
            return candidate
    return roots[0] / relative


def mapped_path_class(mapping: Any, roots: Sequence[Path]) -> type[Path]:
    """Build a native Path-compatible subclass for constructor, join and resolve."""
    anchors = tuple(_normalized(Path(root)) for root in roots)
    if not anchors or any(not root.is_absolute() for root in anchors):
        raise ValueError("Final input roots must be nonempty absolute paths")
    if len(set(anchors)) != len(anchors):
        raise ValueError("Final input roots must be distinct")
    native = type(Path())

    class NativePathMeta(ABCMeta):
        def __instancecheck__(cls, instance: object) -> bool:
            return isinstance(instance, Path)

    class FinalInputPath(native, metaclass=NativePathMeta):  # type: ignore[misc, valid-type]
        def __new__(cls, *args: str | os.PathLike[str]) -> Path:
            candidate = Path(*(mapping.translate(arg) for arg in args))
            return super().__new__(cls, route(candidate, anchors))

        def __truediv__(self, key: str | os.PathLike[str]) -> Path:
            candidate = Path(os.fspath(self)) / mapping.translate(key)
            return type(self)(route(candidate, anchors))

        def joinpath(self, *segments: str | os.PathLike[str]) -> Path:
            result: Path = self
            for segment in segments:
                result = result / segment
            return result

        def resolve(self, strict: bool = False) -> Path:
            candidate = route(Path(os.fspath(self)), anchors)
            # Construct using the superclass to avoid recursively resolving roots.
            resolved = candidate.resolve(strict=strict)
            return super().__new__(type(self), resolved)

    return FinalInputPath


def install(roots: Sequence[Path]) -> None:
    """Install before migration.runtime.install imports scientific consumers."""
    from operational.runpod_migration import paths, runtime

    def factory(mapping: Any) -> type[Path]:
        return mapped_path_class(mapping, roots)

    factory._final_input_composition_v1 = True  # type: ignore[attr-defined]
    paths.mapped_path_class = factory
    runtime.mapped_path_class = factory


__all__ = ["install", "mapped_path_class", "route"]
