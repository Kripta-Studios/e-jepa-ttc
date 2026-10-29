"""Translate bound Windows locations without modifying their original bytes."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Protocol


class PathTranslator(Protocol):
    def translate(self, value: str | os.PathLike[str]) -> str: ...


class PathMap:
    """A longest-prefix, fail-closed mapping into a dedicated bundle directory."""

    def __init__(self, roots: dict[str, str], destination: str) -> None:
        self.destination = PurePosixPath(destination)
        if not self.destination.is_absolute():
            raise ValueError("Bundle destination must be absolute")
        self.roots = sorted(roots.items(), key=lambda item: len(item[0]), reverse=True)
        for original, relative in self.roots:
            if not PureWindowsPath(original).is_absolute():
                raise ValueError("Source roots must be absolute Windows paths")
            if PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
                raise ValueError("Destination roots must stay inside the bundle")

    def relative(self, value: str | os.PathLike[str]) -> str:
        """Resolve a source filename to its safe, portable bundle member name."""
        candidate = PureWindowsPath(os.fspath(value))
        if ".." in candidate.parts:
            raise ValueError("Unresolved traversal in source path")
        for original, relative in self.roots:
            try:
                suffix = candidate.relative_to(PureWindowsPath(original))
            except ValueError:
                continue
            return str(PurePosixPath(relative, *suffix.parts))
        raise ValueError(f"Unregistered absolute Windows path: {value}")

    def translate(self, value: str | os.PathLike[str]) -> str:
        """Map absolute Windows paths; preserve existing POSIX and relative paths."""
        raw = os.fspath(value)
        candidate = PureWindowsPath(raw)
        if candidate.drive:
            return str(self.destination / self.relative(raw))
        return raw.replace("\\", "/")


def mapped_path_class(mapping: PathTranslator) -> type[Path]:
    """Create a native Path subclass for explicitly scoped project module globals."""
    native = type(Path())

    class MappedPath(native):
        def __new__(cls, *args: str | os.PathLike[str]) -> Path:
            return super().__new__(cls, *(mapping.translate(arg) for arg in args))

    return MappedPath
