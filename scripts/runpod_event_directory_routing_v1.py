"""Add per-child routing for admitted EVENT supplement directories."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path, PurePath
from typing import Any


def _layout(mapping: Any) -> tuple[Path, tuple[Path, ...]] | None:
    """Extract the verified primary/supplement layout through queue wrappers."""
    seen: set[int] = set()
    current = mapping
    while id(current) not in seen:
        seen.add(id(current))
        primary = getattr(current, "primary", None)
        roots: Sequence[Path] = getattr(current, "roots", ())
        if primary is not None and hasattr(primary, "destination"):
            destination = Path(os.fspath(primary.destination))
            admitted = tuple(Path(root) for root in roots)
            if not destination.is_absolute() or any(not root.is_absolute() for root in admitted):
                raise ValueError("EVENT routing roots must be absolute")
            if len({os.fspath(root) for root in admitted}) != len(admitted):
                raise ValueError("EVENT routing roots must be distinct")
            return destination, admitted
        current = getattr(current, "mapping", None)
        if current is None:
            return None
    raise ValueError("Cyclic EVENT directory mapping")


def _logical_relative(path: Path, anchors: tuple[Path, ...]) -> PurePath | None:
    for anchor in anchors:
        try:
            return path.relative_to(anchor)
        except ValueError:
            continue
    return None


def _route_child(path: Path, primary: Path, roots: tuple[Path, ...]) -> Path:
    """Resolve files uniquely; retain a virtual primary path for split directories."""
    anchors = (primary, *roots)
    relative = _logical_relative(path, anchors)
    if relative is None:
        return path
    if ".." in relative.parts:
        raise ValueError("EVENT child routing contains traversal")
    candidates = tuple(anchor.joinpath(*relative.parts) for anchor in anchors)
    if any(candidate.is_symlink() for candidate in candidates):
        raise ValueError(f"EVENT routed child is a symlink: {relative}")
    files = tuple(candidate for candidate in candidates if candidate.is_file())
    if len(files) > 1:
        raise ValueError(f"Ambiguous EVENT routed file: {relative}")
    if files:
        return files[0]
    # Directories can be physically split. Keep their primary logical location so
    # the next join is resolved across every admitted root instead of choosing one.
    return primary.joinpath(*relative.parts)


def compositional_path_factory(
    original: Callable[[Any], type[Path]], mapping: Any
) -> type[Path]:
    """Return the normal mapped Path type plus per-child supplement composition."""
    base = original(mapping)
    layout = _layout(mapping)
    if layout is None:
        return base
    primary, roots = layout

    class CompositionalEventPath(base):  # type: ignore[misc, valid-type]
        def __truediv__(self, key: str | os.PathLike[str]) -> Path:
            joined = super().__truediv__(key)
            routed = _route_child(Path(os.fspath(joined)), primary, roots)
            return type(self)(routed)

        def joinpath(self, *pathsegments: str | os.PathLike[str]) -> Path:
            result: Path = self
            for segment in pathsegments:
                result = result / segment
            return result

    CompositionalEventPath.__name__ = "CompositionalEventPath"
    return CompositionalEventPath


def install() -> None:
    """Patch only the runtime Path factory before V18 imports scientific modules."""
    from operational.runpod_migration import paths

    current = paths.mapped_path_class
    if getattr(current, "_event_directory_routing_v1", False):
        return

    def factory(mapping: Any) -> type[Path]:
        return compositional_path_factory(current, mapping)

    factory._event_directory_routing_v1 = True  # type: ignore[attr-defined]
    paths.mapped_path_class = factory

