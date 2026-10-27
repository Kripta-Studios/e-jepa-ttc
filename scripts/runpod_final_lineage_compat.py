"""Compare relocated fold-role references without weakening their content identities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


def role_comparator(mapping: Any, roots: Iterable[Path]) -> Callable[[object, object], bool]:
    """Accept only identical roles whose copied path references rehash identically."""
    allowed = tuple(Path(root).resolve() for root in roots)

    def normalize(value: object) -> object:
        if not isinstance(value, dict):
            return value
        result = dict(value)
        for field, digest_field in (
            ("manifest_path", "manifest_sha256"),
            ("rows_path", "rows_sha256"),
        ):
            if field not in result:
                continue
            path = Path(mapping.translate(result[field])).resolve(strict=True)
            relatives = [
                path.relative_to(root).as_posix() for root in allowed if path.is_relative_to(root)
            ]
            if not relatives:
                raise ValueError(f"Role input escaped declared migration roots: {path}")
            if len(set(relatives)) != 1:
                raise ValueError(f"Ambiguous role path across migration roots: {path}")
            expected = result.get(digest_field)
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError(f"Role path lacks a bound SHA-256: {field}")
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"Relocated role content differs: {path}")
            result[field] = {"bundle_relative_path": relatives[0], "sha256": actual}
        return result

    def equivalent(left: object, right: object) -> bool:
        return json.dumps(normalize(left), sort_keys=True) == json.dumps(
            normalize(right), sort_keys=True
        )

    return equivalent


def install(mapping: Any, roots: Iterable[Path]) -> None:
    """Install the path-only comparator in the already-loaded frozen validator."""
    from operational.final_refit import oof_lineage

    oof_lineage._json_equivalent = role_comparator(mapping, roots)
