#!/usr/bin/env python3
"""Verify the exact extracted handoff. No network, mutation, or optimizer."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(root: Path) -> dict:
    root = root.resolve(strict=True)
    manifest = root / 'SHA256SUMS.txt'
    listed: set[str] = set()
    for number, line in enumerate(manifest.read_text(encoding='utf-8').splitlines(), 1):
        if not line:
            continue
        parts = line.split('  ', 1)
        if len(parts) != 2:
            raise ValueError(f'Invalid manifest line {number}')
        sha, name = parts
        rel = PurePosixPath(name)
        if (len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha)
                or rel.is_absolute() or '..' in rel.parts or '\\' in name
                or name in listed or name == 'SHA256SUMS.txt'):
            raise ValueError(f'Invalid or duplicated manifest entry {number}')
        path = root.joinpath(*rel.parts)
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f'Unsafe or missing file: {name}')
        if digest(path) != sha:
            raise ValueError(f'SHA-256 mismatch: {name}')
        listed.add(name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != listed | {'SHA256SUMS.txt'}:
        raise ValueError(f'Inventory differs: {sorted(actual.symmetric_difference(listed | {"SHA256SUMS.txt"}))}')
    return {'status': 'VERIFIED', 'manifest_entries': len(listed), 'root': str(root)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    print(json.dumps(verify(args.root), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
