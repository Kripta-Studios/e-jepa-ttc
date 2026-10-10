#!/usr/bin/env python3
"""Verify and optionally extract the exact attached SOTA evidence ZIP."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

EXPECTED = 'b24b1f68be8e2846805d2b7f3f7cab2bd385864faac7cc662a1ade47940f3917'


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zip', type=Path, default=Path(__file__).resolve().parents[1] / 'evidence/SOTA_ESSENTIAL.zip')
    p.add_argument('--extract', type=Path)
    a = p.parse_args()
    data = a.zip.read_bytes()
    if hashlib.sha256(data).hexdigest() != EXPECTED:
        raise ValueError('Input ZIP does not match the reviewed attachment.')
    count = 0
    with zipfile.ZipFile(a.zip) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP members.')
        for info in archive.infolist():
            rel = PurePosixPath(info.filename)
            if rel.is_absolute() or '..' in rel.parts or '\\' in info.filename:
                raise ValueError('Unsafe ZIP member.')
            if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError('Symlinks are not allowed.')
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f'CRC failure: {bad}')
        for line in archive.read('SHA256SUMS.txt').decode('utf-8-sig').splitlines():
            if not line.strip():
                continue
            sha, name = line.split('  ', 1)
            name = name.lstrip('*')
            if hashlib.sha256(archive.read(name)).hexdigest() != sha:
                raise ValueError(f'Inner manifest mismatch: {name}')
            count += 1
        if a.extract:
            dest = a.extract.resolve()
            if dest.exists():
                raise ValueError('Extraction destination exists; do not overwrite evidence.')
            dest.mkdir(parents=True)
            archive.extractall(dest)
    print(json.dumps({'sha256': EXPECTED, 'zip_bytes': len(data),
                      'members': len(names), 'manifest_entries': count,
                      'crc': 'PASSED', 'extracted': str(a.extract) if a.extract else None}, indent=2))


if __name__ == '__main__':
    main()
