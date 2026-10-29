"""Verify the handoff's exact file manifest; no code execution or downloads."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

def verify(root: Path) -> dict:
    manifest=root/'SHA256SUMS.txt'
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    checked=0; seen=set()
    for line in manifest.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        expected, relative = line.split('  ',1)
        path=PurePosixPath(relative)
        if path.is_absolute() or '..' in path.parts or '\\' in relative or relative in seen:
            raise ValueError('Unsafe/duplicate manifest path: '+relative)
        if len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):
            raise ValueError('Invalid sha256: '+relative)
        seen.add(relative); local=root.joinpath(*path.parts)
        if not local.is_file() or local.is_symlink():
            raise ValueError('Absent/nonregular payload: '+relative)
        actual=hashlib.sha256(local.read_bytes()).hexdigest()
        if actual!=expected: raise ValueError('Hash mismatch: '+relative)
        checked+=1
    return {'status':'VERIFIED_EXACT_PAYLOAD','payload_files':checked,'root':str(root)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=p.parse_args();print(json.dumps(verify(args.root.resolve()),indent=2))
