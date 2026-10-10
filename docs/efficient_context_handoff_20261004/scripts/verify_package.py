"""Verify this handoff's SHA256SUMS.txt without changing any files."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path,PurePosixPath

def safe_name(name: str)->None:
    p=PurePosixPath(name)
    if not name or '\\' in name or ':' in name or p.is_absolute() or '..' in p.parts:
        raise ValueError(f'Unsafe relative name: {name!r}')

def digest(path: Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def verify(root: Path)->dict:
    root=root.resolve(strict=True);manifest=root/'SHA256SUMS.txt'
    seen=set()
    for line in manifest.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():continue
        expected,name=line.split('  ',1);safe_name(name)
        if name in seen or len(expected)!=64:raise ValueError('Invalid or duplicate manifest entry')
        seen.add(name);p=root/name
        if p.is_symlink() or not p.is_file():raise ValueError(f'Missing or linked payload: {name}')
        if digest(p)!=expected.lower():raise ValueError(f'Hash mismatch: {name}')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='SHA256SUMS.txt'}
    if actual!=seen:raise ValueError(f'Unexpected/missing payload: {actual^seen}')
    return dict(status='VERIFIED',files=len(seen),root=str(root))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=parser.parse_args()
    print(json.dumps(verify(a.root),indent=2))
