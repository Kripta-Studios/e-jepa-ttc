"""Verify this handoff's manifest, using only the standard library."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path

def verify(root:Path)->dict:
    manifest=json.loads((root/'MANIFEST.json').read_text(encoding='utf-8'))
    checked=0
    for row in manifest['files']:
        p=(root/row['path']).resolve()
        if not p.is_relative_to(root.resolve()):raise ValueError('Unsafe path')
        if not p.is_file() or p.stat().st_size!=row['bytes']:raise ValueError('Missing/changed file: '+row['path'])
        if hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Hash mismatch: '+row['path'])
        checked+=1
    return {'status':'PASS','files':checked}
if __name__=='__main__':
    print(json.dumps(verify(Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1])))
