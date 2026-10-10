#!/usr/bin/env python3
"""Verify the handoff SHA256SUMS manifest without downloading or executing payloads."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path, PurePosixPath

def verify(root:Path)->dict:
    root=root.resolve();manifest=root/'SHA256SUMS.txt';seen=set()
    for line in manifest.read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1);rel=PurePosixPath(name)
        if len(digest)!=64 or rel.is_absolute() or '..' in rel.parts or '\\' in name or name in seen:
            raise ValueError('Unsafe or duplicate manifest entry')
        seen.add(name);p=root.joinpath(*rel.parts)
        if p.is_symlink() or not p.is_file() or root not in p.resolve().parents:raise ValueError(f'Invalid file {name}')
        if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError(f'Hash mismatch: {name}')
    return {'status':'VERIFIED','files':len(seen)}
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ns=a.parse_args();print(json.dumps(verify(ns.root)))
