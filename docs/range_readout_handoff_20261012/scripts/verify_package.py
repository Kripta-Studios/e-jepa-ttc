"""Verify the new handoff without third-party packages or external access."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json


def verify(root: Path) -> dict:
    root=root.resolve();manifest=root/"SHA256SUMS.txt"
    checked=[]
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected,rel=line.split("  ",1)
        name=PurePosixPath(rel)
        if name.is_absolute() or ".." in name.parts or "\\" in rel:
            raise ValueError("Unsafe manifest path")
        file=(root/rel).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            raise ValueError(f"Missing/unsafe path: {rel}")
        with file.open("rb") as stream:
            actual=hashlib.file_digest(stream,"sha256").hexdigest()
        if actual!=expected:raise ValueError(f"Hash mismatch: {rel}")
        checked.append(rel)
    if len(checked)!=len(set(checked)):raise ValueError("Duplicate manifest paths")
    return {"status":"VERIFIED","manifest_entries":len(checked),"scope":"Handoff files only; not trained models"}


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args();print(json.dumps(verify(args.root),indent=2))
