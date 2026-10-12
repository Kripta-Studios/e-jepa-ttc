"""Verify extracted handoff files against SHA256SUMS.txt; no code execution beyond hashing."""
from pathlib import Path
import hashlib,sys
root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
count=0
for line in (root/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
    expected,name=line.split('  ',1)
    path=(root/name).resolve()
    if not path.is_relative_to(root):raise ValueError('Unsafe manifest path')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError(f'Hash mismatch: {name}')
    count+=1
print(f'Verified {count} payload files')
