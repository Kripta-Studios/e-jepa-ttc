"""Run reference tests against pinned native phase/model in an isolated import tree.

Does not run repository-wide tests, import a trainer, use raw data, or update weights.
Requires NumPy and PyTorch already installed in the selected Python environment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];repo=args.repo.resolve()
    pins=json.loads((root/'evidence/SOURCE_PINS.json').read_text(encoding='utf-8'))
    native={e['path']:e['sha256'] for e in pins['files'] if e['snapshot']=='V13' and '/simplex_t/' in e['path']}
    with tempfile.TemporaryDirectory(prefix='range-transfer-contracts-') as tmp:
        pkg=Path(tmp)/'e_jepa_ttc';(pkg/'simplex_t').mkdir(parents=True)
        (pkg/'__init__.py').write_text('');(pkg/'simplex_t'/'__init__.py').write_text('')
        for relative,expected in native.items():
            source=repo/relative
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
                raise ValueError('Native module does not match audited pin: '+str(source))
            shutil.copy2(source,pkg/'simplex_t'/source.name)
        shutil.copytree(root/'src/e_jepa_ttc/range_transfer',pkg/'range_transfer',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        env=dict(os.environ);env['PYTHONPATH']=tmp;env['PYTHONDONTWRITEBYTECODE']='1';env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
        command=[sys.executable,'-B','-m','unittest','discover','-v','-s',str(root/'tests')]
        result=subprocess.run(command,env=env,cwd=tmp,check=False)
        raise SystemExit(result.returncode)
