#!/usr/bin/env python
"""Package scientific-recovery logs/evidence without copying giant caches."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MAX_COPY=120*1024*1024
KEEP_NAMES={'summary.json','validation_predictions.csv','validation_predictions.parquet','model_best.pt','manifest.json'}
KEEP_EXT={'.json','.yaml','.yml','.csv','.parquet','.txt','.log','.md','.patch','.ps1','.py'}

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def copy_file(src:Path,dst:Path)->bool:
 if not src.is_file() or src.stat().st_size>MAX_COPY:return False
 dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst); return True

def main()->int:
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--master-root',type=Path,required=True)
 p.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[1])
 p.add_argument('--run-dir',type=Path,action='append',default=[])
 p.add_argument('--output-zip',type=Path,required=True)
 a=p.parse_args(); root=a.master_root.resolve(); repo=a.repo_root.resolve(); zip_path=a.output_zip.resolve()
 stage=zip_path.with_suffix('')
 if stage.exists():shutil.rmtree(stage)
 stage.mkdir(parents=True)
 # Master logs/status/audits.
 for src in root.rglob('*'):
  if src.is_file() and src.stat().st_size<=MAX_COPY:
   copy_file(src,stage/'master'/src.relative_to(root))
 # Selected run evidence only; no state dirs/cache shards.
 for r in a.run_dir:
  rr=r.resolve()
  if not rr.is_dir():continue
  label=rr.name
  for src in rr.rglob('*'):
   if not src.is_file():continue
   if 'state' in src.relative_to(rr).parts:continue
   if src.name in KEEP_NAMES or src.suffix.lower() in {'.json','.yaml','.yml','.csv','.parquet','.txt','.log'}:
    copy_file(src,stage/'runs'/label/src.relative_to(rr))
 # Current source provenance. Keep the full auditable Python/config/test surface:
 # V3's name filter omitted core src/ plus runner dependencies such as the paired
 # bootstrap and causal-hardening freezer, making the evidence ZIP non-self-contained.
 for rel in ['src/e_jepa_ttc','scripts','configs/model','configs/experiment','tests/unit']:
  base=repo/rel
  if not base.exists():continue
  for src in base.rglob('*'):
   if src.is_file() and src.suffix.lower() in KEEP_EXT:
    copy_file(src,stage/'source_snapshot'/src.relative_to(repo))
 manifest=[]
 for src in sorted(stage.rglob('*')):
  if src.is_file():manifest.append({'path':src.relative_to(stage).as_posix(),'bytes':src.stat().st_size,'sha256':sha(src)})
 (stage/'SHA256_MANIFEST.json').write_text(json.dumps({'created_at_utc':datetime.now(UTC).isoformat(),'files':manifest},indent=2,sort_keys=True)+'\n',encoding='utf-8')
 zip_path.parent.mkdir(parents=True,exist_ok=True)
 if zip_path.exists():zip_path.unlink()
 with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for src in stage.rglob('*'):
   if src.is_file():z.write(src,src.relative_to(stage).as_posix())
 print(json.dumps({'zip':str(zip_path),'sha256':sha(zip_path),'bytes':zip_path.stat().st_size},indent=2))
 return 0
if __name__=='__main__':raise SystemExit(main())
