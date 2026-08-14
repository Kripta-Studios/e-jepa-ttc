#!/usr/bin/env python
"""Mechanically clone the frozen D0--D4 seed-7 configs for optimization seeds 13/23."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
def main()->int:
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--seed',type=int,choices=(13,23),required=True); p.add_argument('--source-dir',type=Path,default=ROOT/'configs/experiment/scientific_recovery_v8_jepa'); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args()
 a.output_dir.mkdir(parents=True,exist_ok=True); outputs=[]
 for src in sorted(a.source_dir.glob('*.yaml')):
  raw=yaml.safe_load(src.read_text(encoding='utf-8'))
  if not isinstance(raw,dict) or raw.get('experiment',{}).get('seed')!=7: raise ValueError(f'JEPA source config is not frozen seed7: {src}')
  clone=json.loads(json.dumps(raw)); clone['experiment']['seed']=a.seed; clone['experiment']['name']=str(clone['experiment']['name'])+f'_seed{a.seed}'; clone['experiment']['multiseed_replication']={'source_seed':7,'no_tuning':True,'no_reselection':True,'external_confirmation':False}
  dst=a.output_dir/src.name; dst.write_text(yaml.safe_dump(clone,sort_keys=False),encoding='utf-8',newline='\n'); outputs.append(str(dst))
 print(json.dumps({'status':'completed','seed':a.seed,'configs':outputs},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
