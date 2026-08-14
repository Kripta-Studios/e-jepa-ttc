#!/usr/bin/env python
"""Bind the frozen seed-7 Garl OOF comparator into the V8-A manifest format."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def main()->int:
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--protocol',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args()
 try:
  protocol=json.loads(a.protocol.read_text(encoding='utf-8'))
  if not isinstance(protocol,dict) or not verify_artifact_hash(protocol): raise ValueError('unsigned protocol')
  src=protocol['sources']['garl_oof_predictions']; path=ROOT/str(src['path'])
  if not path.is_file() or sha(path)!=src['sha256']: raise ValueError('frozen Garl OOF binding mismatch')
  f=pd.read_csv(path)
  rename={'sample_token':'token_id','fold':'outer_fold','target_ttc_s':'target_ttc','point_prediction_ttc_s':'prediction_ttc','prediction_ttc_s':'prediction_ttc'}
  for old,new in rename.items():
   if old in f.columns and new not in f.columns: f[new]=f[old]
  needed=['token_id','sequence_id','track_id','target_ttc','prediction_ttc','outer_fold']
  if any(x not in f.columns for x in needed): raise ValueError('Garl comparator lacks required OOF columns')
  if len(f)!=protocol['sample_contract']['rows'] or f['token_id'].astype(str).duplicated().any(): raise ValueError('Garl comparator is not exact OOF')
  f['seed']=7; f['prediction_log_variance']=0.0; f['guard_margin']=1.0; f['event_rate']=0.0; f['motion_magnitude']=0.0; f['occupancy_entropy']=0.0
  out=a.output_dir/'baseline.csv'; out.parent.mkdir(parents=True,exist_ok=True); f.to_csv(out,index=False,lineterminator='\n')
  payload={'artifact_type':'scientific_recovery_v8_garl_replay_v1','status':'completed_replay_without_optimizer_steps','model_name':'garl','causality_checks':{'optimizer_steps':0,'frozen_oof_only':True},'interventions':{'baseline':{'path':out.name,'sha256':sha(out)}}}
  sign_artifact(payload); (a.output_dir/'manifest.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 except (OSError,ValueError,KeyError) as e: p.exit(2,f'V8 Garl comparator failed closed: {type(e).__name__}: {e}\n')
 print(json.dumps({'status':'completed','manifest':str(a.output_dir/'manifest.json')})); return 0
if __name__=='__main__': raise SystemExit(main())
