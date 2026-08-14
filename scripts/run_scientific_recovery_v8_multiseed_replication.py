#!/usr/bin/env python
# ruff: noqa: E501
"""Execute and aggregate the frozen V8 optimization-stability replication.

Only a signed seed-7 TTC winner opens this stage.  Seeds 13/23 train the nominated
candidate without tuning and train matched A5 controls on the same outer folds.
For a router winner the entire nested A5/C2F/router procedure is repeated per seed.
"""
from __future__ import annotations

import argparse, concurrent.futures, hashlib, json, subprocess, sys
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa:E402
from e_jepa_ttc.evaluation.scientific_recovery_v8_aggregate import _bind_candidate_to_baseline_contract, _metrics, _read_rows  # noqa:E402
from e_jepa_ttc.evaluation.scientific_recovery_v8_runner import verify_frozen_inputs  # noqa:E402
from e_jepa_ttc.training.scientific_recovery_v8_jobs import build_fold_jobs, clone_multiseed_configs, execute_jobs  # noqa:E402

CANDIDATE_TO_ARM={"B1_TIMEVOL20_3":"timevol20_3","B2_EXP6_3":"exp6_3","B3_PAIR20_2":"pair20_2","C1_GATED_EXP6_3":"gated_exp6_3","R":"R"}

def _signed(path:Path)->dict[str,Any]:
 v=json.loads(path.read_text(encoding='utf-8'))
 if not isinstance(v,dict) or not verify_artifact_hash(v): raise ValueError(f'unsigned artifact: {path}')
 return v

def _sha(path:Path)->str:
 h=hashlib.sha256();
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def _source_configs(frozen:Any, arm:str)->list[Path]:
 if arm=='R':
  entries=frozen.manifest['enabled_seed7_configs']; return sorted(ROOT/str(e['path']) for n,e in entries.items() if str(n).startswith('router_fold') and str(n).endswith('_seed7'))
 entries=frozen.manifest.get('enabled_seed7_configs',{})
 result=sorted(ROOT/str(e['path']) for n,e in entries.items() if str(n).startswith(f'{arm}_fold') and str(n).endswith('_seed7'))
 if result: return result
 template=frozen.manifest.get('conditional_templates',{}).get(arm,{})
 result=[ROOT/str(e['path']) for e in template.get('fold_configs',[]) if isinstance(e,Mapping)]
 if len(result)!=3: raise ValueError(f'no three frozen source configs for {arm}')
 return sorted(result)

def _run_logged(command:list[str], output_dir:Path, label:str)->None:
 output_dir.mkdir(parents=True,exist_ok=True); logs=output_dir/'logs'; logs.mkdir(exist_ok=True)
 (logs/'command.json').write_text(json.dumps({'label':label,'command':command},indent=2)+'\n',encoding='utf-8')
 with (logs/'stdout.log').open('a',encoding='utf-8',buffering=1) as out,(logs/'stderr.log').open('a',encoding='utf-8',buffering=1) as err:
  out.write(f'\n=== START {label} ===\nCOMMAND: {" ".join(command)}\n')
  rc=subprocess.run(command,cwd=ROOT,stdout=out,stderr=err,check=False).returncode
  out.write(f'=== END {label} exit={rc} ===\n')
 if rc: raise RuntimeError(f'{label} failed; inspect {logs}')

def _clone_router(frozen:Any, root:Path)->list[Path]:
 return clone_multiseed_configs(candidate='R',source_configs=_source_configs(frozen,'R'),output_dir=root)

def _a5_controls(frozen:Any, router_configs:list[Path], seed:int, root:Path, device:str, max_parallel:int)->list[Path]:
 configs=[p for p in router_configs if f'seed{seed}' in p.name]
 if len(configs)!=3: raise ValueError(f'missing router-derived A5 configs for seed {seed}')
 commands=[]; outputs=[]
 for config in configs:
  import yaml
  raw=yaml.safe_load(config.read_text(encoding='utf-8')); fold=int(raw['outer_fold']); out=root/f'fold{fold}'
  commands.append((['uv','run','--no-sync','python','scripts/train_scientific_recovery_v8_router_expert.py','--config',str(config),'--expert','A5','--role','outer_dev','--output-dir',str(out),'--device',device,'--protocol-sha256',str(frozen.protocol['artifact_sha256'])],out,f'a5_seed{seed}_fold{fold}')); outputs.append(out/'expert_oof.csv')
 with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as ex:
  futures=[ex.submit(_run_logged,*item) for item in commands]
  for f in futures: f.result()
 return outputs

def _simple_candidate(frozen:Any, arm:str, seed:int, config_root:Path, results_root:Path, device:str, max_parallel:int)->list[Path]:
 sources=_source_configs(frozen,arm); cloned=clone_multiseed_configs(candidate=arm,source_configs=sources,output_dir=config_root)
 configs=[p for p in cloned if f'seed{seed}' in p.name]
 jobs=build_fold_jobs(configs=configs,output_root=results_root,device=device,max_parallel=max_parallel,allowed_seeds=(seed,))
 execute_jobs(jobs,protocol_hash=str(frozen.protocol['artifact_sha256']),manifest_hash=str(frozen.manifest['artifact_sha256']),dry_run=False,max_parallel=max_parallel)
 paths=[]
 for job in jobs:
  summary=_signed(job.output_dir/'summary.json'); paths.append(job.output_dir/str(summary['dev_predictions']['path']))
 return paths

def _router_candidate(frozen:Any, router_config_dir:Path, seed:int, seed_root:Path, device:str, max_parallel:int)->tuple[list[Path],list[Path]]:
 command=['uv','run','--no-sync','python','scripts/run_scientific_recovery_v8_nested_router.py','--protocol',str(frozen.protocol_path),'--manifest',str(frozen.manifest_path),'--results-root',str(seed_root),'--config-dir',str(router_config_dir),'--seed',str(seed),'--device',device,'--max-parallel',str(max_parallel),'--execute']
 _run_logged(command,seed_root/'router_stage',f'router_seed{seed}')
 candidate=[seed_root/'runs'/f'router_fold{fold}_seed{seed}'/'dev_predictions.csv' for fold in range(3)]
 a5=[seed_root/'router'/f'outer_fold{fold}_seed{seed}'/'a5'/'outer_dev'/'expert_oof.csv' for fold in range(3)]
 return candidate,a5

def _concat(paths:list[Path])->list[dict[str,str]]:
 frames=[pd.read_csv(p,dtype=str,keep_default_na=False) for p in paths]
 frame=pd.concat(frames,ignore_index=True)
 return [{str(k):str(v) for k,v in row.items()} for row in frame.to_dict(orient='records')]

def _bootstrap_all(seed_pairs:dict[int,tuple[list[dict[str,str]],list[dict[str,str]]]], n:int=5000)->dict[str,Any]:
 rng=np.random.default_rng(20260814); sequences=sorted({r['sequence_id'] for r in next(iter(seed_pairs.values()))[0]}); deltas=[]
 for _ in range(n):
  sampled=rng.choice(sequences,size=len(sequences),replace=True); seed_delta=[]
  for seed,(cand,base) in seed_pairs.items():
   cb={s:[r for r in cand if r['sequence_id']==s] for s in sequences}; bb={r['token_id']:r for r in base}; cr=[]; br=[]
   for draw,s in enumerate(sampled):
    tracks=sorted({r['track_id'] for r in cb[str(s)]}); chosen=rng.choice(tracks,size=len(tracks),replace=True)
    for ti,t in enumerate(chosen):
     for r in cb[str(s)]:
      if r['track_id']==str(t):
       q=dict(r); q['sequence_id']=f'{s}__draw{draw}__track{ti}'; cr.append(q); b=dict(bb[r['token_id']]); b['sequence_id']=q['sequence_id']; br.append(b)
   metrics,_,_=_metrics(cr,br); seed_delta.append(metrics['delta_mid_vs_a5'])
  deltas.append(float(np.mean(seed_delta)))
 arr=np.asarray(deltas); return {'resamples':n,'probability_delta_lt_zero':float(np.mean(arr<0)),'ci95_low':float(np.quantile(arr,.025)),'ci95_high':float(np.quantile(arr,.975)),'mean':float(np.mean(arr))}

def main()->int:
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--candidate'); p.add_argument('--device',default='cuda'); p.add_argument('--max-parallel',type=int,default=2); p.add_argument('--protocol',type=Path,default=ROOT/'configs/protocol/scientific_recovery_v8_temporal.json'); p.add_argument('--manifest',type=Path,default=ROOT/'configs/experiment/scientific_recovery_v8_fold_chain/frozen_manifest.json'); p.add_argument('--results-root',type=Path,default=ROOT/'artifacts/scientific_recovery_v8'); p.add_argument('--dry-run',action='store_true'); a=p.parse_args()
 try:
  if a.max_parallel not in (1,2): raise ValueError('multiseed supports max-parallel 1 or 2')
  frozen=verify_frozen_inputs(a.protocol,a.manifest); aggregate=_signed(a.results_root/'results'/'aggregate_seed7.json'); candidate=a.candidate or str(aggregate.get('candidate_id'))
  if aggregate.get('multiseed_replication_candidate') is not True:
   print(json.dumps({'status':'skipped_no_seed7_candidate','candidate':candidate})); return 0
  if candidate not in CANDIDATE_TO_ARM: raise ValueError(f'unsupported candidate {candidate}')
  if a.dry_run: print(json.dumps({'status':'planned','candidate':candidate,'seeds':[13,23],'a5_controls':True,'max_parallel':a.max_parallel},indent=2)); return 0
  root=a.results_root/'multiseed_replication'; router_configs=_clone_router(frozen,root/'configs'/'router')
  arm=CANDIDATE_TO_ARM[candidate]; pairs={}
  # Seed-7 rows use the frozen screen evidence.
  base7=_read_rows(ROOT/str(frozen.protocol['sources']['a5_oof_predictions']['path']),candidate=False)
  if arm=='R': cand7=_concat([a.results_root/'results'/'runs'/f'router_fold{f}_seed7'/'dev_predictions.csv' for f in range(3)])
  else: cand7=_concat([a.results_root/'results'/'runs'/f'{arm}_fold{f}_seed7'/'dev_predictions.csv' for f in range(3)])
  cand7=_bind_candidate_to_baseline_contract(cand7,base7); pairs[7]=(cand7,base7)
  per_seed={}
  for seed in (13,23):
   seed_root=root/f'seed{seed}'
   if arm=='R': cand_paths,a5_paths=_router_candidate(frozen,root/'configs'/'router',seed,seed_root,a.device,a.max_parallel)
   else:
    a5_paths=_a5_controls(frozen,router_configs,seed,seed_root/'a5',a.device,a.max_parallel)
    cand_paths=_simple_candidate(frozen,arm,seed,root/'configs'/arm,seed_root/'candidate'/'runs',a.device,a.max_parallel)
   base=_concat(a5_paths); cand=_concat(cand_paths); cand=_bind_candidate_to_baseline_contract(cand,base)
   metrics,per_sequence,per_bucket=_metrics(cand,base); per_seed[str(seed)]={'metrics':metrics,'per_sequence':per_sequence,'per_bucket':per_bucket}; pairs[seed]=(cand,base)
  # Seed 7 metrics from exact frozen control.
  m7,s7,b7=_metrics(*pairs[7]); per_seed['7']={'metrics':m7,'per_sequence':s7,'per_bucket':b7}
  boot=_bootstrap_all(pairs); deltas=[per_seed[str(s)]['metrics']['delta_mid_vs_a5'] for s in (7,13,23)]
  passed=all(d<0 for d in deltas) and float(np.mean(deltas))<=-3.0 and boot['ci95_high']<0
  payload={'artifact_type':'scientific_recovery_v8_multiseed_replication_v1','status':'completed','candidate_id':candidate,'seeds':[7,13,23],'per_seed':per_seed,'mean_delta_mid_vs_a5':float(np.mean(deltas)),'bootstrap':boot,'passed':passed,'optimization_stability_replication':True,'external_confirmation':False,'protocol_artifact_sha256':frozen.protocol['artifact_sha256'],'closed_evaluation':frozen.protocol.get('closed_evaluation',{})}
  sign_artifact(payload); out=root/'aggregate.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8'); print(json.dumps({'status':'completed','output':str(out),'passed':passed},sort_keys=True)); return 0
 except (OSError,ValueError,KeyError,RuntimeError) as e: p.exit(2,f'V8 multiseed replication failed closed: {type(e).__name__}: {e}\n')
if __name__=='__main__': raise SystemExit(main())
