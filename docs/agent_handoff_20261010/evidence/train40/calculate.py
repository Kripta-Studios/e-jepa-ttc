from pathlib import Path
from datetime import UTC,datetime
import argparse,json,hashlib,subprocess
import numpy as np
import pandas as pd
from operational.garl_comparison.mid import load_scorer,scores,SCORER_SHA256
from operational.sota_evidence.run import json_safe
parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();root=args.root.resolve();campaign=root/'artifacts/train40_system_20261005'
hashes={}
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def pin(p,expected=None):
 with p.open('rb') as f: value=hashlib.file_digest(f,'sha256').hexdigest()
 if expected is not None and value!=expected: raise ValueError(str(p))
 hashes[str(p)]=value
 return value
index=campaign/'TRAIN40_INDEX.npz';pin(index,read(campaign/'DATA_AUDIT.json')['index_sha256'])
with np.load(index,allow_pickle=False) as x:
 truth=x['ttc_s'];seq=x['sequences']
n=len(truth)
assert n==88744 and len(set(seq))==40
pred={f'H8_seed{s}':np.empty(n,np.float32) for s in (7,13,23)}
binding=pin(campaign/'train_predictions/BINDING.json')
for start in range(0,n,128):
 stop=min(start+128,n);path=campaign/'train_predictions'/f'batch_{start:06d}.npz'
 receipt=read(path.with_suffix('.json'));pin(path,receipt['sha256'])
 assert receipt['binding_sha256']==binding
 with np.load(path,allow_pickle=False) as x:
  assert np.array_equal(x['ordinals'],np.arange(start,stop))
  for seed in (7,13,23):pred[f'H8_seed{seed}'][start:stop]=x[f'ttc_seed{seed}']
pred['H8_median3']=np.median(np.stack(list(pred.values())),axis=0)
manifest=read(campaign/'PUBLIC_GARL_PREDICTION_MANIFEST.json')
assert manifest['status']=='COMPLETE_VERIFIED' and manifest['row_count']==n
assert manifest['binding']['index_sha256']==hashes[str(index)]
entry=manifest['files'][0];path=campaign/entry['path'];pin(path,entry['sha256'])
with np.load(path,allow_pickle=False) as x:
 assert np.array_equal(x['ordinals'],np.arange(n))
 pred['public_Garl_event_lhr']=x['ttc']
scorer_path=root/'artifacts/garlttc_submission_20261010/official_score.py';pin(scorer_path,SCORER_SHA256)
scorer=load_scorer(scorer_path);eligible=np.isfinite(truth)&(truth!=0)
rows=[]
for method,values in pred.items():
 assert np.isfinite(values).all()
 row={'dataset':'eAP_TRAIN40','role':'TRAIN_DIAGNOSTIC_NOT_HOLDOUT','method':method,
      'population_n':n,'gt_ineligible_n':int((~eligible).sum()),
      **scores(scorer,truth[eligible],values[eligible].astype(float))}
 rows.append(row)
 print(json.dumps(json_safe({k:row[k] for k in ('method','mean_MiD','overall_MiD','MiDc','MiDs','MiDl','MiDn','n_c','n_s','n_l','n_n','invalid_mid_total')})),flush=True)
pin(Path(__file__))
pd.DataFrame(rows).to_csv(args.output/'METRICS.csv',index=False)
result={'status':'COMPLETE_TRAIN_DIAGNOSTIC','utc':datetime.now(UTC).isoformat(),
 'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'optimizer_updates':0,'gpu_seconds':0,'rows':rows,'input_sha256':hashes,
 'scorer_sha256':SCORER_SHA256,'official_test_score':False,
 'scope':'88,744 training queries, 40 sequences; H8 trained on these data; no claim of generalization or SOTA',
 'ensemble':'median of three native TTC predictions, consistent with prior Dev32/FCWD score tables',
 'garl_checkpoint_training_sequence_manifest_verified':manifest['binding']['actual_checkpoint_training_sequence_manifest_verified']}
(args.output/'RESULT.json').write_text(json.dumps(json_safe(result),indent=2),encoding='utf-8')
