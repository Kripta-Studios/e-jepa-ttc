#!/usr/bin/env python3
from pathlib import Path
import sys,json,hashlib,platform
import numpy as np,pandas as pd,torch
import argparse
parser=argparse.ArgumentParser(description='Numerical replay of V17 saved metrics and prepared export examples; not raw inference or training.')
parser.add_argument('--bundle-root',type=Path,required=True,help='Extracted V17 directory containing source/ and campaign/')
parser.add_argument('--output',type=Path,required=True,help='A NEW output directory outside the bundle')
args=parser.parse_args()
R=args.bundle_root.resolve(); S=R/'source'; O=args.output.resolve()
if not (R/'campaign').is_dir() or not S.is_dir(): raise SystemExit('Missing source/ or campaign/')
if O==R or R in O.parents: raise SystemExit('Use an output directory outside the historical bundle')
O.mkdir(parents=True,exist_ok=False)
(O/'REPLAY_ENVIRONMENT.json').write_text(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'platform':platform.platform(),'scope':'Saved predictions and prepared export inputs only; cross-runtime numerical comparison'},indent=2))
sys.path[:0]=[str(S),str(S/'src')]
from operational.range_transfer.analyze import metrics
from operational.representation_audit.export_weights import load_export
torch.set_num_threads(1)
def differences(a,b,path=''):
 out=[]
 if isinstance(a,dict) and isinstance(b,dict):
  for k in sorted(a.keys()|b.keys()):
   if k not in a or k not in b:out.append((path+'/'+k,'structure',0))
   else:out.extend(differences(a[k],b[k],path+'/'+k))
 elif isinstance(a,list) and isinstance(b,list):
  if len(a)!=len(b):out.append((path,'length',0))
  else:
   for i,(x,y) in enumerate(zip(a,b)):out.extend(differences(x,y,path+f'/{i}'))
 elif a!=b:
  if isinstance(a,(float,int)) and isinstance(b,(float,int)) and not isinstance(a,bool):
   if not (np.isnan(a) and np.isnan(b)):out.append((path,'numeric',abs(a-b)))
  else:out.append((path,'value',0))
 return out
records=[]
for pat in ['evaluation/*/PREDICTIONS.csv','*_evaluation/*/*/PREDICTIONS.csv']:
 for p in sorted((R/'campaign').glob(pat)):
  df=pd.read_csv(p,converters={'target_ttc':float,'prediction_ttc':np.float32})
  rec=json.loads((p.parent/'RECEIPT.json').read_text())
  
  if hashlib.sha256(p.read_bytes()).hexdigest()!=rec['predictions_sha256']: raise ValueError(f'Prediction hash mismatch: {p}')
  a=metrics(df.target_ttc.to_numpy(np.float64),df.prediction_ttc.to_numpy(np.float32),df.group_id.to_numpy(str));b=json.loads((p.parent/'METRICS.json').read_text())
  ds=differences(a,b);records.append({'path':str(p.relative_to(R/'campaign')),'rows':len(df),'maxabs':max([d[2] for d in ds] or [0]),'different_fields':ds,'exact':a==b})
probe=R/'campaign/group_probe';folds=pd.read_csv(probe/'FOLDS.csv',converters={'phase_mae':float})
for name,b in json.loads((probe/'RESULTS.json').read_text()).items():
 p=probe/f'{name}_OOF.csv';df=pd.read_csv(p,converters={'target_ttc':float,'prediction_ttc':float})
 a=metrics(df.target_ttc.to_numpy(np.float64),df.prediction_ttc.to_numpy(np.float64),df.group_id.to_numpy(str));a['macro_phase_mae']=float(folds.loc[folds.representation==name,'phase_mae'].to_numpy().mean())
 ds=differences(a,b);records.append({'path':str(p.relative_to(R/'campaign')),'rows':len(df),'maxabs':max([d[2] for d in ds] or [0]),'different_fields':ds,'exact':a==b})
result={'tables':len(records),'rows_including_reused_controls':sum(x['rows'] for x in records),'maxabs':max(x['maxabs'] for x in records),'nonnumeric_differences':[x for x in records if any(d[1]!='numeric' for d in x['different_fields'])],'exact_tables':sum(x['exact'] for x in records),'records':records,'new_inference':False}
(O/'NUMERICAL_METRICS_REPLAY.json').write_text(json.dumps(result,indent=2));print('METRICS', {k:v for k,v in result.items() if k!='records'},flush=True)
exports=[]
for p in sorted((R/'campaign/exports').iterdir()):
 if not p.is_dir():continue
 md=json.loads((p/'MODEL_METADATA.json').read_text());rec=json.loads((p/'RECEIPT.json').read_text())
 for fn,key in [('WEIGHTS.pt','weights_sha256'),('EXAMPLE_INPUT.npz','example_input_sha256'),('EXAMPLE_OUTPUT.npz','example_output_sha256')]:
  if hashlib.sha256((p/fn).read_bytes()).hexdigest()!=rec[key]: raise ValueError(f'Export hash mismatch: {p/fn}')
 with np.load(p/'EXAMPLE_INPUT.npz',allow_pickle=False) as arc:ins={k:torch.from_numpy(arc[k].copy()) for k in arc.files}
 
 if any(k.startswith(('target_','dino_')) for k in ins): raise ValueError('Target/teacher key in inference input')
 m=load_export(p/'WEIGHTS.pt')
 with torch.inference_mode():a=m(*(ins[k] for k in md['input_order'])) if md['input_style']=='positional' else m(ins)
 fields=[]
 with np.load(p/'EXAMPLE_OUTPUT.npz',allow_pickle=False) as arc:
  for k in arc.files:
   x=(a[k] if isinstance(a,dict) else getattr(a,k)).detach().numpy();y=arc[k]
   finite=np.isfinite(x)&np.isfinite(y)
   diff=np.abs(x[finite].astype(np.float64)-y[finite].astype(np.float64))
   fields.append({'key':k,'maxabs':float(diff.max()) if len(diff) else 0.,'exact':np.array_equal(x,y,equal_nan=True),'same_finiteness':bool(np.array_equal(np.isfinite(x),np.isfinite(y))),'allclose_1e5':bool(np.allclose(x,y,atol=1e-5,rtol=1e-5,equal_nan=True))})
 exports.append({'name':p.name,'fields':fields})
 print('EXPORT',p.name,max(f['maxabs'] for f in fields), all(f['allclose_1e5'] for f in fields),flush=True)
(O/'NUMERICAL_EXPORTS_REPLAY.json').write_text(json.dumps({'exports':exports,'torch':torch.__version__,'new_optimizer_updates':0,'scope':'prepared examples only; no source parent checkpoint, raw input or training cache loaded'},indent=2))
