from pathlib import Path
import numpy as np,pandas as pd,json
import argparse
ap=argparse.ArgumentParser(description='Post-hoc paired cluster bootstrap; no training')
ap.add_argument('--evidence',type=Path,default=Path(__file__).resolve().parents[1]/'evidence')
ap.add_argument('--output',type=Path,required=True)
args=ap.parse_args()
P=args.evidence;out=[]
rng=np.random.default_rng(20261012)
for name in ['FCWD','DEV32']:
 d=pd.read_csv(P/f'{name}_SOURCE.csv');d=d[np.isfinite(d.truth_ttc_seconds)].copy()
 with np.errstate(divide='ignore',invalid='ignore'):
  ph=lambda x:-np.log1p(-.1/x)
 for b in ['R_CTX','E_CTX_MATCHED','public_Garl_event_lhr','public_Garl_rgb_event_full']:
  x=d[np.isfinite(d[b])].copy();y=x.truth_ttc_seconds.to_numpy();a=x.F_TRUE.to_numpy();bb=x[b].to_numpy()
  if not len(x):continue
  for met in ['MAE','mean_MiD']:
   delta=abs(a-y)-abs(bb-y) if met=='MAE' else 1e4*(abs(ph(a)-ph(y))-abs(ph(bb)-ph(y)))
   v=x[['sequence_id']].assign(delta=delta).groupby('sequence_id').agg(n=('delta','size'),s=('delta','sum'))
   n=v.n.to_numpy();s=v.s.to_numpy();idx=rng.integers(0,len(n),size=(4000,len(n)));draws=s[idx].sum(1)/n[idx].sum(1)
   out.append({'dataset':name,'contrast':'F_TRUE-'+b,'metric':met,'n':len(x),'groups':len(n),'point_micro':delta.mean(),'lo':float(np.quantile(draws,.025)),'hi':float(np.quantile(draws,.975)),'draws':4000,'role':'posthoc_cluster_bootstrap_not_confirmation'})
args.output.mkdir(parents=True,exist_ok=True)
pd.DataFrame(out).to_csv(args.output/'PAIRED_POSTHOC.csv',index=False)
print(pd.DataFrame(out).to_string(index=False))
