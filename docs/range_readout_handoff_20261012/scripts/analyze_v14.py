"""Read-only numeric review of the supplied V14 bundle. No model fitting."""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np
import pandas as pd

BANDS=[('crucial',.1,3,.5),('small',3,6,.3),('large',6,10,.1),('negative',-10,0,.1)]

def metrics(t,p,g):
    t=np.asarray(t,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    if not np.isfinite(t).all() or not np.isfinite(p).all():raise ValueError('nonfinite')
    mid=np.abs(np.log(1-.1/t)-np.log(1-.1/p))*10000
    if not np.isfinite(mid).all():raise ValueError('invalid phase')
    out={}; group_scores={}
    for name,lo,hi,w in BANDS:
        mask=(t>lo)&(t<=hi);gm={str(s):float(mid[mask&(g==s)].mean()) for s in np.unique(g[mask])}
        out[name]=float(np.mean(list(gm.values())));group_scores[name]=gm
    out['overall']=sum(out[n]*w for n,_,_,w in BANDS)
    out['mean']=float(mid.mean());out['mae']=float(np.abs(p-t).mean())
    out['sign_errors']=int((np.signbit(t)!=np.signbit(p)).sum())
    short=(t>.1)&(t<=1);warn=(p>.1)&(p<=1)
    out['urgent_total']=int(short.sum());out['urgent_detected']=int((warn&short).sum())
    out['false_alerts']=int((warn&~short).sum());out['output_cap_count']=int((np.abs(p)>=59.99999).sum())
    return out,group_scores

def run(root,out):
    a=root/'artifacts/range_transfer_20261010';out.mkdir(parents=True,exist_ok=True)
    mismatches=[]; manifest=json.loads((root/'BUNDLE_MANIFEST.json').read_text())
    for rel,expected in manifest.items():
        if hashlib.file_digest((root/rel).open('rb'),'sha256').hexdigest()!=expected:mismatches.append(rel)
    if mismatches: raise ValueError(f"Source manifest mismatches: {mismatches}")
    results=[];recon=[]; clipping=[];frames={}
    for folder in sorted((a/'evaluation/screen').iterdir()):
        f=pd.read_csv(folder/'PREDICTIONS.csv',float_precision='round_trip'); frames[folder.name]=f
        # Historical inference emits float32 before CSV serialization. Recover those bits.
        p=f.prediction_ttc.to_numpy(np.float32).astype(np.float64);t=f.target_ttc.to_numpy(np.float64);g=f.group_id.to_numpy(str)
        s,gs=metrics(t,p,g);s['fit_id']=folder.name;results.append(s)
        m=json.loads((folder/'METRICS.json').read_text())
        for b,_,_,w in BANDS:recon.append({'fit_id':folder.name,'field':b,'delta':s[b]-m['bins'][b]['group_macro_mid']})
        recon.append({'fit_id':folder.name,'field':'overall','delta':s['overall']-m['overall_MiD']})
        recon.append({'fit_id':folder.name,'field':'mae','delta':s['mae']-m['diagnostics']['mae_s']})
        c=pd.read_csv(a/'fits'/folder.name/'CURVE.csv')
        clipping.append({'fit_id':folder.name,'updates':len(c),'clip_fraction':float((c.gradient_l2_before_clip>1).mean()),'last250_clip_fraction':float((c.tail(250).gradient_l2_before_clip>1).mean()),'first100_loss':float(c.head(100).loss.mean()),'penultimate250_loss':float(c.iloc[-500:-250].loss.mean()),'last250_loss':float(c.tail(250).loss.mean())})
    ids0=next(iter(frames.values()))
    for f in frames.values():
        if not f[['sample_token','group_id','target_ttc']].equals(ids0[['sample_token','group_id','target_ttc']]):raise ValueError('alignment')
    pd.DataFrame(results).to_csv(out/'METRICS_RECALCULATED.csv',index=False)
    pd.DataFrame(recon).to_csv(out/'METRIC_DELTAS.csv',index=False)
    pd.DataFrame(clipping).to_csv(out/'CLIPPING_AND_CURVES.csv',index=False)
    # Conditional geometric reach under the documented 0.05 bound, with frozen observed heights.
    controls=pd.read_csv(a/'H_DIAGNOSTIC_CONTROLS.csv',float_precision='round_trip')
    trace=pd.read_csv(a/'matched_H_raw_adapted/PRODUCER_TRACES.csv',float_precision='round_trip')
    current=trace.merge(controls[['sample_token','target_ttc','group_id']],left_on='observation_id',right_on='sample_token',validate='many_to_one')
    rows=[]
    fragment_by_obs={}
    for file in (a/'matched_H_raw_adapted/fragments').glob('*.json'):
        q=json.loads(file.read_text());fragment_by_obs[q['observation_id']]=q
    for _,x in current.iterrows():
        if x.target_ttc<=0:continue
        q=fragment_by_obs[x.observation_id];dt=np.array(q['delta_t_s']);p=q['producers'][x.producer];C=.05
        aa=np.array(p['analytic_log_height_ratio']);rr=np.array(p['residual_log_height_ratio'])
        # Optimistic bound: independent max correction at both endpoints (may not be jointly realizable).
        qupper=np.expm1(aa+C)/dt;den=1-dt[-1]*qupper[-2]
        gamma=float(p['replay_current_pair_weight'][0]);qmax=float(qupper[-1])
        if den <= 1e-4: raise ValueError("Reach bound not supported across a transport pole")
        if den>1e-4:qmax=max(qmax,float(gamma*qupper[-1]+(1-gamma)*qupper[-2]/den))
        tmin=1/qmax if qmax>0 else np.inf
        target_r=math.log1p(float(dt[-1])/float(x.target_ttc))
        rows.append({'sample_token':x.sample_token,'producer':x.producer,'target_ttc':x.target_ttc,'gt_band':'c_0.1_1' if x.target_ttc<=1 else 'c_1_2' if x.target_ttc<=2 else 'c_2_3' if x.target_ttc<=3 else 'other_positive','analytic_last':aa[-1],'residual_last':rr[-1],'residual_over_cap':rr[-1]/C,'tanh_local_slope_fraction':1-(rr[-1]/C)**2,'target_r_current_convention':target_r,'required_last_correction_if_current_pair_only':target_r-aa[-1],'optimistic_smallest_positive_ttc_fixed_geometry':tmin,'target_below_fixed_geometry_reach':bool(x.target_ttc<tmin),'producer_ttc':p['ttc_mean_seconds'][0]})
    reach=pd.DataFrame(rows);reach.to_csv(out/'FIXED_GEOMETRY_REACH_H32.csv',index=False)
    summary=reach.groupby(['gt_band','producer']).agg(n=('sample_token','size'),cap_ratio_mean=('residual_over_cap','mean'),unreachable_under_fixed_geometry=('target_below_fixed_geometry_reach','sum'),ttc_lower_min=('optimistic_smallest_positive_ttc_fixed_geometry','min'),ttc_lower_max=('optimistic_smallest_positive_ttc_fixed_geometry','max')).reset_index()
    summary.to_csv(out/'FIXED_GEOMETRY_REACH_SUMMARY.csv',index=False)
    summary_json={'manifest_entries_verified':len(manifest),'manifest_mismatches':mismatches,'rows_per_fit':len(ids0),'fits':len(frames),'saved_prediction_rows_scored':sum(len(f) for f in frames.values()),'macro_metric_max_abs_delta':max(abs(r['delta']) for r in recon),'heads_not_reinferred':True,'no_new_training':True,'fixed_geometry_reach_scope':'Posthoc diagnostic on the 32 H controls selected in V14. Assumes documented residual cap 0.05 and native current-time blend. NOT an absolute bound on retrained encoder, PAIR or H8. Independent max at both pairs is an optimistic relaxation.','reach_summary':json.loads(summary.to_json(orient='records'))}
    (out/'REVIEW_VERIFICATION.json').write_text(json.dumps(summary_json,indent=2)+'\n')
    print(json.dumps(summary_json,indent=2))
    print(pd.DataFrame(results).round(4).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.root,a.output)
