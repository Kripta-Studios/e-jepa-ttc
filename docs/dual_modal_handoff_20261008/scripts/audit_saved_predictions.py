#!/usr/bin/env python3
"""Read-only descriptive audit of the already-exposed EvTTC CSV. No model fitting."""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path
import numpy as np
MODELS=['H8_seed7','H8_seed13','H8_seed23','public_Garl_event_lhr','public_Garl_rgb_event_full']

def metric(pred:np.ndarray,truth:np.ndarray)->dict:
    finite=np.isfinite(pred)
    result={'n':len(truth),'finite_predictions':int(finite.sum()),'coverage':float(finite.mean()) if len(truth) else None}
    if not len(truth) or not finite.all():
        result.update(status='INCOMPLETE_NO_SILENT_ROW_DROPPING',mae=None,rmse=None,median_ae=None);return result
    error=pred-truth;ae=np.abs(error);se=error**2
    crucial=(truth>0)&(truth<=3)
    valid_phase_truth=(truth<0)|(truth>.1)
    valid_phase_pred=(pred<0)|(pred>.1)
    result.update(status='COMPLETE',mae=float(ae.mean()),rmse=float(np.sqrt(se.mean())),median_ae=float(np.median(ae)),
                  bias=float(error.mean()),q90_ae=float(np.quantile(ae,.90)),q95_ae=float(np.quantile(ae,.95)),q99_ae=float(np.quantile(ae,.99)),
                  sign_error_fraction=float(((truth>0)!=(pred>0)).mean()),
                  largest_ae_share=float(ae.max()/ae.sum()) if ae.sum() else 0.,largest_se_share=float(se.max()/se.sum()) if se.sum() else 0.,
                  crucial_count=int(crucial.sum()),
                  crucial_mae=float(ae[crucial].mean()) if crucial.any() else None,
                  crucial_positive_overestimate_fraction=float(((pred>truth)&(pred>0))[crucial].mean()) if crucial.any() else None,
                  invalid_phase_targets=int((~valid_phase_truth).sum()),invalid_phase_predictions=int((~valid_phase_pred).sum()))
    if valid_phase_truth.all() and valid_phase_pred.all():
        loss=1e4*np.abs(-np.log1p(-.1/pred)+np.log1p(-.1/truth))
        result['unweighted_phase_absolute_error_x1e4']=float(loss.mean())
    else:
        result['unweighted_phase_absolute_error_x1e4']=None
    return result

def audit(path:Path,outdir:Path)->dict:
    raw=path.read_bytes();rows=list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))
    required={'query_id','sequence_id','truth_ttc_seconds',*MODELS}
    if not rows or not required.issubset(rows[0]):raise ValueError('Unexpected scored CSV schema')
    if len({r['query_id'] for r in rows})!=len(rows):raise ValueError('Duplicate query IDs')
    def num(v):
        try:return float(v)
        except (ValueError,TypeError):return float('nan')
    targets=np.array([num(r['truth_ttc_seconds']) for r in rows],dtype=np.float64)
    labelled=np.isfinite(targets);truth=targets[labelled]
    seq=np.array([r['sequence_id'] for r in rows])[labelled]
    report={'role':'posthoc_already_exposed_development_no_training_no_model_selection',
            'source':str(path),'source_sha256':hashlib.sha256(raw).hexdigest(),
            'queries':len(rows),'labelled':int(labelled.sum()),'no_valid_label':int((~labelled).sum()),
            'cap_policy':'common_signed_cap60 is a new diagnostic sidecar, not official score or replacement',
            'phase_metric_note':'unweighted phase error is NOT the sequence/bucket-weighted official MiD',
            'models':{}}
    perseq=[]
    for name in MODELS:
        pred=np.array([num(r[name]) for r in rows],np.float64)[labelled]
        entry={'native_exceeds60':int((np.abs(pred)>60).sum())}
        for mode,x in [('native',pred),('common_signed_cap60',np.clip(pred,-60.,60.))]:
            entry[mode]=metric(x,truth)
            macro=[]
            for group in sorted(set(seq)):
                ix=seq==group;m=metric(x[ix],truth[ix]);perseq.append({'model':name,'mode':mode,'sequence':group,**m})
                if m.get('mae') is not None:macro.append(m['mae'])
            entry[mode]['sequence_macro_mae']=float(np.mean(macro)) if len(macro)==len(set(seq)) else None
        report['models'][name]=entry
    outdir.mkdir(parents=True,exist_ok=True)
    (outdir/'TAIL_AND_SUPPORT_AUDIT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    (outdir/'PER_SEQUENCE_SUPPORT.json').write_text(json.dumps(perseq,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--csv',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.resolve()==a.csv.parent.resolve():raise SystemExit('Use a new audit directory, not the historical evidence root')
    r=audit(a.csv,a.output);print(json.dumps({'source_sha256':r['source_sha256'],'queries':r['queries'],'labelled':r['labelled'],'optimizer_updates':0}))
