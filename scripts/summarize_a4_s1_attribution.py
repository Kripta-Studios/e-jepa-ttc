#!/usr/bin/env python
"""Summarize the preregistered A4-S1 scaling/lambda attribution triangle."""
from __future__ import annotations
import argparse, json, sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from e_jepa_ttc.artifacts.hashing import sign_artifact

def _read(p: Path)->dict[str,Any]:
    x=json.loads(p.read_text(encoding='utf-8'))
    if not isinstance(x,dict): raise ValueError(p)
    if x.get('official_test_opened') is not False: raise ValueError(f'test-open contract missing: {p}')
    return x

def _m(x:dict[str,Any])->dict[str,float]:
    v=x['validation_metrics']
    return {'mid':float(v['sequence_macro']['sequence_macro_paper_MiD_overall']), 'failure':float(v['signed']['failure_rate_pct']), 'pearson':float(v['log_ratio_pearson'])}

def _delta(a:dict[str,float],b:dict[str,float])->dict[str,float]:
    return {'mid':b['mid']-a['mid'],'mid_relative':(b['mid']-a['mid'])/a['mid'],'failure_pp':b['failure']-a['failure'],'pearson':b['pearson']-a['pearson']}

def main()->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--a4-2k-lambda4',type=Path,required=True); p.add_argument('--a4-8k-lambda4',type=Path,required=True); p.add_argument('--a4-8k-lambda8',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    try:
        m2=_m(_read(a.a4_2k_lambda4.resolve())); m4=_m(_read(a.a4_8k_lambda4.resolve())); m8=_m(_read(a.a4_8k_lambda8.resolve()))
        result={'artifact_type':'a4_s1_scaling_lambda_attribution_v1','created_at_utc':datetime.now(UTC).isoformat(),'metrics':{'a4_2k_lambda4':m2,'a4_8k_lambda4':m4,'a4_8k_lambda8':m8},'effects':{'scale_2k_to_8k_at_lambda4':_delta(m2,m4),'lambda4_to_lambda8_at_8k':_delta(m4,m8),'combined_2k_lambda4_to_8k_lambda8':_delta(m2,m8)},'interpretation_contract':{'descriptive_not_factorial_causal_estimate':True,'same_public_validation_reused':True,'private_test_opened':False},'sota_claim_authorized':False}
        sign_artifact(result); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    except Exception as exc:
        print(f'attribution summary failed: {type(exc).__name__}: {exc}',file=sys.stderr); return 1
    print(json.dumps(result,indent=2,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
