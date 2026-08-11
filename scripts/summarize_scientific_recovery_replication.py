#!/usr/bin/env python
"""Aggregate three preregistered scientific-recovery seeds into a replication gate."""
from __future__ import annotations
import argparse, json, statistics, sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from scripts.classify_scientific_recovery_gate import _read, _m, classify
from e_jepa_ttc.artifacts.hashing import sign_artifact

def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('a5','a6','a7'),required=True)
    p.add_argument('--base-summary',type=Path,action='append',required=True)
    p.add_argument('--summary',type=Path,action='append',required=True)
    p.add_argument('--a5-summary',type=Path,action='append')
    p.add_argument('--required-passes',type=int,default=2)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:
        if len(a.summary)!=3: raise ValueError('exactly three --summary values are required')
        if a.stage!='a5' and (not a.a5_summary or len(a.a5_summary) not in {1,3}): raise ValueError('A6/A7 require one or three --a5-summary values')

        if len(a.base_summary) not in {1,3}: raise ValueError('one or three --base-summary values are required')
        rows=[]; pass_name=f'REPLICATE_{a.stage.upper()}'
        for i,path in enumerate(a.summary):
            ref=None
            if a.stage!='a5':
                ref_path=a.a5_summary[0] if len(a.a5_summary)==1 else a.a5_summary[i]
                ref=_read(ref_path.resolve())
            base_path=a.base_summary[0] if len(a.base_summary)==1 else a.base_summary[i]
            base=_read(base_path.resolve())
            s=_read(path.resolve()); gate=classify(a.stage,base,s,ref); metrics=_m(s)
            rows.append({'path':str(path.resolve()),'decision':gate['decision'],'pass':gate['decision']==pass_name,'metrics':metrics,'gate':gate})
        passes=sum(int(r['pass']) for r in rows); passed=passes>=a.required_passes
        mids=[r['metrics']['mid'] for r in rows]; pears=[r['metrics']['pearson'] for r in rows]; fails=[r['metrics']['failure'] for r in rows]
        result={'artifact_type':f'scientific_recovery_{a.stage}_replication_gate_v1','created_at_utc':datetime.now(UTC).isoformat(),'stage':a.stage,'status':'PASS' if passed else 'FAIL','required_passes':a.required_passes,'passes':passes,'rows':rows,'aggregate':{'mid_mean':statistics.fmean(mids),'mid_median':statistics.median(mids),'mid_stdev':statistics.stdev(mids),'pearson_mean':statistics.fmean(pears),'pearson_median':statistics.median(pears),'failure_mean':statistics.fmean(fails),'failure_median':statistics.median(fails)},'contract':{'three_frozen_seeds':True,'public_validation_only':True,'private_test_opened':False,'replication_failure_blocks_promotion_not_diagnostics':True},'sota_claim_authorized':False}
        sign_artifact(result); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    except Exception as exc:
        print(f'replication summary failed: {type(exc).__name__}: {exc}',file=sys.stderr); return 1
    print(json.dumps(result,indent=2,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
