#!/usr/bin/env python
"""Build the honest claim/readiness contract after public-only recovery experiments."""
from __future__ import annotations
import argparse, json, sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from e_jepa_ttc.artifacts.hashing import sign_artifact

def _read(path: Path|None)->dict[str,Any]|None:
    if path is None or not path.is_file(): return None
    x=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(x,dict): raise ValueError(path)
    return x

def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract-audit',type=Path,required=True)
    p.add_argument('--prefix-audit',type=Path,required=True)
    p.add_argument('--candidate-summary',type=Path)
    p.add_argument('--candidate-mode',choices=('legacy','causal_left','none'),default='legacy')
    p.add_argument('--garl-budget-summary',type=Path)
    p.add_argument('--paired-bootstrap',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:
        contract=_read(a.contract_audit); prefix=_read(a.prefix_audit); cand=_read(a.candidate_summary); garl=_read(a.garl_budget_summary); boot=_read(a.paired_bootstrap)
        assert contract and prefix
        oracle_matched=bool(contract['claim_contract']['matched_oracle_roi_comparison_allowed'])
        prefix_ok=bool(prefix['results'].get(a.candidate_mode,{}).get('prefix_invariant',False)) if a.candidate_mode!='legacy' else False
        candidate_complete=cand is not None and cand.get('status')=='completed_public_validation_only' and cand.get('official_test_opened') is False
        garl_complete=garl is not None and garl.get('status') in {'completed_max_epochs','completed_early_stopping','completed_public_validation_only'} and garl.get('sealed_sources',{}).get('private_test_opened') is False
        paired_complete=boot is not None and boot.get('checks',{}).get('exact_sample_tokens') is True
        if not candidate_complete:
            readiness='NO_PROMOTABLE_CANDIDATE'
        elif not prefix_ok:
            readiness='ENDPOINT_WINDOW_ORACLE_ROI_CANDIDATE_ONLY__STRICT_CAUSAL_RETRAIN_REQUIRED'
        elif not garl_complete or not paired_complete:
            readiness='CAUSAL_ORACLE_ROI_CANDIDATE__BUDGET_MATCHED_GARL_COMPARISON_BLOCKED'
        else:
            readiness='READY_FOR_ONE_SHOT_SEALED_MATCHED_ORACLE_ROI_TEST'
        result={
            'artifact_type':'scientific_claim_readiness_v1','created_at_utc':datetime.now(UTC).isoformat(),'readiness':readiness,
            'checks':{
                'candidate_public_validation_complete':candidate_complete,
                'model_prefix_causal':prefix_ok,
                'matched_oracle_roi_protocol':oracle_matched,
                'garl_8192_budget_matched_complete':garl_complete,
                'paired_exact_sample_comparison_complete':paired_complete,
                'public_validation_has_been_adaptively_reused':True,
                'private_test_opened':False,
            },
            'authorized_claims':{
                'event_only_neural_forward_under_oracle_roi':candidate_complete and oracle_matched,
                'model_level_prefix_causal':candidate_complete and prefix_ok,
                'end_to_end_no_oracle_localization':False,
                'public_validation_sota':False,
                'sota':False,
            },
            'required_for_future_sota_claim':[
                'freeze commit/config/checkpoint before sealed evaluation',
                'same sealed samples and targets for E-JEPA and Garl',
                'same oracle-ROI privilege or explicitly scoped claim',
                'same failure/coverage metric implementation',
                'budget-matched Garl comparator',
                'one-shot sealed/private evaluation with no post-result tuning',
            ],
            'honest_wording':{
                'oracle_roi':'event-only neural TTC under matched oracle-box/object-ROI preprocessing',
                'strict_model_causal':'model-prefix-causal event TTC under matched oracle-ROI preprocessing' if prefix_ok else None,
                'forbidden':'end-to-end no-oracle SOTA / strict streaming end-to-end causal',
            },
            'sota_claim_authorized':False,
        }
        sign_artifact(result); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    except Exception as exc:
        print(f'claim readiness failed: {type(exc).__name__}: {exc}',file=sys.stderr); return 1
    print(json.dumps(result,indent=2,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
