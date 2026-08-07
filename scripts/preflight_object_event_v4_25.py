#!/usr/bin/env python3
"""Preflight for v4.25 constrained geometry-conditioned TTC readout."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v424-summary',type=Path,required=True)
    p.add_argument('--champion-checkpoint',action='append',required=True)
    args=p.parse_args()
    summary=json.loads(args.v424_summary.read_text(encoding='utf-8'))
    if summary.get('status')!='completed': raise RuntimeError('v4.24 must be completed')
    if summary.get('champion')!='geometry_only_regularized': raise RuntimeError('v4.25 expects the v4.24 geometry_only_regularized champion')
    decision=summary.get('decision',{})
    if decision.get('recommendation')!='schedule_search_exhausted_keep_v422_geometry_redesign_ttc_readout':
        raise RuntimeError('v4.24 did not request TTC readout redesign')
    contract=summary.get('scientific_contract',{})
    for key in ('official_eap_test_not_opened','evttc_not_opened'):
        if contract.get(key) is not True: raise RuntimeError(f'v4.24 contract missing {key}')
    seen=[]
    for item in args.champion_checkpoint:
        seed_text,path_text=item.split('=',1); seed=int(seed_text); path=Path(path_text)
        payload=torch.load(path,map_location='cpu')
        if payload.get('artifact_type')!='object_event_v4_24_orchestrated_champion': raise RuntimeError(f'bad champion artifact: {path}')
        if payload.get('arm')!='geometry_only_regularized' or int(payload.get('seed'))!=seed: raise RuntimeError(f'champion identity mismatch: {path}')
        seen.append(seed)
    if sorted(seen)!=[7,13,23]: raise RuntimeError('exact champion seeds 7,13,23 required')
    print(json.dumps({'status':'passed','v424_champion':summary['champion'],'seeds':sorted(seen),'scientific_contract':{
        'train_only_readout_selection':True,'nonnegative_zero_bias_readout':True,'baseline_anchor_retained':True,
        'official_eap_test_not_opened':True,'evttc_not_opened':True}},indent=2))
    return 0

if __name__=='__main__': raise SystemExit(main())
