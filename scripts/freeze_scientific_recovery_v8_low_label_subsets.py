#!/usr/bin/env python
"""Freeze exact nested low-label token IDs before any V8 D0--D4 training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa:E402
from e_jepa_ttc.data.scientific_recovery_v8_jepa_data import open_jepa_dataset  # noqa:E402
from e_jepa_ttc.evaluation.scientific_recovery_v8_jepa_attribution import nested_low_label_tokens  # noqa:E402


def _signed(path:Path)->dict[str,Any]:
    value=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value,dict) or not verify_artifact_hash(value): raise ValueError(f'unsigned artifact: {path}')
    return value


def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--winner-artifact',type=Path,required=True)
    p.add_argument('--cache-manifest',type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=ROOT/'configs/protocol/scientific_recovery_v8_temporal.json')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/scientific_recovery_v8/jepa/low_label_subsets.json')
    a=p.parse_args()
    try:
        winner,protocol=_signed(a.winner_artifact),_signed(a.protocol)
        dataset=open_jepa_dataset(cache_manifest=a.cache_manifest,protocol_path=a.protocol)
        result:dict[str,Any]={}
        fractions=(0.01,0.05,0.10,0.25,1.0)
        for fold in (0,1,2):
            records=[dataset[i] for i in range(len(dataset)) if int(dataset[i]['outer_fold'])!=fold]
            subsets=nested_low_label_tokens(records,fractions=fractions,seed=7)
            result[str(fold)]={str(f):sorted(str(x) for x in subsets[f]) for f in fractions}
            previous:set[str]=set()
            for fraction in fractions:
                current=set(result[str(fold)][str(fraction)])
                if not previous.issubset(current): raise ValueError('low-label subsets are not nested')
                previous=current
        payload={
            'artifact_type':'scientific_recovery_v8_low_label_subsets_v1','status':'frozen_before_jepa_training',
            'winner_artifact_sha256':winner['artifact_sha256'],'protocol_artifact_sha256':protocol['artifact_sha256'],
            'fractions':list(fractions),'seed':7,'folds':result,'closed_evaluation':protocol.get('closed_evaluation',{})}
        sign_artifact(payload); a.output.parent.mkdir(parents=True,exist_ok=True)
        tmp=a.output.with_suffix(a.output.suffix+'.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8'); tmp.replace(a.output)
        print(json.dumps({'status':'completed','output':str(a.output)},sort_keys=True)); return 0
    except (OSError,ValueError,KeyError) as e: p.exit(2,f'V8 low-label freeze failed closed: {type(e).__name__}: {e}\n')
if __name__=='__main__': raise SystemExit(main())
