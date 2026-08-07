from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument('--v412-summary',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    a=p.parse_args()
    for path in (a.v412_summary,a.predictions):
        if not path.is_file(): raise FileNotFoundError(path)
    s=json.loads(a.v412_summary.read_text(encoding='utf-8'))
    if s.get('artifact_type')!='object_event_v4_12_reversal_balanced_directional_sign':
        raise RuntimeError('wrong v4.12 artifact')
    if not s.get('scientific_contract',{}).get('exact_descriptor_antisymmetry'):
        raise RuntimeError('v4.12 exact odd symmetry not proven')
    print(json.dumps({'status':'passed','v412_status':s.get('status'),'rows_source':str(a.predictions)},indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())
