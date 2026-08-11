#!/usr/bin/env python
from __future__ import annotations
import argparse, json, statistics, sys
from datetime import UTC, datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e_jepa_ttc.artifacts.hashing import sign_artifact


def metric(d, path):
    cur=d
    for k in path: cur=cur[k]
    return float(cur)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--summary', action='append', required=True)
    ap.add_argument('--output', required=True)
    args=ap.parse_args()
    rows=[]
    for p in args.summary:
        d=json.loads(Path(p).read_text(encoding='utf-8'))
        rows.append({
            'path':str(Path(p)),
            'git_commit':d.get('git_commit'),
            'seed':int(d['training_config']['seed']),
            'sequence_macro_MiD':float(d['selection']['sequence_macro_MiD']),
            'failure_rate_pct':float(d['selection']['failure_rate_pct']),
            'log_ratio_pearson':float(d['validation_metrics']['log_ratio_pearson']),
            'delta_log_height_physical_pearson':metric(d,['validation_metrics','geometry_diagnostics','delta_log_height_vs_physical','global','pearson']),
            'abs_log_height_macro_pearson':metric(d,['validation_metrics','geometry_diagnostics','absolute_log_height','macro_by_sequence','pearson']),
        })
    rows=sorted(rows,key=lambda x:x['seed'])
    if len({r['seed'] for r in rows}) != len(rows): raise ValueError('duplicate seeds')
    metrics={}
    for k in ['sequence_macro_MiD','failure_rate_pct','log_ratio_pearson','delta_log_height_physical_pearson','abs_log_height_macro_pearson']:
        vals=[r[k] for r in rows]
        metrics[k]={'mean':statistics.fmean(vals),'stdev':statistics.stdev(vals) if len(vals)>1 else 0.0,'min':min(vals),'max':max(vals)}
    payload={'artifact_type':'a4_s1_replication_summary_v1','created_at_utc':datetime.now(UTC).isoformat(),'rows':rows,'metrics':metrics,'descriptive_only':True,'private_test_opened':False}
    sign_artifact(payload)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps(payload,indent=2,sort_keys=True))

if __name__=='__main__': main()
