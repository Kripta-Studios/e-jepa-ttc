#!/usr/bin/env python
"""Diagnostic replication gate for the surprising A5 seed-7 signal."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[1]

def read_json(path: Path) -> dict[str, Any]:
    data=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data,dict): raise ValueError(f'{path} is not a JSON object')
    return data

def transport_metric(summary: dict[str,Any], name: str, scope: str='global') -> float:
    node=summary['validation_metrics']['transport_diagnostics']['against_physical_log_ratio'][name]
    if scope=='global': return float(node['global']['pearson'])
    return float(node['macro_by_sequence']['pearson'])

def evaluate(summary: dict[str,Any], protocol: dict[str,Any]) -> dict[str,Any]:
    parent=protocol['parent_metrics']; gate=protocol['diagnostic_replication_gate']
    ratio=float(summary['validation_metrics']['log_ratio_pearson'])
    mid=float(summary['selection']['sequence_macro_MiD'])
    failure=float(summary['selection']['failure_rate_pct'])
    fg=summary['validation_metrics']['transport_diagnostics']['against_physical_log_ratio']['foreground_divergence_y']
    fg_global=float(fg['global']['pearson']); fg_macro=float(fg['macro_by_sequence']['pearson'])
    positive=sum(float(v['pearson'])>0 for v in fg['per_sequence'].values())
    checks={
      'log_ratio': ratio >= float(parent['log_ratio_pearson'])+float(gate['log_ratio_min_gain_vs_a4']),
      'MiD': mid <= float(parent['sequence_macro_MiD'])*float(gate['max_MiD_fraction_of_a4']),
      'failure': failure <= float(gate['maximum_failure_rate_pct']),
      'foreground_divergence_y_global': fg_global >= float(gate['foreground_divergence_y_global_min']),
      'foreground_divergence_y_macro': fg_macro >= float(gate['foreground_divergence_y_macro_min']),
      'foreground_divergence_y_sequences': positive >= int(gate['minimum_positive_foreground_divergence_y_sequences']),
    }
    return {'seed':int(summary['training_config']['seed']),'passed':all(checks.values()),'checks':checks,'metrics':{'log_ratio_pearson':ratio,'sequence_macro_MiD':mid,'failure_rate_pct':failure,'foreground_divergence_y_global':fg_global,'foreground_divergence_y_macro':fg_macro,'positive_foreground_divergence_y_sequences':positive}}

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--protocol',required=True); ap.add_argument('--summary',action='append',required=True); ap.add_argument('--output',required=True); args=ap.parse_args()
    protocol=yaml.safe_load((ROOT/args.protocol).read_text(encoding='utf-8'))
    rows=[evaluate(read_json(ROOT/p),protocol) for p in args.summary]
    expected=set(map(int,protocol['diagnostic_replication_gate']['seeds']))
    if {r['seed'] for r in rows} != expected: raise ValueError('diagnostic replication must contain exactly seeds 7,13,23')
    passing=sum(r['passed'] for r in rows); required=int(protocol['diagnostic_replication_gate']['minimum_passing_seeds'])
    payload={'artifact_type':'a5_transport_signal_diagnostic_replication_v1','passed':passing>=required,'passing_seeds':passing,'required_passing_seeds':required,'runs':sorted(rows,key=lambda x:x['seed']),'private_test_opened':False,'interpretation':'diagnostic replication only; does not overwrite the failed original A5 mechanistic gate'}
    out=ROOT/args.output; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'passed':payload['passed'],'passing_seeds':passing,'required':required}))
    return 0 if payload['passed'] else 6
if __name__=='__main__': raise SystemExit(main())
