#!/usr/bin/env python
"""Gate A5-ANCHOR: temporal gain while preserving the inherited A4 geometry."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from typing import Any
import yaml
ROOT=Path(__file__).resolve().parents[1]

def loadj(p:str)->dict[str,Any]: return json.loads((ROOT/p).read_text(encoding='utf-8'))

def metric(summary:dict[str,Any], path:list[str])->float:
    x:Any=summary
    for k in path:x=x[k]
    return float(x)

def eval_run(s:dict[str,Any], protocol:dict[str,Any])->dict[str,Any]:
    parent=protocol['parent_metrics']; g=protocol['anchor_gate']; tol=float(g['geometry_absolute_tolerance'])
    ratio=metric(s,['validation_metrics','log_ratio_pearson']); mid=metric(s,['selection','sequence_macro_MiD']); failure=metric(s,['selection','failure_rate_pct'])
    dh=metric(s,['validation_metrics','geometry_diagnostics','delta_log_height_vs_physical','global','pearson']); dhm=metric(s,['validation_metrics','geometry_diagnostics','delta_log_height_vs_physical','macro_by_sequence','pearson']); ahm=metric(s,['validation_metrics','geometry_diagnostics','absolute_log_height','macro_by_sequence','pearson'])
    fg=s['validation_metrics']['transport_diagnostics']['against_physical_log_ratio']['foreground_divergence_y']; fgg=float(fg['global']['pearson']); fgm=float(fg['macro_by_sequence']['pearson']); pos=sum(float(v['pearson'])>0 for v in fg['per_sequence'].values())
    init=s.get('initialization',{}); source=(init.get('validated_source') or {}) if isinstance(init,dict) else {}
    checks={
      'initialization_source': source.get('sha256')==protocol['source_evidence']['a4_checkpoint_sha256'],
      'complete_encoder_loaded': init.get('complete_encoder_loaded') is True,
      'encoder_frozen': init.get('freeze_encoder') is True and int(init.get('frozen_parameter_count',0))>0,
      'log_ratio': ratio >= float(parent['log_ratio_pearson'])+float(g['log_ratio_min_gain_vs_a4']),
      'MiD': mid <= float(parent['sequence_macro_MiD'])*float(g['max_MiD_fraction_of_a4']),
      'failure': failure <= float(g['maximum_failure_rate_pct']),
      'delta_height_preserved': abs(dh-float(parent['delta_log_height_vs_physical'])) <= tol,
      'delta_height_macro_preserved': abs(dhm-float(parent['delta_log_height_vs_physical_macro'])) <= tol,
      'absolute_height_macro_preserved': abs(ahm-float(parent['absolute_log_height_macro'])) <= tol,
      'foreground_divergence_y_global': fgg >= float(g['foreground_divergence_y_global_min']),
      'foreground_divergence_y_macro': fgm >= float(g['foreground_divergence_y_macro_min']),
      'foreground_divergence_y_sequences': pos >= int(g['minimum_positive_foreground_divergence_y_sequences']),
    }
    return {'seed':int(s['training_config']['seed']),'passed':all(checks.values()),'checks':checks,'metrics':{'log_ratio_pearson':ratio,'sequence_macro_MiD':mid,'failure_rate_pct':failure,'delta_log_height_vs_physical':dh,'delta_log_height_vs_physical_macro':dhm,'absolute_log_height_macro':ahm,'foreground_divergence_y_global':fgg,'foreground_divergence_y_macro':fgm,'positive_foreground_divergence_y_sequences':pos},'initialization':init}

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--protocol',required=True); ap.add_argument('--summary',action='append',required=True); ap.add_argument('--output',required=True); ap.add_argument('--required-passes',type=int); a=ap.parse_args()
    protocol=yaml.safe_load((ROOT/a.protocol).read_text(encoding='utf-8')); rows=[eval_run(loadj(p),protocol) for p in a.summary]; passes=sum(r['passed'] for r in rows); required=a.required_passes if a.required_passes is not None else int(protocol['anchor_gate']['minimum_passing_seeds']); passed=passes>=required
    payload={'artifact_type':'a5_anchor_replication_gate_v1','passed':passed,'passing_seeds':passes,'required_passing_seeds':required,'runs':sorted(rows,key=lambda x:x['seed']),'private_test_opened':False,'interpretation':'A5-ANCHOR must add temporal signal while inherited A4 foreground geometry remains numerically unchanged'}
    out=ROOT/a.output; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8'); print(json.dumps({'passed':passed,'passing_seeds':passes,'required':required})); return 0 if passed else 7
if __name__=='__main__': raise SystemExit(main())
