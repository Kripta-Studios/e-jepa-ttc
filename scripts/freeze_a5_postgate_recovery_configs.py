#!/usr/bin/env python
"""Freeze A5 post-gate diagnostic-replication and A5-ANCHOR configs."""
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
from typing import Any
import yaml

ROOT=Path(__file__).resolve().parents[1]

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def read_yaml(path:Path)->dict[str,Any]:
    d=yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(d,dict): raise ValueError(f'{path} is not a YAML mapping')
    return d

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--base-config-dir',default='artifacts/configs/a5_transport_suite_v3'); ap.add_argument('--protocol',default='configs/experiment/e_jepa_garl_event_causal_scale_a5_postgate_recovery_v1.yaml'); ap.add_argument('--output-dir',default='artifacts/configs/a5_postgate_recovery_v1'); args=ap.parse_args()
    protocol_path=ROOT/args.protocol; protocol=read_yaml(protocol_path); source=protocol['source_evidence']; anchor_contract=protocol['anchor_contract']
    a4_summary=ROOT/source['a4_summary']; a4=json.loads(a4_summary.read_text(encoding='utf-8'))
    if a4.get('artifact_sha256') != source['a4_summary_artifact_sha256']: raise ValueError('A4 summary signed identity differs from recovery preregistration')
    a4_ckpt=ROOT/source['a4_checkpoint']; observed=sha256(a4_ckpt)
    if observed != source['a4_checkpoint_sha256']: raise ValueError('A4 checkpoint SHA256 differs from recovery preregistration')
    seed7_summary=ROOT/source['observed_a5_seed7_summary']; a5=json.loads(seed7_summary.read_text(encoding='utf-8'))
    if a5.get('artifact_sha256') != source['observed_a5_seed7_artifact_sha256']: raise ValueError('observed A5 seed7 summary identity differs from preregistration')
    if int(a5['training_config']['seed']) != 7: raise ValueError('observed A5 source must be seed 7')
    if int(a5['model_architecture']['transport_radius']) != int(source['selected_transport_radius']): raise ValueError('A5 seed7 radius differs from preregistration')
    if float(a5['model_architecture']['transport_temperature']) != float(source['selected_transport_temperature']): raise ValueError('A5 seed7 temperature differs from preregistration')

    base_dir=ROOT/args.base_config_dir; out=ROOT/args.output_dir; out.mkdir(parents=True,exist_ok=True)
    files={}
    for seed in (7,13,23):
        base_path=base_dir/f'seed{seed}.yaml'
        base=read_yaml(base_path)
        # Exact diagnostic replication config is copied byte-for-byte semantically.
        diag_path=out/f'diagnostic_seed{seed}.yaml'; diag_path.write_text(yaml.safe_dump(base,sort_keys=False),encoding='utf-8',newline='\n'); files[diag_path.name]=sha256(diag_path)

        anchored=copy.deepcopy(base)
        anchored['experiment']['name']=f'e_jepa_garl_event_causal_scale_a5_anchor_v1_seed{seed}'
        anchored['experiment']['protocol_version']='causal_scale_a5_anchor_frozen_A4_endpoint_v1'
        anchored['experiment']['parent_arm']='A4_DINO_RELATIONAL_RGB_V2_seed7_checkpoint'
        anchored['experiment']['single_scientific_difference']='inherit_and_freeze_A4_endpoint_encoder_then_train_existing_A5_transport_fusion'
        tr=anchored['training']; tr['foreground_warmup_epochs']=0; tr['initialization_checkpoint']=source['a4_checkpoint']; tr['initialization_checkpoint_sha256']=source['a4_checkpoint_sha256']; tr['initialization_mode']='shape_compatible'; tr['freeze_encoder']=True
        dc=anchored['decision_contract']; change=dc['representation_change']; change['type']='a4_frozen_endpoint_plus_event_native_local_cross_time_transport'; change['A4_endpoint_encoder_inherited_from_checkpoint']=True; change['A4_endpoint_encoder_frozen_for_entire_run']=True
        dc['anchor_contract']={'initialization_mode':'shape_compatible','initialization_checkpoint':source['a4_checkpoint'],'initialization_checkpoint_sha256':source['a4_checkpoint_sha256'],'parent_encoder_frozen_for_entire_run':True,'geometry_must_equal_parent_by_construction':True,'foreground_warmup_epochs':0,'no_capacity_change':True,'no_resolution_change':True,'no_radius_or_temperature_change':True,'no_DINO_lambda_change':True}
        dc['anchor_gate']=copy.deepcopy(protocol['anchor_gate']); dc['postgate_recovery_protocol']=args.protocol
        anchor_path=out/f'anchor_seed{seed}.yaml'; anchor_path.write_text(yaml.safe_dump(anchored,sort_keys=False),encoding='utf-8',newline='\n'); files[anchor_path.name]=sha256(anchor_path)
    manifest={'artifact_type':'a5_postgate_recovery_frozen_configs_v1','protocol':args.protocol,'protocol_sha256':sha256(protocol_path),'source_A4_checkpoint_sha256':observed,'selected_transport_radius':source['selected_transport_radius'],'selected_transport_temperature':source['selected_transport_temperature'],'files':files,'private_test_opened':False}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(manifest,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
