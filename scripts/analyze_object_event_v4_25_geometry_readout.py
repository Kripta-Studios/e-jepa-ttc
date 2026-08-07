#!/usr/bin/env python3
"""Train-only meta-CV of an explicit geometry-conditioned TTC readout."""
from __future__ import annotations
import argparse, json, shutil, sys, time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
for candidate in (ROOT,ROOT/'src'):
    if str(candidate) not in sys.path: sys.path.insert(0,str(candidate))

from scripts.train_e_jepa_object_event_v4_6 import _materialize
from scripts.train_e_jepa_object_event_v4_8 import _load_config as _load_v48_config
from scripts.train_e_jepa_object_event_v4_12 import _align_ensemble,_load_backbone,_read_ensemble
from scripts.train_e_jepa_object_event_v4_16 import _json_safe,_metrics,_resolve_device
from scripts.analyze_object_event_v4_22_encoder_geometry import _score_backbone
from scripts.analyze_object_event_v4_24_orchestrator import _evaluate_arm_cv, _combine_seed_results, _load_config as _load_v424_config, _sequence_folds as _v424_sequence_folds
from e_jepa_ttc.training.object_event_v4_25 import (
    RidgeSpec,apply_geometry_calibration,design_matrix,fit_geometry_calibration,
    nonnegative_ridge_with_prior,predict_readout,
)


def _sequence_folds(sequence_ids: np.ndarray, fold_count:int, seed:int)->list[np.ndarray]:
    unique=np.array(sorted(set(str(x) for x in sequence_ids)),dtype=object)
    rng=np.random.default_rng(seed); rng.shuffle(unique)
    groups=[set(unique[i::fold_count].tolist()) for i in range(fold_count)]
    seq=np.asarray([str(x) for x in sequence_ids],dtype=object)
    return [np.flatnonzero(np.isin(seq,list(g))) for g in groups]


def _parse_champions(values:list[str])->dict[int,Path]:
    out={}
    for item in values:
        seed_text,path_text=item.split('=',1); out[int(seed_text)]=Path(path_text)
    if sorted(out)!=[7,13,23]: raise ValueError('exact champion seeds 7,13,23 required')
    return out


def _load_champion(v48_config:Path,path:Path,device:torch.device)->Any:
    payload=torch.load(path,map_location='cpu')
    source=Path(payload['source_v422_checkpoint'])
    backbone,_=_load_backbone(v48_config_path=v48_config,checkpoint_path=source)
    backbone.load_state_dict(payload['model_state_dict'],strict=True)
    return backbone.to(device).eval()


def _score_all(champions:dict[int,Path],split:Any,*,v48_config:Path,model_config:Any,batch_size:int,device:torch.device)->tuple[np.ndarray,np.ndarray,list[dict[str,float]]]:
    divs=[]; verts=[]; diagnostics=[]
    for seed,path in champions.items():
        print(f'[v4.25] scoring champion seed={seed}',flush=True)
        backbone=_load_champion(v48_config,path,device)
        div,vert,diag=_score_backbone(backbone,split,batch_size=batch_size,config=model_config,device=device)
        divs.append(div); verts.append(vert); diagnostics.append({'seed':seed,**{k:float(v) for k,v in diag.items()}})
        del backbone
        if device.type=='cuda': torch.cuda.empty_cache()
    return np.median(np.stack(divs),axis=0),np.median(np.stack(verts),axis=0),diagnostics


def _candidate_objective(metrics:dict[str,Any],selection:dict[str,float])->tuple[bool,float]:
    target_std=max(float(metrics['target_std']),1e-8)
    eligible=(float(metrics['positive_accuracy'])>=selection['minimum_positive_accuracy'] and
              float(metrics['negative_accuracy'])>=selection['minimum_negative_accuracy'] and
              float(metrics['minimum_sequence_pearson'])>=selection['minimum_sequence_pearson'])
    objective=(selection['pearson_weight']*float(metrics['pearson'])+
               selection['balanced_sign_weight']*float(metrics['balanced_sign_accuracy'])+
               selection['minimum_sequence_pearson_weight']*float(metrics['minimum_sequence_pearson'])+
               selection['minimum_sequence_negative_weight']*float(metrics['minimum_sequence_negative_accuracy'])-
               selection['normalized_mae_penalty']*float(metrics['expansion_mae'])/target_std)
    return eligible,float(objective)


def _fit_candidate(spec:RidgeSpec,baseline:np.ndarray,div_raw:np.ndarray,vert_raw:np.ndarray,target:np.ndarray,fit_idx:np.ndarray,eval_idx:np.ndarray):
    if spec.name == 'baseline_control':
        return np.asarray(baseline[eval_idx],dtype=np.float64),{'features':('baseline',),'coefficients':[1.0],'divergence_calibration':None,'vertical_calibration':None}
    div_cal=fit_geometry_calibration(div_raw[fit_idx],target[fit_idx])
    vert_cal=fit_geometry_calibration(vert_raw[fit_idx],target[fit_idx])
    div_fit=apply_geometry_calibration(div_raw[fit_idx],div_cal); div_eval=apply_geometry_calibration(div_raw[eval_idx],div_cal)
    vert_fit=apply_geometry_calibration(vert_raw[fit_idx],vert_cal); vert_eval=apply_geometry_calibration(vert_raw[eval_idx],vert_cal)
    x_fit,names,prior=design_matrix(baseline[fit_idx],div_fit,vert_fit,spec.features)
    x_eval,_,_=design_matrix(baseline[eval_idx],div_eval,vert_eval,spec.features)
    coeff=nonnegative_ridge_with_prior(x_fit,target[fit_idx],ridge=spec.ridge,prior=prior)
    return predict_readout(x_eval,coeff),{'features':names,'coefficients':coeff.tolist(),'divergence_calibration':asdict(div_cal),'vertical_calibration':asdict(vert_cal)}


def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-manifest',type=Path,required=True); p.add_argument('--v48-config',type=Path,required=True)
    p.add_argument('--v424-config',type=Path,required=True); p.add_argument('--v424-summary',type=Path,required=True)
    p.add_argument('--ensemble-train',type=Path,required=True); p.add_argument('--ensemble-validation',type=Path,required=True)
    p.add_argument('--champion-checkpoint',action='append',required=True); p.add_argument('--adapted-checkpoint',action='append',required=True); p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',default='cuda'); p.add_argument('--force',action='store_true'); args=p.parse_args()
    if args.output_dir.exists():
        if not args.force: raise FileExistsError(f'{args.output_dir} exists; use --force')
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True); started=time.perf_counter()
    raw=yaml.safe_load((ROOT/'configs/experiment/e_jepa_garl_object_event_geometry_readout_v4_25.yaml').read_text(encoding='utf-8'))
    meta=raw['meta']; selection={k:float(v) for k,v in raw['selection'].items()}
    specs=[RidgeSpec(str(r['name']),tuple(r['features']),float(r['ridge'])) for r in raw['readouts']]
    model_config, orch, arms, _, geometry_loss_config, _ = _load_v424_config(args.v424_config)
    if 'geometry_only_regularized' not in arms: raise RuntimeError('v4.24 geometry_only_regularized arm missing')
    champions=_parse_champions(args.champion_checkpoint); adapted=_parse_champions(args.adapted_checkpoint); device=_resolve_device(args.device)
    base_config,_,_,_,_=_load_v48_config(args.v48_config)
    train_split,train_manifest=_materialize(args.cache_manifest,'train',input_size=base_config.input_size)
    val_split,val_manifest=_materialize(args.cache_manifest,'validation',input_size=base_config.input_size)
    train_frame=_align_ensemble(train_split,_read_ensemble(args.ensemble_train)); val_frame=_align_ensemble(val_split,_read_ensemble(args.ensemble_validation))
    # Rebuild cross-fitted geometry features: 3 independent seeds x 3 grouped folds.
    # Each row's geometry features come from a representation that did not train on that sequence.
    folds=_v424_sequence_folds(train_frame['sequence_id'].astype(str).to_numpy(),int(meta['fold_count']),int(meta['seed']))
    per_seed=[]
    for seed in (7,13,23):
        print(f'[v4.25] cross-fitting geometry seed={seed} over {len(folds)} grouped folds',flush=True)
        result=_evaluate_arm_cv('geometry_only_regularized',arms['geometry_only_regularized'],seed,checkpoint=adapted[seed],
            split=train_split,frame=train_frame,folds=folds,v48_config=args.v48_config,model_config=model_config,
            geometry_loss_config=geometry_loss_config,orch=orch,device=device,output_dir=args.output_dir)
        per_seed.append(result)
    oof=_combine_seed_results('geometry_only_regularized',per_seed,train_frame)
    oof_div=np.asarray(oof.divergence,dtype=np.float64); oof_vert=np.asarray(oof.vertical,dtype=np.float64)
    # Full-train champion representations are used only after the readout family is selected.
    train_div,train_vert,train_diag=_score_all(champions,train_split,v48_config=args.v48_config,model_config=model_config,batch_size=8,device=device)
    val_div,val_vert,val_diag=_score_all(champions,val_split,v48_config=args.v48_config,model_config=model_config,batch_size=8,device=device)
    target=train_frame['target_expansion'].to_numpy(dtype=np.float64); baseline=train_frame['fused_prediction_expansion'].to_numpy(dtype=np.float64)
    all_idx=np.arange(len(train_frame),dtype=np.int64); ranking=[]; predictions={}; fold_records=[]
    for spec in specs:
        oof=np.full(len(train_frame),np.nan,dtype=np.float64); local_records=[]
        for fold,held in enumerate(folds):
            fit=np.setdiff1d(all_idx,held,assume_unique=True)
            pred,record=_fit_candidate(spec,baseline,oof_div,oof_vert,target,fit,held); oof[held]=pred
            record.update({'readout':spec.name,'fold':fold,'held_out_sequences':sorted(train_frame.iloc[held]['sequence_id'].astype(str).unique().tolist())})
            local_records.append(record)
        metrics,_=_metrics(train_frame,oof,minimum_negatives=20); eligible,objective=_candidate_objective(metrics,selection)
        ranking.append({'readout':spec.name,'eligible':eligible,'objective':objective,**{k:metrics[k] for k in ('pearson','expansion_mae','positive_accuracy','negative_accuracy','balanced_sign_accuracy','minimum_sequence_pearson','minimum_sequence_negative_accuracy')}})
        predictions[spec.name]=oof; fold_records.extend(local_records)
    ranking.sort(key=lambda r:(bool(r['eligible']),float(r['objective'])),reverse=True)
    baseline_row=next(r for r in ranking if r['readout']=='baseline_control'); winner=ranking[0]
    min_gain=float(meta['minimum_objective_gain_over_baseline'])
    if winner['readout']!='baseline_control' and float(winner['objective']) < float(baseline_row['objective'])+min_gain:
        winner=baseline_row
    chosen=next(s for s in specs if s.name==winner['readout'])
    print(f"[v4.25] selected readout={chosen.name} objective={winner['objective']:.4f}",flush=True)
    # Final coefficients/calibration are fit on train only. Validation is touched once here.
    fit_idx=np.arange(len(train_frame),dtype=np.int64); eval_idx=np.arange(len(val_frame),dtype=np.int64)
    div_cal=fit_geometry_calibration(train_div,target); vert_cal=fit_geometry_calibration(train_vert,target)
    tr_div_exp=apply_geometry_calibration(train_div,div_cal); va_div_exp=apply_geometry_calibration(val_div,div_cal)
    tr_vert_exp=apply_geometry_calibration(train_vert,vert_cal); va_vert_exp=apply_geometry_calibration(val_vert,vert_cal)
    x_train,names,prior=design_matrix(baseline,tr_div_exp,tr_vert_exp,chosen.features)
    val_baseline=val_frame['fused_prediction_expansion'].to_numpy(dtype=np.float64)
    x_val,_,_=design_matrix(val_baseline,va_div_exp,va_vert_exp,chosen.features)
    if chosen.name == 'baseline_control':
        coeff=np.asarray([1.0],dtype=np.float64)
    else:
        coeff=nonnegative_ridge_with_prior(x_train,target,ridge=chosen.ridge,prior=prior)
    train_pred=predict_readout(x_train,coeff); val_pred=predict_readout(x_val,coeff)
    train_metrics,_=_metrics(train_frame,train_pred,minimum_negatives=20); val_metrics,val_per_seq=_metrics(val_frame,val_pred,minimum_negatives=20)
    baseline_metrics,_=_metrics(val_frame,val_baseline,minimum_negatives=20)
    if (float(val_metrics['pearson'])>=float(baseline_metrics['pearson']) and float(val_metrics['balanced_sign_accuracy'])>=float(baseline_metrics['balanced_sign_accuracy'])):
        recommendation='anchored_geometry_readout_supported_integrate_as_v425_model_readout'
    elif float(val_metrics['pearson'])>=float(baseline_metrics['pearson'])-0.01 and float(val_metrics['negative_accuracy'])>float(baseline_metrics['negative_accuracy']):
        recommendation='geometry_readout_tradeoff_promising_lock_then_multiseed_confirm'
    else:
        recommendation='constrained_readout_insufficient_stop_posthoc_readouts_train_explicit_lhr_head'
    pd.DataFrame(ranking).to_csv(args.output_dir/'train_only_readout_ranking.csv',index=False)
    pd.DataFrame(fold_records).to_json(args.output_dir/'meta_fold_coefficients.jsonl',orient='records',lines=True)
    out=val_frame.loc[:,['sequence_id','sample_token','track_id','target_expansion']].copy(); out['baseline_prediction_expansion']=val_baseline; out['v425_prediction_expansion']=val_pred; out['divergence_expansion_proxy']=va_div_exp; out['vertical_expansion_proxy']=va_vert_exp; out.to_csv(args.output_dir/'validation_predictions.csv',index=False)
    val_per_seq.to_csv(args.output_dir/'validation_per_sequence.csv',index=False)
    summary={'artifact_type':'object_event_v4_25_anchored_geometry_readout','status':'completed','elapsed_seconds':time.perf_counter()-started,
      'selected_readout':chosen.name,'selected_features':list(names),'selected_ridge':chosen.ridge,'selected_coefficients':coeff.tolist(),
      'train_only_ranking':ranking,'train_only_geometry_calibration':{'divergence':asdict(div_cal),'vertical':asdict(vert_cal)},
      'train_metrics':train_metrics,'baseline_validation_metrics':baseline_metrics,'validation_metrics':val_metrics,
      'diagnostics':{'train_seed_scoring':train_diag,'validation_seed_scoring':val_diag},
      'decision':{'recommendation':recommendation,'comparisons':{'baseline_pearson':baseline_metrics['pearson'],'v425_pearson':val_metrics['pearson'],'baseline_negative_accuracy':baseline_metrics['negative_accuracy'],'v425_negative_accuracy':val_metrics['negative_accuracy'],'baseline_balanced_sign':baseline_metrics['balanced_sign_accuracy'],'v425_balanced_sign':val_metrics['balanced_sign_accuracy']}},
      'scientific_contract':{'readout_candidate_selection_train_only':True,'cross_fitted_geometry_features_3seeds_3folds':True,'validation_evaluated_once_after_selection':True,'nonnegative_zero_bias_coefficients':True,'baseline_anchor_prior':True,'no_hidden_mlp_or_sign_router':True,'boxes_not_forward_features':True,'official_eap_test_not_opened':True,'evttc_not_opened':True},
      'train_manifest':train_manifest,'validation_manifest':val_manifest}
    (args.output_dir/'summary.json').write_text(json.dumps(_json_safe(summary),indent=2),encoding='utf-8')
    print(json.dumps(_json_safe({'status':'completed','selected_readout':chosen.name,'coefficients':coeff.tolist(),'validation_metrics':val_metrics,'decision':summary['decision']}),indent=2))
    return 0

if __name__=='__main__': raise SystemExit(main())
