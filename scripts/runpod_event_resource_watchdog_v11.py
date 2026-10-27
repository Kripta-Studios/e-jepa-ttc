"""Admit resource pauses during E_CTX inputs after a completed, verified PAIR_E."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

V10_SHA='3015df79d8220041328382133447e30c4053e2646f74eff2b40f2f4c08bc03ee'

def digest(p: Path) -> str:
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def read(p: Path) -> dict[str,Any]:
    if p.is_symlink():raise ValueError('Symlink evidence rejected')
    return json.loads(p.read_text())

def completed_pair_input_pause(task: dict[str,Any], managed: dict[str,Any]) -> bool:
    """No optimizer is active here: reject any E_CTX fit state or incomplete PAIR."""
    try:
        run=Path(task['run']);r=int(task['rotation']);base=run/f'rotations/{r}'
        pp=run/f'EVENT_DOWNSTREAM_PAUSE_R{r}.json';p=read(pp)
        if not (pp.stat().st_mtime>=float(managed['started_epoch'])-2
                and p.get('schema')=='ejepa_event_downstream_pause_v1'
                and p.get('status')=='PAUSED_RECOVERABLE'
                and p.get('reason')=='resource reserve unavailable'
                and p.get('manual') is False and p.get('rotation')==r
                and p.get('pause_classification')=='AUTOMATIC_RESOURCE_PAUSE'
                and p.get('owner_pid')==managed['pid']
                and p.get('owner_start_ticks')==managed['start_ticks']
                and isinstance(p.get('resources'),dict)):return False
        ctx=base/'fits/E_CTX'
        if ctx.exists() and any(ctx.iterdir()):return False
        fit=base/'fits/PAIR_E';rp=fit/'CHECKPOINT_RECEIPT.json'
        receipt=read(rp);pointer=read(fit/'CHECKPOINT_POINTER.json');journal=read(fit/'UPDATE_JOURNAL.json');endpoint=read(fit/'ENDPOINT_ADMISSION.json')
        completed=receipt.get('completed_updates');identity=receipt.get('identity_sha256')
        checkpoint=Path(receipt['checkpoint_path']).resolve(strict=True)
        version=(fit/'checkpoint_versions'/pointer['version']).resolve(strict=True)
        if not checkpoint.is_relative_to(fit.resolve()) or not version.is_relative_to(fit.resolve()):return False
        checksum=digest(checkpoint)
        if not (receipt.get('status')=='COMPLETE' and receipt.get('scientific_endpoint') is True
                and receipt.get('complete_state') is True and receipt.get('accumulation_index')==0
                and isinstance(completed,int) and completed>0
                and pointer.get('completed_updates')==journal.get('completed_updates')==journal.get('durable_updates')==completed
                and journal.get('pending_update_upper')==0
                and receipt.get('recovery_upper')==journal.get('recovery_upper')
                and checksum==digest(version)==receipt.get('checkpoint_sha256')==pointer.get('checkpoint_sha256')==endpoint.get('checkpoint_sha256')
                and identity==pointer.get('identity_sha256')==journal.get('identity_sha256')==endpoint.get('checkpoint_identity_sha256')
                and receipt.get('execution_binding_sha256')==pointer.get('execution_binding_sha256')==journal.get('execution_binding_sha256')
                and endpoint.get('status')=='PASS' and endpoint.get('scientific_endpoint') is True
                and endpoint.get('fit_id')=='PAIR_E' and endpoint.get('rotation')==r
                and endpoint.get('updates')==completed and endpoint.get('checkpoint_receipt_sha256')==digest(rp)
                and endpoint.get('admission_payload_sha256')==canonical({k:v for k,v in endpoint.items() if k!='admission_payload_sha256'})):return False
        parts=base/'refiner/E_CTX/parts';binding=read(parts/'BINDING.json')
        if binding.get('schema')!='scientific_oof_event_history_v1':return False
        for parent in ['E_A5','E_C2F','PAIR_E']:
            parent_receipt=read(base/f'fits/{parent}/CHECKPOINT_RECEIPT.json')
            if parent_receipt.get('status')!='COMPLETE' or binding['parents'].get(parent)!=parent_receipt.get('checkpoint_sha256'):return False
        names=sorted(parts.glob('part_*.json'))
        if not names:return False
        latest=names[-1];fragment=read(latest);data=latest.with_suffix('.npz')
        return (fragment.get('optimizer_updates')==0 and fragment.get('rows')==1
                and fragment.get('binding_sha256')==canonical(binding)
                and not data.is_symlink() and digest(data)==fragment.get('sha256'))
    except (OSError,ValueError,TypeError,KeyError):return False

def install_v11() -> Any:
    source=Path(__file__).resolve();path=source.with_name('runpod_event_resource_watchdog_v10.py')
    if digest(path)!=V10_SHA:raise ValueError('Frozen V10 differs')
    spec=importlib.util.spec_from_file_location('ejepa_watchdog_v10_v11',path)
    wrapper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=wrapper;spec.loader.exec_module(wrapper)
    module=wrapper.install_v10();old_validate=module.validate_config;old_terminal=module.downstream_terminal
    def validate(path: Path) -> dict[str,Any]:
        config=old_validate(path)
        if config.get('watchdog_completed_pair_input_pause_sha256')!=digest(source):raise ValueError('V11 source differs')
        return config
    def terminal(task: dict[str,Any],managed: dict[str,Any]) -> tuple[str,str]:
        result=old_terminal(task,managed)
        if result in [('BLOCKED','STALE_OR_PARTIAL_HEAD_CHECKPOINT_EVIDENCE'),('BLOCKED','EVENT_PIPELINE_RECEIPT_NOT_FRESH')]:
            if completed_pair_input_pause(task,managed):return 'RESTARTABLE','VERIFIED_COMPLETED_PAIR_E_CTX_INPUT_RESOURCE_PAUSE'
        return result
    module.validate_config=validate;module.downstream_terminal=terminal;module.__file__=str(source)
    return module

if __name__=='__main__':raise SystemExit(install_v11().main())
