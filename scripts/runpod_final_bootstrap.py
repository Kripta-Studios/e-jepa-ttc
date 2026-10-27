"""Additive path-only relocation and bounded final TRAIN40 continuation on Pod2."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PureWindowsPath
import sys
import time

BASE = Path('/opt/ejepa_event_bundle/workspace/e-jepa-ttc-v18-joint-measurement')
OPS = Path('/workspace/ejepa_final_operations')
INPUT = Path('/workspace/ejepa_final_inputs')
RUN = Path('/workspace/ejepa_final_runs')
RGB = Path('/workspace/ejepa_rgb_handoff/extracted_v10')
BUNDLE = Path('/opt/ejepa_event_bundle')
DEADLINE = 1793145600.0


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def setup():
    sys.path[:0] = [str(BASE/'src'), str(BASE)]
    # Same migration boundary as event_runner: the historical optional Windows
    # quota scanner has no sealed configuration on this host. Linux guards below
    # remain authoritative for this new run.
    import types
    import importlib
    package=types.ModuleType('operational.efficient_context')
    package.__file__=str(BASE/'operational/efficient_context/__init__.py')
    package.__package__='operational.efficient_context'
    package.__path__=[str(BASE/'operational/efficient_context')]
    sys.modules['operational.efficient_context']=package
    importlib.import_module('operational.efficient_context.resource_policy').install()
    from operational.runpod_migration.paths import PathMap
    from operational.runpod_migration.runtime import install
    manifest = read(BUNDLE/'BUNDLE_MANIFEST.json')
    windows = PathMap(manifest['roots'], str(BUNDLE))
    sources = {v['source_path']:str(RGB/k) for k,v in read(RGB/'MANIFEST.json')['files'].items()}

    class Mapping:
        def translate(self, value):
            raw = os.fspath(value)
            if raw in sources:
                return sources[raw]
            if PureWindowsPath(raw).drive:
                relative = windows.relative(raw)
            elif raw.startswith('/workspace/ejepa_bundle/'):
                relative = raw.removeprefix('/workspace/ejepa_bundle/')
            else:
                return raw.replace('\\','/')
            if relative == 'drives/E/eAP_dataset' or relative.startswith('drives/E/eAP_dataset/'):
                return str(Path('/workspace/ejepa_event_raw') / relative.removeprefix('drives/E/eAP_dataset').lstrip('/'))
            for root in (INPUT, BUNDLE, Path('/opt/ejepa_final_rgb'), Path('/opt/ejepa_event_authority_supplement_v2'), Path('/opt/ejepa_event_supplement')):
                candidate = root/relative
                if candidate.exists():
                    return str(candidate)
            return str(INPUT/relative)

    mapping = Mapping()
    from runpod_final_path_compat import install as install_path_composition
    install_path_composition((INPUT, BUNDLE, Path('/opt/ejepa_final_rgb'),
        Path('/opt/ejepa_event_authority_supplement_v2'), Path('/opt/ejepa_event_supplement')))
    install(mapping, budget_bytes=60_999_999_488, origin_git_commit=manifest['origin_git_commit'])
    from operational.final_refit import oof_lineage, shared_resources
    def normalize_role(value):
        if not isinstance(value, dict):
            return value
        result=dict(value)
        for field in ('manifest_path','rows_path'):
            if field in result:
                result[field]=str(Path(mapping.translate(result[field])).resolve(strict=True))
        return result
    oof_lineage._json_equivalent = lambda a,b: json.dumps(normalize_role(a),sort_keys=True)==json.dumps(normalize_role(b),sort_keys=True)
    from runpod_final_lineage_compat import install as install_lineage_paths
    install_lineage_paths(mapping, (INPUT, BUNDLE, Path('/opt/ejepa_final_rgb'),
        Path('/opt/ejepa_event_authority_supplement_v2'), Path('/opt/ejepa_event_supplement')))
    # The volume ledger accounts for the entire /workspace mount, including this run.
    runtime = Path('/workspace/ejepa_event_operations/runtime_v12')
    import scripts
    scripts.__path__ = [str(runtime/'scripts'), *scripts.__path__]
    import importlib.util
    spec=importlib.util.spec_from_file_location('final_event_resource_guard',runtime/'operational/runpod_migration/event_resources.py')
    resource=importlib.util.module_from_spec(spec);spec.loader.exec_module(resource)
    for rotation in (2,3,4):
        closure=read(Path(f'/workspace/ejepa_event_runs/EVENT_DOWNSTREAM_RESULT_R{rotation}.json'))
        if closure.get('status')!='COMPLETE':
            raise ValueError('Cannot retire producer disk reservations before completion')
    # Completed producers no longer need the original 50 GiB future-output ledger.
    # Final heads/caches retain 8 GiB plus the unchanged 4 GiB inflight and 10 GB floor.
    resource.OUTPUT_LEDGER_RESERVE_BYTES=8*1024**3

    class Monitor(resource.EventResourceMonitor):
        deadline_timestamp = DEADLINE
        def guard(self, run_root, **kwargs):
            if (RUN/'STOP_REQUEST').exists():
                return False, {'reason':'MANUAL_STOP','host_available_bytes':0,'windows_commit_headroom_bytes':None}
            if not Path(run_root).resolve().is_relative_to(RUN):
                raise ValueError('Final training escaped its run root')
            return super().guard(Path('/workspace/ejepa_event_runs'), **kwargs)
    shared_resources.SharedResourceMonitor=Monitor
    from operational.rgb_port.train_producers import ProducerCheckpoint
    original_begin=ProducerCheckpoint.begin
    def bounded_begin(self):
        if time.time()>=DEADLINE or (RUN/'STOP_REQUEST').exists():
            raise TimeoutError('Final execution deadline or manual stop reached before optimizer update')
        return original_begin(self)
    ProducerCheckpoint.begin=bounded_begin
    return mapping


def link(source: Path, target: Path):
    source=source.resolve(strict=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.resolve()!=source:
            raise ValueError(f'Conflicting final input: {target}')
    else:
        target.symlink_to(source, target_is_directory=source.is_dir())


def assemble():
    local=INPUT/'workspace/e-jepa-ttc-v18-joint-measurement/artifacts/final_refit_20261019_scientific_oof'
    RUN.mkdir(exist_ok=True)
    for name in ('SCIENTIFIC_CONFIG.json','parents','cache'):
        link(local/name,RUN/name)
    for rotation in (0,1):
        link(local/f'rotations/{rotation}',RUN/f'rotations/{rotation}')
    for rotation in (2,3,4):
        target=RUN/f'rotations/{rotation}'
        event=Path('/workspace/ejepa_event_runs')/f'rotations/{rotation}'
        for name in ('E_A5','E_C2F','PAIR_E','E_CTX'):
            link(event/'fits'/name,target/'fits'/name)
        for name,kind in (('R_A5','a5'),('R_C2F','c2f'),('PAIR_R','downstream'),('R_CTX','downstream')):
            link(RGB/f'roots/r{rotation}_{kind}/rotations/{rotation}/fits/{name}',target/'fits'/name)
        link(event/'generation/E_CTX',target/'generation/E_CTX')
        link(RGB/f'roots/r{rotation}_downstream/rotations/{rotation}/generation/R_CTX',target/'generation/R_CTX')


def train_selected(final, run, name):
    """Select one independent fit from the frozen runner; preserve its exact body."""
    import ast
    import fcntl
    import inspect
    from operational.final_refit.recipe import sha256
    from operational.final_refit.scientific_oof_pipeline import make_guard, ResourcePause
    original=inspect.getsource(final.fit_next_oof_stage)
    tree=ast.parse(original)
    assignments=[node for node in ast.walk(tree) if isinstance(node,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='stages' for t in node.targets)]
    if len(assignments)!=1:
        raise ValueError('Frozen stage selector changed')
    expected=('PAIR_E_MATCHED','PAIR_R','E_CTX_MATCHED','R_CTX','PHASE_ONLY_EVENT_S7_TRAIN40')
    if ast.literal_eval(assignments[0].value)!=expected or name not in expected:
        raise ValueError('Unknown final stage')
    assignments[0].value=ast.Tuple(elts=[ast.Constant(name)],ctx=ast.Load())
    tree.body[0].name='_fit_selected_stage'
    # Each worker owns its checkpoint and journal; the coordinator owns aggregate status.
    for node in ast.walk(tree):
        if isinstance(node,ast.Return) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='_write_final_receipt':
            node.value=node.value.args[2]
    ast.fix_missing_locations(tree)
    namespace=dict(vars(final))
    exec(compile(tree,'<registered-final-stage-selection>','exec'),namespace)
    folder=RUN/'final/fits'/name;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'WORKER.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        binding={'stage':name,'source_sha256':sha256(final.Path(final.__file__)),
                 'selector':'ONE_STAGE_ONLY_NO_AGGREGATE_WRITE','checkpoint_updates':100,
                 'deadline_utc':'2026-10-28T00:00:00+00:00','pid':os.getpid()}
        (folder/'WORKER_BINDING.json').write_text(json.dumps(binding,indent=2))
        guard=make_guard(run)
        while time.time()<DEADLINE and not (RUN/'STOP_REQUEST').exists():
            try:
                guard()
                result=namespace['_fit_selected_stage'](run,max_updates=100)
            except ResourcePause as pause:
                (folder/'RESOURCE_WAIT.json').write_text(json.dumps({'timestamp':time.time(),'resources':pause.resources}))
                time.sleep(5)
                continue
            print(json.dumps({'stage':name,**result}),flush=True)
            if read(folder/'CHECKPOINT_RECEIPT.json').get('status')=='COMPLETE':
                return


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','pair-cache','train','preflight','finish','repair'])
    parser.add_argument('--stage',choices=['PAIR_E_MATCHED','PAIR_R','E_CTX_MATCHED','R_CTX','PHASE_ONLY_EVENT_S7_TRAIN40'])
    args=parser.parse_args()
    if time.time()>=DEADLINE:
        raise TimeoutError('Authorized deadline reached')
    mapping=setup()
    import torch
    torch.set_num_threads(2)
    from operational.final_refit import scientific_oof_final as final
    import runpod_final_cache_compat as compat
    compat.install(mapping)
    run=final.Path(str(RUN))
    if args.command=='repair':
        print(json.dumps(compat.repair_generated_caches(run)),flush=True)
    elif args.command=='preflight':
        cfg=final._config(run)
        print(json.dumps({'status':'PASS','config_keys':list(cfg)}),flush=True)
    elif args.command=='prepare':
        assemble()
        result=final.prepare_oof_training(run)
        print(json.dumps(result),flush=True)
    elif args.command=='pair-cache':
        print(final.prepare_deployment_rgb_pair(run),flush=True)
    elif args.command=='finish':
        # Training/export are independently useful durable deliverables. Test12
        # sensor recovery has its own admission and must not masquerade as done.
        final.seal_scientific_final(run, deployment_dependencies=final.deployment_dependency_hashes(run))
        from runpod_final_export_compat import install as install_export_compat
        install_export_compat()
        final.export_scientific_head(run)
        status={'status':'TRAINING_EXPORTED_TEST12_PENDING',
                'test12_predictions_executed':False,'submission_authorized':False,
                'reason':'Public Test12 sensor inputs require staged recovery and validation',
                'model_seal':str(run/'final/MODEL_SEAL.json'),
                'head_export':str(run/'final/PHASE_ONLY_HEAD.ts')}
        target=RUN/'final/TRAINING_EXPORT_STATUS.json'
        temporary=target.with_suffix('.pending')
        with temporary.open('w') as stream:
            json.dump(status,stream,indent=2);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,target)
        print(json.dumps(status),flush=True)
    elif args.stage:
        train_selected(final,run,args.stage)
    else:
        while time.time()<DEADLINE and not (RUN/'STOP_REQUEST').exists():
            result=final.fit_next_oof_stage(run,max_updates=100)
            print(json.dumps(result),flush=True)
            if result.get('status')=='COMPLETE':
                break


if __name__=='__main__':
    main()
