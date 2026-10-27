"""Package the existing final-fit and R0/R1 closure; upload through authenticated Jupyter."""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import time

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'artifacts/runpod_event_migration_20261026/final_20261027'
sys.path.insert(0, str(REPO / 'artifacts/runpod_event_migration_20261026/resume_20261027'))
from rpc import call


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path: Path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    import requests
    OUT.mkdir(parents=True, exist_ok=True)
    run = REPO / 'artifacts/final_refit_20261019_scientific_oof'
    cfg = read(run / 'SCIENTIFIC_CONFIG.json')
    files = {Path(p).resolve() for category in ('input_sha256', 'source_sha256') for p in cfg[category]}
    files.add(run / 'SCIENTIFIC_CONFIG.json')
    for category in ('input_sha256', 'source_sha256'):
        for p, expected in cfg[category].items():
            if digest(Path(p)) != expected:
                raise ValueError(f'Frozen authority changed: {p}')
    for directory in [Path(cfg['fold_contract']).parent, run/'parents']:
        files.update(p for p in directory.rglob('*') if p.is_file())
    for rotation in (0, 1):
        root = run / 'rotations' / str(rotation)
        files.update(root.glob('*.json'))
        files.update(root.glob('fits/*/*.json'))
        files.update(root.glob('fits/*/checkpoint_last.pt'))
        files.update(root.glob('normalizers/*.json'))
        files.update(root.glob('generation/*/fragment.*'))
    for name in ('E_A5', 'E_C2F'):
        admission = read(run / 'parents' / (name + '_ADMISSION.json'))
        files.update(Path(admission[k]) for k in ('checkpoint_path','original_receipt_path'))
    for name in ('R_A5', 'R_C2F'):
        root = Path(cfg['active_rgb_runtime'])/'fits'/name
        files.update(root.glob('*.json'))
        files.add(Path(read(root/'CHECKPOINT_RECEIPT.json')['checkpoint_path']))
    descriptor = read(Path(cfg['rgb_source']))
    for value in descriptor['role_manifests'].values():
        p = (Path(cfg['rgb_source']).parent/Path(value)).resolve()
        files.add(p)
        row_path = Path(read(p)['rows_path'])
        if row_path.is_file():
            files.add(row_path)
    from operational.runpod_migration.paths import PathMap
    mapping = PathMap({str(REPO.parent):'workspace','E:\\':'drives/E'}, '/unused')
    rows = {mapping.relative(p): {'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(files)}
    manifest = {'schema':'ejepa_final_input_closure_v1','files':rows,
                'scientific_config_sha256':digest(run/'SCIENTIFIC_CONFIG.json'),
                'optimizer_updates':0,'created_timestamp':time.time()}
    archive = OUT/'FINAL_INPUTS.tar'
    with tarfile.open(archive,'w') as tar:
        data = json.dumps(manifest,sort_keys=True).encode()
        info = tarfile.TarInfo('MANIFEST.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
        for p in sorted(files):
            tar.add(p,arcname=mapping.relative(p),recursive=False)
    expected=digest(archive)
    (OUT/'FINAL_INPUTS.json').write_text(json.dumps({'bytes':archive.stat().st_size,'sha256':expected,'files':len(files)}))
    print(json.dumps({'phase':'PACKAGED','bytes':archive.stat().st_size,'files':len(files)}),flush=True)
    servers=call("import pathlib,json;print('PODSTATUS:'+json.dumps([json.loads(p.read_text()) for p in pathlib.Path('/root/.local/share/jupyter/runtime').glob('jpserver-*.json')]))")
    token=servers[0]['token']
    session=requests.Session();session.headers['Authorization']='token '+token
    url='https://lreiivpf18vxu7-8888.proxy.runpod.net/api/contents/workspace/ejepa_final_operations/FINAL_INPUTS.tar'
    size=archive.stat().st_size;sent=0;chunk=0
    with archive.open('rb') as stream:
        while data:=stream.read(8*1024*1024):
            chunk+=1;sent+=len(data)
            response=session.put(url,json={'type':'file','format':'base64','content':base64.b64encode(data).decode(),'chunk':-1 if sent==size else chunk},timeout=120)
            response.raise_for_status()
            (OUT/'UPLOAD_STATUS.json').write_text(json.dumps({'status':'UPLOADING','bytes':sent,'total':size,'timestamp':time.time()}))
    code=f"import pathlib,hashlib,json;p=pathlib.Path('/workspace/ejepa_final_operations/FINAL_INPUTS.tar');h=hashlib.file_digest(p.open('rb'),'sha256').hexdigest();assert h=={expected!r};print('PODSTATUS:'+json.dumps({{'status':'VERIFIED','bytes':p.stat().st_size,'sha256':h}}))"
    result=call(code,timeout=60)
    (OUT/'UPLOAD_STATUS.json').write_text(json.dumps(result))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    sys.path.insert(0,str(REPO))
    main()
