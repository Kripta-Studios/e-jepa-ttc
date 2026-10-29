"""Deploy an isolated seven-source Test12 probe; optionally launch its bounded worker."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path, PureWindowsPath


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    historical = repo.parent / "e-jepa-ttc-v16-multimodal-validation"
    freeze_path = (
        historical
        / "artifacts/research_independent_20261012T165833Z/DEPENDENCIES/OFFICIAL_TEST12"
        / "H8_CURRENT_SENSOR_INPUT_FREEZE.json"
    )
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    payload = {}
    for old, expected in frozen["sources"].items():
        parts = PureWindowsPath(old).parts
        index = next(i for i, part in enumerate(parts) if part in ("src", "operational"))
        relative = Path(*parts[index:])
        data = (historical / relative).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected
        payload["code/" + relative.as_posix()] = base64.b64encode(data).decode()
    for name in ("runpod_test12_parity_probe.py", "runpod_test12_single_worker.py"):
        payload[name] = base64.b64encode((repo / "scripts" / name).read_bytes()).decode()
    payload["resource_reader.py"] = base64.b64encode(
        (repo / "scripts/runpod_resources_v3.py").read_bytes()
    ).decode()
    final = repo / "artifacts/runpod_event_migration_20261026/final_20261027"
    payload["TEST12_EXACT_INVENTORY.json"] = base64.b64encode(
        (final / "TEST12_EXACT_INVENTORY.json").read_bytes()
    ).decode()
    sys.path.insert(0, str(repo / "artifacts/runpod_event_migration_20261026/resume_20261027"))
    from rpc import call

    code = r"""
import pathlib,json,base64,hashlib,os,shutil,subprocess,urllib.request
root=pathlib.Path('/workspace/ejepa_test12_probe');root.mkdir(exist_ok=True)
base=pathlib.Path('/opt/ejepa_event_bundle/workspace/e-jepa-ttc-v18-joint-measurement')
for folder in ('src','operational'):
    target=root/'code'/folder
    if not target.exists():
        shutil.copytree(base/folder,target,copy_function=os.symlink,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name,encoded in PAYLOAD.items():
    path=root/name;data=base64.b64decode(encoded)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_symlink():path.unlink()
    if path.exists() and path.read_bytes()!=data:
        if name not in ('runpod_test12_single_worker.py','runpod_test12_parity_probe.py'):
            raise ValueError('Existing deployed input differs: '+str(path))
        previous=path.read_bytes();path.with_suffix('.'+hashlib.sha256(previous).hexdigest()+'.bak').write_bytes(previous);path.unlink()
    if not path.exists():path.write_bytes(data)
public=root/'test_inputs.parquet'
if not public.exists():
    url='https://huggingface.co/datasets/NAIL-HNU/GarlTTC-dataset/resolve/9f477b5c4d5f9cdc32e7cc4bb3b186301979d6b5/data/test_inputs.parquet'
    with urllib.request.urlopen(url,timeout=40) as response:data=response.read(2*1024*1024+1)
    expected='aedb9da5484ceedb642ef1cbc7cf6115dcf7f368e4444b55de947989fc19a362'
    if hashlib.sha256(data).hexdigest()!=expected:
        raise ValueError('Public metadata hash mismatch')
    public.write_bytes(data)
v16=pathlib.Path('/workspace/ejepa_final_inputs/workspace/e-jepa-ttc-v16-multimodal-validation')
argv=['/opt/ejepa_env/bin/python',str(root/'runpod_test12_parity_probe.py'),
'--code-root',str(root/'code'),'--raw-root',str(root/'raw'),'--output',str(root/'result'),
'--frozen-inputs',str(v16/'artifacts/multimodal_research_20261012/test12_phase_only/INPUTS.json'),
'--event-freeze',str(v16/'artifacts/research_independent_20261012T165833Z/DEPENDENCIES/OFFICIAL_TEST12/H8_CURRENT_SENSOR_INPUT_FREEZE.json'),
'--public-inputs',str(public),'--inventory',str(root/'TEST12_EXACT_INVENTORY.json')]
result=subprocess.run(argv,capture_output=True,text=True,timeout=30)
if result.returncode:raise RuntimeError(result.stderr[-3000:])
plan=json.loads(result.stdout)
(root/'PLAN.json').write_text(json.dumps(plan,indent=2));(root/'PROBE_ARGV.json').write_text(json.dumps(argv))
answer={'dry_run':'PASS','query':plan['tensor_name'],'raw_bytes':plan['raw_bytes'],'source_count':len(plan['source_bindings'])}
if LAUNCH:
    with (root/'worker.log').open('a') as log:
        child=subprocess.Popen(['/opt/ejepa_env/bin/python',str(root/'runpod_test12_single_worker.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    answer.update(pid=child.pid,log=str(root/'worker.log'))
print('PODSTATUS:'+json.dumps(answer))
""".replace("PAYLOAD", repr(payload)).replace("LAUNCH", repr(args.launch))
    print(json.dumps(call(code, timeout=90), indent=2))


if __name__ == "__main__":
    main()
