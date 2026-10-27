"""Upload and run the additive PHASE_ONLY diagnostic exporter through Pod RPC."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "artifacts/runpod_event_migration_20261026/resume_20261027"))
    from rpc import call

    source = root / "scripts/runpod_final_export_compat.py"
    payload = base64.b64encode(source.read_bytes()).decode()
    driver = """import hashlib,json,pathlib,sys
ops=pathlib.Path('/workspace/ejepa_final_operations')
sys.path.insert(0,str(ops))
import runpod_final_bootstrap as bootstrap
mapping=bootstrap.setup()
import runpod_final_cache_compat as cachecompat
cachecompat.install(mapping)
import runpod_final_export_compat as compat
output=ops/'diagnostics/PHASE_ONLY_HEAD_COMPAT.ts'
result=compat.diagnostic_export(pathlib.Path('/workspace/ejepa_final_runs'),output)
result['source_sha256']=hashlib.sha256((ops/'runpod_final_export_compat.py').read_bytes()).hexdigest()
print('INNERSTATUS:'+json.dumps(result))"""
    code = f"""import base64,hashlib,json,os,pathlib,subprocess
ops=pathlib.Path('/workspace/ejepa_final_operations')
data=base64.b64decode({payload!r})
compile(data,'runpod_final_export_compat.py','exec')
target=ops/'runpod_final_export_compat.py'
if target.exists() and target.read_bytes()!=data:
    raise ValueError('Conflicting final export compatibility source')
if not target.exists():
    pending=ops/'.runpod_final_export_compat.py.pending'
    pending.write_bytes(data);os.replace(pending,target)
child=subprocess.run(['/opt/ejepa_env/bin/python','-c',{driver!r}],capture_output=True,text=True,timeout=160)
if child.returncode:
    raise RuntimeError(child.stdout[-2000:]+child.stderr[-5000:])
rows=[line for line in child.stdout.splitlines() if line.startswith('INNERSTATUS:')]
if len(rows)!=1:
    raise RuntimeError(child.stdout[-5000:]+child.stderr[-1000:])
result=json.loads(rows[0].split('INNERSTATUS:',1)[1])
print('PODSTATUS:'+json.dumps(result))"""
    result = call(code, timeout=180)
    destination = (
        root
        / "artifacts/runpod_event_migration_20261026/final_20261027"
        / "FINAL_EXPORT_COMPAT_DIAGNOSTIC.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
