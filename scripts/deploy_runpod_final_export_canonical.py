"""Run the reviewed PHASE_ONLY compatibility install and canonical export."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "artifacts/runpod_event_migration_20261026/resume_20261027"))
    from rpc import call

    driver = """import json,pathlib,sys
ops=pathlib.Path('/workspace/ejepa_final_operations')
sys.path.insert(0,str(ops))
import runpod_final_bootstrap as bootstrap
mapping=bootstrap.setup()
import runpod_final_cache_compat as cachecompat
cachecompat.install(mapping)
import runpod_final_export_compat as exportcompat
exportcompat.install()
from operational.final_refit import scientific_oof_final as final
result=final.export_scientific_head(final.Path('/workspace/ejepa_final_runs'))
print('INNERSTATUS:'+json.dumps(result))"""
    code = f"""import json,subprocess
child=subprocess.run(['/opt/ejepa_env/bin/python','-c',{driver!r}],capture_output=True,text=True,timeout=160)
if child.returncode:
    raise RuntimeError(child.stdout[-2000:]+child.stderr[-5000:])
rows=[line for line in child.stdout.splitlines() if line.startswith('INNERSTATUS:')]
if len(rows)!=1:
    raise RuntimeError(child.stdout[-5000:]+child.stderr[-1000:])
print('PODSTATUS:'+rows[0].split('INNERSTATUS:',1)[1])"""
    result = call(code, timeout=180)
    destination = (
        root
        / "artifacts/runpod_event_migration_20261026/final_20261027"
        / "FINAL_EXPORT_CANONICAL.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
