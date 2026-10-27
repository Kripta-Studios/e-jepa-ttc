"""Deploy final queue fixes while adopting, rather than stopping, live workers."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'artifacts/runpod_event_migration_20261026/resume_20261027'))
    from rpc import call

    payload = {
        name: base64.b64encode((root / 'scripts' / name).read_bytes()).decode()
        for name in ('runpod_final_bootstrap.py', 'runpod_final_queue.py')
    }
    code = '''import pathlib,json,base64,hashlib,os,subprocess,time,psutil
ops=pathlib.Path('/workspace/ejepa_final_operations')
entry=ops/'runpod_final_bootstrap.py'
queue=ops/'runpod_final_queue.py'
status=json.loads((ops/'FINAL_QUEUE_STATUS.json').read_text())
workers={}
for name,pid in status['running'].items():
    try:
        p=psutil.Process(pid);command=p.cmdline()
        if str(entry) not in command: raise ValueError('Unexpected worker command')
        workers[name]={'pid':pid,'created':p.create_time(),'command':command}
    except psutil.NoSuchProcess: pass
p=psutil.Process(status['pid'])
if str(queue) not in p.cmdline():raise ValueError('Unexpected coordinator command')
p.terminate();p.wait(10)
temporary=ops/'EXTERNAL_JOBS.pending'
temporary.write_text(json.dumps(workers,indent=2));os.replace(temporary,ops/'EXTERNAL_JOBS.json')
hashes={}
for name,encoded in PAYLOAD.items():
    data=base64.b64decode(encoded);compile(data,name,'exec')
    target=ops/name
    previous=target.read_bytes()
    backup=ops/(name+'.'+hashlib.sha256(previous).hexdigest()+'.bak')
    if not backup.exists():backup.write_bytes(previous)
    temporary=target.with_suffix('.pending');temporary.write_bytes(data);os.replace(temporary,target)
    hashes[name]=hashlib.sha256(data).hexdigest()
with (ops/'coordinator.log').open('a') as stream:
    child=subprocess.Popen(['/opt/ejepa_env/bin/python',str(queue)],stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True)
result={'timestamp':time.time(),'pid':child.pid,'preserved_workers':workers,'deployed_sha256':hashes,'checkpoint_updates':100,'deadline_utc':'2026-10-28T00:00:00+00:00'}
(ops/'FINAL_QUEUE_DEPLOYMENT.json').write_text(json.dumps(result,indent=2))
print('PODSTATUS:'+json.dumps(result))'''.replace('PAYLOAD', repr(payload))
    result = call(code, timeout=60)
    destination = root / 'artifacts/runpod_event_migration_20261026/final_20261027/FINAL_QUEUE_DEPLOYMENT.json'
    destination.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
