"""One-shot exact-checkpoint handoff for the three explicitly identified V18 processes."""
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

out = Path(__file__).resolve().parent
run = out.parents[1]
fit = run / 'adaptive/A1_adaptation_depth/fits/A1-LAST-BLOCK-S7'
expected = {16080: 1792171996.036665, 23684: 1792171960.0495126, 38716: 1792172056.7220757}
processes = {pid: psutil.Process(pid) for pid in expected}
for pid, process in processes.items():
    if abs(process.create_time()-expected[pid]) > 0.001:
        raise RuntimeError('Process identity changed; no process modified')
    module = 'operational.joint_measurement.hourly_observe' if pid == 38716 else 'operational.joint_measurement.adaptation_runtime'
    if module not in process.cmdline():
        raise RuntimeError('Unexpected command; no process modified')
if not (out / 'REGISTRATION.json').exists() or (out / 'ORIGIN.pt').exists():
    raise RuntimeError('Registration missing or origin already exists')

def read(path):
    for attempt in range(20):
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except (PermissionError, FileNotFoundError):
            if attempt == 19: raise
            time.sleep(0.01)

def write(path, data):
    tmp = path.with_suffix('.pending')
    with tmp.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)

def safe(journal, receipt, pointer):
    count=journal['completed_updates']
    return (count == journal['durable_updates'] == receipt['completed_updates'] == pointer['completed_updates']
            and journal['pending_update_upper'] == 0 and receipt['complete_state']
            and receipt['accumulation_index'] == 0
            and journal['identity_sha256'] == receipt['identity_sha256'] == pointer['identity_sha256']
            and receipt['checkpoint_sha256'] == pointer['checkpoint_sha256'])

child, driver, observer = (processes[p] for p in (16080,23684,38716))
child_suspended=False
stopped=False
driver.suspend()
try:
    deadline=time.monotonic()+180
    print('Waiting for completed == durable, zero pending and consistent pointer/receipt', flush=True)
    while time.monotonic()<deadline:
        try:
            journal=read(fit/'UPDATE_JOURNAL.json')
        except (PermissionError, FileNotFoundError):
            time.sleep(0.1); continue
        if journal['completed_updates'] != journal['durable_updates'] or journal['pending_update_upper']:
            time.sleep(0.1); continue
        child.suspend(); child_suspended=True
        try:
            journal=read(fit/'UPDATE_JOURNAL.json')
            receipt=read(fit/'CHECKPOINT_RECEIPT.json')
            pointer=read(fit/'CHECKPOINT_POINTER.json')
        except (PermissionError, FileNotFoundError):
            child.resume();child_suspended=False;time.sleep(0.1);continue
        if not safe(journal,receipt,pointer):
            child.resume(); child_suspended=False; time.sleep(0.1); continue
        data=(fit/'checkpoint_last.pt').read_bytes()
        actual=hashlib.sha256(data).hexdigest()
        if actual != receipt['checkpoint_sha256']:
            raise RuntimeError('Checkpoint hash disagrees with durable receipt')
        with (out/'ORIGIN.pt').open('xb') as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        witness={'utc':datetime.now(timezone.utc).isoformat(),'checkpoint_sha256':actual,
                 'completed_updates':journal['completed_updates'],'journal':journal,
                 'receipt':receipt,'pointer':pointer,'processes':expected,
                 'updates_lost':0,'pending_updates':0,
                 'action_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'scope_accounting':'Retain unclosed scope reservation conservatively'}
        write(out/'CHECKPOINT_CAPTURED.json',witness)
        child.kill(); child.wait(timeout=20); child_suspended=False
        driver.kill();driver.wait(timeout=20)
        observer.kill();observer.wait(timeout=20)
        stopped=True
        write(out/'QUIESCED.json',{**witness,'status':'QUIESCED_AT_EXACT_DURABLE_BOUNDARY'})
        print({'status':'QUIESCED','updates':journal['completed_updates'],'sha256':actual},flush=True)
        break
    else:
        raise TimeoutError('No safe checkpoint observed; resume existing processes')
finally:
    if child_suspended and child.is_running():child.resume()
    if not stopped and driver.is_running():driver.resume()
