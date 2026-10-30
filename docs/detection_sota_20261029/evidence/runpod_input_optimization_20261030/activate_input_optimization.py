"""Admit the new producer, retain the old checkpoint, then start the optimized queue."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import time

root=Path('/workspace/phase_release')
out=root/'optimization_v1'
python=root/'.venv-detector/bin/python'
script=out/'runpod_input_optimized_queue.py'
state=root/'INPUT_OPTIMIZATION_STATUS.json'

def record(status,**kwargs):
    state.write_text(json.dumps({'status':status,'epoch':time.time(),**kwargs},indent=2))

try:
    record('PREPARING_OPTIMIZED_CAMPAIGN')
    subprocess.run([str(python),str(script),'prepare','--root',str(root)],check=True)
    record('OPTIMIZED_GPU_SMOKE')
    subprocess.run([str(python),str(script),'smoke','--root',str(root)],check=True)
    old=json.loads((out/'original_checkpoint_pause.json').read_text())
    checkpoint=Path(old['checkpoint'])
    with checkpoint.open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==old['sha256']
    owner=json.loads((root/'cloud_runs/OWNER.json').read_text())['pid']
    assert 'runpod_release_cloud.py' in Path(f'/proc/{owner}/cmdline').read_text()
    assert 'train_detection_campaign.py' in Path(f"/proc/{old['pid']}/cmdline").read_text()
    os.kill(owner,signal.SIGTERM)
    os.kill(old['pid'],signal.SIGTERM)
    os.kill(old['pid'],signal.SIGCONT)
    time.sleep(3)
    retained={'status':'SUPERSEDED_BY_EXACT_INPUT_OPTIMIZATION','checkpoint':str(checkpoint),
              'checkpoint_sha256':old['sha256'],'new_campaign':str(root/'cloud_runs_input_v1'),
              'restart':'COLD_MATCHED_REPLICATES; NO_CHECKPOINT_PROVENANCE_RELAXATION'}
    (root/'cloud_runs/r0_seed7/INPUT_OPTIMIZATION_HANDOFF.json').write_text(json.dumps(retained,indent=2))
    (root/'cloud_runs/status.json').write_text(json.dumps(retained,indent=2))
    (root/'PIPELINE_STATUS').write_text('SUPERSEDED_BY_INPUT_OPTIMIZATION')
    active={'queue':str(root/'cloud_runs_input_v1'),'status_file':str(root/'cloud_runs_input_v1/status.json'),
            'optimization':'exact_input_residency_v1','jobs':1}
    (root/'ACTIVE_CAMPAIGN.json').write_text(json.dumps(active,indent=2))
    record('OPTIMIZED_QUEUE_RUNNING',**active)
    subprocess.run([str(python),str(script),'run','--root',str(root),'--jobs','1'],check=True)
    record('OPTIMIZED_RGB_QUEUE_COMPLETE',**active)
except BaseException as error:
    record('FAILED_NEEDS_REVIEW',error=str(error))
    raise
