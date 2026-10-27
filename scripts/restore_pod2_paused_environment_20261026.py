"""Restore Pod2's exact disposable runtime, without resuming any training."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

BASE = Path('/workspace/ejepa_event_operations/manual_pause_20261026')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore', action='store_true', help='Restore /opt; never remove STOP or launch training')
    args = parser.parse_args()
    status = json.loads((BASE / 'PRESERVATION_STATUS.json').read_text())
    assert status['status'] == 'PAUSED_PRESERVED'
    assert 'USER_MANUAL_PAUSE' in Path('/workspace/ejepa_event_runs/STOP_REQUEST').read_text()
    archive = Path(status['runtime_archive'])
    assert archive.stat().st_size == status['runtime_bytes']
    if args.restore:
        with archive.open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == status['runtime_sha256']
        subprocess.run(['tar', '-C', '/opt', '-xf', str(archive)], check=True)
        subprocess.run(['rsync', '-a', '/workspace/ejepa_event_bundle/', '/opt/ejepa_event_bundle/'], check=True)
        subprocess.run(['/opt/ejepa_env/bin/python', '-c',
                        'import sys,torch,numpy; print(sys.version); print(torch.__version__,torch.version.cuda); assert sys.version_info[:2]==(3,11); assert torch.__version__.startswith("2.11.0")'], check=True)
    print(json.dumps({'status': 'ENVIRONMENT_RESTORED_STILL_PAUSED' if args.restore else 'RESTORE_PLAN_READY',
                      'runtime_bytes': status['runtime_bytes'], 'training_started': False,
                      'manual_stop_retained': True}))

if __name__ == '__main__':
    main()
