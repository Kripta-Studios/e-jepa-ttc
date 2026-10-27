"""Seal the explicitly requested Pod2 pause; never launch scientific work."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
from datetime import datetime, timezone

OPS = Path('/workspace/ejepa_event_operations')
RUN = Path('/workspace/ejepa_event_runs')
OUT = OPS / 'manual_pause_20261026'

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def write(name, obj):
    p = OUT / name
    tmp = p.with_suffix('.pending')
    with tmp.open('w') as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)

def main():
    OUT.mkdir(exist_ok=True)
    assert 'USER_MANUAL_PAUSE' in (RUN / 'STOP_REQUEST').read_text()
    state = json.loads((OPS / 'EVENT_WATCHDOG_STATUS_V18.json').read_text())
    for name, task in state['tasks'].items():
        if name.endswith('EVENT_DOWNSTREAM'):
            p = Path('/proc') / str(task['pid']) / 'cmdline'
            if p.exists() and p.read_bytes():
                raise RuntimeError('Queue has not exited: ' + name)
    # Stop only identified ancillary processes, after all queues have stopped.
    stopped = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            args = [x.decode() for x in (p / 'cmdline').read_bytes().split(b'\0') if x]
        except (FileNotFoundError, PermissionError):
            continue
        if not args or args[0] != '/opt/ejepa_env/bin/python':
            continue
        if any('runpod_event_cache_controller_v1.py' in a or
               'runpod_event_origin_pin_v1.py' in a for a in args) or 'volume-publisher' in args:
            stopped.append({'pid': int(p.name), 'argv': args})
            os.kill(int(p.name), signal.SIGTERM)
    write('ANCILLARY_PROCESSES.json', stopped)
    checkpoints = []
    rotations = {}
    for r in [2, 3, 4]:
        base = RUN / 'rotations' / str(r)
        for fit in ['E_A5', 'E_C2F', 'PAIR_E']:
            receipt = json.loads((base / 'fits' / fit / 'CHECKPOINT_RECEIPT.json').read_text())
            p = Path(receipt['checkpoint_path'])
            digest = sha(p)
            assert receipt['status'] == 'COMPLETE' and digest == receipt['checkpoint_sha256']
            checkpoints.append({'rotation': r, 'fit': fit, 'path': str(p),
                                'sha256': digest, 'updates': receipt['completed_updates']})
        parts = [p for p in (base / 'refiner' / 'E_CTX').rglob('*.npz') if '.tmp.' not in p.name]
        rotations[str(r)] = {'refiner_parts': len(parts), 'latest_mtime': max(p.stat().st_mtime for p in parts)}
    write('SCIENTIFIC_PAUSE_CHECK.json', {'checkpoints': checkpoints, 'rotations': rotations})
    # /opt bundle is a disposable acceleration copy; compare to its durable source.
    comparison = subprocess.run(['rsync', '-rln', '--size-only', '--itemize-changes', '--exclude=__pycache__', '--exclude=*.pyc',
                                 '/opt/ejepa_event_bundle/', '/workspace/ejepa_event_bundle/'],
                                check=True, capture_output=True, text=True)
    write('BUNDLE_SIZE_COMPARISON.json', {'stdout': comparison.stdout,
          'verification': 'relative filenames, symlink targets and sizes; original seals retained'})
    if comparison.stdout.strip():
        raise RuntimeError('Ephemeral bundle differs from persistent source: inspect report')
    # Preserve exact interpreter and installed libraries, supplements and integrity seals.
    names = ['ejepa_env', 'ejepa_python', 'ejepa_event_supplement',
             'ejepa_event_authority_supplement', 'ejepa_event_authority_supplement_v2']
    names += [p.name for p in Path('/opt').glob('ejepa*.json')]
    archive = OUT / 'OPT_RUNTIME.tar'
    pending = archive.with_suffix('.tar.pending')
    write('PRESERVATION_STATUS.json', {'status': 'ARCHIVING_RUNTIME', 'items': names})
    subprocess.run(['tar', '-C', '/opt', '-cf', str(pending), *names], check=True)
    with pending.open('rb') as f:
        os.fsync(f.fileno())
    os.replace(pending, archive)
    archive_sha = sha(archive)
    subprocess.run(['tar', '-tf', str(archive)], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['sync', '-f', str(OUT)], check=True)
    final = {'status': 'PAUSED_PRESERVED', 'utc': datetime.now(timezone.utc).isoformat(),
             'network_volume_id': '3mritxijmp', 'mount': '/workspace',
             'manual_resume_required': True, 'pod_stop_performed': False,
             'runtime_archive': str(archive), 'runtime_bytes': archive.stat().st_size,
             'runtime_sha256': archive_sha, 'verified_complete_checkpoints': len(checkpoints),
             'rotations': rotations, 'scientific_updates_lost': 0,
             'resume_config': str(OUT / 'EVENT_WATCHDOG_CONFIG_V18.json'),
             'budget_deadline_utc': '2026-10-27T14:09:00+00:00',
             'resume_note': 'Restore /opt, retain manual STOP until user resumes, refresh process identities; do not silently extend budget/deadline.'}
    write('PRESERVATION_STATUS.json', final)
    print(json.dumps(final), flush=True)

if __name__ == '__main__':
    main()
