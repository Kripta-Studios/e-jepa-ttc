"""One-shot recoverable FCWD -> cost -> existing R1 supervisor continuation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from operational.efficient_context.common import atomic_json, digest
from operational.sota_eval.followups import verify_cost, verify_fcwd

ROOT = Path.cwd().resolve()
BASE = ROOT / 'artifacts/sota_campaign_20261008'
OUT = BASE / 'viable_continuation'
OUT.mkdir(exist_ok=True)


def child(name, arguments):
    command = [sys.executable, '-m', *arguments]
    with (OUT / (name + '.log')).open('xb') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                   stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        receipt = dict(command=command, pid=process.pid,
                       create_time=psutil.Process(process.pid).create_time(),
                       status='RUNNING', optimizer_updates=0)
        atomic_json(OUT / (name + '.json'), receipt)
        code = process.wait()
    receipt.update(status='EXITED', returncode=code)
    atomic_json(OUT / (name + '.json'), receipt)
    if code:
        raise RuntimeError(f'{name} exited {code}; work preserved')


def branch(name, action):
    try:
        result = dict(status='COMPLETE', evidence=action())
    except Exception as error:
        result = dict(status='FAILED_PRESERVED', error=str(error),
                      error_type=type(error).__name__, scientific_negative=False)
    atomic_json(OUT / (name + '_RESULT.json'), result)
    return result


def fcwd():
    child('FCWD_SCORE', ['operational.sota_eval.fcwd_run', '--score',
          '--manifest', str(BASE / 'fcwd_population/QUERY_MANIFEST.json'),
          '--output', str(BASE / 'fcwd_inference')])
    return verify_fcwd(BASE / 'fcwd_inference',
                       BASE / 'fcwd_population/QUERY_MANIFEST.json',
                       ROOT / 'artifacts/train40_system_20261005',
                       ROOT / 'artifacts/evttc_rgb_transfer_20261008/public_garl',
                       Path('E:/Garl-TTC'))


def cost():
    child('COST', ['operational.sota_eval.cost', '--manifest',
          str(BASE / 'dev32_expanded/QUERY_MANIFEST.json'), '--campaign',
          str(ROOT / 'artifacts/train40_system_20261005'), '--public-full-dir',
          str(ROOT / 'artifacts/evttc_rgb_transfer_20261008/public_garl'),
          '--code-root', 'E:/Garl-TTC', '--output', str(BASE / 'cost')])
    return verify_cost(BASE / 'cost')


def release_owned_pause():
    receipt_path = BASE / 'r1_resume/PRIORITY_PAUSE_OWNERSHIP.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    stop = Path(receipt['path']).resolve()
    expected = (ROOT / 'artifacts/efficient_context_20261004/r1_20261008/resident_1g/STOP_REQUEST').resolve()
    if stop != expected or digest(stop) != receipt['sha256']:
        raise RuntimeError('R1 priority pause ownership mismatch; marker preserved')
    stop.unlink()
    result = dict(status='OWNED_PAUSE_RELEASED', sha256=receipt['sha256'],
                  released_epoch=time.time(), supervisor_restarted=False,
                  retry_budget_reset=False,
                  note='Existing supervisor resumes after its configured backoff.')
    atomic_json(BASE / 'r1_resume/PRIORITY_PAUSE_RELEASED.json', result)
    return result


if __name__ == '__main__':
    identity = json.loads((BASE / 'LAUNCH_FCWD_DIRECT.json').read_text(encoding='utf-8'))
    deadline = time.time() + 4 * 3600
    while psutil.pid_exists(identity['pid']):
        try:
            if abs(psutil.Process(identity['pid']).create_time() - identity['create_time']) > .1:
                break
        except psutil.NoSuchProcess:
            break
        if time.time() >= deadline:
            raise TimeoutError('FCWD still owns GPU; no downstream work launched')
        time.sleep(10)
    results = dict(fcwd=branch('FCWD', fcwd), cost=branch('COST_VERIFIED', cost))
    results['r1'] = branch('R1_RELEASE', release_owned_pause)
    atomic_json(OUT / 'RESULT.json', dict(status='FINISHED_WITH_BRANCH_STATUS',
                branches=results, optimizer_updates=0))
