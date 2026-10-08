"""Capture scoped CPU QA against exact final source bytes."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

from operational.efficient_context.common import atomic_json, digest

root = Path.cwd()
out = root / 'artifacts/sota_campaign_20261008/root_qa'
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
archive = out / ('prior_qa_' + stamp)
archive.mkdir()
for name in ('QA.json', 'PYTEST_RECEIPT.json', 'PYTEST_JUNIT.xml', 'PYTEST.txt',
             'RUFF.txt', 'PYRIGHT_OUTPUT.txt'):
    path = out / name
    if path.exists():
        shutil.copy2(path, archive / name)
sources = sorted((root/'operational/sota_eval').glob('*.py'))
tests = sorted((root/'tests/unit').glob('test_sota*.py'))
before = {str(p.relative_to(root)): digest(p) for p in sources + tests}
started = datetime.now(timezone.utc).isoformat()
command = [sys.executable, '-m', 'pytest', *(str(p.relative_to(root)) for p in tests),
           '-q', '--junitxml=' + str(out/'PYTEST_JUNIT.xml')]


def capture(cmd, name):
    with (out/name).open('wb') as handle:
        result = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f'{name} failed with {result.returncode}')


capture(command, 'PYTEST.txt')
atomic_json(out/'PYTEST_RECEIPT.json', dict(started_utc=started,
    finished_utc=datetime.now(timezone.utc).isoformat(), command=command,
    source_hashes=before, returncode=0))
ruff = Path(sys.executable).parent/'ruff.exe'
capture([str(ruff), 'check', *(str(p.relative_to(root)) for p in sources + tests)], 'RUFF.txt')
capture(['node', '.venv/Lib/site-packages/pyright/dist/dist/pyright.js', '--project',
         str(out/'PYRIGHT_CONFIG.json')], 'PYRIGHT_OUTPUT.txt')
after = {str(p.relative_to(root)): digest(p) for p in sources + tests}
if before != after:
    raise RuntimeError('Source bytes changed during QA')
suite = ET.parse(out/'PYTEST_JUNIT.xml').getroot()
counts = {key: sum(int(item.get(key, 0)) for item in suite.iter('testsuite'))
          for key in ('tests', 'errors', 'failures', 'skipped')}
commit = subprocess.check_output(['git','rev-parse','HEAD']).decode('ascii').strip()
result = dict(status='PASSED', checked_utc=datetime.now(timezone.utc).isoformat(),
    code_commit=commit, tests=counts,
    test_receipt_sha256=digest(out/'PYTEST_RECEIPT.json'),
    ruff_sha256=digest(out/'RUFF.txt'), pyright_output_sha256=digest(out/'PYRIGHT_OUTPUT.txt'),
    pyright_config_sha256=digest(out/'PYRIGHT_CONFIG.json'),
    source_hashes_current=True, no_runtime_environment_upgrade=True,
    type_check_note='Scoped config uses the actual sibling e-jepa-ttc/.venv; no diagnostic suppression.')
atomic_json(out/'QA.json', result)
print(json.dumps(result))
