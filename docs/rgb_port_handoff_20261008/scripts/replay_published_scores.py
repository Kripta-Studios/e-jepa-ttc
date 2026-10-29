#!/usr/bin/env python3
"""Replay the three published scorers without fitting or editing their evidence.

Requires NumPy, as does the original scorer. --source-root is an extraction of
SOTA_ESSENTIAL.zip. The scorer hash is pinned to the inspected publication.
JSON line endings can differ across platforms; report byte and semantic parity
separately. Do not replace the original publication's JSON or hashes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

SCORER_SHA = 'c19af78b0a3efe27f2b7cf8b5aa074294eec2c16a6e8e09b9f5debd64d23a2ec'


def inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f'Path escapes source: {relative}')
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = args.source_root.resolve(strict=True)
    out = args.output.resolve()
    if out.exists() or out.is_relative_to(root):
        raise ValueError('Use a new output directory outside the source extraction.')
    reg = json.loads((root / 'tables/REGENERATION.json').read_text(encoding='utf-8'))
    scorer = inside(root, reg['scorer'])
    if hashlib.sha256(scorer.read_bytes()).hexdigest() != SCORER_SHA:
        raise ValueError('Published scorer bytes differ from the inspected version.')
    out.mkdir(parents=True)
    records = []
    for job in reg['scoring_jobs']:
        name = job['branch']
        if Path(name).name != name or name in {'.', '..'}:
            raise ValueError('Invalid branch name')
        ref = inside(root, job['reference'])
        source = inside(root, job['input'])
        cfg = json.loads((ref / 'METADATA.json').read_text(encoding='utf-8'))['configuration']
        dest = out / name
        command = [sys.executable, '-B', str(scorer), '--input', str(source),
                   '--output-dir', str(dest), '--models', *cfg['methods'],
                   '--bootstrap-draws', str(cfg['bootstrap_draws']), '--seed', str(cfg['seed'])]
        if cfg.get('group_column'):
            command += ['--group-column', cfg['group_column']]
        subprocess.run(command, check=True)
        record = {'branch': name, 'optimizer_updates': 0, 'files': {}}
        for filename in ('METHOD_METRICS.csv', 'PAIRED.csv', 'PER_SEQUENCE.csv',
                         'SCORED_ROWS.csv', 'METADATA.json', 'REPORT.json', 'SHA256.json'):
            a, b = (dest / filename).read_bytes(), (ref / filename).read_bytes()
            item = {'same_bytes': a == b,
                    'same_newline_normalized': a.replace(b'\r\n', b'\n') == b.replace(b'\r\n', b'\n')}
            if filename.endswith('.json'):
                item['same_parsed_values'] = json.loads(a) == json.loads(b)
            record['files'][filename] = item
        records.append(record)
    target = out / 'REPLAY_COMPARISON.json'
    target.write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f'Comparison saved: {target}')


if __name__ == '__main__':
    main()
