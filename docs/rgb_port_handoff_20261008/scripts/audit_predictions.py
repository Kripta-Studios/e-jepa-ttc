#!/usr/bin/env python3
"""Descriptive audit of the published Dev32 and FCWD predictions; no training.

RTE here is 100*abs(pred-truth)/abs(truth), with finite nonzero published truth.
Native predictions remain untouched. Missing RGB predictions are unavailable,
not zeros or sign errors. The chosen TTC strata are post-hoc diagnostics.
Requires NumPy. No pandas, no network, no model checkpoints.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def as_float(value: str | None) -> float:
    try:
        return float(value) if value is not None and value.strip() else float('nan')
    except ValueError:
        return float('nan')


def save_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.source_root.resolve(strict=True), args.output.resolve()
    if out.exists() or out.is_relative_to(root):
        raise ValueError('Use a new output directory outside the source extraction.')
    sets = {
        'dev32': root / 'evidence/dev32_expanded_rgb/SCORED_PREDICTIONS.csv',
        'fcwd': root / 'evidence/fcwd_inference/scoring/SCORED_PREDICTIONS.csv',
    }
    strata, metrics, checks = [], [], {}
    for name, path in sets.items():
        with path.open(encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fields = list(reader.fieldnames or [])
        y = np.array([as_float(r.get('truth_ttc_seconds')) for r in rows], dtype=np.float64)
        eligible = np.isfinite(y) & (y != 0)
        seq = np.array([r['sequence_id'] for r in rows])
        ids = [r['query_id'] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate query IDs: {name}')
        checks[name] = {
            'rows': len(rows), 'eligible': int(eligible.sum()),
            'sequences': len(set(seq)), 'optimizer_updates': 0,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'unavailable_methods': [],
        }
        bins = {
            'negative': y < 0, '0_1': (y > 0) & (y <= 1),
            '1_3': (y > 1) & (y <= 3), '3_6': (y > 3) & (y <= 6),
            '6_10': (y > 6) & (y <= 10), '10_plus': y > 10,
        }
        for method in [c for c in fields if c.startswith(('H8_seed', 'public_Garl'))]:
            pred = np.array([as_float(r.get(method)) for r in rows], dtype=np.float64)
            valid = eligible & np.isfinite(pred)
            if not valid.any():
                checks[name]['unavailable_methods'].append(method)
                continue
            error = np.abs(pred - y)
            relative = np.full_like(y, np.nan)
            np.divide(error, np.abs(y), out=relative, where=eligible)
            relative *= 100
            metrics.append({
                'dataset': name, 'model': method, 'eligible': int(eligible.sum()),
                'scorable': int(valid.sum()), 'coverage': float(valid.sum() / eligible.sum()),
                'MAE_s': float(error[valid].mean()), 'RTE_percent': float(relative[valid].mean()),
                'median_AE_s': float(np.median(error[valid])),
                'RMSE_s': float(np.sqrt((error[valid] ** 2).mean())),
                'sign_errors': int((np.sign(pred[valid]) != np.sign(y[valid])).sum()),
                'p95_AE_s': float(np.quantile(error[valid], .95)),
                'conditional_if_incomplete': not bool(valid.sum() == eligible.sum()),
            })
            for label, mask in bins.items():
                mask = mask & valid
                if not mask.any():
                    continue
                strata.append({
                    'dataset': name, 'model': method, 'bin': label, 'n': int(mask.sum()),
                    'MAE_s': float(error[mask].mean()), 'RTE_percent': float(relative[mask].mean()),
                    'bias_s': float((pred - y)[mask].mean()),
                    'sign_errors': int((np.sign(pred[mask]) != np.sign(y[mask])).sum()),
                    'AE_share': float(error[mask].sum() / error[valid].sum()),
                    'post_hoc_diagnostic': True,
                })
    out.mkdir(parents=True)
    save_rows(out / 'metrics.csv', metrics)
    save_rows(out / 'strata.csv', strata)
    (out / 'checks.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
