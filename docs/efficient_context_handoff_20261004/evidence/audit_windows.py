"""Recompute window redundancy from the pinned T6 archive, without raw replay.

Usage:
    python evidence/audit_windows.py --archive PATH_TO_T6.zip --output RESULT.json

Reads only known numerical history indices with allow_pickle=False. No training,
network access, archive extraction, or write to the source archive is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_SHA256 = "84e2506584a4847a6fa55d3d02e881689285eee1cfa7278ff218526abd4d7e8a"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze(path: Path) -> dict[str, Any]:
    if file_hash(path) != EXPECTED_SHA256:
        raise ValueError("The source archive does not match the reviewed T6 release.")
    result: dict[str, Any] = {
        "scope": "Bound historical query-context indices included in supplied T6 bundle; not raw replay; no optimizer updates",
        "archive_sha256": EXPECTED_SHA256,
        "pools": {},
    }
    with zipfile.ZipFile(path) as archive:
        for pool in ("D0", "D1", "DENSE_OLD"):
            member = f"history/{pool}/query_context_index.npz"
            content = archive.read(member)
            with np.load(io.BytesIO(content), allow_pickle=False) as arrays:
                base = arrays["base_windows_us"]
                valid = arrays["valid"]
                lag = arrays["lag_us"]
                n = len(base)
                if base.shape != (n, 3, 2) or valid.shape != (n, 16) or lag.shape != (16,):
                    raise ValueError(f"Unexpected historical index shapes: {pool}")
                if not valid[:, -1].all():
                    raise ValueError("A historical query is missing its current observation.")
                out: dict[str, Any] = {
                    "rows": n,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "history_lag_us": lag.tolist(),
                    "contexts": {},
                }
                for length in (1, 8, 16):
                    counts, unique, spans = [], [], []
                    for index in range(n):
                        slots = np.flatnonzero(valid[index] & (np.arange(16) >= 16 - length))
                        intervals = (base[index][None] - lag[slots, None, None]).reshape(-1, 2)
                        counts.append(len(intervals))
                        unique.append(len(set(map(tuple, intervals.tolist()))))
                        spans.append(int(intervals[:, 1].max() - intervals[:, 0].min()))
                    out["contexts"][str(length)] = {
                        "total_window_uses": sum(counts),
                        "unique_windows_within_query": sum(unique),
                        "exact_reuse_fraction": 1 - sum(unique) / sum(counts),
                        "median_raw_union_duration_ms": float(np.median(spans) / 1000),
                        "queries_with_exact_duplicates": int(np.sum(np.array(counts) > unique)),
                    }
                out["anchor_to_last_window_end_us_quantiles"] = np.quantile(
                    base[:, -1, 1] - arrays["anchor_us"], [0, .5, 1]
                ).tolist()
                out["anchor_to_roi_available_us_quantiles"] = np.quantile(
                    arrays["roi_available_us"] - arrays["anchor_us"], [0, .5, 1]
                ).tolist()
                result["pools"][pool] = out
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.archive)
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        if args.output.resolve() == args.archive.resolve():
            raise ValueError("Output must not replace the source archive.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
