"""Run the unchanged D0 numerical pipeline with an audited persistent I/O adapter.

The existing IDENTITY remains the numerical origin. Every newly written block
receipt additionally binds the immutable I/O execution receipt. No completed
block, input, producer, tensor layout or numerical source is rewritten.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.cached_event_reader import CachedEventReader, ReaderPool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--io-qa", type=Path, required=True)
    parser.add_argument("--io-qa-sha256", required=True)
    parser.add_argument("--execution-receipt", type=Path, required=True)
    args, forwarded = parser.parse_known_args()
    work = Path(__file__).resolve().parent.parent
    runner = work / "scripts/build_simplex_t_context_features.py"
    reader = work / "src/e_jepa_ttc/simplex_t/cached_event_reader.py"
    base_hash = "aaa80071284b680d5c8befd6fc788d0c535fb9530aa61554cd1294624c195783"
    if sha256(runner) != base_hash or sha256(args.io_qa) != args.io_qa_sha256:
        raise ValueError("base runner or I/O QA changed")
    qa = json.loads(args.io_qa.read_text("utf-8"))
    if qa["status"] != "EXACT_PASS" or qa["reader_sha256"] != sha256(reader):
        raise ValueError("persistent I/O requires exact current-source real input QA")
    if [row["query"] for row in qa["records"]] != [2522, 1343] or any(
        not row["tensor_equal"] or len(set(row["tensor_sha256"])) != 1 for row in qa["records"]
    ):
        raise ValueError("fixed input QA missing")
    if "--output" not in forwarded:
        raise ValueError("explicit original output required")
    output = Path(forwarded[forwarded.index("--output") + 1]).resolve(strict=True)
    if output != (work / "artifacts/simplex_t/T1/context_features_fp32").resolve():
        raise ValueError("this adapter is only for the existing D0 replay")
    if (output.parent / "CURRENT_REPLAY.lock").exists():
        raise RuntimeError("existing replay owns its lease; no takeover")
    identity = output / "IDENTITY.json"
    execution = {
        "schema": "simplex_t_exact_io_execution_v1",
        "authorization": (
            "User 2026-09-08: implement and apply optimizations, preserve valid caches"
        ),
        "numerical_identity_sha256": sha256(identity),
        "base_runner_sha256": base_hash,
        "wrapper_sha256": sha256(Path(__file__)),
        "reader_sha256": sha256(reader),
        "input_qa": {"path": str(args.io_qa.resolve()), "sha256": args.io_qa_sha256},
        "arguments": forwarded,
        "change": "Retain one read-only HDF5 sequence and five dataset handles",
        "per_dataset_cache_bytes": 8388608,
        "unchanged": ["FP32", "H16 batch16", "TF32 off", "ROI", "time windows", "producers"],
        "optimizer_updates": 0,
    }
    write_new_json(args.execution_receipt, execution)
    execution_hash = sha256(args.execution_receipt)
    namespace = runpy.run_path(str(runner), run_name="simplex_t_pinned_numerical_runner")
    entry = namespace["main"]
    pool = ReaderPool()

    @contextmanager
    def borrowed_reader(path: str | Path) -> Iterator[CachedEventReader]:
        yield pool.get(path)

    def attributed_write(path: Path, payload: dict) -> None:
        if "query" in payload and "family" in payload and "sha256" in payload:
            payload = {
                **payload,
                "io_execution": {
                    "path": str(args.execution_receipt.resolve()),
                    "sha256": execution_hash,
                },
            }
        write_new_json(path, payload)

    entry.__globals__["EAPEventReader"] = borrowed_reader
    entry.__globals__["write_new_json"] = attributed_write
    previous = sys.argv
    try:
        sys.argv = [str(runner), *forwarded]
        entry()
    finally:
        sys.argv = previous
        pool.close()


if __name__ == "__main__":
    main()
