"""Prepare or verify the immutable TRAIN24-only DINO teacher cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

MAXIMUM_RSS_BYTES = 1024**3


def _sha256(path: Path) -> str:
    with path.resolve(strict=True).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _require_owned_path(path: Path, root: Path, name: str) -> Path:
    """Resolve and require a non-root child of the independent output directory."""
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve()
    if resolved == resolved_root or not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{name} must stay below {resolved_root}: {resolved}")
    return resolved


class _RSSMonitor:
    def __init__(self) -> None:
        import psutil

        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="teacher-cache-rss", daemon=True)

    def _run(self) -> None:
        while not self.stop.wait(0.1):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self) -> _RSSMonitor:
        self.thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop.set()
        self.thread.join(timeout=2)
        self.peak = max(self.peak, self.process.memory_info().rss)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse fail-closed conversion arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--weight-path", type=Path)
    parser.add_argument(
        "--fold-contract",
        type=Path,
        default=Path("output/independent_phase/train24_population_v2/FOLD_CONTRACT.json"),
    )
    parser.add_argument(
        "--authorization",
        type=Path,
        help="Root-reviewed SHIP authorization; conversion refuses to start without it.",
    )
    parser.add_argument(
        "--row-locality-receipt",
        type=Path,
        default=Path("output/independent_phase/teacher_row_locality_v1.json"),
    )
    parser.add_argument(
        "--teacher-crosswalk-receipt",
        type=Path,
        default=Path("output/independent_phase/teacher_crosswalk_v3/RECEIPT.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/independent_phase/train24_teacher_cache_v2"),
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--manifest-sha256")
    parser.add_argument(
        "--verification-receipt",
        type=Path,
        help="New JSON receipt written after a successful verify-only invocation.",
    )
    return parser.parse_args(argv)


def main() -> int:
    """Convert atomically or verify an already pinned standalone cache."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from operational.independent_refit.teacher_cache import (
        prepare_teacher_cache,
        verify_teacher_cache,
    )

    args = parse_args()
    allowed_root = Path("output/independent_phase").resolve(strict=True)
    output = _require_owned_path(args.output_dir, allowed_root, "output")
    if args.verify_only:
        if not args.manifest_sha256:
            raise ValueError("--verify-only requires --manifest-sha256")
        verification_receipt = None
        if args.verification_receipt is not None:
            verification_receipt = _require_owned_path(
                args.verification_receipt,
                allowed_root,
                "verification receipt",
            )
            if verification_receipt.exists():
                raise FileExistsError(
                    f"Refusing to overwrite verification receipt: {verification_receipt}"
                )
        verify_started_utc = datetime.now(UTC)
        verify_started = time.perf_counter()
        result = verify_teacher_cache(
            output / "MANIFEST.json",
            expected_manifest_sha256=args.manifest_sha256,
            fold_contract_path=args.fold_contract,
        )
        module_path = Path(__file__).resolve().parents[1] / (
            "operational/independent_refit/teacher_cache.py"
        )
        receipt = {
            "schema": "independent_phase_teacher_cache_post_publish_verification_v1",
            "status": "PASS",
            "started_utc": verify_started_utc.isoformat(),
            "ended_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - verify_started,
            "manifest_sha256": args.manifest_sha256,
            "fold_contract_sha256": _sha256(args.fold_contract),
            "script_sha256": _sha256(Path(__file__)),
            "converter_sha256": _sha256(module_path),
            "result": result,
            "process_id": os.getpid(),
            "gpu_used": False,
            "teacher_model_deserialized": False,
        }
        if verification_receipt is not None:
            verification_receipt.parent.mkdir(parents=True, exist_ok=True)
            verification_receipt.write_text(
                json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.verification_receipt is not None:
        raise ValueError("--verification-receipt is valid only with --verify-only")
    if args.manifest_sha256:
        raise ValueError("--manifest-sha256 is valid only with --verify-only")
    missing = [
        name
        for name, value in (
            ("--source-root", args.source_root),
            ("--weight-path", args.weight_path),
            ("--authorization", args.authorization),
            ("--row-locality-receipt", args.row_locality_receipt),
            ("--teacher-crosswalk-receipt", args.teacher_crosswalk_receipt),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"Conversion requires: {', '.join(missing)}")
    if output.name != "train24_teacher_cache_v2":
        raise ValueError("Conversion output must be the fresh train24_teacher_cache_v2 path")
    assert args.source_root is not None
    assert args.weight_path is not None
    assert args.authorization is not None
    assert args.row_locality_receipt is not None
    assert args.teacher_crosswalk_receipt is not None
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = _require_owned_path(
        output.parent / f".{output.name}.building.{uuid.uuid4().hex}",
        allowed_root,
        "staging",
    )
    started_utc = datetime.now(UTC)
    started = time.perf_counter()
    try:
        with _RSSMonitor() as memory:
            result = prepare_teacher_cache(
                source_root=args.source_root,
                weight_path=args.weight_path,
                fold_contract_path=args.fold_contract,
                authorization_path=args.authorization,
                row_locality_receipt_path=args.row_locality_receipt,
                teacher_crosswalk_receipt_path=args.teacher_crosswalk_receipt,
                output_dir=staging,
            )
        if memory.peak >= MAXIMUM_RSS_BYTES:
            raise MemoryError(
                f"Teacher conversion RSS gate failed: {memory.peak} >= {MAXIMUM_RSS_BYTES}"
            )
        module_path = Path(__file__).resolve().parents[1] / (
            "operational/independent_refit/teacher_cache.py"
        )
        execution = {
            "schema": "independent_phase_teacher_cache_execution_v1",
            "status": "PASS_CPU_CONVERSION_SELF_VERIFIED",
            "started_utc": started_utc.isoformat(),
            "ended_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_bytes": memory.peak,
            "maximum_rss_bytes": MAXIMUM_RSS_BYTES,
            "threads": {
                name: os.environ[name]
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "script_sha256": _sha256(Path(__file__)),
            "converter_sha256": _sha256(module_path),
            "authorization_sha256": _sha256(args.authorization),
            "fold_contract_sha256": _sha256(args.fold_contract),
            "row_locality_receipt_sha256": _sha256(args.row_locality_receipt),
            "teacher_crosswalk_receipt_sha256": _sha256(
                args.teacher_crosswalk_receipt
            ),
            "source_manifest_sha256": _sha256(args.source_root / "TEACHER_MANIFEST.json"),
            "weight_file_sha256": _sha256(args.weight_path),
            "manifest_sha256": result["manifest_sha256"],
            "rows": result["rows"],
            "shards": result["shards"],
            "self_verified_before_publish": result["self_verified_before_publish"],
            "torch_imported_by_converter": False,
            "teacher_model_deserialized": False,
            "gpu_used": False,
            "optimizer_updates": 0,
        }
        (staging / "EXECUTION_RECEIPT.json").write_text(
            json.dumps(execution, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if _sha256(staging / "MANIFEST.json") != result["manifest_sha256"]:
            raise ValueError("Manifest changed after self-verification")
        _require_owned_path(staging, allowed_root, "staging before publish")
        _require_owned_path(output, allowed_root, "output before publish")
        os.replace(staging, output)
    except Exception as error:
        if staging.exists():
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            failed = _require_owned_path(
                output.parent / f"{output.name}.failed.{stamp}.{uuid.uuid4().hex[:8]}",
                allowed_root,
                "failure target",
            )
            _require_owned_path(staging, allowed_root, "failed staging")
            shutil.move(str(staging), str(failed))
            (failed / "FAILED.json").write_text(
                json.dumps(
                    {
                        "status": "FAILED_PRESERVED",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"failed_attempt={failed}", file=sys.stderr)
        raise
    result["manifest"] = str(output / "MANIFEST.json")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
