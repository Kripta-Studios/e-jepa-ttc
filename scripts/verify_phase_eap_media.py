"""Verify local Test12 tensor and RGB bytes against the frozen public inventory."""

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

from operational.evttc_transfer_audit.common import file_sha
from operational.final_refit.test12 import TRUSTED_INPUTS_SHA256, _contains_private_key
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_campaign.data import read


def main() -> None:
    """Hash sensor files without decoding arrays, executing models, or opening targets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh verification directory")
    if file_sha(args.inputs) != TRUSTED_INPUTS_SHA256:
        raise ValueError("Input inventory differs from the frozen public authority")
    inputs = read(args.inputs)
    if _contains_private_key(inputs):
        raise ValueError("Unexpected private field in public inputs")
    files = []
    for token, row in sorted(inputs["event_observations"].items()):
        files.append(
            {
                "kind": "event",
                "identity": token,
                "path": str(Path(row["current_tensor_path"]).resolve()),
                "expected_sha256": row["current_tensor_sha256"],
                "expected_bytes": None,
            }
        )
    for relative, pin in sorted(inputs["rgb_sources"].items()):
        path = (args.media_root / relative).resolve()
        if not path.is_relative_to(args.media_root.resolve()):
            raise ValueError("RGB path escapes media root")
        files.append(
            {
                "kind": "rgb",
                "identity": relative,
                "path": str(path),
                "expected_sha256": pin["sha256"],
                "expected_bytes": pin["size_bytes"],
            }
        )
    if len(inputs["event_observations"]) != 5805 or len(inputs["rgb_sources"]) != 36:
        raise ValueError("Expected the complete physical Test12 sensor inventory")
    if len({row["path"] for row in files}) != len(files):
        raise ValueError("Unexpected duplicate physical file")
    started = time.perf_counter()
    source = Path(__file__)
    atomic_receipt(
        args.output / "FREEZE.json",
        {
            "schema": "phase_eap_media_verification_v1",
            "inputs_sha256": TRUSTED_INPUTS_SHA256,
            "script_sha256": file_sha(source),
            "files": files,
            "start_time": datetime.now(UTC).isoformat(),
            "targets_read": False,
            "npz_arrays_decoded": False,
            "producer_calls": 0,
        },
    )
    results = []
    total_bytes = 0
    try:
        for index, row in enumerate(files):
            path = Path(row["path"])
            before = path.stat()
            if row["expected_bytes"] is not None and before.st_size != row["expected_bytes"]:
                raise ValueError(f"Size differs: {path}")
            digest = file_sha(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"File changed while hashing: {path}")
            if digest != row["expected_sha256"]:
                raise ValueError(f"SHA-256 differs: {path}")
            results.append({"path": str(path), "sha256": digest, "bytes": after.st_size})
            total_bytes += after.st_size
            if (index + 1) % 100 == 0 or row["kind"] == "rgb":
                elapsed = time.perf_counter() - started
                print(
                    f"{index + 1}/{len(files)} {row['kind']} "
                    f"verified={total_bytes / 1e9:.3f}GB elapsed={elapsed:.1f}s",
                    flush=True,
                )
        freeze = read(args.output / "FREEZE.json")
        if file_sha(source) != freeze["script_sha256"]:
            raise ValueError("Verifier source changed during execution")
        if file_sha(args.inputs) != TRUSTED_INPUTS_SHA256:
            raise ValueError("Input inventory changed during execution")
        atomic_receipt(args.output / "FILES.json", {"verified": results})
        atomic_receipt(
            args.output / "COMPLETE.json",
            {
                "status": "COMPLETE",
                "event_tensors": 5805,
                "rgb_shards": 36,
                "verified_bytes": total_bytes,
                "elapsed_seconds": time.perf_counter() - started,
                "end_time": datetime.now(UTC).isoformat(),
                "inputs_sha256": TRUSTED_INPUTS_SHA256,
                "freeze_sha256": file_sha(args.output / "FREEZE.json"),
                "files_sha256": file_sha(args.output / "FILES.json"),
                "targets_read": False,
                "npz_arrays_decoded": False,
                "producer_calls": 0,
                "scope": "Byte verification only; no candidate inference or accuracy claim",
            },
        )
        print("COMPLETE", flush=True)
    except Exception as exc:
        atomic_receipt(
            args.output / "FAILED.json",
            {
                "error": repr(exc),
                "files_verified_before_failure": len(results),
                "targets_read": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
