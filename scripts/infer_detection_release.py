"""Verify and run a sealed Dfull RGB + TRAIN-fitted ranking candidate.

The parent detector and ranking modules execute in separate fresh processes.
No ground-truth argument is accepted. The eAP media reader supplies calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any


def digest(path: Path) -> str:
    """Hash a file with bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json_bytes(path: Path, value: object) -> str:
    """Write exactly the UTF-8 bytes being hashed, including on Windows."""
    encoded = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def safe_member(root: Path, relative: str) -> Path:
    """Reject absolute paths, traversal, Windows drives, and escaping symlinks."""
    name = PurePosixPath(relative)
    if (
        not relative
        or name.is_absolute()
        or ".." in name.parts
        or "\\" in relative
        or ":" in relative
    ):
        raise ValueError("Unsafe release path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Release path escapes package")
    return path


def verify_release(root: Path) -> dict[str, Any]:
    """Verify every declared file and bind the ranker to the exact detector."""
    value = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    if value.get("schema") != "phase_detection_release_v1" or not isinstance(
        value.get("files"), dict
    ):
        raise ValueError("Unsupported release manifest")
    for name, expected in value["files"].items():
        path = safe_member(root, name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Release file hash differs: {name}")
    for key in ("checkpoint", "ranker"):
        if value.get(key) not in value["files"]:
            raise ValueError(f"Unbound release {key}")
    ranker = json.loads(safe_member(root, value["ranker"]).read_text(encoding="utf-8"))
    if ranker.get("provenance", {}).get("checkpoint_sha256") != value["files"][value["checkpoint"]]:
        raise ValueError("Release ranker belongs to a different detector")
    return value


def read_tokens(path: Path) -> list[str]:
    """Accept only a nonempty, unique label-free token population."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        if set(value) != {"sample_tokens"}:
            raise ValueError("Token input must not include labels or metadata")
        value = value["sample_tokens"]
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(t, str) or not t for t in value)
    ):
        raise ValueError("Expected nonempty sample tokens")
    if len(value) != len(set(value)):
        raise ValueError("Duplicate sample tokens")
    return value


def detector_worker(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    """Observe the unchanged hash-verified frozen inference producer."""
    checkpoint = safe_member(args.release, manifest["checkpoint"])
    loader_path = checkpoint.parent.parent / "code/scripts/infer_detection_campaign.py"
    spec = importlib.util.spec_from_file_location("_sealed_detection_loader", loader_path)
    if spec is None or spec.loader is None:
        raise ImportError("Frozen detector loader is unavailable")
    loader = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loader
    spec.loader.exec_module(loader)
    frames: list[dict[str, Any]] = []
    predictions = loader.infer(checkpoint, args.data_root, args.tokens, diagnostic_frames=frames)
    from e_jepa_ttc.detection.diagnostic_export import build_sidecar

    prediction_sha = write_json_bytes(args.output / "baseline.json", predictions)
    sidecar = build_sidecar(
        frames,
        checkpoint_sha256=digest(checkpoint),
        prediction_output_sha256=prediction_sha,
        observer_path=checkpoint.parent.parent
        / "code/src/e_jepa_ttc/detection/diagnostic_export.py",
    )
    write_json_bytes(args.output / "diagnostics.json", sidecar)


def ranking_worker(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    """Apply the frozen TRAIN-only ranker and verify unchanged proposal geometry."""
    sys.path.insert(0, str(args.release / "ranking_runtime/src"))
    from e_jepa_ttc.detection.evaluation_protocol import EvaluationProtocol
    from e_jepa_ttc.detection.quality_ranking import (
        apply_quality_ranker,
        proposal_geometry_sha256,
    )

    baseline_path = args.output / "baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    sidecar = json.loads((args.output / "diagnostics.json").read_text(encoding="utf-8"))
    if sidecar["prediction_output_sha256"] != digest(baseline_path):
        raise ValueError("Diagnostic sidecar differs from prediction bytes")
    checkpoint_sha = manifest["files"][manifest["checkpoint"]]
    if (
        baseline["checkpoint_sha256"] != checkpoint_sha
        or sidecar["checkpoint_sha256"] != checkpoint_sha
    ):
        raise ValueError("Outputs belong to another checkpoint")
    if baseline["annotations_returned"] != 0 or baseline["targets_argument"] is not None:
        raise ValueError("Inference was not label-free")
    bundle = json.loads(safe_member(args.release, manifest["ranker"]).read_text(encoding="utf-8"))
    protocol = EvaluationProtocol.default()
    ranked = apply_quality_ranker(baseline["predictions"], sidecar, bundle, protocol=protocol)
    before = proposal_geometry_sha256(baseline["predictions"], protocol=protocol)
    if proposal_geometry_sha256(ranked, protocol=protocol) != before:
        raise ValueError("Ranker changed proposal geometry")
    output = dict(baseline)
    output.update(
        schema="phase_ttc_detection_ranked_predictions_v1",
        source_schema=baseline["schema"],
        predictions=ranked,
        quality_ranker_bundle_sha256=bundle["bundle_sha256"],
        fixed_proposal_pool=True,
        label_access_during_apply=False,
        score_semantics="relative detection ranking; not a calibrated probability or TTC score",
    )
    write_json_bytes(args.output / "predictions.json", output)


def main() -> int:
    """Validate a release or execute a bounded, fresh-process replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "infer"))
    parser.add_argument("--release", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--tokens", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--worker", choices=("detector", "ranking"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.release = args.release.resolve()
    manifest = verify_release(args.release)
    if args.command == "verify":
        print(json.dumps({"status": "VERIFIED", "files": len(manifest["files"])}))
        return 0
    if args.tokens is None or args.data_root is None or args.output is None:
        parser.error("infer requires --data-root, --tokens and --output")
    tokens = read_tokens(args.tokens)
    if len(tokens) > 128 or not 1 <= args.cpu_threads <= 16:
        parser.error("Use 1–128 frames per call and 1–16 CPU threads; split larger populations")
    args.output = args.output.resolve()
    if args.output.is_relative_to(args.release):
        parser.error("Keep output outside the immutable release directory")
    if args.worker:
        (detector_worker if args.worker == "detector" else ranking_worker)(args, manifest)
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    write_json_bytes(args.output / "tokens.json", {"sample_tokens": tokens})
    started = time.perf_counter()
    env = dict(
        os.environ, OMP_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads)
    )
    for worker in ("detector", "ranking"):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "infer",
            "--release",
            str(args.release),
            "--data-root",
            str(args.data_root.resolve()),
            "--tokens",
            str(args.output / "tokens.json"),
            "--output",
            str(args.output),
            "--cpu-threads",
            str(args.cpu_threads),
            "--worker",
            worker,
        ]
        with (args.output / f"{worker}.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    receipt = {
        "status": "COMPLETE",
        "sample_count": len(tokens),
        "labels_read": False,
        "manifest_sha256": digest(args.release / "MANIFEST.json"),
        "elapsed_seconds_including_startup": time.perf_counter() - started,
        "python": platform.python_version(),
        "host": platform.node(),
        "files": {
            name: digest(args.output / name)
            for name in ("tokens.json", "baseline.json", "diagnostics.json", "predictions.json")
        },
    }
    write_json_bytes(args.output / "RECEIPT.json", receipt)
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
