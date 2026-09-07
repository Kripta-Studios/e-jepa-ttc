"""Run a pinned frozen-phase template with a fresh operational receipt per attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import uuid
from pathlib import Path

from e_jepa_ttc.simplex_t.coordination import shared_write_admission
from e_jepa_ttc.simplex_t.lifecycle import admitted


def pinned_bytes(path: Path, expected: str) -> bytes:
    """Read bounded metadata or entrypoint bytes and check their exact identity."""
    with path.open("rb") as stream:
        payload = stream.read(1_048_577)
    if len(payload) > 1_048_576 or hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("attempt input size or SHA256 differs")
    return payload


def materialize_attempt(template: Path, template_sha256: str, work: Path) -> tuple[Path, str]:
    """Change only resource_receipt; leave all scientific and output bindings intact."""
    payload = pinned_bytes(template, template_sha256)
    config = json.loads(payload)
    if config.get("schema") != "simplex_t_frozen_phase_launch_v2":
        raise ValueError("frozen-phase launch v2 required")
    if Path(config["roots"]["work"]).resolve(strict=True) != work.resolve(strict=True):
        raise ValueError("attempt belongs to another worktree")
    if "resource_receipt" not in config:
        raise ValueError("template resource receipt field required")
    directory = (work / "artifacts/simplex_t/resource_observations").resolve()
    if not directory.is_relative_to(work.resolve(strict=True)):
        raise ValueError("resource directory escapes companion")
    token = uuid.uuid4().hex
    launch = directory / f"attempt_{token}.launch.json"
    config["resource_receipt"] = str(directory / f"attempt_{token}.receipt.json")
    result = (json.dumps(config, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if len(result) > 1_048_576:
        raise ValueError("materialized launch exceeds metadata reservation")
    directory.mkdir(parents=True, exist_ok=True)
    with launch.open("xb") as stream:
        stream.write(result)
    return launch, hashlib.sha256(result).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--template-sha256", required=True)
    parser.add_argument("--worker-sha256", required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    parser.add_argument("--own-reserved-bytes", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.other_reserved_bytes < 0 or args.own_reserved_bytes < 2_097_152:
        parser.error("nonnegative other and at least 2097152 own reserved bytes required")
    work = Path(__file__).resolve().parents[1]
    if args.verify_only and (args.report is not None or args.resume):
        parser.error("verification must not request an attempt report or execution resume")
    if not args.verify_only and args.report is None:
        parser.error("execution requires a new --report")
    report = args.report.resolve() if args.report is not None else None
    if report is not None and (not report.is_relative_to(work / "artifacts") or report.exists()):
        raise ValueError("new attempt report inside companion artifacts required")
    worker = Path(__file__).with_name("execute_simplex_t_frozen_phase.py")
    pinned_bytes(worker, args.worker_sha256)
    snapshot = admitted([work])
    if not snapshot["has_headroom"] or not shared_write_admission(
        snapshot["written_volume_free_bytes"][0],
        args.other_reserved_bytes + args.own_reserved_bytes,
    ):
        print("PAUSED_RESOURCE: attempt metadata admission; no worker started")
        return 3
    if args.verify_only:
        config = json.loads(pinned_bytes(args.template, args.template_sha256))
        if (
            config.get("schema") != "simplex_t_frozen_phase_launch_v2"
            or Path(config["roots"]["work"]).resolve(strict=True) != work
        ):
            raise ValueError("verification template must belong to this companion worktree")
        launch, digest = args.template, args.template_sha256
    else:
        launch, digest = materialize_attempt(args.template, args.template_sha256, work)
    command = [
        sys.executable,
        "-B",
        str(worker),
        "--launch",
        str(launch),
        "--launch-sha256",
        digest,
        "--other-reserved-bytes",
        str(args.other_reserved_bytes),
        "--own-reserved-bytes",
        str(args.own_reserved_bytes),
    ]
    if args.resume:
        command.append("--resume")
    if args.verify_only:
        command.append("--verify-only")
        pinned_bytes(worker, args.worker_sha256)
        pinned_bytes(args.template, args.template_sha256)
        return subprocess.run(command, check=False).returncode
    record = {
        "schema": "simplex_t_phase_attempt_v1",
        "template": str(args.template.resolve()),
        "template_sha256": args.template_sha256,
        "launch": str(launch),
        "launch_sha256": digest,
        "worker_sha256": args.worker_sha256,
        "command": command,
        "campaign_complete": False,
    }
    # Publish lineage before execution, including attempts terminated externally.
    # A missing return code is unknown, never evidence of zero completed work.
    assert report is not None
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False)
    pinned_bytes(worker, args.worker_sha256)
    pinned_bytes(args.template, args.template_sha256)
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
