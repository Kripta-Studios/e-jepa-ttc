"""Retry postprocessing in fresh directories while preserving every partial attempt."""

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


def read_pin(path: Path, digest: str) -> dict:
    """Read a bounded exact JSON input, never a directory-discovered data source."""
    payload = read_bytes(path)
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("postprocessing attempt input SHA256 differs")
    return json.loads(payload)


def read_bytes(path: Path) -> bytes:
    with path.open("rb") as stream:
        payload = stream.read(1_048_577)
    if len(payload) > 1_048_576:
        raise ValueError("postprocessing attempt metadata exceeds 1 MiB")
    return payload


def write_new(path: Path, value: dict) -> str:
    payload = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if len(payload) > 1_048_576:
        raise ValueError("postprocessing attempt metadata exceeds 1 MiB")
    with path.open("xb") as stream:
        stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def relocated(template: dict, directory: Path) -> dict:
    """Only analysis and delivery destinations may differ between attempts."""
    config = json.loads(json.dumps(template))
    config["output"] = str(directory / "analysis")
    config["delivery"]["output"] = str(directory / "delivery")
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--template-sha256", required=True)
    parser.add_argument("--worker-sha256", required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    parser.add_argument("--own-reserved-bytes", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.other_reserved_bytes < 0 or args.own_reserved_bytes < 4_194_304:
        parser.error("nonnegative other and at least 4194304 own reserved bytes required")
    work = Path(__file__).resolve().parents[1]
    root = args.attempt_root.resolve()
    if not root.is_relative_to(work / "artifacts"):
        raise ValueError("attempt root must be inside companion artifacts")
    template = read_pin(args.template, args.template_sha256)
    if (
        template.get("schema") != "simplex_t_postprocessing_launch_v3"
        or Path(template["roots"]["work"]).resolve(strict=True) != work
        or Path(template["output"]).resolve() != root / "analysis"
        or Path(template["delivery"]["output"]).resolve() != root / "delivery"
    ):
        raise ValueError("v3 template must bind this worktree and attempt-root destinations")
    worker = Path(__file__).with_name("postprocess_simplex_t_campaign.py")

    def boundary() -> bool:
        read_pin(args.template, args.template_sha256)
        if hashlib.sha256(read_bytes(worker)).hexdigest() != args.worker_sha256:
            raise ValueError("postprocessing worker SHA256 differs")
        snapshot = admitted([work])
        return snapshot["has_headroom"] and shared_write_admission(
            snapshot["written_volume_free_bytes"][0],
            args.other_reserved_bytes + args.own_reserved_bytes,
        )

    def invoke(launch: Path, digest: str, verify: bool) -> int:
        if not boundary():
            return 3
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
            str(args.own_reserved_bytes - 3_145_728),
        ]
        if verify:
            command.append("--verify-only")
        print(
            json.dumps(
                {"status": "POSTPROCESSING_WORKER_START", "command": command},
                ensure_ascii=True,
            ),
            flush=True,
        )
        # Child stdout/stderr remain attached to the foreground orchestrator log.
        return subprocess.run(command, check=False).returncode

    if not boundary():
        return 3
    if root.exists() and not (args.resume or args.verify_only):
        raise ValueError("existing attempt root requires explicit resume")
    # Inspect only our metadata namespace. Never discover datasets or predictions.
    attempts = sorted(root.glob("attempt_*/ATTEMPT.json")) if root.exists() else []
    for record_path in attempts:
        if not boundary():
            return 3
        record = json.loads(read_bytes(record_path))
        launch = record_path.parent / "launch.json"
        if (
            not record_path.resolve().is_relative_to(root)
            or record.get("schema") != "simplex_t_postprocessing_attempt_v1"
            or record.get("template_sha256") != args.template_sha256
            or record.get("worker_sha256") != args.worker_sha256
            or read_pin(launch, record["launch_sha256"]) != relocated(template, record_path.parent)
        ):
            raise ValueError("attempt lineage or non-output configuration changed")
        code = invoke(launch, record["launch_sha256"], True)
        if code != 10:
            return code
    if args.verify_only:
        return 10
    # Every prior incomplete analysis is retained. No fit endpoint is changed.
    if not boundary():
        return 3
    directory = root / f"attempt_{uuid.uuid4().hex}"
    directory.mkdir(parents=True)
    launch = directory / "launch.json"
    digest = write_new(launch, relocated(template, directory))
    write_new(
        directory / "ATTEMPT.json",
        {
            "schema": "simplex_t_postprocessing_attempt_v1",
            "template_sha256": args.template_sha256,
            "worker_sha256": args.worker_sha256,
            "launch_sha256": digest,
            "optimizer_updates_executed": 0,
        },
    )
    return invoke(launch, digest, False)


if __name__ == "__main__":
    raise SystemExit(main())
