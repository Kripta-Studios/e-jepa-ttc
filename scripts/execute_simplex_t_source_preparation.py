"""Dispatch real source QA only after the full configuration builder verifies its pins."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--builder-sha256", required=True)
    parser.add_argument("--worker-sha256", required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    work = Path(__file__).resolve().parents[1]
    output, config = args.output.resolve(), args.source_config.resolve()
    if args.other_reserved_bytes < 0 or any(
        not path.is_relative_to(work / "artifacts") for path in (output, config)
    ):
        raise ValueError("companion artifact paths and nonnegative pending output bytes required")
    local_bytes = args.local_paths.read_bytes()
    if Path(json.loads(local_bytes)["worktree"]).resolve() != work:
        raise ValueError("local paths name a different worktree")
    builder = Path(__file__).with_name("prepare_simplex_t_source_configuration.py")
    worker = Path(__file__).with_name("run_simplex_t_companion.py")

    def check_pins() -> None:
        if args.local_paths.read_bytes() != local_bytes:
            raise ValueError("local paths changed during source preparation")
        for path, expected in ((builder, args.builder_sha256), (worker, args.worker_sha256)):
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("source preparation script pin changed")

    def invoke(command: list[str]) -> int:
        check_pins()
        print(json.dumps({"command": command}), flush=True)
        result = subprocess.run(command, cwd=work, check=False).returncode
        check_pins()
        return result

    common = [
        "--local-paths",
        str(args.local_paths.resolve()),
        "--other-reserved-bytes",
        str(args.other_reserved_bytes),
    ]
    # The preceding orchestration step creates the configuration. Never create
    # or alter it here, particularly not during read-only verification.
    result = invoke(
        [sys.executable, "-B", str(builder), *common, "--output", str(config), "--verify-only"]
    )
    if result:
        return result
    if config.stat().st_size > 1_048_576:
        raise ValueError("oversized source configuration")
    config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
    command = [
        sys.executable,
        "-B",
        str(worker),
        "prepare",
        *common,
        "--source-identities",
        "--source-config",
        str(config),
        "--source-config-sha256",
        config_hash,
        "--output",
        str(output),
    ]
    if args.verify_only:
        command.append("--verify-only")
    elif (output / "SOURCE_PREPARATION.json").exists():
        command.append("--resume")
    result = invoke(command)
    if hashlib.sha256(config.read_bytes()).hexdigest() != config_hash:
        raise ValueError("source configuration changed during identity preparation")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
