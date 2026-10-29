"""Bind the completed D0 catalogs before dispatching a DENSE replay or verifier.

The launch hash is resolved only after the pinned builder validates current
authoritative inputs. No missing catalog, role or timing gate is bypassed.
"""

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
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--builder-sha256", required=True)
    parser.add_argument("--worker-sha256", required=True)
    parser.add_argument("--compile-outer", type=int, choices=(0, 1, 2))
    parser.add_argument("--compilation-controller-sha256")
    parser.add_argument("--compilation-worker-sha256")
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    parser.add_argument("--max-new-queries", type=int, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.other_reserved_bytes < 0 or args.max_new_queries < 1:
        parser.error("nonnegative pending output bytes and positive replay slice required")
    compiling = args.compile_outer is not None
    if compiling != bool(args.compilation_controller_sha256 and args.compilation_worker_sha256):
        parser.error("compilation requires outer fold and both compilation script pins")
    if not compiling and (args.compilation_controller_sha256 or args.compilation_worker_sha256):
        parser.error("compilation pins require a fold")
    if compiling and args.report is not None:
        parser.error("compilation uses inherited orchestration logs, not a replay report")
    if not compiling and args.verify_only == (args.report is not None):
        parser.error("execution requires a report; verification must not write one")
    work = Path(__file__).resolve().parents[1]
    launch = args.launch.resolve()
    if not launch.is_relative_to(work / "artifacts"):
        raise ValueError("DENSE launch must be inside companion artifacts")
    if args.report is not None and (
        args.report.exists() or not args.report.resolve().is_relative_to(work / "artifacts")
    ):
        raise ValueError("new companion report required")
    local_bytes = args.local_paths.read_bytes()
    if Path(json.loads(local_bytes)["worktree"]).resolve() != work:
        raise ValueError("local paths name a different worktree")
    builder = Path(__file__).with_name("prepare_simplex_t_dense_launch.py")
    worker = Path(__file__).with_name("execute_simplex_t_expanded_replay.py")
    controller = Path(__file__).with_name("execute_simplex_t_compilation_attempt.py")
    compilation_worker = Path(__file__).with_name("run_simplex_t_companion.py")

    def check_pins() -> None:
        if args.local_paths.read_bytes() != local_bytes:
            raise ValueError("local paths changed during DENSE dispatch")
        for path, expected in ((builder, args.builder_sha256), (worker, args.worker_sha256)):
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("DENSE dispatch script pin changed")
        if compiling:
            for path, expected in (
                (controller, args.compilation_controller_sha256),
                (compilation_worker, args.compilation_worker_sha256),
            ):
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError("DENSE compilation script pin changed")

    def invoke(command: list[str]) -> int:
        check_pins()
        print(json.dumps({"command": command}), flush=True)
        code = subprocess.run(command, cwd=work, check=False).returncode
        check_pins()
        return code

    common = [
        "--local-paths",
        str(args.local_paths.resolve()),
        "--other-reserved-bytes",
        str(args.other_reserved_bytes),
    ]
    command = [sys.executable, "-B", str(builder), *common, "--output", str(launch)]
    if launch.exists() or args.verify_only:
        command.append("--verify-only")
    code = invoke(command)
    if code:
        return code
    # Snapshot only an actually validated launch; the worker rechecks this hash.
    if launch.stat().st_size > 1_048_576:
        raise ValueError("oversized validated DENSE launch")
    launch_hash = hashlib.sha256(launch.read_bytes()).hexdigest()
    if compiling:
        config = json.loads(launch.read_bytes())
        catalog = config["d0_reuse"][str(args.compile_outer)]
        compile_common = [
            "--local-paths",
            str(args.local_paths.resolve()),
            "--other-reserved-bytes",
            str(args.other_reserved_bytes + config["reserved_output_bytes"]),
        ]
        command = [
            sys.executable,
            "-B",
            str(controller),
            *compile_common,
            "--worker-sha256",
            args.compilation_worker_sha256,
            "--output",
            str(work / f"artifacts/simplex_t/T1/compiled_dense_context/outer{args.compile_outer}"),
            "--outer",
            str(args.compile_outer),
            "--pool",
            "DENSE_OLD",
            "--reuse-d0-compiled",
            str(work / catalog["compiled"]),
            "--reuse-d0-compiled-sha256",
            catalog["compiled_sha256"],
            "--verify-only" if args.verify_only else "--resume",
        ]
        code = invoke(command)
        if hashlib.sha256(launch.read_bytes()).hexdigest() != launch_hash:
            raise ValueError("validated DENSE launch changed during compilation")
        return code
    command = [
        sys.executable,
        "-B",
        str(worker),
        *common,
        "--config",
        str(launch),
        "--config-sha256",
        launch_hash,
        "--max-new-queries",
        str(args.max_new_queries),
    ]
    if args.verify_only:
        command.append("--verify-only")
    else:
        assert args.report is not None
        command += ["--report", str(args.report.resolve())]
    code = invoke(command)
    if hashlib.sha256(launch.read_bytes()).hexdigest() != launch_hash:
        raise ValueError("validated DENSE launch changed during replay")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
