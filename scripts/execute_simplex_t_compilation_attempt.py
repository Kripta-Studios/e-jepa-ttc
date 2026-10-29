"""Compile in a retained attempt directory and publish only verified complete output.

Interrupted compilation restarts from cached blocks in a fresh attempt; it does not
rerun experts or optimizer updates. Existing canonical outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import uuid
from pathlib import Path

from e_jepa_ttc.simplex_t.coordination import shared_write_admission
from e_jepa_ttc.simplex_t.lifecycle import ExclusiveLease, admitted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--worker-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--pool", choices=("D0", "D1", "DENSE_OLD"), required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    parser.add_argument("--reuse-d0-compiled", type=Path)
    parser.add_argument("--reuse-d0-compiled-sha256")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.other_reserved_bytes < 0:
        parser.error("nonnegative pending output reservation required")
    if bool(args.reuse_d0_compiled) != bool(args.reuse_d0_compiled_sha256):
        parser.error("both D0 reuse pins required")
    if args.reuse_d0_compiled and args.pool != "DENSE_OLD":
        parser.error("reuse is restricted to DENSE_OLD")
    work = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to(work / "artifacts") or output == work / "artifacts":
        raise ValueError("compiled destination must be below companion artifacts")
    local_bytes = args.local_paths.read_bytes()
    paths = json.loads(local_bytes)
    if Path(paths["worktree"]).resolve() != work:
        raise ValueError("local paths name a different worktree")
    worker = Path(__file__).with_name("run_simplex_t_companion.py")

    def boundary() -> bool:
        if args.local_paths.read_bytes() != local_bytes:
            raise ValueError("local paths changed during compilation")
        if hashlib.sha256(worker.read_bytes()).hexdigest() != args.worker_sha256:
            raise ValueError("compilation worker SHA256 changed")
        resource = admitted([work])
        return resource["has_headroom"] and shared_write_admission(
            resource["written_volume_free_bytes"][0], args.other_reserved_bytes + 1_048_576
        )

    def invoke(destination: Path, *, verify: bool) -> int:
        if not boundary():
            return 3
        command = [
            sys.executable,
            "-B",
            str(worker),
            "prepare",
            "--local-paths",
            str(args.local_paths.resolve()),
            "--compile-fold",
            str(args.outer),
            "--compile-pool",
            args.pool,
            "--output",
            str(destination),
            "--other-reserved-bytes",
            str(args.other_reserved_bytes + 1_048_576),
        ]
        if args.reuse_d0_compiled:
            command += [
                "--reuse-d0-compiled",
                str(args.reuse_d0_compiled.resolve()),
                "--reuse-d0-compiled-sha256",
                args.reuse_d0_compiled_sha256,
            ]
        if verify:
            command.append("--verify-only")
        print(json.dumps({"command": command}), flush=True)
        code = subprocess.run(command, cwd=work, check=False).returncode
        if not boundary():
            return 3
        return code

    if not boundary():
        return 3
    if args.verify_only:
        return invoke(output, verify=True)
    if output.exists():
        if not args.resume:
            raise FileExistsError("existing output requires --resume verification")
        return invoke(output, verify=True)
    # The lock names only this publication target, never the GPU replay owner.
    with ExclusiveLease(output.with_name(output.name + ".compile.lock")):
        if output.exists():
            raise FileExistsError("compiled output appeared before compilation lease")
        attempt = output.with_name(output.name + ".attempt_" + uuid.uuid4().hex)
        code = invoke(attempt, verify=False)
        if code:
            return code
        code = invoke(attempt, verify=True)
        if code:
            return code
        if not boundary():
            return 3
        if output.exists():
            raise FileExistsError("refusing to overwrite compiled output")
        # Same-parent rename on Windows refuses an existing destination. The
        # completed child has exited, so its memmaps no longer hold file handles.
        attempt.rename(output)
        print(
            json.dumps(
                {
                    "status": "COMPILED_PUBLISHED_NOT_SCIENTIFIC_FREEZE",
                    "output": str(output),
                    "optimizer_updates": 0,
                }
            ),
            flush=True,
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
