"""Explicit SSH transfer command; no transfer occurs when importing or preparing."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from .inventory import digest, read


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", default="root")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--identity-file", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", args.host) or args.host.startswith("-"):
        raise ValueError("Invalid SSH host")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.user) or args.user.startswith("-"):
        raise ValueError("Invalid SSH user")
    if not 1 <= args.port <= 65535:
        raise ValueError("Invalid SSH port")
    receipt = read(args.archive.with_suffix(args.archive.suffix + ".json"))
    if digest(args.archive) != receipt["sha256"]:
        raise ValueError("Archive changed after packaging")
    command = [
        "ssh",
        "-T",
        "-p",
        str(args.port),
        "-i",
        str(args.identity_file),
        f"{args.user}@{args.host}",
        "set -eu; test ! -e /workspace/ejepa_bundle; "
        "mkdir /workspace/ejepa_bundle; tar -xf - -C /workspace/ejepa_bundle",
    ]
    print("Transfer target: /workspace/ejepa_bundle (must not exist).")
    print(f"Bytes to transfer: {receipt['bytes']}; no remote tar copy is retained.")
    if not args.execute:
        print("DRY RUN: pass --execute only after the Pod and SSH host identity are confirmed.")
        return
    # Binary stdin avoids PowerShell's text-pipeline corruption on older versions.
    # Default OpenSSH host-key verification is retained; no credential is printed.
    with args.archive.open("rb") as stream:
        subprocess.run(command, stdin=stream, check=True)


if __name__ == "__main__":
    main()
