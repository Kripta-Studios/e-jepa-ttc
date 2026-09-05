"""Run the pre-remediation source baseline inside this worktree, without raw data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output_root.resolve()
    if not output.is_relative_to(repo):
        raise ValueError("baseline QA must remain inside the authorized worktree")
    output.mkdir(parents=True, exist_ok=False)
    commit = "a9c2d6e8689e24c4eed4acac7aa9b6eede7f8a84"
    archive_path = output / "baseline_source.zip"
    subprocess.run(
        ["git", "archive", "--format=zip", f"--output={archive_path}", commit], cwd=repo, check=True
    )
    source = output / "source"
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            path = source / name
            if not path.resolve().is_relative_to(source.resolve()) or path.suffix in {
                ".pt",
                ".pth",
                ".h5",
                ".hdf5",
                ".parquet",
                ".npy",
                ".npz",
            }:
                raise ValueError(f"baseline archive contains a data/weight member: {name}")
        archive.extractall(source)
    environment = {
        **os.environ,
        "PYTHONPATH": str(source / "src") + os.pathsep + str(source),
        "UV_PROJECT_ENVIRONMENT": sys.prefix,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    command = [
        sys.executable,
        "-B",
        "-m",
        "pytest",
        "tests",
        "-q",
        "--tb=short",
        "--junitxml",
        str(output / "baseline.xml"),
    ]
    start = time.time()
    log = output / "baseline.log"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=source,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    record = {
        "source_commit": commit,
        "source_archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "command": command,
        "cwd": str(source),
        "python": sys.version,
        "python_executable": sys.executable,
        "exit_code": result.returncode,
        "elapsed_seconds": time.time() - start,
        "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        "scope": "versioned_source_and_metadata_only_no_raw_data_or_checkpoints_copied",
    }
    (output / "BASELINE_QA.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"baseline_exit_code": result.returncode}), flush=True)


if __name__ == "__main__":
    main()
