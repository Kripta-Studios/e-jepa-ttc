"""Materialize the full-run manifest, then delegate to the frozen campaign CLI."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from operational.efficient_context.common import atomic_json, digest

ROOT = Path(__file__).resolve().parents[2]


def materialize_manifest(baseline: Path, full: Path) -> dict[str, object]:
    """Publish a byte-exact manifest once; never replace a differing destination."""
    source = baseline.resolve() / "QUERY_MANIFEST.json"
    destination = full.resolve() / "QUERY_MANIFEST.json"
    if not source.is_file():
        raise FileNotFoundError(source)
    expected = digest(source)
    full.resolve().mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or digest(destination) != expected:
            raise ValueError(f"Existing full manifest differs: {destination}")
    else:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=full.resolve(), delete=False) as output:
                temporary = Path(output.name)
                with source.open("rb") as input_stream:
                    shutil.copyfileobj(input_stream, output)
                output.flush()
                os.fsync(output.fileno())
            if digest(temporary) != expected:
                raise ValueError("Temporary manifest copy failed hash verification")
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                if not destination.is_file() or digest(destination) != expected:
                    raise ValueError(
                        f"Concurrent full manifest differs: {destination}"
                    ) from error
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    receipt: dict[str, object] = {
        "status": "BYTE_EXACT_IMMUTABLE_COPY",
        "source": str(source),
        "destination": str(destination),
        "sha256": expected,
        "bytes": source.stat().st_size,
        "optimizer_updates": 0,
        "labels_read": False,
    }
    receipt_path = full.resolve() / "QUERY_MANIFEST_MATERIALIZATION.json"
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        if existing != receipt:
            raise ValueError("Manifest materialization receipt differs")
    else:
        atomic_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/sota_campaign_20261008"))
    parser.add_argument("--baseline-output", type=Path)
    parser.add_argument("--full-output", type=Path)
    args, forwarded = parser.parse_known_args()
    baseline = args.baseline_output or args.output / "dev32_expanded"
    full = args.full_output or args.output / "dev32_expanded_rgb"
    materialize_manifest(baseline, full)
    command = [
        sys.executable,
        "-m",
        "operational.sota_eval.campaign",
        "--output",
        str(args.output),
        "--baseline-output",
        str(baseline),
        "--full-output",
        str(full),
        *forwarded,
    ]
    if os.name == "nt":
        return subprocess.run(
            command, cwd=ROOT, check=False, creationflags=subprocess.CREATE_NO_WINDOW
        ).returncode
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
