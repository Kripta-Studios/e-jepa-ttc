"""Download planning helpers for public dataset manifests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from e_jepa_ttc.utils.io import read_structured


def build_download_plan(
    *,
    manifest_path: str | Path,
    root: str | Path,
    sequences: tuple[str, ...] = (),
    kinds: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Build a deterministic list of public dataset download actions."""

    manifest = read_structured(manifest_path)
    root_path = Path(root)
    selected_sequences = set(sequences)
    selected_kinds = set(kinds)
    plan: list[dict[str, Any]] = []
    for sequence in manifest.get("sequences", []):
        if not isinstance(sequence, dict):
            continue
        sequence_id = str(sequence.get("sequence_id", ""))
        if selected_sequences and sequence_id not in selected_sequences:
            continue
        local_dir = root_path / str(sequence.get("local_dir", sequence_id))
        assets = sequence.get("assets", {})
        if not isinstance(assets, dict):
            continue
        for kind, asset in assets.items():
            if selected_kinds and kind not in selected_kinds:
                continue
            if not isinstance(asset, dict):
                continue
            url = asset.get("url")
            output = asset.get("output")
            asset_kind = str(asset.get("kind", "file"))
            if not url or not output:
                continue
            output_path = local_dir / str(output)
            plan.append(
                {
                    "sequence_id": sequence_id,
                    "asset": str(kind),
                    "kind": asset_kind,
                    "url": str(url),
                    "output": output_path.as_posix(),
                    "size_gb": asset.get("size_gb"),
                }
            )
    return plan


def build_gdown_command(
    item: dict[str, Any],
    *,
    python: str | Path | None = None,
    quiet: bool = False,
    resume: bool = False,
) -> list[str]:
    """Build the gdown command for one planned download."""

    python_executable = str(python or sys.executable)
    output = Path(str(item["output"]))
    command = [python_executable, "-m", "gdown"]
    if quiet:
        command.append("-q")
    if resume:
        command.append("--continue")
    if item["kind"] == "folder":
        command.append("--folder")
    command.extend([str(item["url"]), "-O", output.as_posix()])
    return command


def run_gdown_plan(
    plan: list[dict[str, Any]],
    *,
    python: str | Path | None = None,
    quiet: bool = False,
    resume: bool = False,
) -> None:
    """Execute a download plan with gdown."""

    for item in plan:
        output = Path(str(item["output"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        command = build_gdown_command(item, python=python, quiet=quiet, resume=resume)
        subprocess.run(command, check=True)
