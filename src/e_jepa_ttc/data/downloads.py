"""Download planning helpers for public dataset manifests."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
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


def google_uc_to_usercontent_url(url: str) -> str:
    """Convert a Google Drive uc URL from gdown listings into a direct download URL."""

    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    file_id = query.get("id", [""])[0]
    if not file_id:
        return url
    return "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(
        {"id": file_id, "export": "download"}
    )


def _safe_listing_path(path: str) -> Path:
    relative_path = PurePosixPath(path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"unsafe listing path: {path}")
    return Path(*relative_path.parts)


def download_gdown_listing(
    *,
    listing_path: str | Path,
    output_dir: str | Path,
    skip_existing: bool = True,
    suffixes: tuple[str, ...] = (),
    retries: int = 3,
    retry_delay_s: float = 2.0,
) -> list[dict[str, str]]:
    """Download files from a `gdown --json` folder listing via direct HTTP URLs."""

    listing = json.loads(Path(listing_path).read_text(encoding="utf-8-sig"))
    output_root = Path(output_dir)
    selected_suffixes = tuple(suffix.lower() for suffix in suffixes)
    records: list[dict[str, str]] = []
    for item in listing:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", ""))
        url = str(item.get("url", ""))
        if not path or not url:
            continue
        if selected_suffixes and not path.lower().endswith(selected_suffixes):
            continue
        relative_path = _safe_listing_path(path)
        output_path = output_root / relative_path
        download_url = google_uc_to_usercontent_url(url)
        if skip_existing and output_path.exists() and output_path.stat().st_size > 0:
            records.append({"status": "skipped", "path": output_path.as_posix(), "url": url})
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _urlretrieve_with_retries(
            download_url,
            output_path,
            retries=retries,
            retry_delay_s=retry_delay_s,
        )
        records.append({"status": "downloaded", "path": output_path.as_posix(), "url": url})
    return records


def _urlretrieve_with_retries(
    url: str,
    output_path: Path,
    *,
    retries: int,
    retry_delay_s: float,
) -> None:
    if retries < 0:
        raise ValueError("retries must be non-negative")
    for attempt in range(retries + 1):
        try:
            urllib.request.urlretrieve(url, output_path)
            return
        except Exception:
            if output_path.exists() and output_path.stat().st_size == 0:
                output_path.unlink()
            if attempt >= retries:
                raise
            time.sleep(retry_delay_s)
