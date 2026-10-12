"""Local paths, immutable receipts and bounded CPU admission."""

from __future__ import annotations

import csv
import hashlib
import json
import os

# ruff: noqa: ANN401
from pathlib import Path
from typing import Any  # Dynamic JSON receipts from historical campaigns.

import psutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/multimodal_validation_20261012"
V13 = ROOT.parent / "e-jepa-ttc-v13-rgb-port"
V12 = ROOT.parent / "e-jepa-ttc-v12-efficient-context"
RUN = V13 / "artifacts/rgb_port_20261008"
REG = Path("E:/EJEPA_results/fcwd_rgb_calibration_20261012/campaign/registration")
MANIFEST = V12 / "artifacts/sota_campaign_20261008/fcwd_population/QUERY_MANIFEST.json"
CODE = Path("E:/Garl-TTC")


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path: str | Path) -> str:
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def csvwrite(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def admission() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    rss = psutil.Process().memory_info().rss
    swap = psutil.swap_memory()
    own_total = 0
    research_total = 0
    for process in psutil.process_iter(["name", "cmdline", "memory_info"]):
        try:
            if not (process.info["name"] or "").lower().startswith("python"):
                continue
            command = " ".join(process.info["cmdline"] or [])
            size = process.info["memory_info"].rss
            if "operational.multimodal_validation" in command:
                own_total += size
            if "operational." in command:
                research_total += size
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    value = dict(
        available_bytes=memory.available,
        own_rss=rss,
        own_queue_rss=own_total,
        research_rss=research_total,
        commit_headroom=swap.free,
        cpu_threads=1,
        device="cpu",
    )
    if (
        memory.available < 2 * 1024**3
        or max(rss, own_total) > 4 * 1024**3
        or research_total > 23 * 1024**3
        or swap.free < 1024**3
        or psutil.disk_usage(str(ROOT)).free < 10 * 1024**3
    ):
        raise MemoryError(f"V16 resource admission failed: {value}")
    return value


def registrations() -> dict[str, dict[str, Any]]:
    seal = read(REG / "REGISTRATION_SEALED.json")
    assert sha(REG / "BINDING.json") == seal["binding_sha256"]
    assert sha(MANIFEST) == read(REG / "BINDING.json")["manifest_sha256"]
    result = {}
    for name, expected in seal["query_sha256"].items():
        path = REG / "queries" / name
        assert sha(path) == expected, name
        value = read(path)
        assert value["binding_sha256"] == seal["binding_sha256"]
        result[value["query_id"]] = value
    assert len(result) == seal["population"] == 630
    return result
