"""Resource and protocol admission shared without importing a training producer."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

import torch

from e_jepa_ttc.detection.campaign_training import CampaignConfig
from e_jepa_ttc.detection.evaluation_protocol import EvaluationProtocol


def load_protocol(value: str) -> EvaluationProtocol:
    """Load the fixed default or a fully serialized protocol."""
    if value == "default":
        return EvaluationProtocol.default()
    return EvaluationProtocol.from_dict(json.loads(Path(value).read_text(encoding="utf-8")))


def resource_admission(config: CampaignConfig) -> dict[str, object]:
    """Refuse model construction when current host headroom is below the fixed gate."""
    if os.name == "nt":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        available_ram, total_ram = status.available_physical, status.total_physical
    else:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_ram = os.sysconf("SC_AVPHYS_PAGES") * page_size
        total_ram = os.sysconf("SC_PHYS_PAGES") * page_size
    free_vram, total_vram = torch.cuda.mem_get_info()
    gib = 1024**3
    minimum_ram = int(config.resources.minimum_available_ram_gb * gib)
    minimum_vram = int(config.resources.minimum_available_vram_gb * gib)
    receipt: dict[str, object] = {
        "available_ram_bytes": available_ram,
        "total_ram_bytes": total_ram,
        "available_vram_bytes": free_vram,
        "total_vram_bytes": total_vram,
        "minimum_available_ram_bytes": minimum_ram,
        "minimum_available_vram_bytes": minimum_vram,
    }
    if available_ram < minimum_ram:
        raise RuntimeError("Available RAM is below the preregistered campaign admission gate")
    if free_vram < minimum_vram:
        raise RuntimeError("Available VRAM is below the preregistered campaign admission gate")
    return receipt | {"status": "PASS"}
