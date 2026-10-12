"""User-authorized, bounded continuation; no ownership of historical campaigns."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import psutil
import torch

from operational.multimodal_validation.common import ROOT, admission, read, sha, write

OUT = ROOT / "artifacts/multimodal_research_20261012"


def authorize() -> dict:
    path = OUT / "AUTHORIZATION.json"
    if not path.exists():
        write(
            path,
            dict(
                utc=datetime.now(UTC).isoformat(),
                user_text=(
                    "sí puedes usar la GPU otras 18h y hasta 1M de updates o lo que necesites"
                ),
                max_physical_updates=1000000,
                max_elapsed_seconds=18 * 3600,
                initial_scientific_plan=5000,
                no_push_merge_submission=True,
                V15_closeout=read(
                    ROOT.parent
                    / (
                        "e-jepa-ttc-v15-range-readout/artifacts/range_readout_20261012/CLOSEOUT.json"
                    )
                ),
            ),
        )
    return read(path)


def accounting() -> dict:
    items = []
    for p in sorted((OUT / "fits").glob("*/UPDATE_JOURNAL.json")):
        item = read(p)
        item["fit"] = p.parent.name
        items.append(item)
    result: dict = dict(
        scientific=sum(v["completed_updates"] for v in items),
        recovery_upper=sum(v["recovery_upper"] for v in items),
        pending_upper=sum(v["pending_update_upper"] for v in items),
        fits=items,
    )
    result["physical_upper"] = (
        result["scientific"] + result["recovery_upper"] + result["pending_upper"]
    )
    write(OUT / "ACCOUNTING.json", result)
    return result


def guard(*, prospective_updates: int = 0) -> None:
    authorization = authorize()
    elapsed = (datetime.now(UTC) - datetime.fromisoformat(authorization["utc"])).total_seconds()
    if elapsed >= authorization["max_elapsed_seconds"]:
        raise RuntimeError("Authorized 18-hour execution window exhausted")
    if accounting()["physical_upper"] + prospective_updates > authorization["max_physical_updates"]:
        raise RuntimeError("Authorized physical optimizer budget exhausted")
    admission()
    current = psutil.Process()
    own = {os.getpid(), *(p.pid for p in current.parents())}
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            if p.pid in own or not (p.info["name"] or "").lower().startswith("python"):
                continue
            args = p.info["cmdline"] or []
            if "-m" in args:
                module = args[args.index("-m") + 1]
                if module.startswith("operational.") and not module.startswith(
                    "operational.multimodal_research"
                ):
                    raise RuntimeError(f"Live research GPU owner must finish: {p.pid} {module}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def setup() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    guard()


def source_hashes() -> dict:
    return {
        str(p.relative_to(ROOT)): sha(p)
        for p in [
            Path(__file__).parent / name
            for name in ("common.py", "model.py", "cache.py", "train.py")
        ]
    }
