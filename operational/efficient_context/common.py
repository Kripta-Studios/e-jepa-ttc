"""Independent roots, durable records and per-update resource admission."""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from e_jepa_ttc.simplex_t.campaign_sources import CampaignSources
    from e_jepa_ttc.simplex_t.registry import FitSpec

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from operational.simplex_t_closure.runtime import (  # noqa: E402
    atomic_bytes,
    atomic_json,
    digest,
    publish_json,
)


def read(path: Path) -> dict:
    """Read a JSON boundary record without touching referenced payloads."""
    return json.loads(path.read_text(encoding="utf-8"))


class Campaign:
    """Resolve explicitly separated read-only historical and new writable roots."""

    def __init__(self, protocol: Path) -> None:
        self.config_path = protocol.resolve()
        self.config = read(self.config_path)
        self.historical = (ROOT / self.config["historical_root"]).resolve()
        self.out = (ROOT / self.config["new_output_root"]).resolve()
        if self.out == self.historical or self.historical in self.out.parents:
            raise ValueError("new outputs must be outside historical roots")
        self.policy_path = ROOT / self.config["policy"]
        self.policy = read(self.policy_path)
        self.out.mkdir(parents=True, exist_ok=True)
        self.h16_path = (
            self.historical / "artifacts/simplex_t/h16_replication_20261003/PROTOCOL.json"
        )
        self.h16 = read(self.h16_path)
        self.launch = self.h16["launch"]
        self.local = read(Path(self.launch["local_paths"]))
        self.raw = Path(self.local["eap_root"]) / "data/train"
        self.last_check = 0.0

    def sources(self) -> CampaignSources:
        """Open the acknowledged immutable D1 caches without invoking any old stage."""
        from e_jepa_ttc.simplex_t.configuration_preflight import (
            open_acknowledged_source_configuration,
        )

        return open_acknowledged_source_configuration(
            Path(self.launch["local_paths"]),
            Path(self.launch["source_configuration"]),
            self.launch["source_configuration_sha256"],
        )[0]

    def check(self) -> bool:
        """Enforce RAM/tree RSS/commit/disk and scientific recovery ceilings."""
        from operational.simplex_t_h16_replication.common import memory

        m = memory()
        reasons = []
        if m["available"] < self.policy["min_host_available_gib"] * 1024**3:
            reasons.append("AVAILABLE_RAM_BELOW_2_GIB")
        if m["tree_rss"] > self.policy["max_tree_rss_gib"] * 1024**3:
            reasons.append("TREE_RSS_ABOVE_4_GIB")
        if m["free_disk"] - 10_000_000_000 < 10_000_000_000:
            reasons.append("DISK_AFTER_10GB_RESERVATION_BELOW_10GB")
        if m["commit_headroom"] < 1024**3:
            reasons.append("COMMIT_HEADROOM_BELOW_1_GIB")
        budget = self.out / "PHYSICAL_WORK.json"
        if budget.exists():
            a = read(budget)["accounting"]
            if a["scientific_uncertain_lost_upper"] > self.policy["recovery_updates_max"]:
                reasons.append("RECOVERY_CAP_EXCEEDED")
            if a["scientific_saved_updates"] > 22500:
                reasons.append("WIDE_CAP_EXCEEDED")
        if reasons or time.monotonic() - self.last_check > 5:
            atomic_json(self.out / "RESOURCES.json", {"snapshot": m, "reasons": reasons})
            self.last_check = time.monotonic()
        return not reasons

    def require_resources(self) -> None:
        """Pause at a safe boundary without changing scientific status."""
        if not self.check():
            raise InterruptedError("PAUSED_RESOURCE: " + str(self.out / "RESOURCES.json"))

    def freeze(self) -> dict:
        """Verify the preregistered scientific protocol before reuse."""
        p = self.out / "PROTOCOL.json"
        pin = (self.out / "PROTOCOL.sha256").read_text().split()[0]
        if digest(p) != pin:
            raise ValueError("campaign freeze changed")
        result = read(p)
        for row in result["science_files"]:
            if digest(Path(row["path"])) != row["sha256"]:
                raise ValueError("frozen scientific implementation changed: " + row["path"])
        if digest(self.config_path) != result["config_sha256"]:
            raise ValueError("campaign configuration changed")
        if digest(self.policy_path) != result["policy_sha256"]:
            raise ValueError("campaign policy changed")
        return result


def spec(s: CampaignSources, fold: int, length: int = 16) -> FitSpec:
    """Address an existing read-only source identity, never add an arm to old graph."""
    from e_jepa_ttc.simplex_t.registry import FitSpec

    return next(
        v
        for v in s.graph
        if v == FitSpec("T3" if length == 16 else "T2", f"TPR-D1-H{length}-C160", fold, 7)
    )


def release(s: CampaignSources) -> None:
    """Release retained memmaps between independent folds."""
    s.release()
    gc.collect()


def npz(path: Path, **arrays: object) -> None:
    """Atomically persist numeric analysis or inference fragments."""
    import io

    import numpy as np

    b = io.BytesIO()
    cast(Callable[..., object], np.savez_compressed)(b, **arrays)
    atomic_bytes(path, b.getvalue())


class Lease:
    """One writer/trainer in this campaign; preserve dead owner receipts."""

    def __init__(self, out: Path) -> None:
        self.path = out / "WRITER.lock"

    def __enter__(self) -> Lease:
        import psutil

        if self.path.exists():
            old = read(self.path)
            if psutil.pid_exists(old["pid"]):
                owner = psutil.Process(old["pid"])
                if owner.create_time() == old["create_time"]:
                    raise RuntimeError("live campaign writer")
            publish_json(
                self.path.with_name("STALE_OWNER_" + digest(self.path)[:12] + ".json"), old
            )
            self.path.unlink()
        with self.path.open("x", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "create_time": psutil.Process().create_time()}, f)
            f.flush()
            os.fsync(f.fileno())
        return self

    def __exit__(self, *args: object) -> None:
        if read(self.path)["pid"] != os.getpid():
            raise ValueError("writer ownership changed")
        self.path.unlink()
