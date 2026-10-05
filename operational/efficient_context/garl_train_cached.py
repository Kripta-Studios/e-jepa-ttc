"""Resume the unchanged scientific producer recipe with separately sealed cache placement."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from .common import ROOT, Campaign, atomic_json, digest, read


class GuardedCampaign(Campaign):
    """Pause our worker at an optimizer boundary if another heavy trainer starts."""

    def __init__(self, protocol: Path) -> None:
        super().__init__(protocol)
        self.heavy_scan_time = float("-inf")
        self.other_trainers: list[int] = []

    def check(self) -> bool:
        """Preserve every original resource check and the one-heavy-trainer contract."""
        import psutil

        if not super().check():
            return False
        if time.monotonic() - self.heavy_scan_time >= 5:
            own = psutil.Process()
            related = {
                os.getpid(),
                *(p.pid for p in own.parents()),
                *(p.pid for p in own.children(recursive=True)),
            }
            self.other_trainers = []
            for process in psutil.process_iter(["pid", "name", "cmdline"]):
                if process.pid in related or not (process.info["name"] or "").lower().startswith(
                    "python"
                ):
                    continue
                line = " ".join(process.info["cmdline"] or []).lower()
                if "train" in line and any(
                    v in line
                    for v in (
                        "operational.efficient_context.run train",
                        "operational.efficient_context.garl_train",
                        "operational.efficient_context.garl_heads",
                        *(f"stage{s}" for s in range(70, 77)),
                    )
                ):
                    self.other_trainers.append(process.pid)
            self.heavy_scan_time = time.monotonic()
            if self.other_trainers:
                atomic_json(
                    self.out / "garl/EXTERNAL_TRAINER_RESOURCE_PAUSE.json",
                    {
                        "other_heavy_trainer_pids": self.other_trainers,
                        "our_worker_will_checkpoint_at_optimizer_boundary": True,
                        "other_processes_modified": False,
                    },
                )
        return not self.other_trainers


def main() -> int:
    """Keep every frozen scientific file, checkpoint identity and budget unchanged."""
    from . import garl_train
    from .exclusive_cache import ExclusiveInputCache

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/campaign/efficient_context_v1.json"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    c = GuardedCampaign(args.protocol)
    c.freeze()
    qa_path = c.out / "garl/CACHE_ENGINEERING_QA.json"
    qa = read(qa_path)
    cache_file = Path(__file__).with_name("exclusive_cache.py")
    if (
        qa["status"] != "PASSED"
        or qa["cache_implementation_sha256"] != digest(cache_file)
        or qa["native_protocol_sha256"] != digest(c.out / "garl/PROTOCOL.json")
        or qa["QA_implementation_sha256"]
        != digest(Path(__file__).with_name("cache_engineering_qa.py"))
    ):
        raise ValueError("cache engineering QA does not match this original scientific protocol")
    value = {
        "native_protocol_sha256": digest(c.out / "garl/PROTOCOL.json"),
        "original_scientific_sources_modified": False,
        "original_recipe_checkpoint_identity_RNG_sampler_unchanged": True,
        "reason": "retain disk-evicted FP32 tensors in the already authorized6GiB RAM cache",
        "memory_disk_capacity_increase": False,
        "CACHE_ENGINEERING_QA_sha256": digest(qa_path),
        "files": [
            {"path": str(p), "sha256": digest(p)}
            for p in (
                Path(__file__),
                cache_file,
                Path(__file__).with_name("cache_engineering_qa.py"),
            )
        ],
        "optimizer_updates_for_QA": 0,
    }
    pin = c.out / "garl/CACHE_ENGINEERING_FREEZE.json"
    if pin.exists() and read(pin) != value:
        raise ValueError("cache engineering implementation changed after its separate freeze")
    if not pin.exists():
        atomic_json(pin, value)
    garl_train.InputCache = ExclusiveInputCache
    garl_train.execute(c)
    if (c.out / "garl/ENDPOINTS.json").exists():
        atomic_json(
            c.out / "garl/PRODUCER_RUNTIME_PROVENANCE.json",
            {
                "native_protocol_sha256": value["native_protocol_sha256"],
                "producer_endpoints_sha256": digest(c.out / "garl/ENDPOINTS.json"),
                "cache_engineering_freeze_sha256": digest(pin),
                "all_original_scientific_files_preserved": True,
                "cache_QA_raw_encodings_and_optimizer_updates": 0,
            },
        )
    return 0 if (c.out / "garl/ENDPOINTS.json").exists() else 3


if __name__ == "__main__":
    raise SystemExit(main())
