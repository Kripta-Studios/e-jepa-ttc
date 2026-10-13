"""Resume existing primary checkpoints with the explicitly admitted fast runtime."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from unittest.mock import patch

import psutil
import torch

from operational.range_transfer.common import digest, freeze, read, sha, utc, write
from operational.rgb_port.train_producers import ProducerCheckpoint

from . import run as primary
from . import train
from .common import ROOT, RUN
from .fast_runtime import PersistentSensors, batched_backward, install_graph
from .profile_fast import micros
from .qa import exact

OUT = RUN / "runtime/optimization_20261013"


def no_other_writer() -> None:
    """Check PID ancestry, worktree and all known primary queue entrypoints."""
    own = psutil.Process()
    excluded = {own.pid, *(p.pid for p in own.parents())}
    modules = {
        "operational.joint_measurement." + name for name in ("run", "fast_run", "mapped_run")
    }
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        if process.pid in excluded or "python" not in (process.info["name"] or "").lower():
            continue
        if modules.intersection(process.info["cmdline"] or []):
            try:
                if Path(process.cwd()).resolve() == ROOT:
                    raise RuntimeError(f"Existing primary writer {process.pid}")
            except psutil.NoSuchProcess:
                continue


def binding() -> dict:
    """Require exact-state admission and unchanged scientific and runtime sources."""
    admission = read(OUT / "ADMISSION.json")
    if admission["status"] != "PASS" or not all(admission["checks"].values()):
        raise ValueError("Runtime did not pass exact continuation and performance admission")
    if admission["numerical_graph_policy_sha256"] != sha(OUT / "NUMERICAL_GRAPH_POLICY.json"):
        raise ValueError("Numerical graph policy changed after admission")
    for name, expected in admission["runtime_sources"].items():
        if sha(ROOT / "operational/joint_measurement" / name) != expected:
            raise ValueError("Runtime source changed after admission")
    scientific = read(RUN / "fits/NATIVE-P2-FT-S7/IDENTITY.json")["sources"]
    if scientific != train.sources():
        raise ValueError("Primary scientific identity changed")
    paths = [
        ROOT / "operational/joint_measurement" / name
        for name in ("fast_run.py", "fast_runtime.py", "profile_fast.py", "run.py")
    ]
    runtime_sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    train.committed_sources(runtime_sources)
    result = {
        "schema": "v18_lossless_fast_runtime_v1",
        "status": "ADMITTED",
        "runtime_sources": runtime_sources,
        "scientific_source_hash": digest(scientific),
        "admission_sha256": sha(OUT / "ADMISSION.json"),
        "graph": admission["graph"],
        "numerical_graph_policy_sha256": admission["numerical_graph_policy_sha256"],
        "eager_bitwise_equivalence": False,
        "transition_sha256": sha(RUN / "scheduling/RUNTIME_OPTIMIZATION/STOP_RESULT.json"),
        "origin_updates": admission["origin_updates"],
        "preserve_existing_primary_identity": True,
        "no_optimizer_reset": True,
        "cache_limit_bytes": 8 * 1024**3,
    }
    freeze(OUT / "RUNTIME_BINDING.json", result)
    return result


def run(*, once: bool = False) -> None:
    """Retain the original fit/queue and record each optimized checkpoint segment."""
    no_other_writer()
    record = binding()
    cache = PersistentSensors(limit_bytes=record["cache_limit_bytes"])

    class FastCheckpoint(ProducerCheckpoint):
        """Restore full scientific state before enabling speculative input."""

        def restore(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler,
            generator: torch.Generator,
        ) -> dict:
            cursor = super().restore(model, optimizer, scheduler, generator)
            if self.identity["fit_id"] != "NATIVE-P2-FT-S7":
                raise ValueError("This continuation is admitted only for the unfinished native arm")
            original_ids = {k: id(v) for k, v in model.named_parameters()}
            if record["graph"]:
                install_graph(model)
                saved_sampler = generator.get_state().clone()
                shadow = torch.Generator().set_state(saved_sampler)
                ids = torch.randint(cache.population, (32,), generator=shadow).tolist()
                batches = micros(cache.load(ids))
                for _ in range(3):
                    optimizer.zero_grad(set_to_none=True)
                    batched_backward(model, batches, None, cache.population)
                cursor_after = super().restore(model, optimizer, scheduler, generator)
                if not exact(cursor, cursor_after) or not torch.equal(
                    saved_sampler, generator.get_state()
                ):
                    raise ValueError("Graph warmup changed cursor or sampler")
                del batches
            if original_ids != {k: id(v) for k, v in model.named_parameters()}:
                raise ValueError("Runtime changed optimizer parameter identities")
            optimizer.zero_grad(set_to_none=True)
            cache.prepare(generator.get_state())
            write(
                OUT / "ACTIVE.json",
                {
                    "utc": utc(),
                    "pid": os.getpid(),
                    "completed_updates": self.completed,
                    "durable_updates": self.durable,
                    "recovery_upper": self.recovery_upper,
                    "binding": digest(record),
                    "graph": record["graph"],
                    "cache": cache.counters(),
                    "status": "RESTORED",
                },
            )
            return cursor

    original_fit = train.fit

    def wrapped_fit(fit_id: str) -> dict:
        folder = RUN / "fits" / fit_id
        receipt = read(folder / "CHECKPOINT_RECEIPT.json")
        if receipt["status"] == "COMPLETE":
            return original_fit(fit_id)
        if fit_id != "NATIVE-P2-FT-S7":
            raise ValueError("Unexpected unfinished primary arm")
        segment = (
            folder / "RUNTIME_SEGMENTS" / f"{receipt['completed_updates']:06d}_{time.time_ns()}"
        )
        freeze(
            segment / "START.json",
            {"utc": utc(), "pid": os.getpid(), "before": receipt, "binding": digest(record)},
        )
        result = original_fit(fit_id)
        freeze(
            segment / "END.json",
            {
                "utc": utc(),
                "after": result,
                "start_sha256": sha(segment / "START.json"),
                "binding": digest(record),
                "cache": cache.counters(),
            },
        )
        return result

    def factory(role: str) -> PersistentSensors:
        if role != "P":
            raise ValueError("Optimized training cache only admits P")
        return cache

    try:
        with (
            patch.object(train, "SensorDataset", factory),
            patch.object(train, "ProducerCheckpoint", FastCheckpoint),
            patch.object(train, "backward", batched_backward),
            patch.object(primary, "fit", wrapped_fit),
        ):
            primary.run(once=once)
    finally:
        cache.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    run(once=parser.parse_args().once)
