"""Single V15 continuation owner; fixed queue, conditional replication, no push."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from operational.range_transfer.common import ownership

from .common import ROOT, RUN, read, utc, write


def worker(module: str, *args: str) -> None:
    """Run one stage and retain its log; a failure never retries silently."""
    log = (
        RUN
        / "logs"
        / f"{module.rsplit('.', 1)[-1]}_{'_'.join(args).replace('-', '')}_{time.time_ns()}.log"
    )
    log.parent.mkdir(exist_ok=True)
    command = [sys.executable, "-B", "-m", module, *args]
    write(RUN / "ACTIVE_STAGE.json", {"command": command, "log": str(log), "started_utc": utc()})
    with log.open("wb") as stream:
        result = subprocess.run(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    if result.returncode:
        raise RuntimeError(f"Stage failed ({result.returncode}), diagnose before resume: {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-active-p-extraction", action="store_true")
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT)
    import torch

    from operational.range_transfer.cache import HeadCache

    from .evaluate import SCREEN, evaluate_queue, freeze_queue, select
    from .ledger import accounting
    from .prepare_families import prepare
    from .train_heads import fit as fit_sampling

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with ownership(RUN):
        try:
            if args.wait_active_p_extraction:
                matches = []
                for p in psutil.process_iter(["cmdline"]):
                    command = " ".join(p.info["cmdline"] or [])
                    if (
                        "operational.range_readout.extract_cached P" in command
                        and p.pid != os.getpid()
                    ):
                        matches.append(p)
                write(
                    RUN / "WAITING_EXTRACTION.json",
                    {"pids": [p.pid for p in matches], "started_utc": utc()},
                )
                while not (RUN / "vector_cache/P/COMPLETE.json").exists():
                    if not any(p.is_running() for p in matches):
                        raise RuntimeError(
                            "P extraction stopped before completion; inspect its log"
                        )
                    time.sleep(5)
            if not (RUN / "vector_cache/P/COMPLETE.json").exists():
                worker("operational.range_readout.extract_cached", "P")
            # Sampling is independent; do not change its finished endpoints.
            binding = read(RUN / "ADMISSION.json")
            cache = HeadCache(Path(binding["cache_path"]))
            for mass in (0, 1):
                result = fit_sampling(cache, RUN, binding, mass_id=mass, loss_id=0, seed=7)
                if result["status"] != "COMPLETE":
                    raise InterruptedError("S fit resource pause")
            del cache
            for bound in (0.05, 0.20):
                worker("operational.range_readout.train_readout", "--bound", str(bound))
            if not (RUN / "vector_cache/H/COMPLETE.json").exists():
                worker("operational.range_readout.extract_cached", "H")
            for bound in (0.05, 0.20):
                worker(
                    "operational.range_readout.prepare_families",
                    "H",
                    "--bound",
                    str(bound),
                    "--fit",
                )
            freeze_queue(SCREEN, "screen")
            if not (RUN / "vector_cache/V/COMPLETE.json").exists():
                worker("operational.range_readout.extract_cached", "V")
            for bound in (0.05, 0.20):
                prepare(bound, "V")
            result = select(evaluate_queue("screen"))
            write(RUN / "NEXT_DECISION.json", result)
            if result["family"] is not None:
                family = result["family"]
                names = []
                for seed in (13, 23):
                    if family == "S":
                        cache = HeadCache(Path(binding["cache_path"]))
                        for mass in (0, 1):
                            receipt = fit_sampling(
                                cache, RUN, binding, mass_id=mass, loss_id=0, seed=seed
                            )
                            if receipt["status"] != "COMPLETE":
                                raise InterruptedError("Replica S paused")
                            names.append(receipt["fit_id"])
                    else:
                        for bound in (0.05, 0.20):
                            worker(
                                "operational.range_readout.train_readout",
                                "--bound",
                                str(bound),
                                "--seed",
                                str(seed),
                            )
                        for bound in (0.05, 0.20):
                            worker(
                                "operational.range_readout.prepare_families",
                                "H",
                                "--bound",
                                str(bound),
                                "--seed",
                                str(seed),
                                "--fit",
                            )
                            names.append(f"R-C{round(bound * 100):02d}-CTX-S{seed}")
                freeze_queue(tuple(names), "replication")
                if family == "R":
                    for seed in (13, 23):
                        for bound in (0.05, 0.20):
                            prepare(bound, "V", seed)
                evaluate_queue("replication")
                from .replication import summarize

                result = summarize(family)
                write(RUN / "NEXT_DECISION.json", result)
            accounting(RUN)
            write(
                RUN / "CAMPAIGN_STATE.json",
                {
                    "status": "SCREEN_AND_CONDITIONAL_REPLICATION_COMPLETE",
                    "decision": result,
                    "utc": utc(),
                    "transfer_executed": False,
                },
            )
            print(result, flush=True)
        except BaseException as error:
            write(
                RUN / f"CAMPAIGN_FAILURE_{time.time_ns()}.json",
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "utc": utc(),
                    "action": "preserve all endpoints and diagnose before resuming",
                },
            )
            accounting(RUN)
            raise


if __name__ == "__main__":
    main()
