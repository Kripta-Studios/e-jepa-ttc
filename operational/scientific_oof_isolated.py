"""Isolate OOF producers in fresh processes, retaining the frozen training code.

Only orchestration changes. The original producer CLI performs all updates,
sampler restoration, resource checks and checkpoint transactions. A worker-owned
OS lock prevents concurrent fits even if the waiting pipeline process crashes.
"""

# ruff: noqa: ANN401 -- adapter preserves the frozen train callable's keyword interface.
from __future__ import annotations

import gc
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from operational.final_refit.continuation_queue import queue_lock
from operational.final_refit.recipe import sha256
from operational.final_refit.scientific_authority import load_config
from operational.final_refit.scientific_oof_queue import validate_plan
from operational.oof_runtime_binding import (
    bind_fit,
    load_admission,
    runtime_sources,
    verify_runtime_endpoint,
)
from operational.rgb_port.accounting import atomic_write_json, read_json_shared


def validate_runtime(run: Path) -> dict[str, Any]:
    """Require the actual external entrypoint in the active queue source closure."""
    config = load_config(run)
    plan = read_json_shared(run / "SCIENTIFIC_OOF_QUEUE_PLAN.json")
    validate_plan(plan)
    own = Path(__file__).resolve()
    if plan["source_bindings"].get(str(own)) != sha256(own):
        raise ValueError("Isolated orchestration is not bound by the queue plan")
    load_admission(run)
    admission_path = (run / "OOF_RUNTIME_ADMISSION.json").resolve()
    if plan["source_bindings"].get(str(admission_path)) != sha256(admission_path):
        raise ValueError("Queue does not bind the global runtime admission")
    for path in runtime_sources():
        if plan["source_bindings"].get(str(path)) != sha256(path):
            raise ValueError(f"Queue omits a runtime dependency: {path}")
    return config


def isolated_train(**kwargs: Any) -> dict[str, Any]:
    """Delegate the original producer call without changing its scientific inputs."""
    run = Path(kwargs["run"]).resolve()
    config = validate_runtime(run)
    source = kwargs["source"]
    rotation = int(source.identity["rotation"])
    fit = kwargs["fit"]
    if kwargs.get("max_updates") is not None or kwargs.get("additional_stops"):
        raise ValueError("Isolated queue accepts full producers and the standard stop contract")
    if (
        Path(kwargs["config"]).resolve() != Path(config["config"]).resolve()
        or Path(kwargs["contract"]).resolve() != Path(config["fold_contract"]).resolve()
        or kwargs.get("device", "cuda") != config["device"]
    ):
        raise ValueError("Isolated launch differs from the frozen configuration")
    # This parent source has only been indexed, never decoded. Leave its lifecycle
    # to the original pipeline finally block; it owns no producer CUDA model.
    gc.collect()
    command = [
        sys.executable,
        "-m",
        "operational.scientific_oof_isolated",
        "producer",
        "--run",
        str(run),
        "--rotation",
        str(rotation),
        "--fit",
        fit,
    ]
    result = subprocess.run(
        command, check=False, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    )
    if result.returncode not in (0, 3):
        raise RuntimeError(f"Isolated {fit} failed with exit code {result.returncode}")
    directory = run / "rotations" / str(rotation) / "fits" / fit
    receipt = read_json_shared(directory / "CHECKPOINT_RECEIPT.json")
    journal = read_json_shared(directory / "UPDATE_JOURNAL.json")
    if (
        receipt["completed_updates"] != journal["completed_updates"]
        or journal["completed_updates"] != journal["durable_updates"]
        or journal["pending_update_upper"] != 0
        or sha256(Path(receipt["checkpoint_path"])) != receipt["checkpoint_sha256"]
    ):
        raise ValueError("Isolated worker did not leave a durable checkpoint")
    if (result.returncode == 0) != (receipt["status"] == "COMPLETE"):
        raise ValueError("Worker exit and checkpoint status disagree")
    pointer = read_json_shared(directory / "CHECKPOINT_POINTER.json")
    if (
        pointer["checkpoint_sha256"] != receipt["checkpoint_sha256"]
        or pointer["completed_updates"] != receipt["completed_updates"]
    ):
        raise ValueError("Worker checkpoint pointer differs")
    import torch

    saved = torch.load(Path(receipt["checkpoint_path"]), map_location="cpu", weights_only=False)
    if (
        saved["identity"]["source_identity"] != dict(source.identity)
        or saved["identity"]["fit_id"] != fit
    ):
        raise ValueError("Worker source identity differs from the pipeline")
    if result.returncode == 0:
        from operational.final_refit.scientific_fold_heads import verify_scientific_admission

        admission_path = directory / "ENDPOINT_ADMISSION.json"
        admission = verify_scientific_admission(admission_path)
        verify_runtime_endpoint(admission_path, admission)
    else:
        pause = read_json_shared(directory / "RESOURCE_PAUSE.json")
        if (
            receipt["status"] != "PAUSED_RESOURCE"
            or pause["completed_updates"] != receipt["completed_updates"]
        ):
            raise ValueError("Worker resource pause is not durable")
    return receipt


def producer_main(arguments: list[str]) -> int:
    """Own a fit lock while invoking the unmodified scientific producer entrypoint."""
    import argparse

    import psutil

    from operational.final_refit import scientific_fold_producers
    from operational.oof_runtime_overlay import FoldPrefetchRuntime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--rotation", type=int, choices=range(5), required=True)
    parser.add_argument("--fit", choices=scientific_fold_producers.FIT_NAMES, required=True)
    args = parser.parse_args(arguments)
    run = args.run.resolve()
    config = validate_runtime(run)
    directory = run / "rotations" / str(args.rotation) / "fits" / args.fit
    with queue_lock(directory / "ISOLATED_PRODUCER.lock"):
        receipt_path = directory / "CHECKPOINT_RECEIPT.json"
        origin = read_json_shared(receipt_path) if receipt_path.exists() else None
        actual = [
            "scientific_fold_producers",
            "--run",
            str(run),
            "--rotation",
            str(args.rotation),
            "--fit",
            args.fit,
            "--device",
            config["device"],
        ]
        for flag, key in [
            ("fold-contract", "fold_contract"),
            ("config", "config"),
            ("rgb-source", "rgb_source"),
            ("teacher", "teacher"),
            ("event-root", "event_root"),
        ]:
            actual.extend([f"--{flag}", str(config[key])])
        record = dict(
            schema="scientific_oof_process_isolation_v1",
            pid=os.getpid(),
            process_create_time=psutil.Process().create_time(),
            started_utc=datetime.now(UTC).isoformat(),
            orchestration_sha256=sha256(Path(__file__)),
            queue_plan_sha256=sha256(run / "SCIENTIFIC_OOF_QUEUE_PLAN.json"),
            original_checkpoint=origin,
            actual_producer_arguments=actual,
            recipe_changed=False,
            checkpoint_interval_updates=100,
        )
        audit = directory / f"PROCESS_SESSION_{os.getpid()}_{datetime.now(UTC):%Y%m%dT%H%M%S}.json"
        atomic_write_json(audit, record)
        previous = sys.argv
        try:
            sys.argv = actual
            if args.fit.startswith("R_"):
                # A fresh fit first creates a real durable origin through the
                # original training path. Those updates count scientifically.
                if not receipt_path.exists():
                    sys.argv = [*actual, "--max-updates", "100"]
                    scientific_fold_producers.main()
                    initial = read_json_shared(receipt_path)
                    if initial["status"] == "PAUSED_RESOURCE":
                        record["exit_code"] = 3
                        return 3
                sys.argv = actual
                binding_kwargs = bind_fit(run, directory)

                def stats(value: dict[str, Any]) -> None:
                    atomic_write_json(directory / "OOF_RUNTIME_STATS.json", value)

                with FoldPrefetchRuntime(
                    checkpoint_binding_kwargs=binding_kwargs, stats_callback=stats
                ):
                    code = scientific_fold_producers.main()
            else:
                code = scientific_fold_producers.main()
            record.update(
                exit_code=code,
                finished_utc=datetime.now(UTC).isoformat(),
                final_receipt=read_json_shared(receipt_path),
            )
            atomic_write_json(audit, record)
            return code
        except BaseException as error:
            import traceback

            record.update(
                status="ERROR",
                error_type=type(error).__name__,
                error=str(error),
                traceback=traceback.format_exc(),
            )
            raise
        finally:
            sys.argv = previous
            record["finished_utc"] = datetime.now(UTC).isoformat()
            if receipt_path.exists():
                try:
                    record["last_receipt"] = read_json_shared(receipt_path)
                except (OSError, ValueError) as error:
                    record["last_receipt_read_error"] = str(error)
            atomic_write_json(audit, record)


def main() -> int:
    """Run the pipeline coordinator or one locked, isolated producer."""
    if len(sys.argv) < 2:
        raise ValueError("Expected pipeline or producer")
    mode, arguments = sys.argv[1], sys.argv[2:]
    if mode == "producer":
        return producer_main(arguments)
    if mode not in ("pipeline", "final"):
        raise ValueError(f"Unknown isolated mode: {mode}")
    from operational.final_refit import (
        scientific_fold_heads,
        scientific_oof_final,
        scientific_oof_pipeline,
    )

    original_verify = scientific_fold_heads.verify_scientific_admission

    def verify(path: Path) -> dict[str, Any]:
        value = original_verify(path)
        verify_runtime_endpoint(path, value)
        return value

    # All ancestor traversals retain the original scientific verification and
    # additionally validate the additive runtime binding when present.
    scientific_fold_heads.verify_scientific_admission = verify
    scientific_oof_pipeline.verify_scientific_admission = verify
    if hasattr(scientific_oof_final, "verify_scientific_admission"):
        scientific_oof_final.verify_scientific_admission = verify
    if mode == "final":
        sys.argv = ["scientific_oof_final", *arguments]
        return scientific_oof_final.main()
    original = scientific_oof_pipeline.train
    previous = sys.argv
    try:
        scientific_oof_pipeline.train = isolated_train
        sys.argv = ["scientific_oof_pipeline", *arguments]
        return scientific_oof_pipeline.main()
    finally:
        scientific_oof_pipeline.train = original
        sys.argv = previous


if __name__ == "__main__":
    raise SystemExit(main())
