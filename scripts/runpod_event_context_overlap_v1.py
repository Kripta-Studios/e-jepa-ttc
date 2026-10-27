"""Schedule the frozen E_CTX fit alongside target-free OOF input preparation.

This operational overlay never edits the sealed scientific source. The training
block is extracted verbatim from the hash-bound context_branch implementation.
Each head owns an OS lock, runs in a separate interpreter, and keeps the original
100-update checkpoint cadence, authority checks, and resource guards.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NoReturn

PIPELINE_SHA256 = "f20f567b128e636ef6d20f7056168b1c160f6ed1845fadadf420db2f9d473021"
WORKER_ENV = "EJEPA_EARLY_E_CTX_WORKER"


class HeadFinished(BaseException):
    """Return from the auxiliary queue without publishing a false OOF receipt."""


def extract_training_block(source: str) -> ast.FunctionDef:
    """Select exactly normalizer, training-cache preparation, and the frozen fit."""
    tree = ast.parse(source)
    branch = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "context_branch"
    )
    start = next(
        i
        for i, n in enumerate(branch.body)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "normalizer"
    )
    end = next(
        i
        for i, n in enumerate(branch.body)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "fit_head"
    )
    args = ast.arguments(
        posonlyargs=[],
        args=[
            ast.arg(arg=x)
            for x in (
                "config",
                "rotation",
                "root",
                "unit",
                "base",
                "name",
                "guard",
                "raws",
                "parents",
            )
        ],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )
    return ast.FunctionDef(
        name="_frozen_head_block",
        args=args,
        body=branch.body[start : end + 1],
        decorator_list=[],
        returns=None,
    )


def compile_training_block(pipeline: Any) -> Any:
    source = Path(pipeline.__file__).read_bytes()
    if hashlib.sha256(source).hexdigest() != PIPELINE_SHA256:
        raise ValueError("Frozen context pipeline differs; scheduling overlay refused")
    module = ast.fix_missing_locations(
        ast.Module(body=[extract_training_block(source.decode())], type_ignores=[])
    )
    namespace = dict(vars(pipeline))
    exec(compile(module, "<hash-bound-early-E_CTX>", "exec"), namespace)
    return namespace["_frozen_head_block"]


def wait_guard(guard: Any, pause_type: type, *, sleep: Any = time.sleep) -> Any:
    """Wait only on automatic resource pressure; manual stops/errors propagate."""

    def check() -> None:
        while True:
            try:
                guard()
                return
            except pause_type as error:
                if error.manual:
                    raise
                sleep(5)

    return check


def lock_busy(path: Path) -> bool:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return False


def install(queue: Any) -> None:
    """Install after admitted path routing and before queue lock acquisition."""
    original_install = queue._install_rgb_routing

    def configure(run: Path, config: dict, mapping: Any) -> None:
        original_install(run, config, mapping)
        from operational.final_refit import continuation_queue, scientific_oof_features
        from operational.final_refit import scientific_oof_pipeline as pipeline
        from operational.rgb_port.accounting import atomic_write_json

        compile_training_block(pipeline)
        rotation = int(sys.argv[sys.argv.index("--rotation") + 1])
        root = Path(run) / "rotations" / str(rotation)
        status_path = root / "EARLY_E_CTX_STATUS.json"
        lock_path = root / "EARLY_E_CTX.lock"
        worker = os.environ.get(WORKER_ENV) == "1"
        original_context = pipeline.context_branch
        original_raw = pipeline.event_raw

        def status(state: str, **details: Any) -> None:
            atomic_write_json(
                status_path,
                {
                    "schema": "ejepa_early_context_scheduler_v1",
                    "status": state,
                    "pid": os.getpid(),
                    "rotation": rotation,
                    "timestamp": time.time(),
                    "pipeline_sha256": PIPELINE_SHA256,
                    "scheduler_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "checkpoint_interval_updates": 100,
                    "normalizer_fit_role": "refiner",
                    "generation_targets_used": False,
                    **details,
                },
            )

        def verified_endpoint() -> bool:
            path = root / "fits/E_CTX/ENDPOINT_ADMISSION.json"
            if not path.exists():
                return False
            pipeline.verify_scientific_admission(path)
            return True

        if worker:
            original_lock = continuation_queue.queue_lock

            @contextlib.contextmanager
            def head_lock(path: Path) -> Iterator[None]:
                expected = Path(run) / f"EVENT_DOWNSTREAM_R{rotation}.lock"
                if path != expected:
                    raise ValueError("Unexpected auxiliary queue lock")
                try:
                    with original_lock(lock_path):
                        yield
                except BlockingIOError:
                    raise HeadFinished() from None

            continuation_queue.queue_lock = head_lock

            def head_only(
                config: Any, rotation: int, root: Path, unit: Any, base: Any, name: str, guard: Any
            ) -> NoReturn:
                if name != "E_CTX":
                    raise ValueError("Early worker only admits E_CTX")
                safe_guard = wait_guard(guard, pipeline.ResourcePause)
                try:
                    status("RUNNING")
                    safe_guard()
                    dependencies = pipeline._stage(unit, name).depends_on
                    parents = {
                        k: pipeline.verify_scientific_admission(
                            root / "fits" / k / "ENDPOINT_ADMISSION.json"
                        )["checkpoint_sha256"]
                        for k in dependencies
                    }
                    rows = pipeline.role_rows(
                        pipeline.Path(config["fold_contract"]), rotation, "refiner"
                    )
                    raw_path = root / "refiner" / name / "raw_phase17.npz"
                    valid = scientific_oof_features._reuse_verified_raw_phase17(
                        raw_path,
                        rows,
                        modality="event",
                        parents=parents,
                        expected_provenance={
                            "historical_h8_sha256": pipeline.sha256(
                                pipeline.Path(config["v12_run"]) / "H8_FEATURES.npz"
                            ),
                            "historical_learned_columns_reused": [],
                        },
                    )
                    if not valid:
                        raise ValueError("Refiner inputs are not sealed and complete")
                    compile_training_block(pipeline)(
                        config,
                        rotation,
                        root,
                        unit,
                        base,
                        name,
                        safe_guard,
                        {"refiner": pipeline.arrays(raw_path)},
                        parents,
                    )
                    if not verified_endpoint():
                        raise ValueError("Early head returned without admitted endpoint")
                    status("COMPLETE", completed_updates=pipeline._stage(unit, name).updates)
                except pipeline.ResourcePause as error:
                    status("PAUSED_MANUAL" if error.manual else "PAUSED_RESOURCE", error=str(error))
                    raise
                except BaseException as error:
                    status("FAILED", error=repr(error), traceback=traceback.format_exc())
                    raise
                raise HeadFinished()

            pipeline.context_branch = head_only
            return

        child = None
        last_launch = 0.0

        def ensure_worker(guard: Any) -> None:
            nonlocal child, last_launch
            guard()
            if verified_endpoint() or lock_busy(lock_path):
                return
            previous = json.loads(status_path.read_text()) if status_path.exists() else {}
            if previous.get("status") in {"FAILED", "PAUSED_MANUAL"}:
                raise RuntimeError(f"Early E_CTX requires inspection: {previous}")
            if child is not None and child.poll() is None:
                return
            if child is not None and child.returncode not in (0, 4):
                raise RuntimeError(
                    f"Early E_CTX bootstrap exited {child.returncode}; see EARLY_E_CTX.log"
                )
            if time.time() - last_launch < 10:
                return
            env = dict(os.environ, **{WORKER_ENV: "1"})
            with (root / "EARLY_E_CTX.log").open("ab", buffering=0) as log:
                child = subprocess.Popen(
                    [sys.executable, *sys.argv],
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            last_launch = time.time()

        def raw_with_overlap(**kwargs: Any) -> Path:
            output = Path(kwargs["output"])
            if output.name != "E_CTX" or output.parent.name != "generation":
                return original_raw(**kwargs)
            guard = kwargs["guard"]
            ensure_worker(guard)
            last_check = time.monotonic()

            def monitored_guard() -> None:
                nonlocal last_check
                guard()
                if time.monotonic() - last_check >= 15:
                    ensure_worker(guard)
                    last_check = time.monotonic()

            kwargs["guard"] = monitored_guard
            result = original_raw(**kwargs)
            while not verified_endpoint() or lock_busy(lock_path):
                ensure_worker(guard)
                time.sleep(2)
            return result

        def context_with_wait(
            config: Any, rotation: int, root: Path, unit: Any, base: Any, name: str, guard: Any
        ) -> Any:
            safe_guard = wait_guard(guard, pipeline.ResourcePause)
            return original_context(config, rotation, root, unit, base, name, safe_guard)

        pipeline.event_raw = raw_with_overlap
        pipeline.context_branch = context_with_wait

    queue._install_rgb_routing = configure
