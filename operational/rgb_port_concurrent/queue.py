"""Run the frozen RGB-PORT DAG with exactly two concurrent event producers."""

# ruff: noqa: ANN401

from __future__ import annotations

import argparse
import copy
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

from operational.rgb_port import run as frozen_queue
from operational.rgb_port.accounting import identity_is_live, read_json_shared
from operational.rgb_port_pipeline_v2 import queue as pipeline_queue

from .contracts import FIT_IDS, validate_concurrent_freeze, validate_concurrent_lineage

CONCURRENT_MODULE = "operational.rgb_port_concurrent.producer"
PIPELINE_MODULE = "operational.rgb_port_pipeline_v2.producer"


class _DetachedChildError(RuntimeError):
    pass


def _fit_id(command: Sequence[Any]) -> str | None:
    values = [str(value) for value in command]
    if "--fit-id" not in values:
        return None
    index = values.index("--fit-id") + 1
    return values[index] if index < len(values) else None


def _route_command(
    original: Callable[[list[Any], Path], list[str]],
    freeze_path: Path,
    command: list[Any],
    repository: Path,
) -> list[str]:
    resolved = original(command, repository)
    try:
        module_index = resolved.index("-m") + 1
    except ValueError:
        return resolved
    if (
        module_index >= len(resolved)
        or resolved[module_index] != PIPELINE_MODULE
        or _fit_id(resolved) not in FIT_IDS
    ):
        return resolved
    return [
        *resolved[:module_index],
        CONCURRENT_MODULE,
        "--concurrent-freeze",
        str(freeze_path.resolve(strict=True)),
        "--delegate-module",
        PIPELINE_MODULE,
        "--",
        *resolved[module_index + 1 :],
    ]


def _augment_config(config: Mapping[str, Any], run: Path) -> dict[str, Any]:
    value = copy.deepcopy(dict(config))
    for field in ("heavy_command_markers", "project_command_markers"):
        markers = value["resources"].setdefault(field, [])
        if CONCURRENT_MODULE not in markers:
            markers.append(CONCURRENT_MODULE)
    # This is a scheduling dependency only. It does not change scientific ancestry.
    for task in value["tasks"]:
        if task.get("heavy") and task.get("fit_id") not in FIT_IDS:
            soft = task.setdefault("soft_depends", [])
            for fit_id in FIT_IDS:
                if fit_id not in soft:
                    soft.append(fit_id)
    members = value["package"]["members"]
    additions = [
        "CONCURRENT_FREEZE.json",
        "repo-tree:operational/rgb_port_concurrent",
        "repo:tests/test_rgb_port_concurrent_queue.py",
    ]
    runtime = run / "concurrent"
    if runtime.is_dir():
        additions.extend(
            str(path.relative_to(run)).replace("\\", "/")
            for path in sorted(runtime.rglob("*"))
            if path.is_file()
        )
    for fit_id in FIT_IDS:
        fit = run / "fits" / fit_id
        if (fit / "CONCURRENT_RUNTIME.json").is_file():
            additions.append(f"fits/{fit_id}/CONCURRENT_RUNTIME.json")
        receipts = fit / "concurrent_checkpoints"
        if receipts.is_dir():
            additions.extend(
                str(path.relative_to(run)).replace("\\", "/")
                for path in sorted(receipts.glob("*.json"))
                if path.name != "PENDING_EXECUTION_RECEIPT.json"
            )
    for member in additions:
        if member not in members:
            members.append(member)
    return value


def _unrelated_heavy_owner(config: Mapping[str, Any], run: Path) -> dict[str, Any] | None:
    import psutil

    state = read_json_shared(run / "RGB_PORT_STATE.json")
    allowed = {
        (int(child["pid"]), float(child["create_time"]))
        for fit_id in FIT_IDS
        if isinstance((child := state["tasks"].get(fit_id, {}).get("child")), dict)
        and identity_is_live(child)
    }
    for pid, _created in list(allowed):
        try:
            descendants = psutil.Process(pid).children(recursive=True)
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            continue
        for process in descendants:
            try:
                allowed.add((process.pid, float(process.create_time())))
            except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                continue
    markers = [str(value).lower() for value in config["resources"]["heavy_command_markers"]]
    for process in psutil.process_iter(["pid", "create_time", "cmdline"]):
        try:
            arguments = [str(value).lower() for value in (process.info.get("cmdline") or [])]
            identity = (process.pid, float(process.info["create_time"]))
            if frozen_queue._module_matches(arguments, markers) and identity not in allowed:
                return {
                    "source": "live_process_scan",
                    "pid": identity[0],
                    "create_time": identity[1],
                }
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            continue
    for entry in config.get("external_heavy_owners", []):
        path = Path(entry["path"])
        if not path.is_file():
            continue
        document = read_json_shared(path)
        for identity in document.get("owners", [document]):
            key = (int(identity.get("pid", -1)), float(identity.get("create_time", -1)))
            arguments = [str(value).lower() for value in identity.get("cmdline", [])]
            selectors = [str(value).lower() for value in entry.get("command_markers", [])]
            if (
                key not in allowed
                and (not selectors or frozen_queue._module_matches(arguments, selectors))
                and identity_is_live(identity)
            ):
                return {"path": str(path), "identity": identity}
    return None


def _detach_factory(original: Callable[..., Any]) -> Callable[..., Any]:
    def launch(command: Sequence[Any], *args: Any, **kwargs: Any) -> Any:
        process = original(command, *args, **kwargs)
        values = [str(value) for value in command]
        try:
            module = values[values.index("-m") + 1]
        except (ValueError, IndexError):
            return process
        if module != CONCURRENT_MODULE or _fit_id(values) not in FIT_IDS:
            return process

        class DetachedProcess:
            pid = process.pid

            def __getattr__(self, name: str) -> Any:
                return getattr(process, name)

            def wait(self, *_args: Any, **_kwargs: Any) -> int:
                raise _DetachedChildError

        return DetachedProcess()

    return launch


@contextmanager
def _patched_runtime(run: Path) -> Iterator[None]:
    freeze_path = run / "CONCURRENT_FREEZE.json"
    validate_concurrent_freeze(freeze_path)
    with pipeline_queue._patched_runtime(run):
        original_load = frozen_queue._load_config
        original_resolve = frozen_queue._resolve_command
        original_freeze = frozen_queue._freeze_inputs
        original_cycle = frozen_queue._one_cycle
        original_popen = frozen_queue.subprocess.Popen

        def load(path: Path) -> dict[str, Any]:
            return _augment_config(original_load(path), run)

        def resolve(command: list[Any], repository: Path) -> list[str]:
            return _route_command(original_resolve, freeze_path, command, repository)

        def check_inputs(
            config_path: Path, config: dict[str, Any], run_root: Path
        ) -> dict[str, Any]:
            result = original_freeze(config_path, config, run_root)
            current = validate_concurrent_freeze(freeze_path)
            state = read_json_shared(run_root / "RGB_PORT_STATE.json")
            record = state["tasks"][FIT_IDS[1]]
            adopted = current["adopted_c2f_child"]
            if (
                identity_is_live(adopted)
                and record.get("status") == "RUNNING"
                and record.get("child") != adopted
            ):
                raise RuntimeError("Running C2F child differs from concurrent adoption freeze")
            for fit_id in FIT_IDS:
                validate_concurrent_lineage(run_root, freeze_path, fit_id)
            return result

        def cycle(*args: Any, **kwargs: Any) -> bool:
            with patch.object(frozen_queue.subprocess, "Popen", _detach_factory(original_popen)):
                try:
                    return bool(original_cycle(*args, **kwargs))
                except _DetachedChildError:
                    return True

        with ExitStack() as stack:
            stack.enter_context(patch.object(frozen_queue, "_load_config", load))
            stack.enter_context(patch.object(frozen_queue, "_resolve_command", resolve))
            stack.enter_context(patch.object(frozen_queue, "_freeze_inputs", check_inputs))
            stack.enter_context(
                patch.object(
                    frozen_queue,
                    "_external_heavy_owner",
                    lambda config: _unrelated_heavy_owner(config, run),
                )
            )
            stack.enter_context(patch.object(frozen_queue, "_one_cycle", cycle))
            yield


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    resume = sub.add_parser("resume")
    resume.add_argument("--run", type=Path, required=True)
    resume.add_argument("--once", action="store_true", help=argparse.SUPPRESS)
    package = sub.add_parser("package")
    package.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    run = args.run.resolve(strict=True)
    validate_concurrent_freeze(run / "CONCURRENT_FREEZE.json")
    forwarded = [args.action, "--run", str(run)]
    if getattr(args, "once", False):
        forwarded.append("--once")
    with _patched_runtime(run):
        return frozen_queue.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
