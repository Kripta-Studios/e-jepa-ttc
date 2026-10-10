"""Add the admitted C2F CUDA graph runtime to the concurrent RGB-PORT queue."""

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
from operational.rgb_port_concurrent import queue as concurrent_queue

C2F_FIT_ID = "E_C2F_MATCHED"
CONCURRENT_MODULE = "operational.rgb_port_concurrent.producer"
GRAPH_MODULE = "operational.rgb_port_c2f_graph.producer"


def _detach_factory(original: Callable[..., Any]) -> Callable[..., Any]:
    """Detach both frozen concurrent children and the new graph wrapper."""

    def launch(command: Sequence[Any], *args: Any, **kwargs: Any) -> Any:
        process = original(command, *args, **kwargs)
        values = [str(value) for value in command]
        try:
            module = values[values.index("-m") + 1]
        except (ValueError, IndexError):
            return process
        if module not in {CONCURRENT_MODULE, GRAPH_MODULE}:
            return process

        class DetachedProcess:
            pid = process.pid

            def __getattr__(self, name: str) -> Any:
                return getattr(process, name)

            def wait(self, *_args: Any, **_kwargs: Any) -> int:
                raise concurrent_queue._DetachedChildError

        return DetachedProcess()

    return launch


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
    """Wrap only C2F after the frozen concurrent resolver has run."""
    resolved = original(command, repository)
    try:
        module_index = resolved.index("-m") + 1
    except ValueError:
        return resolved
    if (
        module_index >= len(resolved)
        or resolved[module_index] != CONCURRENT_MODULE
        or _fit_id(resolved) != C2F_FIT_ID
    ):
        return resolved
    return [
        *resolved[:module_index],
        GRAPH_MODULE,
        "--graph-freeze",
        str(freeze_path.resolve(strict=True)),
        "--delegate-module",
        CONCURRENT_MODULE,
        "--",
        *resolved[module_index + 1 :],
    ]


def _runtime_members(run: Path) -> list[str]:
    members = [
        "C2F_GRAPH_FREEZE.json",
        "repo-tree:operational/rgb_port_c2f_graph",
        "repo:tests/test_rgb_port_c2f_graph_queue.py",
        "repo:tests/test_rgb_port_c2f_graph_producer.py",
    ]
    fit = run / "fits" / C2F_FIT_ID
    if (fit / "C2F_GRAPH_RUNTIME.json").is_file():
        members.append(f"fits/{C2F_FIT_ID}/C2F_GRAPH_RUNTIME.json")
    receipts = fit / "c2f_graph_checkpoints"
    if receipts.is_dir():
        members.extend(
            str(path.relative_to(run)).replace("\\", "/")
            for path in sorted(receipts.glob("*.json"))
            if path.name != "PENDING_EXECUTION_RECEIPT.json"
        )
    graph_root = run / "c2f_graph"
    if graph_root.is_dir():
        members.extend(
            str(path.relative_to(run)).replace("\\", "/")
            for path in sorted(graph_root.rglob("*"))
            if path.is_file()
        )
    return members


def _augment_config(config: Mapping[str, Any], run: Path) -> dict[str, Any]:
    value = copy.deepcopy(dict(config))
    for field in ("heavy_command_markers", "project_command_markers"):
        markers = value["resources"].setdefault(field, [])
        if GRAPH_MODULE not in markers:
            markers.append(GRAPH_MODULE)
    members = value["package"]["members"]
    for member in _runtime_members(run):
        if member not in members:
            members.append(member)
    return value


@contextmanager
def _patched_runtime(run: Path) -> Iterator[None]:
    from .contracts import validate_c2f_graph_freeze, validate_c2f_graph_lineage

    freeze_path = run / "C2F_GRAPH_FREEZE.json"
    validate_c2f_graph_freeze(freeze_path)
    with concurrent_queue._patched_runtime(run):
        original_load = frozen_queue._load_config
        original_resolve = frozen_queue._resolve_command
        original_freeze = frozen_queue._freeze_inputs
        original_fit_complete = frozen_queue._fit_complete

        def load(path: Path) -> dict[str, Any]:
            return _augment_config(original_load(path), run)

        def resolve(command: list[Any], repository: Path) -> list[str]:
            return _route_command(original_resolve, freeze_path, command, repository)

        def check_inputs(
            config_path: Path, config: dict[str, Any], run_root: Path
        ) -> dict[str, Any]:
            result = original_freeze(config_path, config, run_root)
            validate_c2f_graph_freeze(freeze_path)
            validate_c2f_graph_lineage(run_root, freeze_path)
            return result

        def fit_complete(task: Mapping[str, Any], run_root: Path) -> bool:
            complete = original_fit_complete(task, run_root)
            if complete and task.get("fit_id") == C2F_FIT_ID:
                validate_c2f_graph_lineage(run_root, freeze_path, require_endpoint=True)
            return complete

        with ExitStack() as stack:
            stack.enter_context(patch.object(concurrent_queue, "_detach_factory", _detach_factory))
            stack.enter_context(patch.object(frozen_queue, "_load_config", load))
            stack.enter_context(patch.object(frozen_queue, "_resolve_command", resolve))
            stack.enter_context(patch.object(frozen_queue, "_freeze_inputs", check_inputs))
            stack.enter_context(patch.object(frozen_queue, "_fit_complete", fit_complete))
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
    forwarded = [args.action, "--run", str(run)]
    if getattr(args, "once", False):
        forwarded.append("--once")
    with _patched_runtime(run):
        return frozen_queue.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
