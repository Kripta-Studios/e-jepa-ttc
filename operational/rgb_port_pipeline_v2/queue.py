"""Compose pipeline V2 over the frozen acceleration V1 supervisor."""

from __future__ import annotations

import argparse
import copy
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

from operational.rgb_port import run as frozen_queue
from operational.rgb_port_acceleration import queue as acceleration_queue

from .contracts import EVENT_FIT_IDS, validate_pipeline_freeze
from .receipts import validate_fit_lineage, write_report_extension

ACCELERATED_MODULE = "operational.rgb_port_acceleration.producer"
PIPELINE_MODULE = "operational.rgb_port_pipeline_v2.producer"


def _runtime_members(run: Path) -> list[str]:
    members = [
        "PIPELINE_FREEZE.json",
        "pipeline_v2/admissions/CPU_ADMISSION.json",
        "pipeline_v2/admissions/GPU_ADMISSION.json",
        "pipeline_v2/REPORT_EXTENSION.json",
        "current_bottleneck_20261009/V1_RATE_BASELINE.json",
        "repo-tree:operational/rgb_port_pipeline_v2",
        "repo:tests/test_rgb_port_pipeline_queue.py",
        "repo:tests/test_rgb_port_pipeline_prefetch.py",
    ]
    integration = run / "pipeline_v2/PIPELINE_INTEGRATION_QA.json"
    members.append(
        "pipeline_v2/PIPELINE_INTEGRATION_QA.json"
        if integration.is_file()
        else "optional:pipeline_v2/PIPELINE_INTEGRATION_QA.json"
    )
    producer_test = Path(__file__).resolve().parents[2] / "tests/test_rgb_port_pipeline_producer.py"
    if producer_test.is_file():
        members.append("repo:tests/test_rgb_port_pipeline_producer.py")
    pipeline_root = run / "pipeline_v2"
    if pipeline_root.is_dir():
        members.extend(
            str(path.relative_to(run)).replace("\\", "/")
            for path in sorted(pipeline_root.rglob("*"))
            if path.is_file()
            and path.name != "PENDING_EXECUTION_RECEIPT.json"
            and path.suffix.lower() in {".json", ".md", ".py", ".pt"}
        )
    for fit_id in EVENT_FIT_IDS:
        origin = run / "pipeline_v2/origins"
        members.extend(
            str(path.relative_to(run)).replace("\\", "/")
            for path in sorted(origin.glob(f"{fit_id}_*.pt"))
        )
        fit = run / "fits" / fit_id
        for name in ("PIPELINE_RUNTIME.json", "PIPELINE_CANARY.json"):
            if (fit / name).is_file():
                members.append(f"fits/{fit_id}/{name}")
        receipts = fit / "pipeline_checkpoints"
        if receipts.is_dir():
            members.extend(
                str(path.relative_to(run)).replace("\\", "/")
                for path in sorted(receipts.glob("*.json"))
                if path.name != "PENDING_EXECUTION_RECEIPT.json"
            )
    return members


def _augmented_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(config))
    for field in ("heavy_command_markers", "project_command_markers"):
        markers = value["resources"].setdefault(field, [])
        if PIPELINE_MODULE not in markers:
            markers.append(PIPELINE_MODULE)
    members = value["package"]["members"]
    for member in _runtime_members(Path(value["run_root"])):
        if member not in members:
            members.append(member)
    return value


def _route_command(
    original: Callable[[list[Any], Path], list[str]],
    freeze_path: Path,
    command: list[Any],
    repository: Path,
) -> list[str]:
    """Wrap exactly the two event producers after acceleration V1 resolution."""
    resolved = original(command, repository)
    try:
        module_index = resolved.index("-m") + 1
    except ValueError:
        return resolved
    if (
        module_index >= len(resolved)
        or resolved[module_index] != ACCELERATED_MODULE
        or "--fit-id" not in resolved
    ):
        return resolved
    fit_index = resolved.index("--fit-id")
    if fit_index + 1 >= len(resolved) or resolved[fit_index + 1] not in EVENT_FIT_IDS:
        return resolved
    return [
        *resolved[:module_index],
        PIPELINE_MODULE,
        "--pipeline-freeze",
        str(freeze_path.resolve(strict=True)),
        "--",
        *resolved[module_index + 1 :],
    ]


@contextmanager
def _patched_runtime(run: Path) -> Iterator[None]:
    freeze_path = run / "PIPELINE_FREEZE.json"
    freeze = validate_pipeline_freeze(freeze_path)
    with acceleration_queue._patched_runtime(run):
        original_load = frozen_queue._load_config
        original_resolve = frozen_queue._resolve_command
        original_freeze = frozen_queue._freeze_inputs
        original_fit_complete = frozen_queue._fit_complete

        def load(path: Path) -> dict[str, Any]:
            return _augmented_config(original_load(path))

        def resolve(command: list[Any], repository: Path) -> list[str]:
            return _route_command(original_resolve, freeze_path, command, repository)

        def check_inputs(
            config_path: Path, config: dict[str, Any], run_root: Path
        ) -> dict[str, Any]:
            result = original_freeze(config_path, config, run_root)
            current = validate_pipeline_freeze(freeze_path)
            for fit_id in EVENT_FIT_IDS:
                validate_fit_lineage(run_root, current, freeze_path, fit_id)
            write_report_extension(run_root, current, freeze_path)
            return result

        def fit_complete(task: Mapping[str, Any], run_root: Path) -> bool:
            complete = original_fit_complete(task, run_root)
            fit_id = task.get("fit_id")
            if complete and fit_id in EVENT_FIT_IDS:
                validate_fit_lineage(
                    run_root,
                    freeze,
                    freeze_path,
                    str(fit_id),
                    require_endpoint=True,
                )
            return complete

        with ExitStack() as stack:
            stack.enter_context(patch.object(frozen_queue, "_load_config", load))
            stack.enter_context(patch.object(frozen_queue, "_resolve_command", resolve))
            stack.enter_context(patch.object(frozen_queue, "_freeze_inputs", check_inputs))
            stack.enter_context(patch.object(frozen_queue, "_fit_complete", fit_complete))
            yield


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("execute", "resume"):
        item = sub.add_parser(action)
        item.add_argument("--run", type=Path, required=True)
        if action == "execute":
            item.add_argument("--config", type=Path, required=True)
        item.add_argument("--once", action="store_true", help=argparse.SUPPRESS)
    package = sub.add_parser("package")
    package.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    run = args.run.resolve(strict=True)
    freeze_path = run / "PIPELINE_FREEZE.json"
    freeze = validate_pipeline_freeze(freeze_path)
    if args.action == "package":
        for fit_id in acceleration_queue.METRICS_FIT_IDS:
            if (run / "fits" / fit_id / "CHECKPOINT_POINTER.json").is_file():
                acceleration_queue._validate_runtime_receipts(
                    run,
                    require_endpoint=fit_id == acceleration_queue.FIT_ID,
                    fit_id=fit_id,
                )
        for fit_id in EVENT_FIT_IDS:
            validate_fit_lineage(run, freeze, freeze_path, fit_id)
        write_report_extension(run, freeze, freeze_path)
    forwarded = [args.action]
    if args.action == "execute":
        forwarded.extend(("--config", str(args.config)))
    else:
        forwarded.extend(("--run", str(run)))
    if getattr(args, "once", False):
        forwarded.append("--once")
    with _patched_runtime(run):
        return frozen_queue.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
