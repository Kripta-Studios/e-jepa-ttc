"""Reset EVENT producer retry budgets only after a new durable checkpoint."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

V1_SHA256 = "8eb97e5683c5061b918da0be17ee996f1d5abe0939c01af68aca671b77ce7d5c"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def durable_producer_updates(module: Any, task: dict[str, Any], anchor: object) -> int | None:
    """Return a stable sealed boundary newer than anchor, or no reset authority."""
    if task.get("fit") not in module.PRODUCERS:
        return None
    directory = Path(task["run"]) / f"rotations/{task['rotation']}/fits/{task['fit']}"
    receipt_path = directory / "CHECKPOINT_RECEIPT.json"
    pointer_path = directory / "CHECKPOINT_POINTER.json"
    journal_path = directory / "UPDATE_JOURNAL.json"
    if any(not path.is_file() for path in (receipt_path, pointer_path, journal_path)):
        return None
    try:
        receipt = module.read(receipt_path)
        completed = receipt.get("completed_updates")
        if isinstance(anchor, int) and isinstance(completed, int) and completed <= anchor:
            return None
        pointer, journal = module.read(pointer_path), module.read(journal_path)
        version = (directory / "checkpoint_versions" / str(pointer.get("version"))).resolve(
            strict=True
        )
        checkpoint = Path(str(receipt.get("checkpoint_path"))).resolve(strict=True)
        root = directory.resolve(strict=True)
        if not checkpoint.is_relative_to(root) or not version.is_relative_to(root):
            return None
        before_checkpoint, before_version = checkpoint.stat(), version.stat()
        checkpoint_sha, version_sha = digest(checkpoint), digest(version)
        after_checkpoint, after_version = checkpoint.stat(), version.stat()

        def stable(a: Any, b: Any) -> bool:
            return (a.st_dev, a.st_ino, a.st_size, a.st_mtime_ns) == (
                b.st_dev,
                b.st_ino,
                b.st_size,
                b.st_mtime_ns,
            )
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return None
    identity = receipt.get("identity_sha256")
    binding = receipt.get("execution_binding_sha256")
    journal_completed = journal.get("completed_updates")
    if not (
        stable(before_checkpoint, after_checkpoint)
        and stable(before_version, after_version)
        and receipt.get("schema") == "rgb_port_checkpoint_receipt_v1"
        and pointer.get("schema") == "rgb_port_checkpoint_pointer_v1"
        and journal.get("schema") == "rgb_port_update_journal_v1"
        and receipt.get("complete_state") is True
        and receipt.get("accumulation_index") == 0
        and isinstance(completed, int)
        and completed >= 0
        and isinstance(journal_completed, int)
        and journal_completed >= completed
        and completed == pointer.get("completed_updates") == journal.get("durable_updates")
        and journal.get("pending_update_upper") == 0
        and receipt.get("recovery_upper") == journal.get("recovery_upper") == 0
        and checkpoint_sha
        == version_sha
        == receipt.get("checkpoint_sha256")
        == pointer.get("checkpoint_sha256")
        and identity == pointer.get("identity_sha256") == journal.get("identity_sha256")
        and binding
        == pointer.get("execution_binding_sha256")
        == journal.get("execution_binding_sha256")
    ):
        return None
    return completed


def install_v2() -> Any:
    source = Path(__file__).resolve()
    v1_path = source.with_name("runpod_event_resource_watchdog_v1.py")
    if digest(v1_path) != V1_SHA256:
        raise ValueError("Frozen EVENT watchdog V1 bytes differ")
    spec = importlib.util.spec_from_file_location("ejepa_event_resource_watchdog_v1_v2", v1_path)
    if spec is None or spec.loader is None:
        raise ImportError(str(v1_path))
    module: Any = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original_validate = module.validate_config
    original_adopt = module.Watchdog._adopt
    original_restart = module.Watchdog._restart

    def validate_config(path: Path) -> dict[str, Any]:
        value = original_validate(path)
        if value.get("watchdog_source_sha256") != digest(source):
            raise ValueError("EVENT watchdog V2 source binding differs")
        return value

    def observe(self: Any, task: dict[str, Any], current: dict[str, Any]) -> None:
        anchor = current.get("retry_anchor_completed_updates")
        completed = durable_producer_updates(module, task, anchor)
        if completed is None:
            return
        if not isinstance(anchor, int):
            current.update(
                retry_anchor_completed_updates=completed,
                retry_anchor_initialized_without_reset=True,
            )
            return
        previous = int(current.get("retries", 0))
        current.update(
            retries=0,
            retry_anchor_completed_updates=completed,
            retry_reset_previous_retries=previous,
            retry_reset_from_anchor=anchor,
            retry_reset_current_durable=completed,
            retry_reset_count=int(current.get("retry_reset_count", 0)) + 1,
        )

    def adopt(self: Any, task: dict[str, Any], current: dict[str, Any]) -> Any:
        observe(self, task, current)
        return original_adopt(self, task, current)

    def restart(self: Any, task: dict[str, Any], current: dict[str, Any]) -> None:
        observe(self, task, current)
        original_restart(self, task, current)

    module.validate_config = validate_config
    module.Watchdog._adopt = adopt
    module.Watchdog._restart = restart
    module.Watchdog._observe_durable_retry_boundary = observe
    module.__file__ = str(source)
    return module


def main() -> int:
    return int(install_v2().main())


if __name__ == "__main__":
    raise SystemExit(main())
