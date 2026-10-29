"""Admit a fresh durable HEAD pause despite an intentionally old pipeline receipt."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

V7_SHA256 = "aeb8a855a98d58e62f267f017817b8b16d9010d2bff048cdce35eba435093592"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _stable_digest(path: Path) -> str | None:
    if not path.is_file() or path.is_symlink():
        return None
    before = path.stat()
    value = digest(path)
    after = path.stat()

    def identity(stat: Any) -> tuple[int, int, int, int]:
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    return value if identity(before) == identity(after) else None


def _durable_paused_head(
    module: Any,
    task: dict[str, Any],
    managed: dict[str, Any],
    fit: str,
    *,
    required: bool,
) -> bool:
    directory = Path(task["run"]) / f"rotations/{int(task['rotation'])}/fits/{fit}"
    names = (
        "CHECKPOINT_RECEIPT.json",
        "CHECKPOINT_POINTER.json",
        "UPDATE_JOURNAL.json",
        "RESOURCE_PAUSE.json",
        "PROGRESS.json",
    )
    paths = {name: directory / name for name in names}
    present = [path.exists() for path in paths.values()]
    if not any(present):
        return not required
    started = float(managed["started_epoch"])
    if not all(module.fresh(path, started) and not path.is_symlink() for path in paths.values()):
        return False
    try:
        receipt = module.read(paths["CHECKPOINT_RECEIPT.json"])
        pointer = module.read(paths["CHECKPOINT_POINTER.json"])
        journal = module.read(paths["UPDATE_JOURNAL.json"])
        pause = module.read(paths["RESOURCE_PAUSE.json"])
        progress = module.read(paths["PROGRESS.json"])
        root = directory.resolve(strict=True)
        checkpoint = Path(str(receipt.get("checkpoint_path"))).resolve(strict=True)
        version = (directory / "checkpoint_versions" / str(pointer.get("version"))).resolve(
            strict=True
        )
        if not checkpoint.is_relative_to(root) or not version.is_relative_to(root):
            return False
        checkpoint_sha = _stable_digest(checkpoint)
        version_sha = _stable_digest(version)
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return False
    completed = receipt.get("completed_updates")
    identity = receipt.get("identity_sha256")
    binding = receipt.get("execution_binding_sha256")
    return bool(
        receipt.get("schema") == "rgb_port_checkpoint_receipt_v1"
        and receipt.get("status") == "PAUSED_RESOURCE"
        and receipt.get("complete_state") is True
        and receipt.get("accumulation_index") == 0
        and isinstance(completed, int)
        and completed >= 0
        and receipt.get("recovery_upper") == 0
        and pointer.get("schema") == "rgb_port_checkpoint_pointer_v1"
        and pointer.get("completed_updates") == completed
        and journal.get("schema") == "rgb_port_update_journal_v1"
        and journal.get("completed_updates") == completed
        and journal.get("durable_updates") == completed
        and journal.get("pending_update_upper") == 0
        and journal.get("recovery_upper") == 0
        and pause.get("manual") is False
        and pause.get("completed_updates") == completed
        and progress.get("pid") == int(managed["pid"])
        and progress.get("completed_updates") == completed
        and checkpoint_sha is not None
        and checkpoint_sha
        == version_sha
        == receipt.get("checkpoint_sha256")
        == pointer.get("checkpoint_sha256")
        and identity == pointer.get("identity_sha256") == journal.get("identity_sha256")
        and binding
        == pointer.get("execution_binding_sha256")
        == journal.get("execution_binding_sha256")
    )


def install_v8() -> Any:
    source = Path(__file__).resolve()
    v7_path = source.with_name("runpod_event_resource_watchdog_v7.py")
    if digest(v7_path) != V7_SHA256:
        raise ValueError("Frozen EVENT watchdog V7 bytes differ")
    spec = importlib.util.spec_from_file_location("ejepa_event_resource_watchdog_v7_v8", v7_path)
    if spec is None or spec.loader is None:
        raise ImportError(v7_path)
    wrapper: Any = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = wrapper
    spec.loader.exec_module(wrapper)
    module: Any = wrapper.install_v7()
    original_validate = module.validate_config
    original_terminal = module.downstream_terminal

    def validate_config(path: Path) -> dict[str, Any]:
        value = original_validate(path)
        if value.get("watchdog_head_pause_overlay_source_sha256") != digest(source):
            raise ValueError("EVENT watchdog V8 HEAD pause binding differs")
        return value

    def downstream_terminal(task: dict[str, Any], managed: dict[str, Any]) -> tuple[str, str]:
        terminal, reason = original_terminal(task, managed)
        if (terminal, reason) == (
            "RESTARTABLE",
            "FRESH_AUTOMATIC_PRE_HEAD_RESOURCE_PAUSE",
        ):
            run = Path(task["run"])
            if any(
                (
                    run / f"rotations/{int(task['rotation'])}/fits/{fit}/CHECKPOINT_RECEIPT.json"
                ).exists()
                for fit in ("PAIR_E", "E_CTX")
            ):
                return "BLOCKED", "STALE_OR_PARTIAL_HEAD_CHECKPOINT_EVIDENCE"
            return terminal, reason
        if (terminal, reason) != ("BLOCKED", "EVENT_PIPELINE_RECEIPT_NOT_FRESH"):
            return terminal, reason
        pause_path = Path(task["run"]) / f"EVENT_DOWNSTREAM_PAUSE_R{int(task['rotation'])}.json"
        if not module.fresh(pause_path, float(managed["started_epoch"])) or pause_path.is_symlink():
            return terminal, reason
        pause = module.read(pause_path)
        if not (
            pause.get("schema") == "ejepa_event_downstream_pause_v1"
            and pause.get("status") == "PAUSED_RECOVERABLE"
            and pause.get("rotation") == int(task["rotation"])
            and pause.get("reason") == "resource reserve unavailable"
            and pause.get("manual") is False
            and pause.get("pause_classification") == "AUTOMATIC_RESOURCE_PAUSE"
            and pause.get("owner_pid") == int(managed["pid"])
            and pause.get("owner_start_ticks") == managed.get("start_ticks")
            and isinstance(pause.get("resources"), dict)
        ):
            return terminal, reason
        if not _durable_paused_head(module, task, managed, "PAIR_E", required=True):
            return terminal, reason
        if not _durable_paused_head(module, task, managed, "E_CTX", required=False):
            return terminal, reason
        return "RESTARTABLE", "FRESH_DURABLE_HEAD_PAUSE_WITH_STALE_PIPELINE_RECEIPT"

    module.validate_config = validate_config
    module.downstream_terminal = downstream_terminal
    module.__file__ = str(source)
    return module


def main() -> int:
    return int(install_v8().main())


if __name__ == "__main__":
    raise SystemExit(main())
