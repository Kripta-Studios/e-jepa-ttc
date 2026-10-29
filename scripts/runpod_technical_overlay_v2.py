"""Local POSIX technical preflight only; cannot launch scientific cloud training."""

# ruff: noqa: ANN401 -- preserve the frozen runtime interface.

from __future__ import annotations

import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


def select_mount(path: str, text: str) -> dict[str, str]:
    """Resolve the longest matching Linux mount and reject unverified filesystems."""
    target = PurePosixPath(path)
    matches = []
    for line in text.splitlines():
        left, right = line.split(" - ", 1)
        fields, fs = left.split(), right.split()
        mount = fields[4].replace("\\040", " ").replace("\\134", "\\")
        if target.is_relative_to(PurePosixPath(mount)):
            matches.append({"mount": mount, "filesystem": fs[0]})
    if not matches:
        raise ValueError("No Linux mount information for technical output")
    result = max(matches, key=lambda item: len(PurePosixPath(item["mount"]).parts))
    if result["filesystem"] not in {"overlay", "ext4", "xfs", "btrfs"}:
        raise ValueError(f"Technical checkpoint filesystem is not admitted: {result}")
    return result


def validate_command(arguments: list[str]) -> None:
    """Reject scientific commands before importing any training module."""
    if not arguments or arguments[0] not in {"preflight", "resume-probe"}:
        raise ValueError("This overlay permits only preflight or budgeted resume-probe")


def validate_run(run: str, bundle: str) -> None:
    """Restrict the technical sandbox independently of the production runner."""
    target, inputs = PurePosixPath(run), PurePosixPath(bundle)
    if (
        ".." in target.parts
        or not target.is_relative_to(PurePosixPath("/opt/ejepa_technical"))
        or target.is_relative_to(inputs)
    ):
        raise ValueError("Technical output must be separate under /opt/ejepa_technical")


def probe_filesystem(run: Path) -> dict[str, Any]:
    """Exercise local fsync/rename/locking; does not claim power-loss durability."""
    import fcntl

    mount = select_mount(str(run), Path("/proc/self/mountinfo").read_text())
    run.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fs_probe_", dir=run) as temporary:
        folder = Path(temporary)
        pending, committed = folder / "pending", folder / "committed"
        with pending.open("xb") as stream:
            stream.write(b"ejepa-local-posix-probe-v2\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, committed)
        descriptor = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if committed.read_bytes() != b"ejepa-local-posix-probe-v2\n":
            raise ValueError("Read-after-rename mismatch")
        with committed.open("rb") as first, committed.open("rb") as second:
            fcntl.flock(first.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                fcntl.flock(second.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise ValueError("Independent descriptor did not respect exclusive lock")
            finally:
                fcntl.flock(first.fileno(), fcntl.LOCK_UN)
    return {
        **mount,
        "file_fsync_rename_directory_fsync_lock_probe": "PASS",
        "pod_stop_survival": False,
        "power_loss_durability_verified": False,
        "scientific_execution_allowed": False,
    }


def prepare(bundle: Path, run: Path, budget_bytes: int) -> dict[str, Any]:
    """Bind a technical-only local run without changing the frozen source files."""
    from operational.runpod_migration import runner
    from operational.runpod_migration.bundle import verify
    from operational.runpod_migration.inventory import digest
    from operational.runpod_migration.paths import PathMap
    from operational.runpod_migration.resources import LinuxResourceMonitor
    from operational.runpod_migration.runtime import install

    if platform.system() != "Linux":
        raise RuntimeError("Technical overlay requires Linux")
    bundle, run = bundle.resolve(strict=True), run.resolve()
    validate_run(str(run), str(bundle))
    if budget_bytes <= 2 * 1024**3:
        raise ValueError("Memory budget must exceed the physical reserve")
    storage = probe_filesystem(run)
    manifest = verify(bundle)
    mapping = PathMap(manifest["roots"], str(bundle))
    original = Path(mapping.translate(manifest["source_run"])) / "SCIENTIFIC_CONFIG.json"
    if digest(original) != manifest["scientific_config_sha256"]:
        raise ValueError("Authority differs from the inventory")
    copied = run / "SCIENTIFIC_CONFIG.json"
    if copied.exists() and digest(copied) != digest(original):
        raise ValueError("Technical run has a different scientific authority")
    if not copied.exists():
        shutil.copyfile(original, copied)
    migration = {
        "schema": "ejepa_linux_technical_migration_binding_v2",
        "original_authority_sha256": digest(original),
        "bundle_manifest_sha256": digest(bundle / "BUNDLE_MANIFEST.json"),
        "runtime_sources": runner.source_identity(),
        "technical_overlay_sha256": digest(Path(__file__)),
        "bundle_root": str(bundle),
        "run_root": str(run),
        "roots": manifest["roots"],
        "pod_memory_budget_bytes": budget_bytes,
        "scientific_files_rewritten": False,
        "historical_checkpoint_import": False,
        "precision": "bf16",
        "seed": 7,
        "checkpoint_updates": 100,
        "historical_prefetch_overlay_enabled": False,
        "storage": storage,
        "scientific_execution_allowed": False,
    }
    runner._write_once(run / "MIGRATION_BINDING.json", migration)
    install(mapping, budget_bytes=budget_bytes)
    from operational.final_refit.scientific_authority import load_config

    config = load_config(run)
    import torch

    if platform.python_version() != "3.11.15" or str(torch.__version__) != "2.11.0+cu128":
        raise ValueError("Expected Python3.11.15 and torch2.11.0+cu128 exactly")
    if torch.version.cuda != "12.8" or not torch.cuda.is_available():
        raise ValueError("CUDA12.8 PyTorch runtime and a visible GPU are required")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise ValueError("Expose exactly one BF16-capable GPU per worker")
    allowed, resources = LinuxResourceMonitor(budget_bytes=budget_bytes).guard(
        run, checkpoint_reservation=1024**3, completed_updates=0
    )
    resources["storage"] = storage
    if not allowed:
        raise RuntimeError(f"Resource guard denied technical preflight: {resources}")
    return {
        "manifest": manifest,
        "config": config,
        "migration": migration,
        "resources": resources,
        "gpu": torch.cuda.get_device_name(),
    }


def main() -> int:
    """Delegate only admitted technical commands to the unchanged frozen runner."""
    validate_command(sys.argv[1:])
    from operational.runpod_migration import runner

    runner.prepare = prepare
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
