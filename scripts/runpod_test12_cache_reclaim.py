"""Release clean cache pages of receipt-bound Test12 inputs and completed backups.

No file writes to admitted sources, deletion, chmod, global drop_caches, or fresh
multi-gigabyte hashing. Content identity comes from immutable producer receipts;
device/inode/size/mtime/mode are pinned at admission and checked around each
read-only O_NOFOLLOW advice operation. Dirty pages are left to the kernel.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import time
from collections.abc import Callable
from pathlib import Path

ROOT = Path("/workspace/ejepa_test12_cache_controller")
PROBE = Path("/workspace/ejepa_test12_probe")
EVENTS = Path("/workspace/ejepa_test12_events")
RGB = Path("/workspace/ejepa_test12_rgb")
PRESERVATION = Path("/workspace/ejepa_preservation")
INPUTS_SHA = "b78730abe68de7fbe06c6c98b25d5fa025c04d9303720b680ce0c0f575d2e526"
GIB = 1024**3
DEADLINE = 1793145600.0


def identity(value: os.stat_result) -> tuple[int, ...]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_mode


def advise(item: dict, callback: Callable | None = None) -> None:
    """Only advise a regular, unchanged file through a read-only descriptor."""
    path = Path(item["path"])
    if path.is_symlink():
        raise ValueError("Symlink is outside cache admission contract")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags)
    try:
        if identity(os.fstat(fd)) != tuple(item["stat"]):
            raise ValueError("Admitted file changed before cache advice")
        function = callback or os.posix_fadvise
        function(fd, 0, 0, getattr(os, "POSIX_FADV_DONTNEED", 4))
        if identity(os.fstat(fd)) != tuple(item["stat"]):
            raise ValueError("Admitted file changed during cache advice")
    finally:
        os.close(fd)


def observe() -> dict:
    sys.path.insert(0, str(PROBE)) if str(PROBE) not in sys.path else None
    from resource_reader import memory_snapshot

    room, reports = memory_snapshot(
        Path("/sys/fs/cgroup"), Path("/proc/self/cgroup").read_text(), 2**63 - 1
    )
    leaf = reports[0]
    stats = leaf["memory_stat"]
    return {
        "headroom_bytes": room,
        "charged_bytes": leaf["current"],
        "cache_bytes": stats.get("total_cache", stats.get("file", stats.get("cache", 0))),
        "rss_bytes": stats.get("total_rss", stats.get("anon", stats.get("rss", 0))),
    }


def atomic(path: Path, value: dict) -> None:
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(pending, path)


class Registry:
    """Incrementally register closed producer outputs without rereading payloads."""

    def __init__(self) -> None:
        self.files: dict[str, dict] = {}
        self.seen: dict[str, int] = {}
        self.rejected: list[dict] = []

    def read_changed(self, path: Path) -> dict | None:
        try:
            stamp = path.stat().st_mtime_ns
        except FileNotFoundError:
            return None
        if self.seen.get(str(path)) == stamp:
            return None
        value = json.loads(path.read_text())
        self.seen[str(path)] = stamp
        return value

    def add(
        self,
        path: Path,
        root: Path,
        sha: str,
        source: Path,
        priority: int,
        size: int | None = None,
        mtime: int | None = None,
    ) -> None:
        key = str(path)
        if key in self.files:
            return
        if len(sha) != 64 or not all(c in "0123456789abcdef" for c in sha):
            raise ValueError("Invalid receipt content SHA")
        try:
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Receipt source escaped owned scope")
            observed = path.stat()
        except FileNotFoundError:
            return
        if not stat.S_ISREG(observed.st_mode):
            raise ValueError("Nonregular cache source")
        if size is not None and observed.st_size != size:
            raise ValueError("Receipt size changed")
        if mtime is not None and observed.st_mtime_ns != mtime:
            raise ValueError("Receipt mtime changed")
        self.files[key] = {
            "path": key,
            "stat": identity(observed),
            "sha256": sha,
            "source_receipt": str(source),
            "priority": priority,
            "last_advised": 0.0,
            "stat_binding": "captured_at_admission_after_verified_receipt",
        }

    def discover(self) -> None:
        manifest = EVENTS / "EVENT_INPUT_MANIFEST.json"
        data = self.read_changed(manifest)
        if data:
            if data["inputs_sha256"] != INPUTS_SHA:
                raise ValueError("Event manifest binding changed")
            for row in data["tensors"].values():
                self.add(Path(row["path"]), EVENTS / "tensors", row["sha256"], manifest, 1)
        for receipt in (EVENTS / "raw_receipts").glob("*.json"):
            data = self.read_changed(receipt)
            if data and data.get("storage_state") == "PRESENT":
                if data["inputs_sha256"] != INPUTS_SHA or data["status"] != "SHA256_VERIFIED":
                    raise ValueError("Raw source unverified")
                self.add(
                    Path(data["path"]),
                    PROBE / "raw",
                    data["sha256"],
                    receipt,
                    3,
                    data["size_bytes"],
                )
        for receipt in (RGB / "receipts").glob("*/*.json"):
            data = self.read_changed(receipt)
            if data and data.get("status") == "PASS":
                if data["inputs_sha256"] != INPUTS_SHA:
                    raise ValueError("RGB source binding changed")
                self.add(
                    Path(data["path"]),
                    RGB / "data/test",
                    data["sha256"],
                    receipt,
                    2,
                    data["size_bytes"],
                    data["actual_mtime_ns"],
                )
        for receipt in PRESERVATION.glob("*.json"):
            data = self.read_changed(receipt)
            if (
                data
                and data.get("schema") == "pod2_preservation_archive_receipt_v1"
                and data.get("status") == "READY"
            ):
                self.add(
                    Path(data["path"]),
                    Path("/tmp/ejepa_preservation"),
                    data["sha256"],
                    receipt,
                    0,
                    data["bytes"],
                )


def main() -> None:
    import fcntl

    ROOT.mkdir(exist_ok=True)
    with (ROOT / "WORKER.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registry = Registry()
        while time.time() < DEADLINE and not (ROOT / "STOP_REQUEST").exists():
            registry.discover()
            before = observe()
            advised_bytes = 0
            advised_count = 0
            errors = []
            started = time.monotonic()
            if before["headroom_bytes"] < 4 * GIB:
                candidates = sorted(
                    registry.files.values(),
                    key=lambda x: (x["priority"], x["last_advised"], -x["stat"][2]),
                )
                for item in candidates:
                    if time.monotonic() - started > 30 or time.time() >= DEADLINE:
                        break
                    try:
                        advise(item)
                        item["last_advised"] = time.time()
                        advised_bytes += item["stat"][2]
                        advised_count += 1
                    except FileNotFoundError:
                        registry.files.pop(item["path"], None)
                    except (ValueError, OSError) as error:
                        registry.files.pop(item["path"], None)
                        errors.append({"path": item["path"], "error": str(error)})
                    if advised_count and (
                        advised_count % 32 == 0 or item["stat"][2] >= 256 * 1024**2
                    ):
                        if observe()["headroom_bytes"] >= 8 * GIB:
                            break
            after = observe()
            status = {
                "schema": "test12_scoped_cache_controller_v1",
                "timestamp": time.time(),
                "pid": os.getpid(),
                "status": "ADVISED" if advised_count else "IDLE",
                "before": before,
                "after": after,
                "advised_bytes": advised_bytes,
                "advised_files": advised_count,
                "registered_files": len(registry.files),
                "errors": errors[:20],
                "seconds": time.monotonic() - started,
                "source_mutations": 0,
                "global_cache_controls": False,
            }
            atomic(ROOT / "STATUS.json", status)
            atomic(
                ROOT / "REGISTRY.json", {"files": registry.files, "receipt_bindings": registry.seen}
            )
            print(json.dumps(status), flush=True)
            time.sleep(5)
        atomic(
            ROOT / "STATUS.json",
            {"status": "DEADLINE_OR_STOP", "pid": os.getpid(), "timestamp": time.time()},
        )


if __name__ == "__main__":
    main()
