"""Cache verified RGB media admissions behind an exact per-process stat barrier."""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

V6_SHA256 = "88bf04a2ad2b2938b93b1a1a2a8099de07365992ea0a1ff75ef6bd81684d19ca"


def digest(path: Path) -> str:
    import hashlib

    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value: object) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _stat(path: Path) -> tuple[int, int, int, int, int]:
    value = path.stat()
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _barrier_paths(config: dict[str, Any], rotation: int, role: str) -> tuple[Path, ...]:
    scope = config["scope_map"].get(f"R{rotation}_{role}")
    if scope is None:
        return ()
    plan_path, manifest_path = Path(str(scope["plan_path"])), Path(
        str(scope["manifest_path"])
    )
    if not plan_path.is_file() or not manifest_path.is_file():
        return (plan_path, manifest_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    media_root = Path(str(config["media_root"]))
    paths: list[Path] = [plan_path, manifest_path]
    for item in plan.get("tars", []):
        relative = Path(str(item["relative"]).replace("\\", "/"))
        tar_path = media_root / relative
        paths.extend((tar_path, tar_path.with_suffix(tar_path.suffix + ".json")))
    return tuple(paths)


def _snapshot(
    paths: tuple[Path, ...],
) -> tuple[tuple[str, tuple[int, int, int, int, int]], ...] | None:
    try:
        return tuple((str(path), _stat(path)) for path in paths)
    except (FileNotFoundError, NotADirectoryError):
        return None


class AdmissionCache:
    """Memoize PASS only while every admitted identity and filesystem stat is exact."""

    def __init__(self, verifier: Any) -> None:
        self.verifier = verifier
        self.entries: dict[tuple[int, str], dict[str, Any]] = {}

    def verify(self, config: dict[str, Any], rotation: int, role: str) -> dict[str, Any] | None:
        key = (rotation, role)
        scope = config["scope_map"].get(f"R{rotation}_{role}")
        authority_sha = str(config.get("authority_sha256"))
        scope_sha = canonical(scope) if scope is not None else None
        cached = self.entries.get(key)
        if cached is not None:
            if (
                cached["authority_sha256"] != authority_sha
                or cached["scope_evidence_sha256"] != scope_sha
            ):
                raise ValueError("Cached RGB media authority/scope changed")
            current = _snapshot(cached["paths"])
            if current is None or current != cached["barrier"]:
                raise ValueError("Cached RGB media stat barrier changed")
            return dict(cached["evidence"])

        paths_before = _barrier_paths(config, rotation, role)
        before = _snapshot(paths_before)
        evidence = self.verifier(config, rotation, role)
        if evidence is None:
            return None
        paths_after = _barrier_paths(config, rotation, role)
        after = _snapshot(paths_after)
        if (
            not paths_before
            or paths_before != paths_after
            or before is None
            or after is None
            or before != after
        ):
            raise ValueError("RGB media changed during full admission verification")
        self.entries[key] = {
            "authority_sha256": authority_sha,
            "scope_evidence_sha256": scope_sha,
            "paths": paths_after,
            "barrier": after,
            "evidence": dict(evidence),
        }
        return dict(evidence)


@contextmanager
def compatible_v6_composed_v5_hook(
    v6: Any, v5: Any, media_config: Path
) -> Iterator[None]:
    """Compose V6 outside V5 while preserving V5's internal keyword protocol."""
    original_v5 = v5.install_queue_heartbeat

    def install(module: Any, clock: Any = time.monotonic, **kwargs: Any) -> None:
        def invoke_v5(inner: Any, clock: Any = clock) -> None:
            original_v5(inner, clock=clock, **kwargs)

        v6.install_queue_heartbeat(
            module,
            clock=clock,
            media_config=media_config,
            heartbeat_installer=invoke_v5,
        )

    v5.install_queue_heartbeat = install
    try:
        yield
    finally:
        v5.install_queue_heartbeat = original_v5


def main() -> None:
    source, ops = Path(__file__).resolve(), Path(__file__).resolve().parent
    v6_path = ops / "runpod_rgb_context_v6.py"
    if digest(v6_path) != V6_SHA256:
        raise ValueError("Frozen RGB context V6 bytes differ")
    sys.path.insert(0, str(ops))
    import runpod_rgb_context_v6 as v6

    run = Path(sys.argv[sys.argv.index("--run") + 1]).resolve(strict=True)
    binding = {
        "schema": "ejepa_rgb_media_stat_cache_v7",
        "status": "PASS",
        "source_sha256": digest(source),
        "context_v6_sha256": digest(v6_path),
        "migration_binding_sha256": digest(run / "MIGRATION_BINDING.json"),
        "barrier_fields": ["st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"],
        "barrier_scope": "plan_manifest_all_tars_all_sidecars",
        "cache_scope": "current_process_only",
        "missing_or_changed_after_admission": "FAIL_CLOSED",
        "missing_before_admission": "NO_CACHE_WAIT",
    }
    target = run / "RGB_OPERATIONAL_RECOVERY_V7.json"
    if target.is_file() and json.loads(target.read_text(encoding="utf-8")) != binding:
        raise ValueError("RGB recovery V7 binding differs")
    if not target.exists():
        atomic_json(target, binding)
    original = v6.verify_scope
    original_composer = v6.composed_v5_hook
    cache = AdmissionCache(original)
    v6.verify_scope = cache.verify
    v6.composed_v5_hook = lambda v5, media_config: compatible_v6_composed_v5_hook(
        v6, v5, media_config
    )
    try:
        v6.main()
    finally:
        v6.composed_v5_hook = original_composer
        v6.verify_scope = original


if __name__ == "__main__":
    main()
