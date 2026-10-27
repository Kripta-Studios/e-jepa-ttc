"""Extend the existing page-cache controller to verified, read-only RAW inputs."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

MANIFEST_SHA = '6e766252c8e4ab90ce6c17107cdde353a18c471211fedabeaffa8efecc5fbd20'


def identity(stat: os.stat_result) -> tuple[int, ...]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_mode


def advise_raw(item: dict[str, Any], advise: Any = None) -> None:
    """Evict clean pages only; dataset permissions need not be immutable."""
    fn = advise or os.posix_fadvise
    fd = os.open(item['path'], os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        if identity(os.fstat(fd)) != item['stat']:
            raise ValueError('RAW file identity changed before cache advice')
        fn(fd, 0, 0, getattr(os, 'POSIX_FADV_DONTNEED', 4))
        if identity(os.fstat(fd)) != item['stat']:
            raise ValueError('RAW file identity changed during cache advice')
    finally:
        os.close(fd)


def inventory() -> list[dict[str, Any]]:
    root = Path('/workspace/ejepa_event_raw')
    manifest_path = Path('/workspace/ejepa_event_operations/EVENT_RAW_DEFERRED_MANIFEST.json')
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != MANIFEST_SHA:
        raise ValueError('RAW manifest changed')
    coverage = json.loads((root/'EVENT_RAW_COVERAGE_RECEIPT.json').read_text())
    payload = {k: v for k, v in coverage.items() if k != 'receipt_sha256'}
    canonical = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if not (coverage['status'] == 'VERIFIED' and coverage['hashes_verified'] == 41
            and coverage['raw_deferred_manifest_sha256'] == MANIFEST_SHA
            and coverage['receipt_sha256'] == canonical):
        raise ValueError('RAW coverage not verified')
    result = []
    for row in json.loads(manifest_path.read_text())['files']:
        relative = row['relative'].removeprefix('drives/E/eAP_dataset/')
        lexical = root / relative
        path = lexical.resolve(strict=True)
        if not path.is_relative_to(root) or lexical.is_symlink():
            raise ValueError('RAW path outside admitted root or symlink')
        if path.suffix != '.h5':
            continue
        stat = path.stat()
        if stat.st_size != row['bytes']:
            raise ValueError('RAW size differs from verified manifest')
        result.append({'path': str(path), 'stat': identity(stat), 'sha256': row['sha256']})
    if len(result) != 40:
        raise ValueError('Expected exactly 40 verified event H5 inputs')
    return result


def main() -> int:
    import scripts

    ops = Path('/workspace/ejepa_event_operations')
    scripts.__path__ = [str(ops/'runtime_v12/scripts'), *scripts.__path__]
    path = ops/'resume_20261027/runpod_event_cache_controller_relocated.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != 'a337b72526a762e55990506c8aa1cca2d95dec2d399d8af4b38d7364c5db6b25':
        raise ValueError('Original relocated cache controller changed')
    spec = importlib.util.spec_from_file_location('ejepa_cache_with_raw_scope', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    rows = inventory()
    original = module.advise_cycle
    cursor = 0

    def cycle(*args: Any, **kwargs: Any) -> Any:
        nonlocal cursor
        before = module.cgroup_observation()
        advised = []
        # Hysteresis: reclaim below 4 GiB, refill free headroom to 8 GiB.
        # The training guards still enforce their original 2 GiB reserve.
        if before['effective_headroom_bytes'] < 4 * 1024**3:
            for _ in range(len(rows)):
                item = rows[cursor]
                advise_raw(item)
                advised.append(item['path'])
                cursor = (cursor + 1) % len(rows)
                if module.cgroup_observation()['effective_headroom_bytes'] >= 8 * 1024**3:
                    break
        kwargs['target_headroom_bytes'] = 4 * 1024**3
        next_cursor, value = original(*args, **kwargs)
        value.update(raw_advised_files=advised, raw_manifest_sha256=MANIFEST_SHA,
                     raw_cursor=cursor, raw_before=before, raw_after=module.cgroup_observation(),
                     raw_cache_overlay_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return next_cursor, value

    module.advise_cycle = cycle
    return module.main()


if __name__ == '__main__':
    raise SystemExit(main())
