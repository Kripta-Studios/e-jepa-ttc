"""Preserve native PID-time precision in the frozen concurrent owner scanner."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from unittest.mock import patch

import psutil

from operational.rgb_port.accounting import identity_is_live
from operational.rgb_port_concurrent import queue


def native_owner_scan(config: dict, run: Path) -> dict | None:
    """Use exact OS times in a private copy after validating recorded identities."""
    reader = queue.read_json_shared

    def read(path: Path) -> dict:
        value = reader(path)
        if path == run / 'RGB_PORT_STATE.json':
            value = copy.deepcopy(value)
            for fit_id in queue.FIT_IDS:
                child = value['tasks'].get(fit_id, {}).get('child')
                if isinstance(child, dict) and identity_is_live(child):
                    process = psutil.Process(int(child['pid']))
                    native = process.create_time()
                    if abs(native - float(child['create_time'])) >= 0.01:
                        raise RuntimeError('Recorded child no longer matches its native process')
                    child['create_time'] = native
        return value

    with patch.object(queue, 'read_json_shared', read):
        return _original_scan(config, run)


_original_scan = queue._unrelated_heavy_owner


if __name__ == '__main__':
    with patch.object(queue, '_unrelated_heavy_owner', native_owner_scan):
        raise SystemExit(queue.main(sys.argv[1:]))
