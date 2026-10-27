from __future__ import annotations

import inspect

from operational.runpod_migration import pod2_rgb_handoff as runtime
from scripts.runpod_rgb_pod2_handoff_v2 import (
    NEW_GUARD,
    OLD_GUARD,
    V2_CHUNK_BYTES,
    activate_v2,
)


def _restore(original: tuple[int, str]) -> None:
    runtime.CHUNK_BYTES, runtime.REMOTE_HELPER = original


def test_v2_chunk_and_remote_guard_are_bounded_to_8mib() -> None:
    original = activate_v2()
    try:
        assert runtime.CHUNK_BYTES == V2_CHUNK_BYTES == 8 * 1024 * 1024
        assert runtime.REMOTE_HELPER.count(NEW_GUARD) == 1
        assert OLD_GUARD not in runtime.REMOTE_HELPER
    finally:
        _restore(original)


def test_v2_keeps_exact_quota_and_reservation_contract() -> None:
    constants = (
        runtime.EVENT_RESERVED_BYTES,
        runtime.SYSTEM_RESERVED_BYTES,
        runtime.RESERVE_BYTES,
        runtime.QUOTA_SCHEMA,
        runtime.QUOTA_RUN_ROOT,
        runtime.QUOTA_VOLUME_ROOT,
    )
    original = activate_v2()
    try:
        assert constants == (
            runtime.EVENT_RESERVED_BYTES,
            runtime.SYSTEM_RESERVED_BYTES,
            runtime.RESERVE_BYTES,
            runtime.QUOTA_SCHEMA,
            runtime.QUOTA_RUN_ROOT,
            runtime.QUOTA_VOLUME_ROOT,
        )
    finally:
        _restore(original)


def test_v2_does_not_replace_resume_or_finalize_implementation() -> None:
    run_source = inspect.getsource(runtime.Pod2Handoff.run)
    remote_source = inspect.getsource(runtime.Pod2Handoff._remote_call)
    original = activate_v2()
    try:
        assert inspect.getsource(runtime.Pod2Handoff.run) == run_source
        assert inspect.getsource(runtime.Pod2Handoff._remote_call) == remote_source
        assert 'probe = self._remote_call("probe"' in run_source
        assert 'stream.seek(offset)' in run_source
        assert '"finalize", name, total, total, sha' in run_source
    finally:
        _restore(original)
