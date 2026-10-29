"""External, identity-preserving shared Inputs policy for event fold sources.

This operational overlay aliases only the decoded-input cache used by the P/H/V
storage views.  Role tokens, native event indices, batches, scientific identity,
and the original 2 GiB cache limit remain untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MethodType
from typing import Any

from operational.final_refit.recipe import canonical_sha256
from operational.rgb_port.accounting import atomic_write_json

EXPECTED_ROLES = frozenset({"P", "H", "V"})
EXPECTED_LIMIT_BYTES = 2 * 1024**3
_ORIGINAL_LOADER: Callable[..., Any] | None = None


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _compatibility(source: Any) -> dict[str, Any]:
    inputs = source._inputs
    identity = dict(source.identity)
    return {
        "event_source_root": str(Path(identity["event_source_root"]).resolve(strict=True)),
        "inputs_output": str(Path(inputs.output).resolve(strict=True)),
        "inputs_directory": str(Path(inputs.directory).resolve(strict=True)),
        "prepare_freeze_sha256": identity["prepare_freeze_sha256"],
        "train40_rows_sha256": identity["train40_rows_sha256"],
        "train40_index_sha256": identity["train40_index_sha256"],
        "dino_teacher_freeze_sha256": identity["dino_teacher_freeze_sha256"],
        "dino_targets_sha256": identity["dino_targets_sha256"],
        "decoded_event_cache_limit_bytes": int(identity["decoded_event_cache_limit_bytes"]),
        "metadata_sha256": canonical_sha256(list(inputs.metadata)),
    }


def share_fold_inputs(source: Any, receipt_path: Path) -> Any:
    """Alias compatible empty P/H/V Inputs caches and preserve source identity."""
    sources = source.sources
    if set(sources) != EXPECTED_ROLES:
        raise ValueError("Shared Inputs policy requires exact P/H/V historical views")
    roles = sorted(sources)
    original_identity = json.loads(json.dumps(dict(source.identity)))
    compatibility = {role: _compatibility(sources[role]) for role in roles}
    if any(value != compatibility[roles[0]] for value in compatibility.values()):
        raise ValueError("P/H/V Inputs do not bind the same event/teacher storage")
    if compatibility[roles[0]]["decoded_event_cache_limit_bytes"] != EXPECTED_LIMIT_BYTES:
        raise ValueError("Shared Inputs policy cannot change the frozen 2 GiB cache limit")

    pools = [sources[role]._inputs for role in roles]
    if len({id(inputs) for inputs in pools}) != len(pools):
        raise ValueError("Fold Inputs were already aliased before policy admission")
    for inputs in pools:
        with inputs.lock:
            if inputs.cache or inputs.cache_bytes or inputs.reads or inputs.hits:
                raise ValueError("Inputs sharing must occur before the first cache read")

    shared = pools[0]
    for duplicate in pools[1:]:
        duplicate.close()
    for role in roles:
        sources[role]._inputs = shared

    closed = False

    def close_once(_fold: Any) -> None:
        nonlocal closed
        if not closed:
            shared.close()
            closed = True

    source.close = MethodType(close_once, source)
    if dict(source.identity) != original_identity:
        close_once(source)
        raise ValueError("Shared Inputs policy changed scientific source identity")

    policy = {
        "schema": "ejepa_event_shared_inputs_policy_v1",
        "status": "ADMITTED",
        "roles": roles,
        "shared_inputs_instances": 1,
        "closed_empty_duplicate_inputs": len(pools) - 1,
        "decoded_event_cache_limit_bytes": EXPECTED_LIMIT_BYTES,
        "role_tokens_unchanged": True,
        "role_event_indices_unchanged": True,
        "data_filter_changed": False,
        "scientific_source_identity_changed": False,
        "original_source_identity_sha256": original_identity["identity_sha256"],
        "storage_compatibility": compatibility[roles[0]],
        "overlay_source_sha256": _digest(Path(__file__)),
        "observed_timestamp": time.time(),
    }
    policy["policy_sha256"] = canonical_sha256(
        {key: value for key, value in policy.items() if key != "observed_timestamp"}
    )
    try:
        atomic_write_json(receipt_path, policy)
    except Exception:
        close_once(source)
        raise
    return source


def install(receipt_path: Path) -> Mapping[str, Any]:
    """Install the loader overlay before importing any scientific runner."""
    from operational.final_refit import fold_sources

    global _ORIGINAL_LOADER
    if _ORIGINAL_LOADER is not None:
        raise RuntimeError("Shared Inputs loader policy is already installed")
    _ORIGINAL_LOADER = fold_sources.load_fold_event_source

    def load_shared(fold_contract: Path, rotation: int, event_source_root: Path) -> Any:
        assert _ORIGINAL_LOADER is not None
        source = _ORIGINAL_LOADER(fold_contract, rotation, event_source_root)
        try:
            return share_fold_inputs(source, receipt_path)
        except Exception:
            source.close()
            raise

    fold_sources.load_fold_event_source = load_shared
    return {
        "schema": "ejepa_event_shared_inputs_install_v1",
        "status": "INSTALLED",
        "overlay_source_sha256": _digest(Path(__file__)),
        "receipt_path": str(receipt_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-receipt", type=Path, required=True)
    parser.add_argument("--entrypoint", type=Path, required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    entrypoint = args.entrypoint.resolve(strict=True)
    install(args.policy_receipt)
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    sys.argv = [str(entrypoint), *arguments]
    runpy.run_path(str(entrypoint), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
