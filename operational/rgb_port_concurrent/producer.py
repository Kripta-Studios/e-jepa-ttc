"""Additive two-producer cache and lineage wrapper for RGB-PORT."""

# ruff: noqa: ANN401

from __future__ import annotations

import argparse
import importlib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import operational.rgb_port.train_producers as training
from operational.rgb_port.accounting import atomic_write_json, read_json_shared, sha256_file

FIT_IDS = ("E_A5_MATCHED", "E_C2F_MATCHED")
DELEGATES = {
    "operational.rgb_port_acceleration.producer",
    "operational.rgb_port_pipeline_v2.producer",
}


class Runtime:
    """Apply the frozen cache cap and bind every subsequent native checkpoint."""

    def __init__(
        self,
        source: training.ProducerSource,
        recipe: Any,
        run: Path,
        freeze_path: Path,
        freeze: Mapping[str, Any],
    ) -> None:
        self.source = source
        self.run = run
        self.fit_id = str(recipe.fit_id)
        if self.fit_id not in FIT_IDS:
            raise ValueError("Concurrent wrapper is restricted to the two event producers")
        self.fit = run / "fits" / self.fit_id
        self.freeze = dict(freeze)
        self.freeze_sha256 = str(freeze["identity_sha256"])
        self.freeze_file_sha256 = sha256_file(freeze_path)
        self.receipt_dir = self.fit / "concurrent_checkpoints"
        self.pending = self.receipt_dir / "PENDING_EXECUTION_RECEIPT.json"
        self.runtime_path = self.fit / "CONCURRENT_RUNTIME.json"
        limits = freeze["cache_policy"]["per_fit_max_bytes"]
        self.cache_limit = int(limits[self.fit_id])
        inputs = getattr(source, "_inputs", None)
        if inputs is None or not hasattr(inputs, "limit"):
            raise TypeError("Concurrent event producer lacks the canonical bounded Inputs cache")
        self.inputs = inputs
        self.prior_cache_limit = int(inputs.limit)
        inputs.limit = self.cache_limit
        self._write_runtime("ACTIVE")

    def _mode(self) -> str:
        graph = self.freeze["cache_policy"]["graphs"][self.fit_id]
        return str(graph)

    def _common(self) -> dict[str, Any]:
        return {
            "fit_id": self.fit_id,
            "concurrent_freeze_sha256": self.freeze_sha256,
            "concurrent_freeze_file_sha256": self.freeze_file_sha256,
            "cache_limit_bytes": self.cache_limit,
            "graph_mode": self._mode(),
            "optimizer_updates": 0,
            "source_sha256": self.freeze["source_sha256"],
        }

    def _cache(self) -> dict[str, int]:
        with self.inputs.lock:
            return {
                "reads": int(self.inputs.reads),
                "hits": int(self.inputs.hits),
                "entries": len(self.inputs.cache),
                "bytes": int(self.inputs.cache_bytes),
                "limit_bytes": int(self.inputs.limit),
            }

    def _matches_common(self, value: Mapping[str, Any]) -> bool:
        """Require every immutable execution binding, including the cache policy."""
        return all(value.get(name) == item for name, item in self._common().items())

    def _pointer_matches(self, state: Any, pointer: Mapping[str, Any]) -> bool:
        return (
            pointer.get("schema") == "rgb_port_checkpoint_pointer_v1"
            and pointer.get("completed_updates") == state.completed
            and pointer.get("version") == f"checkpoint_{state.completed:06d}.pt"
            and pointer.get("identity_sha256") == state.identity_sha256
            and pointer.get("checkpoint_sha256") == sha256_file(state.path)
        )

    def _write_runtime(self, status: str, state: Any | None = None) -> None:
        value: dict[str, Any] = {
            "schema": "rgb_port_concurrent_runtime_v1",
            "status": status,
            **self._common(),
            "prior_cache_limit_bytes": self.prior_cache_limit,
            "cache": self._cache(),
        }
        if state is not None:
            value.update(
                completed_updates=state.completed,
                checkpoint_sha256=sha256_file(state.path),
                checkpoint_identity_sha256=state.identity_sha256,
            )
        atomic_write_json(self.runtime_path, value)

    def before_save(self, state: Any) -> None:
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            self.pending,
            {
                "schema": "rgb_port_concurrent_pending_v1",
                "status": "PENDING",
                **self._common(),
                "start_update": state.durable,
                "end_update": state.completed,
                "checkpoint_version": f"checkpoint_{state.completed:06d}.pt",
                "checkpoint_identity_sha256": state.identity_sha256,
            },
        )

    def _receipt(self, state: Any, status: str) -> dict[str, Any]:
        return {
            "schema": "rgb_port_concurrent_checkpoint_receipt_v1",
            "status": status,
            **self._common(),
            "completed_updates": state.completed,
            "checkpoint_version": f"checkpoint_{state.completed:06d}.pt",
            "checkpoint_sha256": sha256_file(state.path),
            "checkpoint_identity_sha256": state.identity_sha256,
            "cache": self._cache(),
        }

    def after_save(self, state: Any) -> None:
        self._write_runtime("ACTIVE", state)
        atomic_write_json(
            self.receipt_dir / f"checkpoint_{state.completed:06d}.json",
            self._receipt(state, "COMPLETE"),
        )
        self.pending.unlink(missing_ok=True)

    def after_restore(self, state: Any) -> None:
        receipt_path = self.receipt_dir / f"checkpoint_{state.completed:06d}.json"
        if receipt_path.is_file():
            value = read_json_shared(receipt_path)
            status = value.get("status")
            expected = self._receipt(state, str(status))
            if (
                status
                not in {
                    "COMPLETE",
                    "RECOVERED_FROM_PENDING",
                    "ADOPTED_V2_LINEAGE",
                    "ADOPTED_V1_LINEAGE",
                }
                or any(
                    value.get(name) != item for name, item in expected.items() if name != "cache"
                )
                or not isinstance(value.get("cache"), Mapping)
                or value["cache"].get("limit_bytes") != self.cache_limit
            ):
                raise RuntimeError("Concurrent checkpoint receipt differs")
        elif self.pending.is_file():
            pending = read_json_shared(self.pending)
            pointer = read_json_shared(state.pointer)
            if (
                pending.get("schema") != "rgb_port_concurrent_pending_v1"
                or pending.get("status") != "PENDING"
                or pending.get("fit_id") != self.fit_id
                or not self._matches_common(pending)
                or not isinstance(pending.get("start_update"), int)
                or pending["start_update"] > state.completed
                or pending.get("end_update") != state.completed
                or pending.get("checkpoint_version") != pointer.get("version")
                or pending.get("checkpoint_identity_sha256") != state.identity_sha256
                or not self._pointer_matches(state, pointer)
            ):
                raise RuntimeError("Concurrent pending proof cannot repair checkpoint")
            atomic_write_json(receipt_path, self._receipt(state, "RECOVERED_FROM_PENDING"))
            self.pending.unlink()
        else:
            origin = self.freeze["origins"][self.fit_id]
            if (
                origin["completed_updates"] != state.completed
                or origin["checkpoint_sha256"] != sha256_file(state.path)
                or origin["identity_sha256"] != state.identity_sha256
            ):
                self._adopt_c2f_lineage(state, origin)
        self._write_runtime("READY", state)

    def _adopt_c2f_lineage(self, state: Any, origin: Mapping[str, Any]) -> None:
        """Bind progress made by the explicitly adopted live C2F child.

        The freeze is taken while C2F is running, so that child may reach a later
        Pipeline V2 checkpoint before its first launch through this wrapper.
        Only independently validated V2 lineage can bridge that one-time gap.
        """
        if self.fit_id != "E_C2F_MATCHED" or state.completed <= int(origin["completed_updates"]):
            raise RuntimeError("Concurrent restore lacks origin or supplemental receipt")
        v2_root = self.fit / "pipeline_checkpoints"
        v2_receipt_path = v2_root / f"checkpoint_{state.completed:06d}.json"
        if not v2_receipt_path.is_file():
            self._adopt_c2f_v1(state)
            return
        from operational.rgb_port_pipeline_v2.contracts import validate_pipeline_freeze
        from operational.rgb_port_pipeline_v2.receipts import validate_fit_lineage

        pipeline_path = Path(self.freeze["pipeline_freeze_path"]).resolve(strict=True)
        pipeline = validate_pipeline_freeze(pipeline_path)
        validate_fit_lineage(self.run, pipeline, pipeline_path, self.fit_id)
        value = read_json_shared(v2_receipt_path)
        timing_path = v2_root / str(value.get("timings_snapshot", ""))
        if (
            value.get("schema") != "rgb_port_pipeline_checkpoint_receipt_v2"
            or value.get("status") not in {"COMPLETE", "RECOVERED_FROM_PENDING"}
            or value.get("fit_id") != self.fit_id
            or value.get("completed_updates") != state.completed
            or value.get("checkpoint_version") != f"checkpoint_{state.completed:06d}.pt"
            or value.get("checkpoint_sha256") != sha256_file(state.path)
            or value.get("checkpoint_identity_sha256") != state.identity_sha256
            or not timing_path.is_file()
            or value.get("timings_sha256") != sha256_file(timing_path)
        ):
            raise RuntimeError("Adopted C2F Pipeline V2 evidence differs")
        atomic_write_json(
            self.receipt_dir / f"checkpoint_{state.completed:06d}.json",
            {
                **self._receipt(state, "ADOPTED_V2_LINEAGE"),
                "adopted_pipeline_v2_receipt": str(v2_receipt_path.resolve()),
                "adopted_pipeline_v2_receipt_sha256": sha256_file(v2_receipt_path),
                "adopted_pipeline_v2_timings": str(timing_path.resolve()),
                "adopted_pipeline_v2_timings_sha256": sha256_file(timing_path),
            },
        )

    def _adopt_c2f_v1(self, state: Any) -> None:
        """Bridge the live C2F child when Pipeline V2 deliberately rolled back."""
        from operational.rgb_port_acceleration.queue import _validate_runtime_receipts

        _validate_runtime_receipts(self.run, require_endpoint=False, fit_id=self.fit_id)
        root = self.fit / "acceleration_checkpoints"
        receipt_path = root / f"checkpoint_{state.completed:06d}.json"
        if not receipt_path.is_file():
            raise RuntimeError("Adopted C2F progress lacks a V1 acceleration receipt")
        value = read_json_shared(receipt_path)
        snapshot_path = root / str(value.get("runtime_snapshot", ""))
        if (
            value.get("schema") != "rgb_port_acceleration_checkpoint_receipt_v1"
            or value.get("status") not in {"COMPLETE", "RECOVERED_FROM_PENDING"}
            or value.get("fit_id") != self.fit_id
            or value.get("completed_updates") != state.completed
            or value.get("checkpoint_version") != f"checkpoint_{state.completed:06d}.pt"
            or value.get("checkpoint_sha256") != sha256_file(state.path)
            or value.get("checkpoint_identity_sha256") != state.identity_sha256
            or value.get("mode") != "batched_metrics_only"
            or value.get("backend") is not None
            or value.get("shape") is not None
            or not snapshot_path.is_file()
            or value.get("runtime_receipt_sha256") != sha256_file(snapshot_path)
        ):
            raise RuntimeError("Adopted C2F V1 acceleration evidence differs")
        atomic_write_json(
            self.receipt_dir / f"checkpoint_{state.completed:06d}.json",
            {
                **self._receipt(state, "ADOPTED_V1_LINEAGE"),
                "adopted_acceleration_receipt": str(receipt_path.resolve()),
                "adopted_acceleration_receipt_sha256": sha256_file(receipt_path),
                "adopted_acceleration_runtime": str(snapshot_path.resolve()),
                "adopted_acceleration_runtime_sha256": sha256_file(snapshot_path),
            },
        )


@contextmanager
def installed(runtime: Runtime) -> Iterator[None]:
    original_restore = training.ProducerCheckpoint.restore
    original_save = training.ProducerCheckpoint.save

    def restore(state: Any, *args: Any, **kwargs: Any) -> Any:
        cursor = original_restore(state, *args, **kwargs)
        runtime.after_restore(state)
        return cursor

    def save(state: Any, *args: Any, **kwargs: Any) -> None:
        runtime.before_save(state)
        original_save(state, *args, **kwargs)
        runtime.after_save(state)

    cast(Any, training.ProducerCheckpoint).restore = restore
    cast(Any, training.ProducerCheckpoint).save = save
    try:
        yield
    finally:
        cast(Any, training.ProducerCheckpoint).restore = original_restore
        cast(Any, training.ProducerCheckpoint).save = original_save


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrent-freeze", type=Path, required=True)
    parser.add_argument("--delegate-module", choices=sorted(DELEGATES), required=True)
    parser.add_argument("forward", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.forward or args.forward[0] != "--":
        parser.error("delegate arguments must follow --")
    forwarded = args.forward[1:]
    if "--fit-id" not in forwarded:
        parser.error("delegate command lacks --fit-id")
    fit_id = forwarded[forwarded.index("--fit-id") + 1]
    if fit_id not in FIT_IDS:
        parser.error("concurrent wrapper is restricted to E_A5/E_C2F")
    from operational.rgb_port_concurrent.contracts import validate_concurrent_freeze

    freeze_path = args.concurrent_freeze.resolve(strict=True)
    freeze = validate_concurrent_freeze(freeze_path)
    original_fit = training.fit_producer

    def fit(
        source: training.ProducerSource, recipe: Any, run: Path, **kwargs: Any
    ) -> dict[str, Any]:
        runtime = Runtime(source, recipe, run, freeze_path, freeze)
        with installed(runtime):
            return original_fit(source, recipe, run, **kwargs)

    training.fit_producer = fit
    try:
        delegate = importlib.import_module(args.delegate_module)
        return int(delegate.main(forwarded))
    finally:
        training.fit_producer = original_fit


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FIT_IDS", "Runtime", "installed", "main"]
