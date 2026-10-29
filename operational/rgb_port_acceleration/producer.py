"""Additive, checkpoint-safe CUDA-graph runtime for RGB-PORT E_A5 only."""

# ruff: noqa: ANN401

from __future__ import annotations

import argparse
import copy
import random
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn

import operational.rgb_port.train_producers as training
from operational.rgb_port.accounting import atomic_write_json, read_json_shared, sha256_file
from operational.rgb_port.recipe import canonical_sha256

FIT_ID = "E_A5_MATCHED"
PRODUCER_IDS = {"E_A5_MATCHED", "E_C2F_MATCHED", "R_A5", "R_C2F"}
BACKEND = "cudagraphs"
FREEZE_SCHEMA = "rgb_port_acceleration_freeze_v1"


class ForwardDispatcher:
    """Use the captured callable only for the admitted B32T3 signature."""

    def __init__(self, original: Any, compiled: Any) -> None:
        self.original, self.compiled = original, compiled
        self.compiled_calls = self.eager_calls = 0

    def __call__(
        self, inputs: Tensor, delta_t_s: Tensor, *, return_dense_features: bool = False
    ) -> object:
        if inputs.shape[:2] == (32, 3):
            self.compiled_calls += 1
            return self.compiled(inputs, delta_t_s, return_dense_features=return_dense_features)
        self.eager_calls += 1
        return self.original(inputs, delta_t_s, return_dense_features=return_dense_features)

    def snapshot(self) -> dict[str, int]:
        return {
            "compiled_calls": self.compiled_calls,
            "eager_calls": self.eager_calls,
        }


class LazyCPUComponents(dict[str, Tensor]):
    """Materialize all ordered component scalars in one GPU-to-CPU transfer."""

    def __init__(self, values: Mapping[str, Tensor], counter: dict[str, int]) -> None:
        super().__init__(values)
        self._names = tuple(values)
        self._values = tuple(values.values())
        self._counter = counter
        self._materialized: dict[str, Tensor] | None = None

    def _cpu(self) -> dict[str, Tensor]:
        if self._materialized is None:
            stacked = torch.stack([value.detach().float() for value in self._values]).cpu()
            self._materialized = dict(zip(self._names, stacked.unbind(), strict=True))
            self._counter["component_cpu_transfers"] += 1
        return self._materialized

    def items(self) -> Any:
        return self._cpu().items()


def _same(left: object, right: object) -> bool:
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return (
            left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return tuple(left) == tuple(right) and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _freeze(path: Path) -> dict[str, Any]:
    value = read_json_shared(path)
    required = {
        "schema",
        "status",
        "fit_id",
        "metrics_fit_ids",
        "graph_fit_id",
        "original_source_freeze_sha256",
        "admission_path",
        "admission_sha256",
        "source_sha256",
        "torch_backend_source_sha256",
        "admitted_shape",
        "origin",
        "identity_sha256",
    }
    if not required.issubset(value) or value["schema"] != FREEZE_SCHEMA:
        raise ValueError("Acceleration freeze schema/fields differ")
    if value["status"] != "FROZEN" or value["fit_id"] != FIT_ID:
        raise ValueError("Acceleration freeze is not the frozen E_A5 runtime")
    if set(value["metrics_fit_ids"]) != PRODUCER_IDS or value["graph_fit_id"] != FIT_ID:
        raise ValueError("Acceleration freeze producer scopes differ")
    if value["admitted_shape"] != {"batch_size": 32, "frames": 3}:
        raise ValueError("Acceleration freeze shape differs from B32T3")
    if value["identity_sha256"] != canonical_sha256(
        {name: item for name, item in value.items() if name != "identity_sha256"}
    ):
        raise ValueError("Acceleration freeze canonical identity differs")
    sources = value["source_sha256"]
    expected_sources = {
        str(Path(__file__).resolve()): sha256_file(Path(__file__)),
        str(Path(training.__file__).resolve()): sha256_file(Path(training.__file__)),
    }
    from operational.rgb_port_revision.migration import inventory_matches

    if not inventory_matches(sources, expected_sources, path.parent):
        raise ValueError("Acceleration executed-source hashes differ")
    admission = Path(value["admission_path"]).resolve(strict=True)
    if sha256_file(admission) != value["admission_sha256"]:
        raise ValueError("Acceleration admission bytes differ")
    admitted = read_json_shared(admission)
    if admitted.get("status") != "PASSED" or admitted.get("optimizer_updates") != 0:
        raise ValueError("Acceleration admission did not pass at zero optimizer updates")
    backend = Path(torch.__file__).resolve().parent / "_dynamo/backends/cudagraphs.py"
    graphs = Path(cast(str, torch.cuda.graphs.__file__)).resolve(strict=True)
    expected_backend = {
        str(backend): sha256_file(backend),
        str(graphs): sha256_file(graphs),
    }
    if value["torch_backend_source_sha256"] != expected_backend:
        raise ValueError("Installed cudagraph backend differs from freeze")
    return value


def verify_acceleration_freeze(path: Path) -> dict[str, Any]:
    """Validate the additive executed-source, backend, admission and scope binding."""
    return _freeze(path.resolve(strict=True))


class Runtime:
    def __init__(
        self,
        source: Any,
        recipe: Any,
        run: Path,
        freeze_path: Path,
        freeze: Mapping[str, Any],
    ) -> None:
        self.source, self.recipe, self.run = source, recipe, run
        self.fit_id = str(recipe.fit_id)
        self.graph_enabled = self.fit_id == FIT_ID
        if sha256_file(run / "SOURCE_FREEZE.json") != freeze["original_source_freeze_sha256"]:
            raise ValueError("Original RGB-PORT source freeze differs")
        self.freeze_path, self.freeze = freeze_path, dict(freeze)
        self.freeze_sha256 = str(freeze["identity_sha256"])
        self.freeze_file_sha256 = sha256_file(freeze_path)
        self.dispatcher: ForwardDispatcher | None = None
        self.counters = {"component_cpu_transfers": 0, "warmup_backward_passes": 0}
        self.runtime_path = run / "fits" / self.fit_id / "ACCELERATION_RUNTIME.json"
        self.prewarm_path = run / "fits" / self.fit_id / "PREWARM_QA.json"
        self.receipt_dir = run / "fits" / self.fit_id / "acceleration_checkpoints"
        self.pending_path = self.receipt_dir / "PENDING_EXECUTION_RECEIPT.json"

    def _ready(self, completed: int, checkpoint_sha: str, exact: Mapping[str, bool]) -> None:
        atomic_write_json(
            self.runtime_path,
            {
                "schema": "rgb_port_acceleration_runtime_v1",
                "status": "READY",
                "mode": "graph_and_batched_metrics"
                if self.graph_enabled
                else "batched_metrics_only",
                "fit_id": self.fit_id,
                "completed_updates": completed,
                "optimizer_updates": 0,
                "acceleration_freeze_sha256": self.freeze_sha256,
                "acceleration_freeze_file_sha256": self.freeze_file_sha256,
                "admission_sha256": (
                    self.freeze["admission_sha256"] if self.graph_enabled else None
                ),
                "checkpoint_sha256": checkpoint_sha,
                "backend": BACKEND if self.graph_enabled else None,
                "shape": {"batch_size": 32, "frames": 3} if self.graph_enabled else None,
                "post_warm_exact": dict(exact),
                "counters": self.counters,
                "source_sha256": self.freeze["source_sha256"],
                "torch_backend_source_sha256": self.freeze["torch_backend_source_sha256"],
            },
        )

    def _sync_runtime_to_checkpoint(self, state: Any) -> dict[str, Any]:
        """Make mutable session state describe the durable checkpoint exactly."""
        checkpoint_sha = sha256_file(state.path)
        runtime = read_json_shared(self.runtime_path)
        runtime.update(
            status="ACTIVE",
            completed_updates=state.completed,
            checkpoint_sha256=checkpoint_sha,
            last_saved_update=state.completed,
            last_checkpoint_sha256=checkpoint_sha,
            counters={
                **self.counters,
                **(self.dispatcher.snapshot() if self.dispatcher is not None else {}),
            },
        )
        atomic_write_json(self.runtime_path, runtime)
        return runtime

    def _archive_runtime(self, state: Any, runtime: Mapping[str, Any]) -> tuple[Path, str]:
        """Seal one immutable runtime snapshot for one durable checkpoint version."""
        archive = self.receipt_dir / f"runtime_{state.completed:06d}.json"
        if archive.is_file():
            existing = read_json_shared(archive)
            if (
                existing.get("last_saved_update") != state.completed
                or existing.get("last_checkpoint_sha256") != sha256_file(state.path)
                or existing.get("acceleration_freeze_sha256") != self.freeze_sha256
                or existing.get("acceleration_freeze_file_sha256") != self.freeze_file_sha256
            ):
                raise RuntimeError("Immutable acceleration runtime snapshot differs")
        else:
            atomic_write_json(archive, dict(runtime))
        return archive, sha256_file(archive)

    def _checkpoint_receipt(
        self, state: Any, *, status: str, runtime_snapshot: Path, runtime_sha: str
    ) -> dict[str, Any]:
        return {
            "schema": "rgb_port_acceleration_checkpoint_receipt_v1",
            "status": status,
            "fit_id": self.fit_id,
            "completed_updates": state.completed,
            "checkpoint_version": f"checkpoint_{state.completed:06d}.pt",
            "checkpoint_sha256": sha256_file(state.path),
            "checkpoint_identity_sha256": state.identity_sha256,
            "acceleration_freeze_sha256": self.freeze_sha256,
            "acceleration_freeze_file_sha256": self.freeze_file_sha256,
            "runtime_receipt_sha256": runtime_sha,
            "runtime_snapshot": runtime_snapshot.name,
            "admission_sha256": self.freeze["admission_sha256"] if self.graph_enabled else None,
            "source_sha256": self.freeze["source_sha256"],
            "torch_backend_source_sha256": self.freeze["torch_backend_source_sha256"],
            "backend": BACKEND if self.graph_enabled else None,
            "shape": {"batch_size": 32, "frames": 3} if self.graph_enabled else None,
            "mode": "graph_and_batched_metrics" if self.graph_enabled else "batched_metrics_only",
            "counters": {
                **self.counters,
                **(self.dispatcher.snapshot() if self.dispatcher is not None else {}),
            },
        }

    def before_save(self, state: Any) -> None:
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            self.pending_path,
            {
                "schema": "rgb_port_acceleration_pending_v1",
                "status": "PENDING",
                "fit_id": self.fit_id,
                "mode": (
                    "graph_and_batched_metrics" if self.graph_enabled else "batched_metrics_only"
                ),
                "start_update": state.durable,
                "end_update": state.completed,
                "checkpoint_version": f"checkpoint_{state.completed:06d}.pt",
                "checkpoint_identity_sha256": state.identity_sha256,
                "acceleration_freeze_sha256": self.freeze_sha256,
                "acceleration_freeze_file_sha256": self.freeze_file_sha256,
                "admission_sha256": (
                    self.freeze["admission_sha256"] if self.graph_enabled else None
                ),
                "source_sha256": self.freeze["source_sha256"],
                "torch_backend_source_sha256": self.freeze["torch_backend_source_sha256"],
            },
        )

    def verify_or_repair_lineage(self, state: Any) -> None:
        origin = self.freeze["origin"]
        if self.graph_enabled and state.completed == int(origin["completed_updates"]):
            if (
                sha256_file(state.path) != origin["checkpoint_sha256"]
                or state.identity_sha256 != origin["identity_sha256"]
            ):
                raise ValueError("Native acceleration origin differs")
            return
        receipt_path = self.receipt_dir / f"checkpoint_{state.completed:06d}.json"
        if receipt_path.is_file():
            receipt = read_json_shared(receipt_path)
            snapshot_name = f"runtime_{state.completed:06d}.json"
            runtime_snapshot = self.receipt_dir / snapshot_name
            expected = {
                "schema": "rgb_port_acceleration_checkpoint_receipt_v1",
                "fit_id": self.fit_id,
                "completed_updates": state.completed,
                "checkpoint_sha256": sha256_file(state.path),
                "checkpoint_identity_sha256": state.identity_sha256,
                "acceleration_freeze_sha256": self.freeze_sha256,
                "acceleration_freeze_file_sha256": self.freeze_file_sha256,
            }
            if any(receipt.get(name) != value for name, value in expected.items()):
                raise ValueError("Acceleration checkpoint receipt differs")
            if (
                receipt.get("runtime_snapshot") != snapshot_name
                or not runtime_snapshot.is_file()
                or runtime_snapshot.name != snapshot_name
                or sha256_file(runtime_snapshot) != receipt.get("runtime_receipt_sha256")
            ):
                raise ValueError("Acceleration checkpoint runtime snapshot differs")
            return
        if not self.pending_path.is_file() or not self.runtime_path.is_file():
            raise RuntimeError("Acceleration checkpoint lacks receipt and pre-save pending proof")
        pending = read_json_shared(self.pending_path)
        pointer = read_json_shared(state.pointer)
        expected_pending = {
            "schema": "rgb_port_acceleration_pending_v1",
            "status": "PENDING",
            "fit_id": self.fit_id,
            "end_update": state.completed,
            "checkpoint_version": pointer["version"],
            "checkpoint_identity_sha256": state.identity_sha256,
            "acceleration_freeze_sha256": self.freeze_sha256,
            "acceleration_freeze_file_sha256": self.freeze_file_sha256,
        }
        if any(pending.get(name) != value for name, value in expected_pending.items()):
            raise ValueError("Pending acceleration proof differs from restored pointer")
        if pointer.get("checkpoint_sha256") != sha256_file(state.path):
            raise ValueError("Pending repair pointer/checkpoint SHA differs")
        runtime = self._sync_runtime_to_checkpoint(state)
        runtime_snapshot, runtime_sha = self._archive_runtime(state, runtime)
        atomic_write_json(
            receipt_path,
            self._checkpoint_receipt(
                state,
                status="RECOVERED_FROM_PENDING",
                runtime_snapshot=runtime_snapshot,
                runtime_sha=runtime_sha,
            ),
        )
        self.pending_path.unlink()

    def after_restore(
        self,
        state: Any,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        generator: torch.Generator,
        cursor: dict[str, Any],
        original_restore: Any,
    ) -> dict[str, Any]:
        if (
            not self.graph_enabled
            and not self.runtime_path.is_file()
            and self.pending_path.is_file()
        ):
            self._ready(
                state.completed,
                sha256_file(state.path),
                {"metrics_runtime_recovered_from_pending": True},
            )
        self.verify_or_repair_lineage(state)
        if not self.graph_enabled:
            self._ready(
                state.completed,
                sha256_file(state.path),
                {"canonical_restore_used_without_graph_warmup": True},
            )
            return cursor
        checkpoint_sha = sha256_file(state.path)
        protected = {
            path: sha256_file(path)
            for path in (state.path, state.pointer, state.receipt, state.journal)
        }
        parameter_ids = {name: id(value) for name, value in model.named_parameters()}
        state_keys = tuple(model.state_dict())
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        scheduler_before = copy.deepcopy(scheduler.state_dict())
        generator_before = generator.get_state().clone()
        cursor_before = copy.deepcopy(cursor)
        python_before, numpy_before = random.getstate(), np.random.get_state()
        cpu_before = torch.get_rng_state().clone()
        cuda_before = [value.clone() for value in torch.cuda.get_rng_state_all()]

        original_forward = model.forward
        compiled = torch.compile(original_forward, backend=BACKEND, dynamic=False, fullgraph=False)
        self.dispatcher = ForwardDispatcher(original_forward, compiled)
        model.forward = self.dispatcher  # type: ignore[method-assign]
        order = cast(Tensor, cursor["order"])
        position = int(cursor["position"])
        ids = order[position : position + 32].tolist()
        if len(ids) != 32:
            ids = order[:32].tolist()
        batch = training._to_device(self.source.batch(ids, "event"), torch.device("cuda"))
        training._validate_batch(batch, self.recipe)
        loss_config = training.CausalScaleTTCLossConfig(**self.recipe.loss_config)
        if int(cursor["epoch"]) <= 3:
            from e_jepa_ttc.training.causal_scale_eap import _foreground_only_loss_config

            loss_config = _foreground_only_loss_config(loss_config)
        from e_jepa_ttc.training.causal_scale_eap import _autocast

        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with _autocast(torch.device("cuda"), "bf16"):
                total, _ = training._producer_loss(
                    cast(training.CausalScaleTTC, model), batch, loss_config
                )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            self.counters["warmup_backward_passes"] += 1
        torch.cuda.synchronize()

        restored = original_restore(state, model, optimizer, scheduler, generator)
        optimizer.zero_grad(set_to_none=True)
        exact = {
            "parameter_identities": parameter_ids
            == {name: id(value) for name, value in model.named_parameters()},
            "state_keys": state_keys == tuple(model.state_dict()),
            "optimizer": _same(optimizer_before, optimizer.state_dict()),
            "scheduler": _same(scheduler_before, scheduler.state_dict()),
            "sampler": torch.equal(generator_before, generator.get_state()),
            "cursor": _same(cursor_before, restored),
            "python_rng": _same(python_before, random.getstate()),
            "numpy_rng": _same(numpy_before, np.random.get_state()),
            "cpu_rng": torch.equal(cpu_before, torch.get_rng_state()),
            "cuda_rng": all(
                torch.equal(a, b)
                for a, b in zip(cuda_before, torch.cuda.get_rng_state_all(), strict=True)
            ),
            "protected_files": protected == {path: sha256_file(path) for path in protected},
        }
        if not all(exact.values()):
            raise RuntimeError(f"Post-warm restore was not exact: {exact}")
        self._ready(state.completed, checkpoint_sha, exact)
        prewarm = {
            "schema": "rgb_port_acceleration_prewarm_qa_v1",
            "status": "PASSED",
            "fit_id": self.fit_id,
            "completed_updates": state.completed,
            "optimizer_updates": 0,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_identity_sha256": state.identity_sha256,
            "acceleration_freeze_sha256": self.freeze_sha256,
            "acceleration_freeze_file_sha256": self.freeze_file_sha256,
            "runtime_receipt_sha256": sha256_file(self.runtime_path),
            "post_warm_exact": exact,
            "counters": {**self.counters, **self.dispatcher.snapshot()},
        }
        if self.prewarm_path.is_file():
            existing = read_json_shared(self.prewarm_path)
            if (
                existing.get("schema") != prewarm["schema"]
                or existing.get("status") != "PASSED"
                or existing.get("fit_id") != self.fit_id
                or existing.get("acceleration_freeze_sha256") != self.freeze_sha256
                or existing.get("acceleration_freeze_file_sha256") != self.freeze_file_sha256
            ):
                raise RuntimeError("Existing immutable prewarm QA differs")
        else:
            atomic_write_json(self.prewarm_path, prewarm)
        return restored

    def after_save(self, state: Any) -> None:
        if self.graph_enabled and self.dispatcher is None:
            raise RuntimeError("Acceleration save occurred before a proven warm restore")
        checkpoint_sha = sha256_file(state.path)
        if not self.runtime_path.is_file():
            self._ready(
                state.completed,
                checkpoint_sha,
                {"fresh_metrics_only_checkpoint": True},
            )
        runtime = self._sync_runtime_to_checkpoint(state)
        runtime_snapshot, runtime_sha = self._archive_runtime(state, runtime)
        receipt = self._checkpoint_receipt(
            state,
            status="COMPLETE",
            runtime_snapshot=runtime_snapshot,
            runtime_sha=runtime_sha,
        )
        atomic_write_json(self.receipt_dir / f"checkpoint_{state.completed:06d}.json", receipt)
        self.pending_path.unlink()


@contextmanager
def installed(runtime: Runtime) -> Iterator[None]:
    original_restore = training.ProducerCheckpoint.restore
    original_save = training.ProducerCheckpoint.save
    original_loss = training._producer_loss

    def restore(state: Any, model: Any, optimizer: Any, scheduler: Any, generator: Any) -> Any:
        cursor = original_restore(state, model, optimizer, scheduler, generator)
        return runtime.after_restore(
            state, model, optimizer, scheduler, generator, cursor, original_restore
        )

    def save(state: Any, *args: Any, **kwargs: Any) -> None:
        runtime.before_save(state)
        original_save(state, *args, **kwargs)
        runtime.after_save(state)

    def loss(*args: Any, **kwargs: Any) -> tuple[Tensor, Mapping[str, Tensor]]:
        total, components = original_loss(*args, **kwargs)
        return total, LazyCPUComponents(components, runtime.counters)

    cast(Any, training.ProducerCheckpoint).restore = restore
    cast(Any, training.ProducerCheckpoint).save = save
    cast(Any, training)._producer_loss = loss
    try:
        yield
    finally:
        cast(Any, training.ProducerCheckpoint).restore = original_restore
        cast(Any, training.ProducerCheckpoint).save = original_save
        cast(Any, training)._producer_loss = original_loss


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceleration-freeze", type=Path, required=True)
    parser.add_argument("forward", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.forward or args.forward[0] != "--":
        parser.error("original train_producers arguments must follow --")
    forwarded = args.forward[1:]
    if (
        "--fit-id" not in forwarded
        or forwarded[forwarded.index("--fit-id") + 1] not in PRODUCER_IDS
    ):
        parser.error("acceleration is restricted to the four RGB-PORT producers")
    if "--device" not in forwarded or forwarded[forwarded.index("--device") + 1] != "cuda":
        parser.error("acceleration requires explicit --device cuda")
    freeze_path = args.acceleration_freeze.resolve(strict=True)
    freeze = verify_acceleration_freeze(freeze_path)
    original_fit = training.fit_producer

    def fit(source: Any, recipe: Any, run: Path, **kwargs: Any) -> dict[str, Any]:
        runtime = Runtime(source, recipe, run, freeze_path, freeze)
        with installed(runtime):
            return original_fit(source, recipe, run, **kwargs)

    training.fit_producer = fit
    try:
        return training.main(forwarded)
    finally:
        training.fit_producer = original_fit


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ForwardDispatcher",
    "LazyCPUComponents",
    "Runtime",
    "installed",
    "main",
    "verify_acceleration_freeze",
]
