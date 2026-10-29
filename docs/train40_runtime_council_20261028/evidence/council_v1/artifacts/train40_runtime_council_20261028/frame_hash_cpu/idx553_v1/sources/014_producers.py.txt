"""Durable fresh RGB producers for the prospective all-TRAIN40 final refit."""

# ruff: noqa: ANN401 -- adapters wrap the validated dynamic prepared-cache contract

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
import zipfile
from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import MethodType
from typing import Any, Literal, cast

import torch
from torch import Tensor

from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig
from e_jepa_ttc.training.causal_scale_eap import _autocast, _foreground_only_loss_config
from operational.rgb_port.accounting import atomic_write_json, read_json_shared
from operational.rgb_port.recipe import ProducerRecipe, file_sha256, resolved_recipe
from operational.rgb_port.train_producers import (
    ProducerCheckpoint,
    _epoch_order,
    _producer_loss,
    _validate_batch,
    build_producer,
)
from operational.rgb_port_revision.loss import EffectiveBatch

from .producer_plan import SCHEMA as PRODUCER_PLAN_SCHEMA
from .producer_plan import load_plan, runtime_source_paths
from .recipe import canonical_sha256, load_recipe, stage
from .resource_monitor import ResourceMonitor, assert_recovery_budget
from .rgb import AllTrainRGBSource, load_all_train_source
from .runtime_migration import (
    resolve_runtime_checkpoint_identity,
    runtime_checkpoint_binding_kwargs,
)

RGBFit = Literal["R_A5", "R_C2F"]
RGB_FITS: tuple[RGBFit, ...] = ("R_A5", "R_C2F")


def configure_rgb_runtime() -> dict[str, Any]:
    """Apply the numerically admitted process-local CPU threading policy."""
    torch.set_num_threads(1)
    return {
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "resource_guard": "cached_command_inventory_fresh_pid_birth_and_rss",
        "checkpoint_interval_updates": 100,
    }


def _runtime_optimization_options(run_root: Path, plan_path: Path) -> dict[str, bool]:
    """Load only source-bound, separately admitted execution optimizations."""
    plan = read_json_shared(plan_path)
    digest = plan.get("input_sha256", {}).get("runtime_optimization_admission")
    if digest is None:
        return {}
    path = run_root / "RUNTIME_OPTIMIZATION_ADMISSION.json"
    if not path.is_file() or file_sha256(path) != digest:
        raise ValueError("Runtime optimization admission is missing or changed")
    value = read_json_shared(path)
    unsigned = dict(value)
    claimed = unsigned.pop("admission_sha256", None)
    if (
        claimed != canonical_sha256(unsigned)
        or value.get("schema") != "final_refit_rgb_optimization_admission_v1"
        or value.get("status") != "PASS"
        or value.get("scientific_optimizer_updates") != 0
        or value.get("checks")
        != {
            "numerical_parity": "PASS",
            "state_parity": "PASS",
            "resume_parity": "PASS",
            "rng_parity": "PASS",
        }
    ):
        raise ValueError("Runtime optimization admission is not a verified PASS")
    options = value.get("admitted_options")
    names = {
        "shared_resource_snapshot",
        "host_loss_metadata",
        "transport_reuse",
        "scalar_transfer",
        "merge_parts_by_t",
        "cudagraph_local_transport",
    }
    if (
        not isinstance(options, dict)
        or set(options) != names
        or any(type(flag) is not bool for flag in options.values())
    ):
        raise ValueError("Runtime optimization option contract differs")
    if options["merge_parts_by_t"] or options["cudagraph_local_transport"]:
        raise ValueError("Experimental batching/graphs are not executable producer options")
    closure = value.get("source_sha256_before_after", {})
    for name, digest_after in plan["executed_source_sha256"].items():
        if closure.get(name, {}).get("after_sha256") != digest_after:
            raise ValueError(f"Runtime optimization source closure differs: {name}")
    return cast(dict[str, bool], options)


def _requires_runtime_migration(run_root: Path, plan_path: Path, fit_id: str) -> bool:
    """Fail closed if a declared continuation lacks its per-fit migration proof."""
    plan = read_json_shared(plan_path)
    if "runtime_optimization_admission" in plan.get("input_sha256", {}):
        from .runtime_migration import validate_runtime_batch_migration

        batch = validate_runtime_batch_migration(
            run_root / "RUNTIME_MIGRATION_BATCH.json",
            new_plan_path=plan_path,
            optimization_admission_path=run_root / "RUNTIME_OPTIMIZATION_ADMISSION.json",
        )
        if fit_id not in batch["fits"]:
            raise ValueError("Producer is absent from the complete runtime migration")
        return True
    digest = plan.get("input_sha256", {}).get("runtime_migration_admission")
    present = (run_root / "fits" / fit_id / "RUNTIME_MIGRATION.json").is_file()
    if digest is None:
        if present:
            raise ValueError("Runtime migration is not declared by the producer plan")
        return False
    admission_path = run_root / "RUNTIME_MIGRATION_ADMISSION.json"
    if not admission_path.is_file() or file_sha256(admission_path) != digest:
        raise ValueError("Declared runtime migration admission is missing or changed")
    admission = read_json_shared(admission_path)
    if (
        admission.get("schema") != "final_refit_runtime_migration_admission_v1"
        or admission.get("status") != "ADMITTED_TECHNICAL_RUNTIME_MIGRATION"
        or admission.get("fit_id") not in RGB_FITS
    ):
        raise ValueError("Invalid runtime migration admission for this producer plan")
    required = admission["fit_id"] == fit_id
    if required != present:
        raise ValueError("Producer continuation requires its exact migration receipt")
    return required


def _final_memory_allowed(allowed: bool, values: Mapping[str, Any]) -> bool:
    """Apply the tighter final-refit physical/commit reserve to historical guards."""
    commit = values.get("windows_commit_headroom_bytes")
    return bool(
        allowed
        and int(values.get("host_available_bytes", 0)) >= 2 * 1024**3
        and (commit is None or int(commit) >= 2 * 1024**3)
    )


class ParallelPreparedCache:
    """Bounded parallel reads that retain PreparedRGBCache's full validation path."""

    def __init__(
        self,
        cache: Any,
        dataset: Any,
        pool: ThreadPoolExecutor,
        *,
        max_pending_bytes: int,
    ) -> None:
        self.cache, self.dataset, self.pool = cache, dataset, pool
        self.max_pending_bytes = max_pending_bytes
        self.pending: dict[str, tuple[Future[dict[str, Any]], int]] = {}
        self.reservations: dict[str, int] = {}
        self.pending_bytes = self.peak_pending_bytes = 0
        self.reads = self.hits = self.fallback_reads = 0
        self.wait_seconds = self.decode_seconds = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.cache, name)

    def _reservation(self, key: str) -> int:
        if key not in self.reservations:
            with zipfile.ZipFile(key) as archive:
                raw = sum(member.file_size for member in archive.infolist())
            self.reservations[key] = 4 * raw + 2 * 1024**2
        return self.reservations[key]

    def _read(self, index: int, token: str, key: str) -> dict[str, Any]:
        started = time.perf_counter()
        shell = object.__new__(type(self.cache))
        shell.__dict__.update(vars(self.cache))
        shell._loaded = OrderedDict()
        shell._loaded_bytes = 0
        try:
            shell.get(index, token)
            self.reads += 1
            return cast(dict[str, Any], shell._loaded[key])
        finally:
            self.decode_seconds += time.perf_counter() - started

    def plan(self, indices: Sequence[int]) -> None:
        if self.pending:
            raise RuntimeError("Previous parallel RGB cache plan was not drained")
        seen: set[str] = set()
        for index in indices:
            item = self.cache._lookup.get(int(index))
            if item is None:
                continue
            key = str(item[0]["path"])
            if key in seen or key in self.cache._loaded:
                continue
            seen.add(key)
            reserve = self._reservation(key)
            if (
                reserve > self.max_pending_bytes
                or self.pending_bytes + reserve > self.max_pending_bytes
            ):
                continue
            token = str(self.dataset.rows[int(index)]["sample_token"])
            self.pending[key] = (self.pool.submit(self._read, int(index), token, key), reserve)
            self.pending_bytes += reserve
            self.peak_pending_bytes = max(self.peak_pending_bytes, self.pending_bytes)

    def get(self, index: int, token: str) -> dict[str, Any]:
        item = self.cache._lookup.get(index)
        if item is None:
            return cast(dict[str, Any], self.cache.get(index, token))
        key = str(item[0]["path"])
        if key in self.cache._loaded:
            self.hits += 1
            return cast(dict[str, Any], self.cache.get(index, token))
        future = self.pending.pop(key, None)
        if future is None:
            self.fallback_reads += 1
            return cast(dict[str, Any], self.cache.get(index, token))
        task, reserve = future
        started = time.perf_counter()
        try:
            stored = task.result()
        finally:
            self.wait_seconds += time.perf_counter() - started
            self.pending_bytes -= reserve
        self.cache._loaded[key] = stored
        self.cache._loaded_bytes += sum(value.nbytes for value in stored.values())
        while (
            self.cache._loaded_bytes > self.cache._max_loaded_bytes and len(self.cache._loaded) > 1
        ):
            _, stale = self.cache._loaded.popitem(last=False)
            self.cache._loaded_bytes -= sum(value.nbytes for value in stale.values())
        return cast(dict[str, Any], self.cache.get(index, token))

    def drain(self) -> None:
        for future, _ in self.pending.values():
            if not future.cancel():
                try:
                    future.result()
                except Exception:
                    pass
        self.pending.clear()
        self.pending_bytes = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "parallel_reads": self.reads,
            "hits": self.hits,
            "fallback_reads": self.fallback_reads,
            "peak_pending_bytes": self.peak_pending_bytes,
            "consumer_wait_seconds": self.wait_seconds,
            "decode_worker_seconds": self.decode_seconds,
        }


class ParallelRGBDecoder:
    """Install four bounded readers on source instances; never patch global classes."""

    def __init__(
        self,
        source: AllTrainRGBSource,
        *,
        workers: int = 4,
        max_pending_bytes: int = 128 * 1024**2,
    ) -> None:
        if workers < 1 or max_pending_bytes < 1:
            raise ValueError("Positive RGB decoder bounds are required")
        self.source = source
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="final-rgb-decode")
        self.original_batch = source.batch
        self.original_caches = {
            role: dataset.prepared_cache for role, dataset in source.datasets.items()
        }
        if any(cache is None for cache in self.original_caches.values()):
            raise ValueError("All P/H/V lossless RGB caches are required for final training")
        self.decoders = {
            role: ParallelPreparedCache(
                cache,
                source.datasets[role],
                self.pool,
                max_pending_bytes=max_pending_bytes,
            )
            for role, cache in self.original_caches.items()
        }
        for role, decoder in self.decoders.items():
            source.datasets[role].prepared_cache = cast(Any, decoder)

        def batch(_source: Any, indices: Sequence[int], modality: str) -> Any:
            rows = [source.row(int(index)) for index in indices]
            roles = {str(row["historical_role"]) for row in rows}
            if len(roles) != 1:
                raise ValueError("Parallel RGB physical batch must preserve one historical role")
            role = next(iter(roles))
            local = [int(row["historical_role_ordinal"]) for row in rows]
            decoder = self.decoders[role]
            decoder.plan(local)
            try:
                return self.original_batch(indices, modality)
            finally:
                decoder.drain()

        source.batch = MethodType(batch, source)  # type: ignore[method-assign]

    def snapshot(self) -> dict[str, Any]:
        return {role: decoder.snapshot() for role, decoder in self.decoders.items()}

    def close(self) -> None:
        vars(self.source).pop("batch", None)
        for role, cache in self.original_caches.items():
            self.source.datasets[role].prepared_cache = cache
        for decoder in self.decoders.values():
            decoder.drain()
        self.pool.shutdown(wait=True, cancel_futures=True)


def build_rgb_recipe(
    config_path: Path,
    fit_id: str,
    source_descriptor: Path,
    source: AllTrainRGBSource,
    *,
    microbatch_size: int = 32,
) -> ProducerRecipe:
    """Resolve the validated V13 recipe with a prospective explicit TRAIN teacher scope."""
    if fit_id not in RGB_FITS:
        raise ValueError(f"Final RGB producer must be one of {RGB_FITS}")
    descriptor_sha = file_sha256(source_descriptor.resolve(strict=True))
    recipe = resolved_recipe(
        config_path.resolve(strict=True),
        fit_id=fit_id,
        producer_population=source.population_size,
        role_manifest_sha256=descriptor_sha,
        microbatch_size=microbatch_size,
    )
    teacher = {
        **recipe.teacher_policy,
        "roles": ["TRAIN"],
        "native_roles_preserved": ["P", "H", "V"],
        "lookup": "native_source_ordinal_plus_sample_token",
        "teacher_manifest_sha256": source.teacher.manifest_sha256,
        "teacher_rows_sha256": source.teacher.rows_sha256,
    }
    training = {
        **recipe.training_config,
        "runtime_input_route": "lossless_cache_bounded_prefetch_v1",
        "cuda_graphs": "DISABLED_UNADMITTED_VARIABLE_ROLE_T_SIGNATURES",
        "deterministic_transport_sampling": "ENABLED_FINAL_REFIT_PROSPECTIVE",
    }
    model = {
        **recipe.model_config,
        "deterministic_transport_sampling": True,
    }
    return replace(
        recipe,
        model_config=model,
        teacher_policy=teacher,
        training_config=training,
    )


def make_effective_batch(
    source: AllTrainRGBSource,
    indices: Sequence[int],
    recipe: ProducerRecipe,
) -> EffectiveBatch:
    """Preserve one B32 boundary while separating role/T-specific physical batches."""
    if not indices:
        raise ValueError("An effective RGB batch cannot be empty")
    groups: dict[tuple[str, int], list[int]] = {}
    for index in indices:
        groups.setdefault(source.batch_keys[int(index)], []).append(int(index))
    parts = [
        source.batch(group[start : start + recipe.microbatch_size], "rgb")
        for key, group in sorted(groups.items())
        for start in range(0, len(group), recipe.microbatch_size)
    ]
    if not parts:
        raise RuntimeError("RGB effective batch partition produced no parts")
    return EffectiveBatch(parts)


def producer_objective(
    model: torch.nn.Module,
    batch: EffectiveBatch,
    recipe: ProducerRecipe,
    epoch: int,
) -> tuple[Tensor, Mapping[str, Tensor]]:
    """Evaluate the exact historical objective and its three-epoch foreground warmup."""
    config = CausalScaleTTCLossConfig(**recipe.loss_config)
    selected = _foreground_only_loss_config(config) if epoch <= 3 else config
    total, components = _producer_loss(cast(Any, model), batch, selected)
    return total, components


class BoundedRGBPrefetch:
    """One CPU owner prepares at most two canonical effective batches ahead."""

    def __init__(
        self,
        source: AllTrainRGBSource,
        recipe: ProducerRecipe,
        cursor: Mapping[str, Any],
        *,
        depth: int = 2,
        prepare_host_metadata: bool = False,
    ) -> None:
        if depth not in {1, 2}:
            raise ValueError("RGB prefetch depth must be one or two")
        self.source, self.recipe, self.cursor, self.depth = source, recipe, cursor, depth
        self.prepare_host_metadata = prepare_host_metadata
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="final-rgb-prefetch")
        self.queue: deque[tuple[int, tuple[int, ...], Future[EffectiveBatch]]] = deque()
        self.order: Any = None
        self.next_position = 0
        self.submitted = self.consumed = self.prepared = 0
        self.wait_seconds = self.prepare_seconds = 0.0
        self.closed = False
        self._reset()

    def _reset(self) -> None:
        if self.queue:
            raise RuntimeError("Cannot reset RGB prefetch with outstanding batches")
        self.order = self.cursor["order"]
        self.next_position = int(self.cursor["position"])
        if self.order.ndim != 1 or len(self.order) != self.source.population_size:
            raise ValueError("RGB prefetch received an invalid canonical order")
        self._fill()

    def _prepare(self, ids: tuple[int, ...]) -> EffectiveBatch:
        started = time.perf_counter()
        try:
            batch = make_effective_batch(self.source, ids, self.recipe)
            if self.prepare_host_metadata:
                from .host_loss import prepare_effective_batch

                batch = prepare_effective_batch(batch, self.recipe)
            return batch
        finally:
            self.prepare_seconds += time.perf_counter() - started
            self.prepared += 1

    def _fill(self) -> None:
        while len(self.queue) < self.depth and self.next_position < self.source.population_size:
            start = self.next_position
            stop = min(start + self.recipe.effective_batch_size, self.source.population_size)
            ids = tuple(map(int, self.order[start:stop].tolist()))
            self.queue.append((start, ids, self.pool.submit(self._prepare, ids)))
            self.next_position = stop
            self.submitted += 1

    def get(self, indices: Sequence[int]) -> EffectiveBatch:
        if self.closed:
            raise RuntimeError("RGB prefetch is closed")
        if self.cursor["order"] is not self.order:
            for _, _, future in self.queue:
                future.cancel()
            self.queue.clear()
            self._reset()
        if not self.queue:
            raise RuntimeError("RGB prefetch has no canonical batch")
        start, expected, future = self.queue.popleft()
        if int(self.cursor["position"]) != start or tuple(indices) != expected:
            raise RuntimeError("RGB prefetch order differs from the saved sampler cursor")
        started = time.perf_counter()
        try:
            result = future.result()
        finally:
            self.wait_seconds += time.perf_counter() - started
        self.consumed += 1
        self._fill()
        return result

    def snapshot(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "host_metadata_prepared": self.prepare_host_metadata,
            "worker_threads": 1,
            "submitted": self.submitted,
            "consumed": self.consumed,
            "prepared": self.prepared,
            "outstanding": len(self.queue),
            "source_wait_seconds": self.wait_seconds,
            "source_prepare_seconds": self.prepare_seconds,
        }

    def close(self) -> None:
        if not self.closed:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.queue.clear()
            self.closed = True


def _identity(
    recipe: ProducerRecipe,
    source: AllTrainRGBSource,
    source_descriptor: Path,
    recipe_path: Path,
    contract_schema: str,
    device: str,
) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    executed = {
        **runtime_source_paths(repository),
        "configs/rgb_port/producers.json": repository / "configs/rgb_port/producers.json",
    }
    value: dict[str, Any] = {
        **recipe.identity,
        "schema": "final_refit_rgb_producer_identity_v1",
        "recipe_identity_sha256": recipe.identity_sha256,
        "producer_contract_path": str(recipe_path.resolve(strict=True)),
        "producer_contract_sha256": file_sha256(recipe_path.resolve(strict=True)),
        "producer_contract_schema": contract_schema,
        "source_descriptor_path": str(source_descriptor.resolve(strict=True)),
        "source_descriptor_sha256": file_sha256(source_descriptor.resolve(strict=True)),
        "source_identity": dict(source.identity),
        "population_role": "TRAIN",
        "historical_roles_preserved": ["P", "H", "V"],
        "initialization": "fresh_seed7_no_event_or_historical_rgb_weights",
        "runtime": {
            "precision": "bf16",
            "effective_batch_size": 32,
            "prefetch_depth": 2,
            "teacher_lru_bytes": source.teacher.max_bytes,
            "cuda_graphs": "disabled_unadmitted_variable_role_T_signatures",
        },
        "environment": {
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
        },
        "device": device,
        "executed_source_sha256": {name: file_sha256(path) for name, path in executed.items()},
    }
    value["identity_sha256"] = canonical_sha256(value)
    return value


def _validate_producer_contract(
    contract_path: Path,
    fit_id: str,
    producer: ProducerRecipe,
    *,
    source_descriptor: Path,
    teacher_manifest: Path,
    config_path: Path,
) -> str:
    """Validate a full-chain recipe or the standalone two-producer contract."""
    raw = json.loads(contract_path.read_text(encoding="utf-8"))
    schema = str(raw.get("schema"))
    if schema == PRODUCER_PLAN_SCHEMA:
        plan = load_plan(contract_path)
        if not plan.launch_authorized or plan.status != "FROZEN_AUTHORIZED_PRODUCER_SCOPE":
            raise ValueError("Producer plan has not passed runtime admission for scientific launch")
        matches = [item for item in plan.stages if item.fit_id == fit_id]
        if len(matches) != 1:
            raise ValueError(f"Producer plan contains no unique {fit_id} stage")
        selected = matches[0]
        if (
            selected.modality != "rgb"
            or selected.base_arm != producer.base_arm.upper()
            or selected.seed != producer.seed
            or selected.epochs != producer.epochs
            or selected.population != producer.producer_population
            or selected.effective_batch_size != producer.effective_batch_size
            or selected.microbatch_size != producer.microbatch_size
            or selected.updates != producer.updates
            or not selected.fresh_initialization
            or selected.parent_checkpoints
        ):
            raise ValueError(f"Producer-only plan differs from executable {fit_id} recipe")
        expected_inputs = {
            "source_descriptor": file_sha256(source_descriptor.resolve(strict=True)),
            "teacher_manifest": file_sha256(teacher_manifest.resolve(strict=True)),
            "producer_config": file_sha256(config_path.resolve(strict=True)),
            "train40_rows": str(producer.teacher_policy["teacher_rows_sha256"]),
        }
        mismatched_inputs = {
            name: digest
            for name, digest in expected_inputs.items()
            if plan.input_sha256.get(name) != digest
        }
        descriptor = json.loads(source_descriptor.read_text(encoding="utf-8"))
        bindings = descriptor.get("cache_bindings")
        if not isinstance(bindings, dict) or set(bindings) != {"P", "H", "V"}:
            raise ValueError("Scientific producer launch requires P/H/V cache bindings")
        for role, reference in sorted(bindings.items()):
            cache_path = (source_descriptor.resolve().parent / str(reference)).resolve(strict=True)
            name = f"rgb_cache_binding_{role}"
            digest = file_sha256(cache_path)
            if plan.input_sha256.get(name) != digest:
                mismatched_inputs[name] = digest
        admission_digest = str(plan.input_sha256.get("runtime_admission", ""))
        if len(admission_digest) != 64 or any(
            character not in "0123456789abcdef" for character in admission_digest
        ):
            mismatched_inputs["runtime_admission"] = "required SHA-256"
        expected_input_names = {
            *expected_inputs,
            "rgb_cache_binding_P",
            "rgb_cache_binding_H",
            "rgb_cache_binding_V",
            "runtime_admission",
        }
        if "runtime_optimization_admission" in plan.input_sha256:
            expected_input_names.add("runtime_optimization_admission")
            digest = plan.input_sha256["runtime_optimization_admission"]
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                mismatched_inputs["runtime_optimization_admission"] = "required SHA-256"
        if "runtime_migration_admission" in plan.input_sha256:
            expected_input_names.add("runtime_migration_admission")
            digest = plan.input_sha256["runtime_migration_admission"]
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                mismatched_inputs["runtime_migration_admission"] = "required SHA-256"
        if set(plan.input_sha256) != expected_input_names:
            mismatched_inputs["input_names"] = str(sorted(expected_input_names))
        if mismatched_inputs:
            raise ValueError(
                f"Producer plan input hashes differ from execution inputs: {mismatched_inputs}"
            )
        frozen_sources = plan.executed_source_sha256
    else:
        final = load_recipe(contract_path)
        selected = stage(final, fit_id)
        if (
            selected.kind != "rgb_producer"
            or selected.updates != producer.updates
            or selected.batch_size != producer.effective_batch_size
            or selected.weight_decay != float(producer.training_config["weight_decay"])
            or selected.depends_on
            or not selected.fresh_initialization
        ):
            raise ValueError(f"Frozen final-refit stage differs from executable {fit_id} recipe")
        frozen_sources = final.source_sha256
    repository = Path(__file__).resolve().parents[2]
    required = {
        "operational/final_refit/producer_plan.py": file_sha256(
            repository / "operational/final_refit/producer_plan.py"
        ),
        **{name: file_sha256(path) for name, path in runtime_source_paths(repository).items()},
    }
    missing = {
        name: digest for name, digest in required.items() if frozen_sources.get(name) != digest
    }
    if missing:
        raise ValueError(
            f"FINAL_REFIT_RECIPE does not bind every executed RGB producer source: {missing}"
        )
    return schema


def _git_commit(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if len(commit) != 40:
        raise ValueError("Git HEAD is not a full commit identity")
    return commit


def run_rgb_producer(
    *,
    fit_id: RGBFit,
    all_train_manifest: Path,
    teacher_manifest: Path,
    event_source_root: Path | None = None,
    run_root: Path,
    device: str = "cuda",
    max_updates_this_call: int | None = None,
    microbatch_size: int = 32,
    config_path: Path = Path("configs/rgb_port/producers.json"),
    recipe_path: Path | None = None,
) -> dict[str, Any]:
    """Run or resume one fresh RGB producer to the fixed 18-epoch TRAIN40 endpoint.

    ``event_source_root`` is accepted for the shared final-refit engine contract;
    RGB never reads it.
    """
    del event_source_root
    if fit_id not in RGB_FITS:
        raise ValueError(f"Unsupported final RGB fit: {fit_id}")
    run_root = run_root.resolve()
    recipe_path = (recipe_path or run_root / "FINAL_REFIT_RECIPE.json").resolve(strict=True)
    runtime_settings = configure_rgb_runtime()
    optimization_options = _runtime_optimization_options(run_root, recipe_path)
    runtime_settings["optimizations"] = optimization_options
    requires_migration = _requires_runtime_migration(run_root, recipe_path, fit_id)
    source = load_all_train_source(all_train_manifest, teacher_manifest)
    try:
        recipe = build_rgb_recipe(
            config_path,
            fit_id,
            all_train_manifest,
            source,
            microbatch_size=microbatch_size,
        )
        contract_schema = _validate_producer_contract(
            recipe_path,
            fit_id,
            recipe,
            source_descriptor=all_train_manifest,
            teacher_manifest=teacher_manifest,
            config_path=config_path,
        )
        target = torch.device(device)
        if target.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA RGB producer requested but unavailable")
        resource_monitor: Any
        if optimization_options.get("shared_resource_snapshot", False):
            from .shared_resources import SharedResourceMonitor

            resource_monitor = SharedResourceMonitor(
                Path(__file__).resolve().parents[2],
                session_path=run_root / "RESOURCE_SESSION.json",
                snapshot_path=run_root / "RESOURCE_SNAPSHOT.json",
                run_root=run_root,
            )
            runtime_settings["resource_guard"] = "shared_inventory_fresh_local_guards_fail_closed"
        else:
            resource_monitor = ResourceMonitor(Path(__file__).resolve().parents[2])
        allowed, resources = resource_monitor.guard(
            run_root, checkpoint_reservation=1 * 1024**3, completed_updates=0
        )
        allowed = _final_memory_allowed(allowed, resources)
        if not allowed:
            raise RuntimeError(f"Final RGB producer resource preflight denied: {resources}")
        model = build_producer(recipe).to(target)
        model.runtime_reuse_reverse_sample = optimization_options.get("transport_reuse", False)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(recipe.training_config["learning_rate"]),
            weight_decay=float(recipe.training_config["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=recipe.epochs,
            eta_min=float(recipe.training_config["minimum_learning_rate"]),
        )
        generator = torch.Generator().manual_seed(recipe.seed)
        fit_dir = run_root / "fits" / fit_id
        identity = _identity(
            recipe, source, all_train_manifest, recipe_path, contract_schema, device
        )
        execution_identity = identity
        binding_kwargs: dict[str, str] = {}
        if requires_migration:
            if optimization_options:
                from .runtime_migration import (
                    resolve_runtime_checkpoint_identity_v3,
                    runtime_checkpoint_binding_kwargs_v3,
                )

                identity = resolve_runtime_checkpoint_identity_v3(
                    run_root=run_root,
                    fit_id=fit_id,
                    new_identity=execution_identity,
                    new_plan_path=recipe_path,
                )
                binding_kwargs = runtime_checkpoint_binding_kwargs_v3(
                    run_root=run_root,
                    fit_id=fit_id,
                )
            else:
                identity = resolve_runtime_checkpoint_identity(
                    run_root=run_root,
                    fit_id=fit_id,
                    new_identity=execution_identity,
                    new_plan_path=recipe_path,
                )
                binding_kwargs = runtime_checkpoint_binding_kwargs(run_root=run_root, fit_id=fit_id)
        state = ProducerCheckpoint(fit_dir, identity, recipe.updates, **binding_kwargs)
        cursor: dict[str, Any] = {
            "epoch": 1,
            "position": 0,
            "order": _epoch_order(source, generator),
            "loss_sums": {},
            "examples": 0,
            "pending_curve": [],
        }
        if state.pointer.is_file():
            cursor = state.restore(model, optimizer, scheduler, generator)
        else:
            state.save(model, optimizer, scheduler, generator, cursor, status="READY")
        provenance_path = fit_dir / "RUN_PROVENANCE.json"
        repository = Path(__file__).resolve().parents[2]
        if not provenance_path.exists():
            atomic_write_json(
                provenance_path,
                {
                    "schema": "final_refit_rgb_run_provenance_v1",
                    "experiment_id": "FINAL_REFIT_TRAIN40",
                    "run_id": fit_id,
                    "fit_id": fit_id,
                    "git_commit": _git_commit(repository),
                    "config_hash": recipe.identity_sha256,
                    "dataset_manifest_hash": file_sha256(all_train_manifest),
                    "split_version": source.teacher.rows_sha256,
                    "identity_sha256": state.identity_sha256,
                    "execution_identity": execution_identity,
                    "execution_binding_sha256": binding_kwargs.get("execution_binding_sha256"),
                    "runtime_settings": runtime_settings,
                    "source_identity": dict(source.identity),
                    "precision": recipe.training_config["precision"],
                    "seed": recipe.seed,
                    "host": platform.node(),
                    "python_version": platform.python_version(),
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "gpu_name": (
                        torch.cuda.get_device_name(target) if target.type == "cuda" else None
                    ),
                    "device": device,
                    "start_time": datetime.now(UTC).isoformat(),
                    "end_time": None,
                    "status": "RUNNING",
                    "checkpoint_path": None,
                    "metrics_path": None,
                },
            )
        model.train()
        call_updates = 0
        last_losses: dict[str, float] = {}
        checkpoint_reservation = max(1 * 1024**3, 2 * state.path.stat().st_size)
        decoder = ParallelRGBDecoder(source, workers=4, max_pending_bytes=128 * 1024**2)
        prefetch = BoundedRGBPrefetch(
            source,
            recipe,
            cursor,
            depth=2,
            prepare_host_metadata=optimization_options.get("host_loss_metadata", False),
        )
        try:
            while int(cursor["epoch"]) <= recipe.epochs:
                allowed, resources = resource_monitor.guard(
                    run_root,
                    checkpoint_reservation=checkpoint_reservation,
                    completed_updates=state.completed,
                )
                allowed = _final_memory_allowed(allowed, resources)
                if not allowed or (fit_dir / "STOP_REQUEST").is_file():
                    state.save(
                        model, optimizer, scheduler, generator, cursor, status="PAUSED_RESOURCE"
                    )
                    atomic_write_json(fit_dir / "RESOURCE_PAUSE.json", resources)
                    break
                epoch = int(cursor["epoch"])
                start = int(cursor["position"])
                stop = min(start + recipe.effective_batch_size, source.population_size)
                indices = cast(Tensor, cursor["order"])[start:stop].tolist()
                optimizer.zero_grad(set_to_none=True)
                try:
                    host_batch = prefetch.get(indices)
                except OSError as error:
                    state.save(
                        model, optimizer, scheduler, generator, cursor, status="PAUSED_RESOURCE"
                    )
                    atomic_write_json(
                        fit_dir / "RESOURCE_PAUSE.json",
                        {"kind": "transient_input_io", "message": str(error)},
                    )
                    break
                batch = host_batch.to(target)
                if not optimization_options.get("host_loss_metadata", False):
                    for part in batch.parts:
                        _validate_batch(part, recipe)
                with _autocast(target, str(recipe.training_config["precision"])):
                    total, components = producer_objective(model, batch, recipe, epoch)
                if not bool(torch.isfinite(total)):
                    raise FloatingPointError("Nonfinite final RGB producer loss")
                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(recipe.training_config["grad_clip_norm"]),
                    error_if_nonfinite=True,
                )
                assert_recovery_budget(run_root, prospective_pending=1)
                state.begin()
                optimizer.step()
                state.commit()
                call_updates += 1
                if optimization_options.get("scalar_transfer", False):
                    from .host_loss import collect_loss_scalars

                    last_losses = collect_loss_scalars(total, components)
                else:
                    last_losses = {
                        **{
                            name: float(value.detach().float().cpu())
                            for name, value in components.items()
                        },
                        "total": float(total.detach().float().cpu()),
                    }
                for name, value in last_losses.items():
                    cursor["loss_sums"][name] = cursor["loss_sums"].get(name, 0.0) + value * len(
                        indices
                    )
                cursor["examples"] += len(indices)
                cursor["position"] = stop
                cursor["pending_curve"].append(
                    {
                        "update": state.completed,
                        "epoch": epoch,
                        "examples": len(indices),
                        "losses": last_losses,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )
                if stop == source.population_size:
                    atomic_write_json(
                        fit_dir / f"epoch_{epoch:02d}.json",
                        {
                            "epoch": epoch,
                            "completed_updates": state.completed,
                            "examples": cursor["examples"],
                            "losses": {
                                name: value / cursor["examples"]
                                for name, value in cursor["loss_sums"].items()
                            },
                        },
                    )
                    scheduler.step()
                    cursor.update(
                        epoch=epoch + 1,
                        position=0,
                        order=_epoch_order(source, generator),
                        loss_sums={},
                        examples=0,
                    )
                complete = int(cursor["epoch"]) > recipe.epochs
                requested = (
                    max_updates_this_call is not None and call_updates >= max_updates_this_call
                )
                if state.completed % 100 == 0 or complete or requested:
                    if cursor["pending_curve"]:
                        atomic_write_json(
                            fit_dir / f"curve_{state.completed:06d}.json",
                            {"updates": cursor["pending_curve"]},
                        )
                        cursor["pending_curve"] = []
                    status = (
                        "COMPLETE" if complete else "PAUSED_REQUESTED" if requested else "RUNNING"
                    )
                    state.save(model, optimizer, scheduler, generator, cursor, status=status)
                    atomic_write_json(
                        fit_dir / "RGB_RUNTIME.json",
                        {
                            "schema": "final_refit_rgb_runtime_v1",
                            "status": status,
                            "fit_id": fit_id,
                            "completed_updates": state.completed,
                            "prefetch": prefetch.snapshot(),
                            "parallel_decoder": decoder.snapshot(),
                            "teacher_cache": source.teacher.snapshot(),
                            "resource_monitor": resource_monitor.statistics(),
                            "runtime_settings": runtime_settings,
                            "cuda_graphs": "DISABLED_UNADMITTED_VARIABLE_ROLE_T_SIGNATURES",
                        },
                    )
                    if complete or requested:
                        break
        finally:
            prefetch.close()
            decoder.close()
        receipt = read_json_shared(state.receipt)
        provenance = read_json_shared(provenance_path)
        provenance.update(
            end_time=datetime.now(UTC).isoformat(),
            status=receipt["status"],
            completed_updates=receipt["completed_updates"],
            checkpoint_path=str(state.path.resolve()),
            checkpoint_sha256=receipt["checkpoint_sha256"],
            metrics_path=str(fit_dir.resolve()),
            last_losses=last_losses,
        )
        atomic_write_json(provenance_path, provenance)
        return receipt
    finally:
        source.close()


def run_rgb_producers(
    *,
    fit_ids: Sequence[RGBFit] = RGB_FITS,
    all_train_manifest: Path,
    teacher_manifest: Path,
    run_root: Path,
    device: str = "cuda",
    max_updates_this_call: int | None = None,
    microbatch_size: int = 32,
    config_path: Path = Path("configs/rgb_port/producers.json"),
    recipe_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Run requested RGB stages sequentially so one GPU has one scientific owner."""
    if not fit_ids or any(fit_id not in RGB_FITS for fit_id in fit_ids):
        raise ValueError(f"RGB fit sequence must use {RGB_FITS}")
    return {
        fit_id: run_rgb_producer(
            fit_id=fit_id,
            all_train_manifest=all_train_manifest,
            teacher_manifest=teacher_manifest,
            run_root=run_root,
            device=device,
            max_updates_this_call=max_updates_this_call,
            microbatch_size=microbatch_size,
            config_path=config_path,
            recipe_path=recipe_path,
        )
        for fit_id in fit_ids
    }


def load_rgb_endpoint(
    fit_directory: Path,
    recipe: ProducerRecipe,
    *,
    device: str = "cpu",
) -> torch.nn.Module:
    """Load a complete final-refit RGB endpoint without historical P-only migration rules."""
    fit_directory = fit_directory.resolve(strict=True)
    checkpoint = fit_directory / "checkpoint_last.pt"
    receipt = read_json_shared(fit_directory / "CHECKPOINT_RECEIPT.json")
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("scientific_endpoint") is not True
        or int(receipt.get("completed_updates", -1)) != recipe.updates
        or receipt.get("checkpoint_sha256") != file_sha256(checkpoint)
    ):
        raise ValueError("Final RGB producer endpoint is incomplete or changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    identity = payload.get("identity", {})
    if (
        payload.get("schema") != ProducerCheckpoint.schema
        or identity.get("schema") != "final_refit_rgb_producer_identity_v1"
        or identity.get("recipe_identity_sha256") != recipe.identity_sha256
        or identity.get("population_role") != "TRAIN"
        or identity.get("historical_roles_preserved") != ["P", "H", "V"]
        or payload.get("identity_sha256") != receipt.get("identity_sha256")
    ):
        raise ValueError("Final RGB producer checkpoint identity differs")
    model = build_producer(recipe)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).float().eval().requires_grad_(False)


def benchmark_rgb_source(
    source: AllTrainRGBSource,
    recipe: ProducerRecipe,
    *,
    batches: int = 4,
) -> dict[str, Any]:
    """Bounded zero-update CPU benchmark using the exact effective-batch adapter."""
    if not 1 <= batches <= 32:
        raise ValueError("Benchmark batches must be in [1,32]")
    generator = torch.Generator().manual_seed(recipe.seed)
    order = _epoch_order(source, generator)
    durations: list[float] = []
    examples = 0
    signatures: dict[str, int] = {}
    decoder = ParallelRGBDecoder(source, workers=4, max_pending_bytes=128 * 1024**2)
    try:
        for batch_index in range(batches):
            start = batch_index * recipe.effective_batch_size
            ids = order[start : start + recipe.effective_batch_size].tolist()
            tick = time.perf_counter()
            effective = make_effective_batch(source, ids, recipe)
            durations.append(time.perf_counter() - tick)
            examples += len(ids)
            for part in effective.parts:
                signature = f"B{len(part.events)}T{part.events.shape[1]}"
                signatures[signature] = signatures.get(signature, 0) + 1
        decoder_snapshot = decoder.snapshot()
    finally:
        decoder.close()
    total = sum(durations)
    return {
        "schema": "final_refit_rgb_source_benchmark_v1",
        "status": "COMPLETE",
        "optimizer_updates": 0,
        "batches": batches,
        "examples": examples,
        "elapsed_seconds": total,
        "examples_per_second": examples / total,
        "batch_signatures": signatures,
        "teacher_cache": source.teacher.snapshot(),
        "parallel_decoder": decoder_snapshot,
        "cuda_graphs": "NOT_EXERCISED",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/rgb_port/producers.json"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--fit-id", choices=RGB_FITS, default="R_A5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--microbatch", type=int, default=32)
    parser.add_argument("--max-updates-this-call", type=int)
    parser.add_argument("--benchmark-batches", type=int)
    args = parser.parse_args(argv)
    if args.benchmark_batches is not None:
        source = load_all_train_source(args.source, args.teacher_manifest)
        try:
            recipe = build_rgb_recipe(
                args.config, args.fit_id, args.source, source, microbatch_size=args.microbatch
            )
            result = benchmark_rgb_source(source, recipe, batches=args.benchmark_batches)
        finally:
            source.close()
    else:
        result = run_rgb_producer(
            fit_id=args.fit_id,
            all_train_manifest=args.source,
            teacher_manifest=args.teacher_manifest,
            run_root=args.run,
            device=args.device,
            max_updates_this_call=args.max_updates_this_call,
            microbatch_size=args.microbatch,
            config_path=args.config,
            recipe_path=args.recipe,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "COMPLETE" else 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RGB_FITS",
    "BoundedRGBPrefetch",
    "benchmark_rgb_source",
    "build_rgb_recipe",
    "load_rgb_endpoint",
    "make_effective_batch",
    "producer_objective",
    "run_rgb_producer",
    "run_rgb_producers",
]
