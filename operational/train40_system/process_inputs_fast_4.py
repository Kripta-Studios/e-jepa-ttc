"""Spawned TRAIN40 decoder with bounded group cache and shared batch arenas."""

from __future__ import annotations

import multiprocessing as mp
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from operational.train40_system.contracts import read

if TYPE_CHECKING:
    from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch

GROUP_ROWS = 256
SHARD_ROWS = 32
SHARDS_PER_GROUP = GROUP_ROWS // SHARD_ROWS
CACHE_BUDGET_BYTES = 1_320_000_000
DEFAULT_SLOT_BYTES = 80 * 1024**2


def _value_bytes(value: dict[str, np.ndarray]) -> int:
    return sum(array.nbytes for array in value.values())


def _manifest_identities(output: Path) -> tuple[Path, dict[Path, tuple[int, int]]]:
    input_manifest = read(output / "INPUT_MANIFEST.json")
    teacher_manifest = read(output / "TEACHER_MANIFEST.json")
    input_directory = Path(input_manifest["directory"])
    identities: dict[Path, tuple[int, int]] = {}
    for manifest in (input_manifest, teacher_manifest):
        directory = Path(manifest["directory"])
        for entry in manifest["ordered_shards"]:
            identities[directory / entry["name"]] = (int(entry["bytes"]), int(entry["mtime_ns"]))
    return input_directory, identities


def _verify_path(path: Path, identities: dict[Path, tuple[int, int]]) -> None:
    expected = identities.get(path)
    if expected is None:
        raise ValueError(f"Shard is absent from sealed manifest: {path}")
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != expected:
        raise ValueError(f"Sealed TRAIN40 shard changed: {path}")


def _decode_shard(
    output: Path,
    input_directory: Path,
    identities: dict[Path, tuple[int, int]],
    number: int,
) -> dict[str, np.ndarray]:
    name = f"shard_{number:05d}.npz"
    input_path = input_directory / name
    teacher_path = output / "teacher" / name
    _verify_path(input_path, identities)
    _verify_path(teacher_path, identities)
    with np.load(input_path, allow_pickle=False) as stored:
        value = {key: stored[key] for key in stored.files}
    with np.load(teacher_path, allow_pickle=False) as stored:
        if not np.array_equal(value["ordinals"], stored["ordinals"]):
            raise ValueError("Teacher/event ordinals differ")
        if not np.array_equal(value["tokens"], stored["tokens"]):
            raise ValueError("Teacher/event object tokens differ")
        value["relation_targets"] = stored["relation_targets"]
        value["relation_valid"] = stored["relation_valid"]
    return value


class _GroupCache:
    def __init__(
        self,
        output: Path,
        input_directory: Path,
        identities: dict[Path, tuple[int, int]],
        shard_count: int,
    ) -> None:
        self.output = output
        self.input_directory = input_directory
        self.identities = identities
        self.shard_count = shard_count
        self.groups: OrderedDict[int, dict[int, dict[str, np.ndarray]]] = OrderedDict()
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="train40-decode")
        self.cache_bytes = 0
        self.reads = 0
        self.hits = 0
        self.group_requests = 0
        self.decode_seconds = 0.0
        self.decoded_groups = 0

    def _verify_shard(self, number: int) -> None:
        name = f"shard_{number:05d}.npz"
        _verify_path(self.input_directory / name, self.identities)
        _verify_path(self.output / "teacher" / name, self.identities)

    def group(self, number: int) -> dict[int, dict[str, np.ndarray]]:
        self.group_requests += 1
        start = number * SHARDS_PER_GROUP
        stop = min(start + SHARDS_PER_GROUP, self.shard_count)
        if start >= self.shard_count:
            raise ValueError(f"Logical shard group is outside the manifest: {number}")
        if number in self.groups:
            for shard in range(start, stop):
                self._verify_shard(shard)
            self.groups.move_to_end(number)
            self.hits += stop - start
            return self.groups[number]
        # Evict before decoding a third group so resident decoded arrays never
        # transiently exceed the two-group design bound.
        while len(self.groups) >= 2:
            _, old = self.groups.popitem(last=False)
            self.cache_bytes -= sum(_value_bytes(shard) for shard in old.values())
            del old
        decode_started = time.perf_counter()
        futures = {
            shard: self.pool.submit(
                _decode_shard,
                self.output,
                self.input_directory,
                self.identities,
                shard,
            )
            for shard in range(start, stop)
        }
        value = {shard: future.result() for shard, future in futures.items()}
        self.decode_seconds += time.perf_counter() - decode_started
        self.decoded_groups += 1
        self.reads += stop - start
        size = sum(_value_bytes(shard) for shard in value.values())
        while self.groups and (
            len(self.groups) >= 2 or self.cache_bytes + size > CACHE_BUDGET_BYTES
        ):
            _, old = self.groups.popitem(last=False)
            self.cache_bytes -= sum(_value_bytes(shard) for shard in old.values())
            del old
        self.groups[number] = value
        self.cache_bytes += size
        return value

    def shard(self, number: int) -> dict[str, np.ndarray]:
        return self.group(number // SHARDS_PER_GROUP)[number]

    def retain(self, groups: tuple[int, ...]) -> None:
        wanted = set(groups[:2])
        # Discard groups outside the new plan first. This preserves a still-wanted
        # group across transitions such as short-tail -> next shuffled group.
        for number in list(self.groups):
            if number not in wanted:
                old = self.groups.pop(number)
                self.cache_bytes -= sum(_value_bytes(shard) for shard in old.values())
                del old
        for number in groups[:2]:
            self.group(number)

    def snapshot(self) -> dict[str, Any]:
        return {
            "cache_groups": list(self.groups),
            "cache_bytes": self.cache_bytes,
            "reads": self.reads,
            "hits": self.hits,
            "group_requests": self.group_requests,
            "cache_budget_bytes": CACHE_BUDGET_BYTES,
            "decode_seconds": self.decode_seconds,
            "decoded_groups": self.decoded_groups,
        }

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.groups.clear()
        self.cache_bytes = 0


def _collate(
    ids: list[int],
    cache: _GroupCache,
    metadata: list[dict[str, str]],
) -> ObjectEventV4Batch:
    from operational.train40_system.fast_collate import collate_object_event_v4

    group_numbers = sorted({value // GROUP_ROWS for value in ids})
    groups = {number: cache.group(number) for number in group_numbers}
    records = []
    for ordinal in ids:
        if ordinal < 0 or ordinal >= len(metadata):
            raise ValueError(f"TRAIN40 ordinal is outside the sealed population: {ordinal}")
        shard_number = ordinal // SHARD_ROWS
        value = groups[ordinal // GROUP_ROWS][shard_number]
        local = ordinal % SHARD_ROWS
        if local >= len(value["ordinals"]) or int(value["ordinals"][local]) != ordinal:
            raise ValueError("Event shard ordinal drifted")
        row = metadata[ordinal]
        if str(value["tokens"][local]) != row["sample_token"]:
            raise ValueError("Event shard sample identity drifted")
        records.append(
            {
                **row,
                "event_v4_common_roi": value["events"][local],
                "garl_delta_t_s": value["delta"][local],
                "observable_motion": value["motion"][local],
                "garl_visible_heights_px": value["visible_heights"][local],
                "ttc_s": value["target_ttc"][local],
                "event_v4_boxes_xyxy": value["boxes"][local],
                "event_v4_common_square_xyxy": value["square"][local],
                "dinov3_relation_targets": value["relation_targets"][local],
                "dinov3_relation_valid": value["relation_valid"][local],
            }
        )
    return collate_object_event_v4(records)


def _write_batch(batch: ObjectEventV4Batch, arena: SharedMemory) -> dict[str, Any]:
    descriptors: dict[str, Any] = {}
    offset = 0
    for field in fields(batch):
        value = getattr(batch, field.name)
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            raw = tensor.reshape(-1).view(torch.uint8).numpy()
            size = int(raw.nbytes)
            if offset + size > arena.size:
                raise MemoryError(
                    f"Collated batch requires {offset + size} bytes; arena has {arena.size}"
                )
            buffer = arena.buf
            if buffer is None:
                raise RuntimeError("Shared arena closed during worker write")
            target = np.ndarray((size,), dtype=np.uint8, buffer=buffer, offset=offset)
            np.copyto(target, raw, casting="no")
            descriptors[field.name] = {
                "kind": "tensor",
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "offset": offset,
                "bytes": size,
            }
            offset += size
        else:
            descriptors[field.name] = {"kind": "metadata", "value": value}
    return {"fields": descriptors, "used_bytes": offset}


def _worker(
    connection: Connection,
    output_text: str,
    arena_names: list[str],
    slot_bytes: int,
) -> None:
    arenas: list[SharedMemory] = []
    cache: _GroupCache | None = None
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        output = Path(output_text)
        input_directory, identities = _manifest_identities(output)
        input_manifest = read(output / "INPUT_MANIFEST.json")
        shard_count = len(input_manifest["ordered_shards"])
        import pandas as pd

        metadata = cast(
            list[dict[str, str]],
            pd.read_parquet(
                output / "TRAIN40_ROWS.parquet",
                columns=["sequence_id", "sample_token", "track_id"],
            ).to_dict(orient="records"),
        )
        cache = _GroupCache(output, input_directory, identities, shard_count)
        arenas = [SharedMemory(name=name, create=False, size=slot_bytes) for name in arena_names]
        available = deque(range(len(arenas)))
        collate_seconds = 0.0
        arena_write_seconds = 0.0
        requested_batch_count = 0

        def worker_snapshot() -> dict[str, Any]:
            return {
                **cache.snapshot(),
                "collate_seconds": collate_seconds,
                "arena_write_seconds": arena_write_seconds,
                "requested_batch_count": requested_batch_count,
            }

        while True:
            request = connection.recv()
            kind = request.get("kind")
            request_id = int(request.get("id", -1))
            try:
                if kind == "close":
                    connection.send({"id": request_id, "kind": "closed", **worker_snapshot()})
                    break
                if kind == "release":
                    slot = int(request["slot"])
                    if slot in available:
                        raise ValueError("Shared arena slot released twice")
                    available.append(slot)
                    continue
                if kind == "prepare":
                    cache.retain(tuple(int(value) for value in request["groups"]))
                    connection.send({"id": request_id, "kind": "prepared", **worker_snapshot()})
                    continue
                if kind != "batch":
                    raise ValueError(f"Unknown process-input request: {kind}")
                if not available:
                    raise RuntimeError("No released shared arena slot is available")
                ids = [int(value) for value in request["ids"]]
                if not ids:
                    raise ValueError("Process input batch cannot be empty")
                slot = available.popleft()
                collate_started = time.perf_counter()
                batch = _collate(ids, cache, metadata)
                collate_seconds += time.perf_counter() - collate_started
                write_started = time.perf_counter()
                payload = _write_batch(batch, arenas[slot])
                arena_write_seconds += time.perf_counter() - write_started
                requested_batch_count += 1
                connection.send(
                    {
                        "id": request_id,
                        "kind": "batch",
                        "slot": slot,
                        "ids": ids,
                        "payload": payload,
                        **worker_snapshot(),
                    }
                )
            except BaseException as error:
                connection.send(
                    {
                        "id": request_id,
                        "kind": "error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
    except (EOFError, BrokenPipeError):
        pass
    finally:
        if cache is not None:
            cache.close()
        for arena in arenas:
            arena.close()
        connection.close()


_TORCH_DTYPES = {
    str(value): value
    for value in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
}


class ProcessInputs:
    """Serve exact typed batches from one persistent spawned CPU decoder."""

    def __init__(
        self,
        output: Path,
        *,
        slots: int = 3,
        slot_bytes: int = DEFAULT_SLOT_BYTES,
        timeout_seconds: float = 120.0,
    ) -> None:
        if slots not in (2, 3) or slot_bytes <= 0 or timeout_seconds <= 0:
            raise ValueError("Process input arena configuration is invalid")
        self.output = output
        self.timeout_seconds = timeout_seconds
        self._lock = Lock()
        self._plan_lock = Lock()
        self._failure: BaseException | None = None
        self._closed = False
        self._request_id = 0
        self._last_plan: tuple[int, ...] | None = None
        self._pending_plan: tuple[int, ...] | None = None
        self._arenas: list[SharedMemory] = []
        self.cache_bytes = 0
        self.reads = 0
        self.hits = 0
        self._snapshot: dict[str, Any] = {
            "cache_groups": [],
            "cache_bytes": 0,
            "reads": 0,
            "hits": 0,
            "group_requests": 0,
            "cache_budget_bytes": CACHE_BUDGET_BYTES,
            "decode_seconds": 0.0,
            "decoded_groups": 0,
            "collate_seconds": 0.0,
            "arena_write_seconds": 0.0,
            "requested_batch_count": 0,
            "parent_ipc_wait_seconds": 0.0,
            "parent_reconstruct_seconds": 0.0,
        }
        context = mp.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        self._connection = parent
        try:
            for _ in range(slots):
                self._arenas.append(SharedMemory(create=True, size=slot_bytes))
            self._process = context.Process(
                target=_worker,
                args=(child, str(output), [arena.name for arena in self._arenas], slot_bytes),
                name="train40-process-inputs",
                daemon=True,
            )
            self._process.start()
        except BaseException:
            child.close()
            parent.close()
            self._release_arenas()
            raise
        child.close()

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _poison(self, error: BaseException) -> None:
        if self._failure is None:
            self._failure = error

    def _check(self) -> None:
        if self._failure is not None:
            raise RuntimeError("Process input loader is poisoned") from self._failure
        if self._closed:
            raise RuntimeError("Process input loader is closed")
        if not self._process.is_alive():
            error = RuntimeError(f"Process input worker exited with code {self._process.exitcode}")
            self._poison(error)
            raise error

    def _send(self, value: dict[str, Any]) -> None:
        self._check()
        try:
            self._connection.send(value)
        except BaseException as error:
            self._poison(error)
            raise RuntimeError("Process input worker send failed") from error

    def _receive(self, expected_id: int, expected_kind: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._connection.poll(min(0.1, max(0.0, remaining))):
                if not self._process.is_alive():
                    error = RuntimeError(
                        f"Process input worker exited with code {self._process.exitcode}"
                    )
                    self._poison(error)
                    raise error
                if remaining > 0:
                    continue
                error = TimeoutError("Process input worker response timed out")
                self._poison(error)
                raise error
            try:
                response = cast(dict[str, Any], self._connection.recv())
            except BaseException as error:
                self._poison(error)
                raise RuntimeError("Process input worker receive failed") from error
            if response.get("kind") == "error":
                error = RuntimeError(
                    f"Process input worker {response.get('error_type')}: {response.get('error')}"
                )
                self._poison(error)
                raise error
            response_id = int(response.get("id", -1))
            if response_id != expected_id:
                if response.get("kind") == "prepared" and response_id < expected_id:
                    self._update(response)
                    continue
                error = RuntimeError("Process input response order differs from request FIFO")
                self._poison(error)
                raise error
            if response.get("kind") != expected_kind:
                error = RuntimeError("Process input response kind differs from request")
                self._poison(error)
                raise error
            self._update(response)
            return response

    def _update(self, response: dict[str, Any]) -> None:
        for name in (
            "cache_groups",
            "cache_bytes",
            "reads",
            "hits",
            "group_requests",
            "cache_budget_bytes",
            "decode_seconds",
            "decoded_groups",
            "collate_seconds",
            "arena_write_seconds",
            "requested_batch_count",
        ):
            if name in response:
                self._snapshot[name] = response[name]
        self.cache_bytes = int(self._snapshot["cache_bytes"])
        self.reads = int(self._snapshot["reads"])
        self.hits = int(self._snapshot["hits"])

    @staticmethod
    def _planned_groups(order: torch.Tensor, start: int, batch_size: int) -> tuple[int, ...]:
        if order.ndim != 1 or start < 0 or batch_size <= 0 or start >= len(order):
            raise ValueError("Invalid process-input lookahead cursor")
        values = order[start : min(len(order), start + GROUP_ROWS + batch_size)].tolist()
        groups: list[int] = []
        for value in values:
            group = int(value) // GROUP_ROWS
            if group not in groups:
                groups.append(group)
                if len(groups) == 2:
                    break
        return tuple(groups)

    def prepare_order(self, order: torch.Tensor, start: int, batch_size: int) -> None:
        """Publish a nonblocking current/next-group hint without touching sampler state."""
        groups = self._planned_groups(order, start, batch_size)
        with self._plan_lock:
            self._check()
            if groups != self._last_plan:
                self._pending_plan = groups

    def _reconstruct(self, response: dict[str, Any]) -> ObjectEventV4Batch:
        from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch

        slot = int(response["slot"])
        arena = self._arenas[slot]
        values: dict[str, Any] = {}
        try:
            for name, descriptor in response["payload"]["fields"].items():
                if descriptor["kind"] == "metadata":
                    values[name] = descriptor["value"]
                    continue
                dtype = _TORCH_DTYPES.get(descriptor["dtype"])
                if dtype is None:
                    raise TypeError(f"Unsupported shared tensor dtype: {descriptor['dtype']}")
                size = int(descriptor["bytes"])
                offset = int(descriptor["offset"])
                buffer = arena.buf
                if buffer is None:
                    raise RuntimeError("Shared arena closed during parent copy")
                raw = np.ndarray((size,), dtype=np.uint8, buffer=buffer, offset=offset)
                tensor = torch.empty(tuple(descriptor["shape"]), dtype=dtype)
                tensor.reshape(-1).view(torch.uint8).copy_(torch.from_numpy(raw))
                values[name] = tensor
            return ObjectEventV4Batch(**values)
        finally:
            self._send({"id": self._next_id(), "kind": "release", "slot": slot})

    def batch(self, ids: list[int]) -> ObjectEventV4Batch:
        """Return an owned byte-identical batch and release its shared slot."""
        with self._lock:
            with self._plan_lock:
                groups = self._pending_plan
                self._pending_plan = None
            if groups is not None and groups != self._last_plan:
                prepare_id = self._next_id()
                self._send({"id": prepare_id, "kind": "prepare", "groups": groups})
                self._last_plan = groups
            request_id = self._next_id()
            self._send({"id": request_id, "kind": "batch", "ids": list(ids)})
            wait_started = time.perf_counter()
            response = self._receive(request_id, "batch")
            self._snapshot["parent_ipc_wait_seconds"] = float(
                self._snapshot["parent_ipc_wait_seconds"]
            ) + (time.perf_counter() - wait_started)
            if response.get("ids") != list(ids):
                error = RuntimeError("Process input worker changed batch IDs")
                self._poison(error)
                raise error
            reconstruct_started = time.perf_counter()
            try:
                return self._reconstruct(response)
            finally:
                self._snapshot["parent_reconstruct_seconds"] = float(
                    self._snapshot["parent_reconstruct_seconds"]
                ) + (time.perf_counter() - reconstruct_started)

    def snapshot(self) -> dict[str, Any]:
        """Return retained JSON-safe worker/cache/resource telemetry."""
        # Telemetry is advisory and each assignment is atomic under CPython.
        # Avoid waiting behind a long decode/IPC batch during checkpointing.
        return {
            "schema": "train40_process_inputs_runtime_v1",
            **dict(self._snapshot),
            "worker_pid": self._process.pid,
            "worker_alive": self._process.is_alive(),
            "closed": self._closed,
            "poisoned": self._failure is not None,
            "arena_count": len(self._arenas),
            "arena_bytes_each": self._arenas[0].size if self._arenas else 0,
        }

    def _release_arenas(self) -> None:
        for arena in self._arenas:
            try:
                arena.close()
            finally:
                try:
                    arena.unlink()
                except FileNotFoundError:
                    pass

    def close(self) -> None:
        """Stop the worker and unlink every shared arena; safe after failures."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self._process.is_alive():
                    request_id = self._next_id()
                    try:
                        self._connection.send({"id": request_id, "kind": "close"})
                        deadline = time.monotonic() + min(self.timeout_seconds, 10.0)
                        while self._process.is_alive() and time.monotonic() < deadline:
                            if self._connection.poll(0.1):
                                response = cast(dict[str, Any], self._connection.recv())
                                self._update(response)
                                if response.get("kind") == "closed":
                                    break
                    except (EOFError, BrokenPipeError, OSError):
                        pass
                    self._process.join(timeout=2.0)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=5.0)
            finally:
                self._connection.close()
                self._release_arenas()


__all__ = ["ProcessInputs"]
