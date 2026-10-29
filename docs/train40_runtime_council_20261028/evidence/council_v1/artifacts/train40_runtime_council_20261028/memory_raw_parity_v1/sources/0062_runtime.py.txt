"""Operational persistence only; the frozen statistical recipe remains unchanged."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    """Replace one own output atomically; preserve failed temporary bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".pending", dir=path.parent)
    with os.fdopen(handle, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(name, path)


def atomic_json(path: Path, value: dict) -> None:
    atomic_bytes(
        path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    )


def publish_json(path: Path, value: dict) -> None:
    """An identical publication is reusable; a different publication fails closed."""
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"publication changed: {path}")
        return
    atomic_json(path, value)


def json_record(value: dict) -> dict:
    """Use the durable JSON type representation, preserving full float precision."""
    return json.loads(json.dumps(value, allow_nan=False))


def atomic_parquet_write(
    frame: pd.DataFrame,
    path: Path,
    writer: Callable[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    """Keep the previous complete table until the new bytes have been flushed."""
    pending = path.with_name(path.name + ".pending")
    result = writer(frame, pending, *args, **kwargs)
    with pending.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.replace(pending, path)
    return result


def require_roots(roots: dict[str, Path]) -> None:
    """Report the exact failed operation and root, including non-E: failures."""
    for role, path in roots.items():
        try:
            path.stat()
        except OSError as error:
            if isinstance(error, FileNotFoundError):
                raise FileNotFoundError(
                    error.errno, f"required root stat ({role}): {error.strerror}", str(path)
                ) from error
            raise


def repair_torn_journal(path: Path) -> None:
    """Retain a torn last append separately; reject corruption in complete lines."""
    if not path.exists():
        return
    raw = path.read_bytes()
    end = raw.rfind(b"\n") + 1
    for line in raw[:end].splitlines():
        json.loads(line)
    if end != len(raw):
        evidence = path.with_name(path.name + ".torn_" + hashlib.sha256(raw).hexdigest()[:12])
        if not evidence.exists():
            atomic_bytes(evidence, raw)
        atomic_bytes(path, raw[:end])


def resumable_hierarchical_losses(
    frame: pd.DataFrame,
    losses: np.ndarray,
    output: Path,
    check: Callable[[], None],
    *,
    draws_path: Path,
    binding: dict,
    chunk_size: int = 256,
    publication_hook: Callable[[str], None] = lambda _: None,
) -> tuple[np.ndarray, dict]:
    """Replay registered draws, committing complete fragments and accepted IDs.

    Whole-sequence/track population, bucket weights and floating-point operation
    order match hierarchical_losses. No new random draw is generated: the full
    immutable T2 draw log is the authority for pending generation. A torn legacy
    line is retained as evidence and is never a committed draw.
    """
    from e_jepa_ttc.evaluation.stage63_65 import BUCKETS

    check()
    seq = frame.sequence_id.to_numpy()
    tracks = frame.track_id.to_numpy()
    target = frame.target_ttc_s.to_numpy()
    sequences = sorted(np.unique(seq))
    groups = {}
    for s in sequences:
        names = sorted(np.unique(tracks[seq == s]))
        counts = np.zeros((len(names), 4))
        sums = np.zeros((len(names), 4, losses.shape[1]))
        for t, name in enumerate(names):
            for b, (_, lo, hi, _) in enumerate(BUCKETS):
                mask = (seq == s) & (tracks == name) & (target > lo) & (target <= hi)
                counts[t, b] = mask.sum()
                sums[t, b] = losses[mask].sum(0)
        groups[s] = (counts, sums)
    raw = draws_path.read_bytes()
    lines = raw.splitlines(keepends=True)
    if not lines or any(not line.endswith(b"\n") for line in lines):
        raise ValueError("canonical draw publication is truncated")
    h = hashlib.sha256()
    for line in lines:
        h.update(line.rstrip(b"\r\n"))
    recipe = {
        "schema": "simplex_t_bootstrap_fragments_v1",
        "binding": binding,
        "draw_file_sha256": hashlib.sha256(raw).hexdigest(),
        "frame_sha256": hashlib.sha256(
            cast(Callable[..., pd.Series], hash_pandas_object)(frame, index=True)
            .to_numpy()
            .tobytes()
        ).hexdigest(),
        "losses_sha256": hashlib.sha256(np.asarray(losses, dtype=np.float64).tobytes()).hexdigest(),
        "losses_shape": list(losses.shape),
        "sequence_order": sequences,
        "chunk_size": chunk_size,
        "rng": "not pending; replay sealed T2 attempts including rejections",
        "numpy_version": np.__version__,
    }
    output.mkdir(parents=True, exist_ok=True)
    control = output / ".resume"
    control.mkdir(exist_ok=True)
    publish_json(control / "INPUTS.json", recipe)
    legacy = output / "HIERARCHICAL_DRAWS.jsonl"
    if legacy.exists():
        old = legacy.read_bytes()
        complete = old[: old.rfind(b"\n") + 1]
        if not raw.startswith(complete):
            raise ValueError("partial draw log differs from canonical T2 attempts")
        if complete != old:
            publish_json(
                control / "LEGACY_TORN_TAIL.json",
                {
                    "sha256": hashlib.sha256(old).hexdigest(),
                    "committed_prefix_bytes": len(complete),
                },
            )
    chunks = []
    accepted = 0
    for start in range(0, len(lines), chunk_size):
        stop = min(start + chunk_size, len(lines))
        directory = control / f"fragment_{start:05d}_{stop:05d}"
        receipt = directory / "RECEIPT.json"
        check()
        if receipt.exists():
            record = json.loads(receipt.read_text(encoding="utf-8"))
            if (
                record["start"] != start
                or record["stop"] != stop
                or record["input_sha256"] != digest(control / "INPUTS.json")
            ):
                raise ValueError("fragment offset or input identity changed")
            for name, pin in record["files"].items():
                if digest(directory / name) != pin:
                    raise ValueError(f"fragment bytes changed: {directory / name}")
            result = np.load(directory / "LOSSES.npy", allow_pickle=False)
        else:
            values, accepted_ids, rejected_ids = [], [], []
            for draw_id in range(start, stop):
                encoded = json.loads(lines[draw_id])
                scores = []
                valid = True
                if len(encoded[0]) != len(sequences):
                    raise ValueError("draw sequence count differs")
                for position, index in enumerate(encoded[0], 1):
                    counts, sums = groups[sequences[index]]
                    selection = np.asarray(encoded[position], dtype=np.int64)
                    if (
                        len(selection) != len(counts)
                        or np.any(selection < 0)
                        or np.any(selection >= len(counts))
                    ):
                        raise ValueError("draw track population differs")
                    total = counts[selection].sum(0)
                    if np.any(total == 0):
                        valid = False
                        break
                    scores.append(
                        np.array([0.5, 0.3, 0.1, 0.1]) @ (sums[selection].sum(0) / total[:, None])
                    )
                if valid:
                    if len(encoded) != len(sequences) + 1:
                        raise ValueError("valid draw has trailing/missing track selections")
                    values.append(np.mean(scores, axis=0))
                    accepted_ids.append(draw_id)
                else:
                    rejected_ids.append(draw_id)
            result = np.asarray(values, dtype=np.float64).reshape(-1, losses.shape[1])
            directory.mkdir(exist_ok=True)
            buffer = tempfile.SpooledTemporaryFile()
            np.save(buffer, result, allow_pickle=False)
            buffer.seek(0)
            atomic_bytes(directory / "LOSSES.npy", buffer.read())
            buffer.close()
            atomic_bytes(directory / "DRAWS.jsonl", b"".join(lines[start:stop]))
            record = {
                "start": start,
                "stop": stop,
                "accepted_ids": accepted_ids,
                "rejected_ids": rejected_ids,
                "input_sha256": digest(control / "INPUTS.json"),
                "files": {name: digest(directory / name) for name in ("LOSSES.npy", "DRAWS.jsonl")},
            }
            publication_hook("before_fragment_commit")
            publish_json(receipt, record)
            print(
                json.dumps(
                    {
                        "status": "BOOTSTRAP_FRAGMENT_COMMITTED",
                        "path": str(receipt),
                        "attempts": stop,
                        "valid_draws": accepted + len(result),
                    }
                ),
                flush=True,
            )
        chunks.append(result)
        accepted += len(result)
    sampled = np.concatenate(chunks)
    if len(sampled) != 8192 or len(sampled) / len(lines) < 0.99:
        raise RuntimeError("BOOTSTRAP_SUPPORT_BLOCKED")
    check()
    atomic_bytes(legacy, raw)
    publication_hook("after_draw_publication")
    buffer = tempfile.SpooledTemporaryFile()
    np.save(buffer, sampled, allow_pickle=False)
    buffer.seek(0)
    atomic_bytes(output / "BOOTSTRAP_LOSSES.npy", buffer.read())
    buffer.close()
    report = dict(
        draws_sha256=h.hexdigest(),
        valid_draws=len(sampled),
        attempts=len(lines),
        discarded=len(lines) - len(sampled),
        validity_fraction=len(sampled) / len(lines),
    )
    publish_json(control / "COMPLETE.json", report)
    return sampled, report
