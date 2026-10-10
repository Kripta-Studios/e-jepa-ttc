"""Resumable label-free feature replay followed by an explicit scoring join."""

# ruff: noqa: ANN401 -- experiment manifests and Torch records are heterogeneous.
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from operational.efficient_context.common import digest
from operational.evttc_transfer.models import FrozenModels
from operational.simplex_t_shared_route.adapter import inputs
from operational.train40_system.contracts import environment
from operational.train40_system.durable_io import atomic_json
from operational.ttc_revision.direct_runtime import DirectRuntime
from operational.ttc_revision.head import DirectTTCHead
from operational.ttc_revision.inputs import EventPreparer
from operational.ttc_revision.runtime import H8Runtime


def prefetched_queries(
    pending: list[tuple[int, dict[str, Any], Path]],
    prepare: Callable[[dict[str, Any]], dict[str, Any]],
) -> Iterator[tuple[int, dict[str, Any], Path, dict[str, Any]]]:
    """Overlap one CPU preparation with GPU replay; never call a model in the worker."""
    if not pending:
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="ttc-input") as pool:
        future = pool.submit(prepare, pending[0][1])
        for position, (number, row, path) in enumerate(pending):
            prepared = future.result()
            if position + 1 < len(pending):
                future = pool.submit(prepare, pending[position + 1][1])
            yield number, row, path, prepared


def replay(
    manifest_path: Path,
    output: Path,
    *,
    fcwd: bool = False,
    compiled_backend: bool = False,
    cache_mib: int = 512,
    deadline_utc: str | None = None,
) -> None:
    """Cache every query's raw features without opening targets or fitting models."""
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    deadline = datetime.fromisoformat(deadline_utc) if deadline_utc else None
    if deadline is not None and deadline.tzinfo is None:
        raise ValueError("deadline must include a timezone")
    if not 0 <= cache_mib <= 1024:
        raise ValueError("raw cache limit must be between 0 and 1024 MiB")
    rows = sorted(document["rows"], key=lambda r: (r["sequence_id"], r["anchor_us"]))
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    frozen = FrozenModels(Path("artifacts/train40_system_20261005"), "cuda")
    canonical, compact = H8Runtime(frozen), H8Runtime(frozen, batch=8, on_device=True)
    direct_canonical = DirectRuntime(frozen, output.parent, compact=False)
    direct_compact = DirectRuntime(frozen, output.parent)
    if compiled_backend:
        from operational.ttc_revision.compiled import CompiledH8Runtime

        receipt_path = output.parent / "compiled" / "COMPILED_PROBE.json"
        admission = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            admission["status"] != "COMPLETE"
            or admission["source_sha256"] != digest(Path(__file__).with_name("compiled.py"))
            or admission["model_bindings"] != frozen.bindings
            or not all(
                row["admitted"] and row["canonical16_admitted"] for row in admission["admission"]
            )
        ):
            raise ValueError("complete matching compiler admission required")
        canonical = CompiledH8Runtime(frozen, batch=16, on_device=False)
        compact = CompiledH8Runtime(frozen)
    identity = {
        "manifest_sha256": digest(manifest_path),
        "models": frozen.bindings,
        "source_sha256": {
            name: digest(Path(__file__).with_name(name))
            for name in ("evaluate.py", "inputs.py", "runtime.py", "direct_runtime.py", "head.py")
        },
        "direct_checkpoints_sha256": {
            str(seed): digest(output.parent / f"direct_seed{seed}.pt") for seed in (7, 13, 23)
        },
        "fcwd": fcwd,
        "compiled_backend": compiled_backend,
        "offline_cpu_prefetch_depth": 1,
        "raw_cache_mib": cache_mib if not fcwd else None,
        "compiled_source_sha256": digest(Path(__file__).with_name("compiled.py"))
        if compiled_backend
        else None,
    }
    identity_path = output / "REPLAY_IDENTITY.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("feature replay identity changed")
    atomic_json(identity_path, identity)
    fragment_dir = output / "fragments"
    fragment_dir.mkdir(exist_ok=True)
    begun = time.perf_counter()
    pending = []
    for number, row in enumerate(rows):
        path = fragment_dir / f"{number:06d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if str(saved["query_id"]) != row["query_id"]:
                    raise ValueError("resumed fragment query mismatch")
        else:
            pending.append((number, row, path))
    with EventPreparer(cache_bytes=cache_mib * 1024**2) as preparer, torch.inference_mode():
        prepare = preparer.prepare
        if fcwd:
            from operational.sota_eval.fcwd_inputs import prepare as prepare_fcwd

            prepare = prepare_fcwd
        for number, row, path, prepared in prefetched_queries(pending, prepare):
            if deadline is not None and datetime.now(UTC) >= deadline:
                atomic_json(
                    output / "REPLAY_INTERRUPTED.json",
                    {
                        "reason": "GPU budget deadline",
                        "next_query": number,
                        "deadline_utc": deadline.isoformat(),
                    },
                )
                raise TimeoutError("GPU budget reached; completed query fragments remain resumable")
            raw = canonical.features(prepared["own_events"])
            expected = canonical.predict_from_features(raw).cpu().numpy()
            raw_compact = compact.features(prepared["own_events"])
            actual = compact.predict_from_features(raw_compact).cpu().numpy()
            direct_gpu = direct_canonical.predict_from_features(raw).cpu().numpy()
            direct_compact_gpu = direct_compact.predict_from_features(raw_compact).cpu().numpy()
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    query_id=np.asarray(row["query_id"]),
                    features=raw.cpu().numpy(),
                    compact_features=raw_compact.cpu().numpy(),
                    canonical=expected,
                    compact=actual,
                    direct_gpu=direct_gpu,
                    direct_compact_gpu=direct_compact_gpu,
                    feature_abs_max=np.max(np.abs(raw.cpu().numpy() - raw_compact.cpu().numpy())),
                )
                handle.flush()
            temporary.replace(path)
            if number % 50 == 0 or number + 1 == len(rows):
                progress = {
                    "completed": number + 1,
                    "population": len(rows),
                    "elapsed_seconds_this_run": time.perf_counter() - begun,
                }
                atomic_json(output / "REPLAY_PROGRESS.json", progress)
                print(json.dumps(progress), flush=True)
    atomic_json(
        output / "REPLAY_COMPLETE.json",
        {
            "status": "COMPLETE",
            "population": len(rows),
            "identity_sha256": digest(identity_path),
            "environment": environment(),
            "fragments": {p.name: digest(p) for p in fragment_dir.glob("*.npz")},
        },
    )


def predict_and_join(replay_dir: Path, campaign: Path, original: Path, output: Path) -> None:
    """Evaluate fixed complete checkpoints, then join labels solely for reporting."""
    receipt = json.loads((replay_dir / "REPLAY_COMPLETE.json").read_text(encoding="utf-8"))
    if receipt["status"] != "COMPLETE":
        raise ValueError("complete feature replay required")
    records = []
    raw_features = []
    compact_features = []
    for name, sha in sorted(receipt["fragments"].items()):
        path = replay_dir / "fragments" / name
        if digest(path) != sha:
            raise ValueError("feature replay bytes changed")
        with np.load(path, allow_pickle=False) as stored:
            records.append(
                {
                    "query_id": str(stored["query_id"]),
                    **{
                        f"replayed_H8_seed{seed}": float(stored["canonical"][i])
                        for i, seed in enumerate((7, 13, 23))
                    },
                    **{
                        f"{prefix}_seed{seed}": float(stored[key][i])
                        for key, prefix in (
                            ("direct_gpu", "GPU_Direct"),
                            ("direct_compact_gpu", "GPU_compact_Direct"),
                        )
                        for i, seed in enumerate((7, 13, 23))
                    },
                    "compact_max_abs_seconds": float(
                        np.max(np.abs(stored["canonical"] - stored["compact"]))
                    ),
                    "compact_admitted": bool(
                        np.allclose(stored["canonical"], stored["compact"], atol=0.01, rtol=0.0001)
                    ),
                }
            )
            raw_features.append(stored["features"])
            compact_features.append(stored["compact_features"])
    raw = np.stack(raw_features)
    for item, feature in zip(records, raw, strict=True):
        for column, index in (
            ("roi_log_event_count", 0),
            ("roi_log_event_rate", 1),
            ("a5_transport", 2),
            ("a5_confidence", 3),
            ("a5_log_variance", 4),
            ("c2f_transport", 5),
            ("c2f_confidence", 6),
            ("c2f_log_variance", 7),
        ):
            item[column] = float(feature[-1, index])
    with np.load("artifacts/train40_system_20261005/H8_FEATURES.npz", allow_pickle=False) as stored:
        mean, scale = stored["mean"], stored["scale"]
    gathered = [
        inputs(
            "H8_SEED7",
            row,
            np.ones(8, bool),
            np.arange(350000, -1, -50000, dtype=np.int64),
            0,
            0,
            mean,
            scale,
        )[:3]
        for row in np.concatenate((raw, np.stack(compact_features)))
    ]
    xs = [torch.cat([row[k] for row in gathered]) for k in range(3)]
    torch.set_num_threads(2)
    bindings: dict[str, Any] = {}
    for seed in (7, 13, 23):
        path = campaign / f"direct_seed{seed}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        expected = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if digest(path) != expected["sha256"] or checkpoint["status"] != "COMPLETE":
            raise ValueError("complete verified trained head required")
        if checkpoint["update"] != checkpoint["contract"]["recipe"]["updates"]:
            raise ValueError("not the predeclared endpoint")
        model = DirectTTCHead().eval()
        model.load_state_dict(checkpoint["model"])
        with torch.inference_mode():
            prediction = torch.cat(
                [model(*(x[i : i + 256] for x in xs)) for i in range(0, len(xs[0]), 256)]
            ).numpy()
        for item, value, compact_value in zip(
            records, prediction[: len(raw)], prediction[len(raw) :], strict=True
        ):
            item[f"Direct_seed{seed}"] = float(value)
            item[f"compact_features_Direct_seed{seed}"] = float(compact_value)
        bindings[str(seed)] = expected["sha256"]
    # This is the first label access: no subsequent training or endpoint selection.
    source = pd.read_csv(original)
    predictions = pd.DataFrame(records)
    if set(predictions.query_id) != set(source.query_id) or source.query_id.duplicated().any():
        raise ValueError("source and replay population differ")
    joined = source.merge(predictions, on="query_id", how="left", validate="one_to_one")
    # Fixed prediction-only aggregation; never pick a favourable seed using GT.
    for prefix in ("H8", "Direct"):
        joined[f"{prefix}_median3"] = np.median(
            np.asarray(joined[[f"{prefix}_seed{seed}" for seed in (7, 13, 23)]], dtype=float),
            axis=1,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output, index=False)
    atomic_json(
        output.with_suffix(".json"),
        {
            "source_sha256": digest(original),
            "replay_receipt_sha256": digest(replay_dir / "REPLAY_COMPLETE.json"),
            "checkpoint_sha256": bindings,
            "output_sha256": digest(output),
            "compact_admitted_every_query": bool(joined.compact_admitted.all()),
            "compact_max_abs_seconds": float(joined.compact_max_abs_seconds.max()),
            "direct_compact_features_admitted": all(
                bool(
                    np.allclose(
                        joined[f"Direct_seed{s}"],
                        joined[f"compact_features_Direct_seed{s}"],
                        atol=0.01,
                        rtol=0.0001,
                    )
                )
                for s in (7, 13, 23)
            ),
            "direct_compact_features_max_abs_seconds": max(
                float(
                    np.max(
                        np.abs(
                            joined[f"Direct_seed{s}"] - joined[f"compact_features_Direct_seed{s}"]
                        )
                    )
                )
                for s in (7, 13, 23)
            ),
            "direct_gpu_admitted_every_query": all(
                bool(
                    np.allclose(
                        joined[f"Direct_seed{s}"],
                        joined[f"{prefix}_seed{s}"],
                        atol=0.01,
                        rtol=0.0001,
                    )
                )
                for prefix in ("GPU_Direct", "GPU_compact_Direct")
                for s in (7, 13, 23)
            ),
            "direct_gpu_max_abs_seconds": max(
                float(np.max(np.abs(joined[f"Direct_seed{s}"] - joined[f"{prefix}_seed{s}"])))
                for prefix in ("GPU_Direct", "GPU_compact_Direct")
                for s in (7, 13, 23)
            ),
            "canonical_original_max_abs_seconds": {
                str(s): float(
                    np.max(np.abs(joined[f"H8_seed{s}"] - joined[f"replayed_H8_seed{s}"]))
                )
                for s in (7, 13, 23)
            },
            "canonical_original_admitted_every_query": all(
                bool(
                    np.allclose(
                        joined[f"H8_seed{s}"],
                        joined[f"replayed_H8_seed{s}"],
                        atol=0.01,
                        rtol=0.0001,
                    )
                )
                for s in (7, 13, 23)
            ),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("replay", help="Label-free raw-event replay; resumes per query")
    cache.add_argument("--manifest", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--fcwd", action="store_true")
    cache.add_argument("--cache-mib", type=int, default=512)
    cache.add_argument("--deadline-utc", help="Stop between queries at an aware ISO timestamp")
    cache.add_argument(
        "--compiled-backend",
        action="store_true",
        help="Use only after a successful matching COMPILED_PROBE receipt",
    )
    evaluate = sub.add_parser("join", help="Predict fixed heads then join frozen labels")
    evaluate.add_argument("--replay-dir", type=Path, required=True)
    evaluate.add_argument("--campaign", type=Path, required=True)
    evaluate.add_argument("--original", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "replay":
        replay(
            args.manifest,
            args.output,
            fcwd=args.fcwd,
            compiled_backend=args.compiled_backend,
            cache_mib=args.cache_mib,
            deadline_utc=args.deadline_utc,
        )
    else:
        predict_and_join(args.replay_dir, args.campaign, args.original, args.output)
