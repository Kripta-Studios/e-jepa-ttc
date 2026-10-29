"""Finite recoverable TRAIN route measurements; no optimizer or sealed evaluation."""

# Manifest and scientific model payloads are dynamic checked boundary objects.
# ruff: noqa: ANN401
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from operational.simplex_t_post_campaign.contracts import checked, digest, read, save  # noqa: E402
from operational.simplex_t_post_campaign.run import guard_function  # noqa: E402
from operational.simplex_t_route_readiness.inspect import inspect as inspect_sources  # noqa: E402

OUT = ROOT / "artifacts/simplex_t/shared_gpu_route_20261004"
NIGHT = ROOT / "artifacts/simplex_t/nocturnal_20261003"
H16 = ROOT / "artifacts/simplex_t/h16_replication_20261003"
MODES = ("application_hdf5_cold", "warm_block1", "warm_block2")


def guard() -> dict:
    """Retain the active RAM/RSS/commit/disk limits under shared GPU permission."""
    return guard_function()(OUT)


def now() -> str:
    """UTC boundary timestamp."""
    return datetime.now(UTC).isoformat()


def progress(status: str, **values: Any) -> None:
    """Persist completed route fragments and the next concrete operation."""
    save(
        OUT / "PROGRESS.json",
        dict(
            status=status,
            observed_utc=now(),
            optimizer_updates=0,
            fragments=len(list((OUT / "fragments").glob("*.json"))),
            **values,
        ),
    )
    print(json.dumps(dict(status=status, **values), ensure_ascii=False), flush=True)


def setup_runtime() -> None:
    """Apply the unchanged historical numerical policy after resource admission."""
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    # Restore the recorded cache-generator policy, rather than different cuDNN kernels.
    torch.use_deterministic_algorithms(False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the verified frozen producers")


def prepare() -> None:
    """Freeze the shared-resource amendment, schedule, tolerances and source pins."""
    import numpy as np
    import torch

    from operational.simplex_t_h16_replication.common import historical_spec, sources

    guard()
    path = OUT / "PROTOCOL.json"
    if path.exists():
        p = read(path)
        for name, sha in p["source_pins"].items():
            checked(ROOT / name, sha)
        print("Existing prospective protocol retained", flush=True)
        return
    readiness = inspect_sources(ROOT)
    if any(not row["exists"] for row in readiness["checkpoints"] + readiness["raw_references"]):
        raise ValueError("required route source absent")
    # The user prospectively waived exclusivity; this does not rewrite an ACK,
    # local source identity, old delivery, scientific objective or candidate.
    h16 = read(H16 / "PROTOCOL.json")
    s = sources(h16)
    try:
        source = s.train(historical_spec(s, 0, 16))
        if source.identity_sha256 != h16["sources"]["0"]["train_sha256"]:
            raise ValueError("historical TRAIN/normalizer identity differs")
        np.savez(OUT / "NORMALIZER.npz", mean=source.normalizer.mean, scale=source.normalizer.scale)
    finally:
        s.release()
        gc.collect()
    profile = read(NIGHT / "PROFILE_INPUT_EXPORT.json")
    for row in profile["models"].values():
        for kind in ("weights", "inputs"):
            checked(NIGHT / row[kind + "_path"], row[kind + "_sha256"])
    raw = [
        {**row, "mtime_ns": Path(row["path"]).stat().st_mtime_ns}
        for row in readiness["raw_references"]
    ]
    local = read(Path(h16["launch"]["local_paths"]))
    config = read(Path(h16["launch"]["source_configuration"]))
    launch = read(ROOT / "artifacts/simplex_t/T0/D1_EXACT_RESOURCE_LAUNCH.json")
    prep = ROOT.parent / launch["preprocessing"]
    checked(prep, "063980fdae5fda0b2836befc662fdd1cd5659bf06f10d9760dfc0d566fac8e39")
    code = [
        *sorted(Path(__file__).parent.glob("*.py")),
        ROOT / "src/e_jepa_ttc/simplex_t/expert_features.py",
        ROOT / "src/e_jepa_ttc/simplex_t/expert_phase.py",
        ROOT / "src/e_jepa_ttc/simplex_t/context_raw_union.py",
        ROOT / "src/e_jepa_ttc/simplex_t/query_context_voxel.py",
        ROOT / "src/e_jepa_ttc/simplex_t/cached_event_reader.py",
        ROOT / "src/e_jepa_ttc/simplex_t/model.py",
        ROOT / "src/e_jepa_ttc/simplex_t/phase.py",
        ROOT / "operational/simplex_t_cost_context/model.py",
        ROOT / "operational/simplex_t_post_campaign/route_policy.py",
        ROOT / "src/e_jepa_ttc/models/causal_scale_ttc.py",
        ROOT / "src/e_jepa_ttc/data/eap_representation.py",
        ROOT / "src/e_jepa_ttc/evaluation/scientific_recovery_v8.py",
        ROOT / "src/e_jepa_ttc/training/stage61_pair_head.py",
    ]
    protocol = dict(
        schema="prospective_shared_gpu_route_profile_v1",
        created_utc=now(),
        authorization="User explicitly permits shared GPU and waives exclusive access for P3.",
        optimizer_updates_authorized=0,
        registered_candidate="TPR-D1-H8-C160",
        source_pins={str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in code},
        head_index_sha256=digest(NIGHT / "PROFILE_INPUT_EXPORT.json"),
        selection_sha256=readiness["selection_sha256"],
        normalizer_sha256=digest(OUT / "NORMALIZER.npz"),
        train_identity_sha256=h16["sources"]["0"]["train_sha256"],
        queries=readiness["queries"],
        checkpoints=readiness["checkpoints"],
        raw=raw,
        models=list(profile["models"]),
        modes=list(MODES),
        measurements=64 * 9 * 3,
        warmups_per_query_model=1,
        warmups=64 * 9,
        order="family0/1/2, original selection rank; models rotate by query rank+mode",
        producer_batch=16,
        padding="fixed historical producer batch; inactive history zeroed",
        producer_precision="CUDA_FP32_TF32_OFF",
        head_precision="CPU_FP32",
        threads=4,
        interop=2,
        cache_scope="cold closes application HDF5 handles; warm retains them; no OS cache flush",
        resident_models=True,
        model_load_cost_reported_separately=True,
        inference_input_parity=dict(feature_atol=1e-4, phase_atol=1e-5, ttc_atol_seconds=0.01),
        parity=(
            "all 64 raw H16 contexts against cached inputs; every measured reduced route checked"
        ),
        timer=(
            "direct total plus raw HDF5 reads, ROI/voxel, producer/transfers, "
            "normalization, head/TTC"
        ),
        gpu_shared=True,
        performance_claim="observational shared-resource measurements, not isolated causality",
        roi_scope="supplied current ROI; excludes detection, tracking and AEB",
        raw_hash_scope=(
            "historical source bindings; current size/mtime, endpoints, and decoded-input parity"
        ),
        raw_train_root=str(Path(local["eap_root"]) / "data/train"),
        index_dirs={
            "D0": config["original"]["index_root"]["relative_path"],
            "D1": str(Path(config["expansion"]["0"]["index_manifest"]["relative_path"]).parent),
        },
        preprocessing_path=str(prep),
        preprocessing_sha256=digest(prep),
        resources_before=guard(),
        torch=str(torch.__version__),
    )
    save(path, protocol)
    save(OUT / "SOURCE_ADMISSION.json", readiness)
    progress("PROTOCOL_FROZEN", planned_fragments=1728, raw_queries=64)


class TimedReader:
    """Observe canonical chunk reads without changing events, crop or boundaries."""

    def __init__(self, reader: Any) -> None:
        self.reader, self.raw_ns = reader, 0

    def iter_window_chunks(self, *args: Any, **kwargs: Any) -> Any:
        iterator = iter(self.reader.iter_window_chunks(*args, **kwargs))
        while True:
            start = time.perf_counter_ns()
            try:
                chunk = next(iterator)
            except StopIteration:
                self.raw_ns += time.perf_counter_ns() - start
                break
            self.raw_ns += time.perf_counter_ns() - start
            yield chunk


class Runtime:
    """One resident producer family, one HDF5 reader, nine frozen CPU heads."""

    def __init__(self, p: dict) -> None:
        import numpy as np
        import torch

        from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool
        from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
        from operational.simplex_t_cost_context.model import build_model

        self.p, self.torch, self.np = p, torch, np
        self.pool, self.models, self.family = ReaderPool(), {}, None
        self.counts = {name: 0 for name in ("A5", "C2F", "PAIR")}
        self.raw_verified: set[str] = set()
        self.attempt_counter = 0
        self.heads, self.cached, self.indexes = {}, {}, {}
        with np.load(OUT / "NORMALIZER.npz", allow_pickle=False) as z:
            self.mean, self.scale = z["mean"], z["scale"]
        for name, relative in p["index_dirs"].items():
            directory = ROOT / relative
            manifest = read(directory / "INDEX_MANIFEST.json")
            checked(directory / "query_context_index.npz", manifest["index_sha256"])
            with np.load(directory / "query_context_index.npz", allow_pickle=False) as z:
                self.indexes[name] = {
                    k: z[k]
                    for k in (
                        "tokens",
                        "sequences",
                        "base_windows_us",
                        "square_xyxy",
                        "anchor_us",
                        "roi_available_us",
                        "lag_us",
                        "valid",
                    )
                }
        self.prep = read(Path(p["preprocessing_path"]))["config"]
        for label, row in read(NIGHT / "PROFILE_INPUT_EXPORT.json")["models"].items():
            c = row["constructor"]
            head = (
                TemporalRefiner(TemporalConfig(**c["config"]))
                if c["kind"] == "historical_temporal"
                else build_model(c["arm"])
            )
            with np.load(NIGHT / row["weights_path"], allow_pickle=False) as z:
                head.load_state_dict(
                    {k: torch.from_numpy(z[k].copy()) for k in z.files}, strict=True
                )
            self.heads[label] = head.cpu().float().eval()
            with np.load(NIGHT / row["inputs_path"], allow_pickle=False) as z:
                self.cached[label] = {k: z[k] for k in z.files}

    def load(self, query: dict) -> None:
        """Load only the pinned inner-fold family; log initialization outside timers."""
        from e_jepa_ttc.evaluation.scientific_recovery_v8 import load_causal_scale_replay_checkpoint
        from e_jepa_ttc.training.stage61_pair_head import load_pair_head

        if self.family == query["family_id"]:
            return
        self.close_family()
        guard()
        free, _total = self.torch.cuda.mem_get_info()
        if free < 2 * 1024**3:
            raise InterruptedError(f"Shared GPU free memory below 2 GiB before loading: {free}")
        paths = {r["registered_sha256"]: Path(r["path"]) for r in self.p["checkpoints"]}
        start = time.perf_counter_ns()
        for name in ("A5", "C2F", "PAIR"):
            path = paths[query["experts"][name]]
            checked(path, query["experts"][name])
            loader = load_pair_head if name == "PAIR" else load_causal_scale_replay_checkpoint
            self.models[name] = loader(path, device=self.torch.device("cuda"))

            def count_call(module: Any, inputs: Any, output: Any, name: str = name) -> None:
                self.counts[name] += 1

            self.models[name].register_forward_hook(count_call)
        self.torch.cuda.synchronize()
        self.family = query["family_id"]
        save(
            OUT / f"model_load/family{self.family}_{os.getpid()}.json",
            dict(
                family=self.family,
                milliseconds=(time.perf_counter_ns() - start) / 1e6,
                device="CUDA_FP32",
                optimizer_updates=0,
                observed_utc=now(),
            ),
        )

    def close_family(self) -> None:
        """Release only this process's model tensors and HDF5 handles."""
        self.pool.close()
        self.models.clear()
        self.family = None
        gc.collect()
        self.torch.cuda.empty_cache()

    def execute(
        self,
        query: dict,
        rank: int,
        label: str,
        cold: bool,
        purpose: str = "measurement",
    ) -> tuple[dict, dict]:
        """Time a fresh supplied-ROI request including raw reads and real experts."""
        from e_jepa_ttc.simplex_t.context_raw_union import encode_context_union
        from e_jepa_ttc.simplex_t.phase import phase_to_ttc
        from operational.simplex_t_post_campaign.route_policy import ALLOWED
        from operational.simplex_t_shared_route.adapter import extract, inputs

        torch, np = self.torch, self.np
        self.attempt_counter += 1
        attempt_path = OUT / f"attempts/p{os.getpid()}_{self.attempt_counter:05d}.json"
        attempt = dict(
            status="STARTED",
            purpose=purpose,
            rank=rank,
            model=label,
            observed_utc=now(),
            optimizer_updates=0,
        )
        save(attempt_path, attempt)
        index, qi = self.indexes[query["pool"]], query["index_row"]
        if str(index["tokens"][qi]) != query["sample_token"]:
            raise ValueError("profile query binding changed")
        length = 1 if label == "H1_SEED7" else 16 if label == "H16_SEED7" else 8
        valid = index["valid"][qi].copy()
        valid[:-length] = False
        raw_path = Path(self.p["raw_train_root"]) / query["sequence_id"] / "events.h5"
        receipt = next(r for r in self.p["raw"] if r["path"] == str(raw_path))
        stat = raw_path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (receipt["bytes"], receipt["mtime_ns"]):
            raise ValueError("authorized raw stream changed since admission")
        if query["sequence_id"] not in self.raw_verified:
            import h5py

            # Register any existing HDF5 compression plugins via the canonical reader.
            self.pool.get(raw_path)
            with h5py.File(raw_path, "r") as stream:
                times = stream["events/t"]
                if not isinstance(times, h5py.Dataset) or times.ndim != 1 or not len(times):
                    raise ValueError("nonempty one-dimensional raw timestamp dataset required")
                first, last = int(times[0]), int(times[-1])
            bound = [first, last + int(query["pool"] == "D0")]
            manifest = read(ROOT / self.p["index_dirs"][query["pool"]] / "INDEX_MANIFEST.json")
            if manifest["stream_bounds_us"][query["sequence_id"]] != bound:
                raise ValueError("raw endpoints differ from registered context support")
            self.raw_verified.add(query["sequence_id"])
            save(
                OUT / f"raw_support/{query['sequence_id']}.json",
                dict(
                    **receipt,
                    first_us=first,
                    last_us=last,
                    bounds_match=True,
                    full_raw_hash_recomputed=False,
                    optimizer_updates=0,
                ),
            )
        if cold:
            self.pool.close()
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        open_start = time.perf_counter_ns()
        reader = TimedReader(self.pool.get(raw_path))
        open_ns = time.perf_counter_ns() - open_start
        events = encode_context_union(
            cast(Any, reader),
            index["base_windows_us"][qi],
            index["lag_us"],
            valid,
            tuple(index["square_xyxy"][qi]),
            sequence_id=query["sequence_id"],
            roi_size=self.prep["roi_size"],
            event_pixel_diff=self.prep["event_pixel_diff"],
        )
        prep_end = time.perf_counter_ns()
        delta = torch.tensor(
            np.diff(index["base_windows_us"][qi][:, 1]) / 1e6, dtype=torch.float32
        ).repeat(16, 1)
        calls_before = self.counts.copy()
        raw = extract(label, self.models, events.to("cuda"), delta.to("cuda"))
        torch.cuda.synchronize()
        expert_end = time.perf_counter_ns()
        actual_calls = {key: self.counts[key] - calls_before[key] for key in self.counts}
        xs = inputs(
            label,
            raw,
            valid,
            index["lag_us"],
            int(index["anchor_us"][qi]),
            int(index["roi_available_us"][qi]),
            self.mean,
            self.scale,
        )
        feature_end = time.perf_counter_ns()
        result = self.heads[label](*xs)
        ttc = phase_to_ttc(result["point_phase"].to(torch.float64))
        end = time.perf_counter_ns()
        if purpose == "validation":
            canonical_path = (
                OUT / f"canonical_parity_historical_runtime/family{query['family_id']}.json"
            )
            if not canonical_path.exists():
                from e_jepa_ttc.simplex_t.expert_features import extract_family

                canonical = extract_family(
                    self.models["A5"],
                    self.models["C2F"],
                    self.models["PAIR"],
                    events.to("cuda"),
                    delta.to("cuda"),
                )
                difference = float(np.max(np.abs(raw - canonical["features145"][:, :17])))
                if difference != 0:
                    raise ValueError("real full-route adapter differs from canonical extraction")
                save(
                    canonical_path,
                    dict(
                        family=query["family_id"],
                        max_raw_feature_difference=0,
                        additional_validation_producer_calls=3,
                        optimizer_updates=0,
                    ),
                )
        if not bool(torch.isfinite(ttc).all()):
            raise ValueError("nonfinite measured TTC")
        errors = {}
        for name, value in zip(("features", "times", "valid", "experts"), xs, strict=True):
            expected = self.cached[label][name][rank : rank + 1]
            actual = value.numpy()
            error = float(np.max(np.abs(actual.astype(float) - expected.astype(float))))
            errors[name] = error
            tolerance = (
                self.p["inference_input_parity"]["feature_atol"]
                if name == "features"
                else (self.p["inference_input_parity"]["phase_atol"] if name == "experts" else 1e-7)
            )
            if error > tolerance:
                raise ValueError(f"real route {label} {name} differs from cached contract: {error}")
        reference = self.heads[label](
            *(
                torch.from_numpy(self.cached[label][k][rank : rank + 1].copy())
                for k in ("features", "times", "valid", "experts")
            )
        )
        ref_ttc = phase_to_ttc(reference["point_phase"].to(torch.float64))
        phase_error = float((reference["point_phase"] - result["point_phase"]).abs().max())
        ttc_error = float((ref_ttc - ttc).abs().max())
        if phase_error > self.p["inference_input_parity"]["phase_atol"] or (
            ttc_error > self.p["inference_input_parity"]["ttc_atol_seconds"]
        ):
            raise ValueError("real route emitted point differs from cached frozen head")
        payload = {
            key: value.numpy()
            for key, value in zip(
                ("features", "times", "valid", "experts"),
                xs,
                strict=True,
            )
        }
        identity = hashlib.sha256(
            b"".join(
                key.encode()
                + value.dtype.str.encode()
                + repr(value.shape).encode()
                + value.tobytes()
                for key, value in payload.items()
            )
        ).hexdigest()
        input_path = OUT / f"inference_inputs/{identity}.npz"
        if not input_path.exists():
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **payload)
            input_path.parent.mkdir(parents=True, exist_ok=True)
            with input_path.open("xb") as stream:
                stream.write(buffer.getvalue())
                stream.flush()
                os.fsync(stream.fileno())
        raw_ns = open_ns + reader.raw_ns
        row: dict[str, Any] = dict(
            model=label,
            rank=rank,
            sample_token=query["sample_token"],
            sequence_id=query["sequence_id"],
            family=query["family_id"],
            total_ms=(end - start) / 1e6,
            raw_read_ms=raw_ns / 1e6,
            roi_voxel_ms=(prep_end - start - raw_ns) / 1e6,
            experts_transfers_ms=(expert_end - prep_end) / 1e6,
            normalize_ms=(feature_end - expert_end) / 1e6,
            head_emission_ms=(end - feature_end) / 1e6,
            ttc_seconds=float(ttc[0]),
            point_phase=float(result["point_phase"][0]),
            allowed_producers=sorted(ALLOWED[label]),
            producer_calls=len(ALLOWED[label]),
            a5_encodings=int("A5" in ALLOWED[label]),
            valid_slots=int(valid.sum()),
            gpu_allocated_bytes=torch.cuda.memory_allocated(),
            gpu_reserved_bytes=torch.cuda.memory_reserved(),
            shared_gpu=True,
            actual_producer_calls=actual_calls,
            inputs_relative_path=str(input_path.relative_to(OUT)).replace("\\", "/"),
            inputs_sha256=digest(input_path),
        )
        if {key for key, calls in row["actual_producer_calls"].items() if calls} != ALLOWED[label]:
            raise ValueError("actual producer calls violate the reduced route contract")
        parity = dict(
            input_errors=errors, point_phase_error=phase_error, ttc_error_seconds=ttc_error
        )
        save(
            attempt_path,
            {
                **attempt,
                "status": "COMPLETED",
                "completed_utc": now(),
                "actual_producer_calls": actual_calls,
            },
        )
        return row, parity


def run(max_new: int | None) -> None:
    """Validate all 64 raw contexts, then collect the fixed recoverable route queue."""
    import torch

    p = read(OUT / "PROTOCOL.json")
    for name, sha in p["source_pins"].items():
        checked(ROOT / name, sha)
    checked(OUT / "NORMALIZER.npz", p["normalizer_sha256"])
    runtime = Runtime(p)
    queries = sorted(enumerate(p["queries"]), key=lambda v: (v[1]["family_id"], v[0]))
    count = 0
    try:
        with torch.inference_mode():
            for rank, query in queries:
                path = OUT / f"parity_historical_runtime/q{rank:02d}.json"
                if path.exists():
                    continue
                guard()
                runtime.load(query)
                row, parity = runtime.execute(query, rank, "H16_SEED7", True, "validation")
                save(
                    path,
                    dict(
                        query=query,
                        observation=row,
                        parity=parity,
                        protocol_sha256=digest(OUT / "PROTOCOL.json"),
                    ),
                )
                progress(
                    "RAW_PARITY_FRAGMENT",
                    parity_queries=len(list((OUT / "parity_historical_runtime").glob("*.json"))),
                )
            progress("RAW_PARITY_COMPLETE", parity_queries=64)
            for rank, query in queries:
                for mode_id, mode in enumerate(MODES):
                    shift = (rank + mode_id) % 9
                    order = p["models"][shift:] + p["models"][:shift]
                    for label in order:
                        path = OUT / f"fragments/q{rank:02d}_{mode}_{label}.json"
                        if path.exists():
                            if read(path)["protocol_sha256"] != digest(OUT / "PROTOCOL.json"):
                                raise ValueError("committed timing fragment binding changed")
                            continue
                        if max_new is not None and count >= max_new:
                            progress("FRAGMENT_LIMIT", next_id=path.stem)
                            return
                        resources = guard()
                        runtime.load(query)
                        if mode_id == 1:
                            runtime.execute(query, rank, label, False, "warmup")
                        row, parity = runtime.execute(query, rank, label, mode_id == 0)
                        row.update(
                            mode=mode,
                            resources=resources,
                            parity=parity,
                            protocol_sha256=digest(OUT / "PROTOCOL.json"),
                            observed_utc=now(),
                        )
                        save(path, row)
                        count += 1
                    progress("QUERY_BLOCK_COMMITTED", rank=rank, mode=mode)
            progress("MEASUREMENTS_COMPLETE", confirmed_measurements=1728)
    finally:
        runtime.close_family()


def main() -> None:
    """Explicit prepare/run commands; retain a live-owner lock for this output root."""
    import psutil

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--max-new", type=int)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    owner = OUT / "ACTIVE_OWNER.json"
    if owner.exists():
        old = read(owner)
        try:
            alive = abs(psutil.Process(old["pid"]).create_time() - old["created_unix"]) < 0.01
        except psutil.NoSuchProcess:
            alive = False
        if alive:
            raise RuntimeError("shared-route writer already live")
        save(OUT / "owners" / f"stale_{old['pid']}.json", old)
        owner.unlink()
    with owner.open("x", encoding="utf-8") as stream:
        json.dump(dict(pid=os.getpid(), created_unix=psutil.Process().create_time()), stream)
    try:
        guard()
        setup_runtime()
        if args.command == "prepare":
            prepare()
        else:
            run(args.max_new)
    except BaseException as error:
        progress(
            "PAUSED" if isinstance(error, InterruptedError) else "ERROR",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    finally:
        owner.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
