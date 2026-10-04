"""Measured native Garl routes on the unchanged64 TRAIN profiling queries."""

from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from .common import ROOT, Campaign, Lease, atomic_bytes, atomic_json, digest, read

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor
    from torch.nn import Module

    from e_jepa_ttc.efficient_context.garl_head import NativeHeadSource


def prepared_forward(
    producer: Module,
    tensors: list[Tensor],
    counts: list[np.ndarray],
    native: NativeHeadSource,
    xs: tuple[Tensor, ...],
    *,
    label: str,
    refiner: Module,
) -> tuple[tuple[float, float], tuple[Tensor, ...] | None]:
    """Execute unchanged batch1 native forwards and the fixed CPU correction head."""
    import numpy as np
    import torch

    from e_jepa_ttc.simplex_t.phase import phase_to_ttc

    valid = xs[2].numpy()[0]
    if not valid[-1]:
        raise ValueError("native runtime requires its current observation")
    raw = np.zeros((1, native.length, 3), np.float32)
    current_ttc = float("nan")
    for j, slot in enumerate(np.flatnonzero(valid)):
        with torch.inference_mode():
            heights, _ = producer(tensors[j][None].cuda())
        pair = heights.double().cpu().numpy()[0]
        ratio = pair[0] / pair[1]
        if not np.isfinite(ratio) or ratio <= 0:
            raise ArithmeticError("invalid native runtime phase; no query dropping")
        raw[0, slot, 0] = -np.log(ratio)
        current_ttc = float(np.inf if ratio == 1 else 0.1 / (1 - ratio))
        if label != "GARL_NATIVE":
            raw[0, slot, 1:] = counts[j]
    if label == "GARL_NATIVE":
        return (float(raw[0, -1, 0]), current_ttc), None
    normalized = ((raw - native.normalizer.mean) / native.normalizer.scale).astype(np.float32)
    normalized[0, ~valid] = 0
    experts = np.repeat(raw[:, -1, :1], 3, axis=1)
    fresh = (torch.from_numpy(normalized), xs[1], xs[2], torch.from_numpy(experts))
    with torch.inference_mode():
        output = refiner(*fresh)
        value = phase_to_ttc(output["point_phase"].double())
    return (float(output["point_phase"][0]), float(value[0])), fresh


def run(c: Campaign) -> None:
    """Profile only sealed native producers and heads; retain every parity receipt."""
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import torch

    from e_jepa_ttc.efficient_context.garl_head import model
    from e_jepa_ttc.efficient_context.garl_input import inference_record
    from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool
    from e_jepa_ttc.simplex_t.context_raw_union import _read_crop
    from e_jepa_ttc.simplex_t.phase import phase_to_ttc
    from e_jepa_ttc.simplex_t.query_context_voxel import encode_query_window
    from e_jepa_ttc.simplex_t.training import load_checkpoint, state_digest

    from .garl_heads import source
    from .native_garl import model_class, native_config

    c.freeze()
    c.require_resources()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    route = c.historical / "artifacts/simplex_t/shared_gpu_route_20261004/PROTOCOL.json"
    parent = read(route)
    producers = read(c.out / "garl/ENDPOINTS.json")
    heads = read(c.out / "garl_heads/ENDPOINTS.json")
    if not producers["all_twelve_frozen_before_evaluation"] or len(producers["fits"]) != 12:
        raise ValueError("all twelve sealed native producers required for runtime")
    if not heads["all_six_frozen_before_evaluation"] or len(heads["fits"]) != 6:
        raise ValueError("all six sealed native heads required for runtime")
    queries = parent["queries"]
    if len(queries) != 64:
        raise ValueError("only the unchanged64 authorized TRAIN profiling queries")
    folder = c.out / "garl_runtime"
    protocol = {
        "parent_route_sha256": digest(route),
        "producer_endpoints_sha256": digest(c.out / "garl/ENDPOINTS.json"),
        "head_endpoints_sha256": digest(c.out / "garl_heads/ENDPOINTS.json"),
        "queries": [q["sample_token"] for q in queries],
        "blocks": ["application_cold", "warm_block1", "warm_block2"],
        "CPU_head_warmups_per_head": 10,
        "model_family_initialization_excluded_from_R0": True,
        "systems": ["GARL_NATIVE", "GARL_H1", "GARL_H8"],
        "regimes": ["R0", "R2_PREPARED_PRODUCER_AND_HEAD", "R2_HEAD_ONLY_CPU_FP32"],
        "native_producer_dispatch": "one unchanged native observation per forward",
        "count_rate": "unchanged common100ms ROI encoding, scalar reduction on CUDA",
        "parity_limits": {"features": 1e-4, "phase": 1e-5, "times": 1e-7},
        "OLD_DEV_opened": False,
        "optimizer_updates": 0,
        "implementation_sha256": digest(Path(__file__)),
    }
    pin = folder / "PROTOCOL.json"
    if pin.exists() and read(pin) != protocol:
        raise ValueError("native runtime freeze changed")
    if not pin.exists():
        atomic_json(pin, protocol)
    features = c.out / "garl/features/fold0/inner_oof"
    with np.load(features / "SOURCE.npz", allow_pickle=False) as z:
        tokens = z["tokens"].astype(str)
    positions = {t: i for i, t in enumerate(tokens)}
    if any(q["sample_token"] not in positions for q in queries):
        raise ValueError("runtime query is outside native fold0 TRAIN")
    rows = pq.read_table(
        Path(c.local["garl_annotations_candidate"]),
        columns=["sample_token", "sequence_id", "event_windows_us", "boxes_xyxy"],
        filters=[("sample_token", "in", protocol["queries"])],
        use_threads=False,
    ).to_pylist()
    records = {r["sample_token"]: r for r in rows}
    indexes = {}
    for key, relative in parent["index_dirs"].items():
        path = c.historical / relative
        if (
            digest(path / "query_context_index.npz")
            != read(path / "INDEX_MANIFEST.json")["index_sha256"]
        ):
            raise ValueError("runtime TRAIN context index changed")
        with np.load(path / "query_context_index.npz", allow_pickle=False) as z:
            indexes[key] = {k: z[k].copy() for k in ("lag_us", "base_windows_us", "square_xyxy")}
    preprocessing = read(Path(parent["preprocessing_path"]))["config"]
    inputs, refiners = {}, {}
    for length in (1, 8):
        inputs[length] = source(c, 0, "inner_oof", length)
        endpoint = next(r for r in heads["fits"] if r["fold"] == 0 and r["length"] == length)
        if digest(Path(endpoint["checkpoint"])) != endpoint["checkpoint_sha256"]:
            raise ValueError("native runtime head endpoint changed")
        refiners[length] = model().float().eval()
        refiners[length].load_state_dict(load_checkpoint(Path(endpoint["checkpoint"]))["model"])
        warm = inputs[length].gather(torch.tensor([positions[queries[0]["sample_token"]]]))[:4]
        for _ in range(10):
            with torch.inference_mode():
                refiners[length](*warm)
    endpoints = {r["key"]: r for r in producers["fits"]}
    admission = read(c.out / "garl/ADMISSION.json")
    families = [r for r in admission["fits"] if r["outer"] == 0 and r["inner"] is not None]
    active, producer = None, None
    pool = ReaderPool()
    measurements = []
    try:
        for block in protocol["blocks"]:
            for qi, query in enumerate(queries):
                token = query["sample_token"]
                record = records[token]
                candidates = [
                    r["key"] for r in families if record["sequence_id"] in r["excluded_inner"]
                ]
                if len(candidates) != 1:
                    raise ValueError("native runtime query lacks unique INNER-OOF producer")
                key = candidates[0]
                if record["sequence_id"] in endpoints[key]["train_sequences"]:
                    raise ValueError("native runtime producer has seen its query holdout")
                if active != key:
                    producer = None
                    torch.cuda.empty_cache()
                    endpoint = endpoints[key]
                    if digest(Path(endpoint["checkpoint"])) != endpoint["sha256"]:
                        raise ValueError("native runtime producer endpoint changed")
                    state = torch.load(
                        endpoint["checkpoint"], map_location="cpu", weights_only=True
                    )
                    seal = state.pop("state_sha256")
                    if state_digest(state) != seal or state["status"] != "COMPLETE":
                        raise ValueError("unsealed native runtime producer")
                    producer = (
                        model_class(c)(native_config(c), is_train=False).float().cuda().eval()
                    )
                    producer.load_state_dict(state["model"])
                    del state
                    active = key
                index = indexes[query["pool"]]
                row = query["index_row"]
                for label, length in (("GARL_NATIVE", 1), ("GARL_H1", 1), ("GARL_H8", 8)):
                    c.require_resources()
                    receipt = folder / f"fragments/{block}_{qi:02d}_{label}.json"
                    if receipt.exists():
                        saved = read(receipt)
                        if saved["status"] != "PASSED":
                            raise ValueError("failed native runtime parity requires inspection")
                        measurements.extend(saved["measurements"])
                        continue
                    native = inputs[length]
                    xs = native.gather(torch.tensor([positions[token]]))[:4]
                    valid = xs[2].numpy()[0]
                    lags = index["lag_us"][-length:]
                    if not np.allclose(
                        xs[1].numpy()[0, valid, 0], lags[valid] / 1e6, atol=1e-7, rtol=0
                    ):
                        raise ValueError("native runtime timing differs from registered context")
                    if block == "application_cold":
                        pool.close()
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    tensors, counts = [], []
                    for slot in np.flatnonzero(valid):
                        lag = int(lags[slot])
                        shifted = dict(
                            record,
                            event_windows_us=(
                                np.asarray(record["event_windows_us"], np.int64) - lag
                            ).tolist(),
                        )
                        tensors.append(inference_record(shifted, pool, c.raw))
                        if label != "GARL_NATIVE":
                            start, end = map(int, index["base_windows_us"][row, -1] - lag)
                            square = tuple(index["square_xyxy"][row])
                            reader = pool.get(c.raw / record["sequence_id"] / "events.h5")
                            crop = _read_crop(
                                reader,
                                start,
                                end,
                                square,
                                preprocessing["event_pixel_diff"],
                                256 * 1024**2,
                            )
                            if crop is None:
                                raise InterruptedError(
                                    "native runtime count/rate crop exceeds bounded capacity"
                                )
                            common = encode_query_window(
                                crop,
                                square_xyxy=square,
                                start_us=start,
                                end_us=end,
                                sequence_id=record["sequence_id"],
                                roi_size=preprocessing["roi_size"],
                                bins_per_polarity=5,
                                event_pixel_diff=preprocessing["event_pixel_diff"],
                            )
                            counts.append(common[-2:].cuda().mean((-2, -1)).cpu().numpy())
                    torch.cuda.synchronize()
                    prepared = time.perf_counter()

                    if producer is None:
                        raise ValueError("missing frozen native producer")
                    forward = partial(
                        prepared_forward,
                        producer,
                        tensors,
                        counts,
                        native,
                        xs,
                        label=label,
                        refiner=refiners[length],
                    )
                    value, fresh = forward()
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                    feature_error = 0.0
                    phase_error = 0.0
                    if fresh is not None:
                        feature_error = max(
                            float((a.double() - b.double()).abs().max())
                            for a, b in zip(fresh, xs, strict=True)
                        )
                        with torch.inference_mode():
                            expected = refiners[length](*xs)
                        phase_error = abs(value[0] - float(expected["point_phase"][0]))
                    else:
                        expected_phase = float(
                            native.features[native.history[positions[token], -1], 0]
                        )
                        phase_error = abs(value[0] - expected_phase)
                    measured = [
                        dict(
                            label=label,
                            block=block,
                            query=token,
                            regime="R0",
                            milliseconds=(end - begin) * 1000,
                            raw_ROI_ms=(prepared - begin) * 1000,
                            prepared_modules_ms=(end - prepared) * 1000,
                        )
                    ]
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    repeat_value, _ = forward()
                    torch.cuda.synchronize()
                    measured.append(
                        dict(
                            label=label,
                            block=block,
                            query=token,
                            regime="R2_PREPARED_PRODUCER_AND_HEAD",
                            milliseconds=(time.perf_counter() - start) * 1000,
                        )
                    )
                    repeat_error = (
                        0.0
                        if np.array_equal(value, repeat_value)
                        else float(np.max(np.abs(np.asarray(value) - np.asarray(repeat_value))))
                    )
                    if label != "GARL_NATIVE":
                        start = time.perf_counter()
                        with torch.inference_mode():
                            output = refiners[length](*xs)
                            phase_to_ttc(output["point_phase"].double())
                        measured.append(
                            dict(
                                label=label,
                                block=block,
                                query=token,
                                regime="R2_HEAD_ONLY_CPU_FP32",
                                milliseconds=(time.perf_counter() - start) * 1000,
                            )
                        )
                    passed = feature_error <= 1e-4 and phase_error <= 1e-5 and repeat_error == 0
                    atomic_json(
                        receipt,
                        dict(
                            status="PASSED" if passed else "FAILED_INTEGRITY",
                            producer_sha256=endpoints[key]["sha256"],
                            measurements=measured,
                            feature_max_abs=feature_error,
                            phase_max_abs=phase_error,
                            repeated_output_max_abs=repeat_error,
                            optimizer_updates=0,
                        ),
                    )
                    if not passed:
                        raise ValueError("native runtime parity failed: " + str(receipt))
                    measurements.extend(measured)
                    print(f"GARL_RUNTIME_{block}_{qi + 1}/64_{label}", flush=True)
                    del tensors, counts, forward
        table = pd.DataFrame(measurements)
        atomic_bytes(folder / "MEASUREMENTS.csv", table.to_csv(index=False).encode())
        summary = (
            table.groupby(["regime", "label", "block"])
            .milliseconds.agg(requests="count", p50_ms="median", p95_ms=lambda x: x.quantile(0.95))
            .reset_index()
        )
        atomic_bytes(folder / "RUNTIME.csv", summary.to_csv(index=False).encode())
        atomic_json(
            c.out / "GARL_RUNTIME_RESULTS.json",
            dict(
                status="COMPLETE",
                measurements=len(measurements),
                raw_requests=576,
                queries=64,
                blocks=3,
                optimizer_updates=0,
                producer_parameters=sum(p.numel() for p in producer.parameters())
                if producer is not None
                else None,
                parity="PASSED",
                information_budget_equivalent=False,
                native_span_ms=200,
                historical_span_ms=300,
                ingestion_and_ROI_detector_excluded=True,
            ),
        )
    finally:
        pool.close()


def main() -> int:
    """Execute bounded native cost measurements after all producer/head seals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/campaign/efficient_context_v1.json"
    )
    args = parser.parse_args()
    c = Campaign(args.protocol)
    with Lease(c.out):
        run(c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
