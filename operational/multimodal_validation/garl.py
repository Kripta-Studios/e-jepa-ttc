"""Q1 frozen Garl FCWD inference. Seal all predictions before scoring."""

from __future__ import annotations

import argparse

# ruff: noqa: ANN401 -- upstream native Garl dynamic model boundary.
import gc
import hashlib
import time
from typing import Any

import h5py
import numpy as np
import torch
import yaml

from e_jepa_ttc.data.garl_official_preprocessing import official_resize_roi, official_square_box
from e_jepa_ttc.efficient_context.garl_input import native_feature
from operational.evttc_rgb_transfer.inputs import _nearest_frame_indices, _normalized_rgb
from operational.evttc_rgb_transfer.model import (
    _NATIVE_SOURCES,
    FullGarl,
    _constructor,
    release_ttc,
)
from operational.sota_eval.fcwd_inputs import FCWDEventReader, _validate_row
from operational.sota_eval.fcwd_run import validate_manifest

from .common import (
    CODE,
    MANIFEST,
    OUT,
    REG,
    V12,
    admission,
    read,
    registrations,
    sha,
    write,
)


def timing(row: dict, clock: np.ndarray) -> dict:
    ends = np.asarray(row["windows_us"], np.int64)[-2:, 1]
    nearest = _nearest_frame_indices(clock, ends)
    if nearest is None:
        return {"available": False, "reason": "RGB_ENDPOINT_WITHOUT_FRAME_WITHIN_1MS"}
    indices, offsets = nearest
    stamps = clock[indices]
    interval = int(stamps[1] - stamps[0])
    available = abs(interval - 100000) <= 1000
    return dict(
        available=available,
        reason="" if available else "RGB_INTERVAL_OUTSIDE_100MS_PLUS_MINUS_1MS",
        indices=indices.tolist(),
        timestamps_us=stamps.tolist(),
        offsets_us=offsets.tolist(),
        interval_us=interval,
        available_us=max(int(row["anchor_us"]), int(stamps[-1])),
    )


def prepare(row: dict, reg: dict, temporal: dict, full: bool) -> tuple[np.ndarray | None, dict]:
    start = time.perf_counter()
    path = _validate_row(row)
    rgb_square = official_square_box(reg["rgb_boxes_xyxy3"][-2:], 1)
    rgb = []
    with h5py.File(path, "r") as h:
        for index in temporal["indices"]:
            # The pinned manifest verifies this HDF5 path is an image Dataset.
            rgb.append(_normalized_rgb(np.asarray(h["blackflys/right/data"][index], np.uint8)))  # pyright: ignore[reportIndexIssue]
    decoded = time.perf_counter()
    # Same linear sampling applied independently to channels avoids a 6-plane full-frame copy.
    resized = np.concatenate(
        [official_resize_roi(frame, rgb_square, (128, 128)).numpy() for frame in rgb]
    )
    del rgb
    cropped = time.perf_counter()
    parts = []
    if full:
        with FCWDEventReader(path) as reader:
            for i, w in enumerate(row["windows_us"][-2:]):
                raw = reader.read_window(*w)
                if not len(raw["t"]):
                    return None, {"reason": "EMPTY_EVENT_ENDPOINT"}
                square = official_square_box(row["boxes_xyxy3"][-2:], i)
                parts.append(native_feature(raw, square, size=128).numpy())
    sensor = np.ascontiguousarray(
        np.concatenate([resized, *parts]) if full else resized, dtype=np.float32
    )
    assert sensor.shape == ((46 if full else 6), 128, 128) and np.isfinite(sensor).all()
    return sensor, dict(
        rgb_read_decode_normalize_ms=(decoded - start) * 1000,
        rgb_crop_ms=(cropped - decoded) * 1000,
        event_read_native_voxels_ms=(time.perf_counter() - cropped) * 1000,
        rgb_square_xyxy=list(rgb_square),
        sensor_sha256=hashlib.sha256(sensor.tobytes()).hexdigest(),
    )


def load(kind: str) -> tuple[Any, dict]:
    if kind == "full":
        wrapper = FullGarl(V12 / "artifacts/evttc_rgb_transfer_20261008/public_garl", CODE, "cpu")
        return wrapper.model, wrapper.bindings
    public = OUT / "public_garl"
    ckpt = public / "paper_visual_only_lhr.pth"
    config = public / "configs/ablation/visual_lhr.yaml"
    for name, digest in _NATIVE_SOURCES.items():
        assert sha(CODE / name) == digest
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert cfg["dataset"]["mode"] == "image_only" and cfg["dataset"]["sync"] == "front"
    assert cfg["model"]["input_feat_num"] == 6 and cfg["model"]["frame_num"] == 2
    assert cfg["model"]["mode"] == "height_ratio" and cfg["dataset"]["window_interval"] == 1
    assert cfg["dataset"]["img_mean"] == [0.485, 0.456, 0.406] and cfg["dataset"]["img_std"] == [
        0.229,
        0.224,
        0.225,
    ]
    for name in ("pretrained_ckpt_rgb", "pretrained_ckpt_event"):
        cfg["model"].pop(name, None)
    # Compare actual download bytes with the Hub's immutable LFS oid metadata.
    metadata = (
        (public / ".cache/huggingface/download/paper_visual_only_lhr.pth.metadata")
        .read_text()
        .splitlines()
    )
    assert metadata[0] == "b676fcdaf26c04bcf896cdb2b208c9c424e8462a"
    assert sha(ckpt) == metadata[1]
    model = _constructor(CODE)(cfg, is_train=False)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.float().eval().requires_grad_(False), dict(
        checkpoint_sha256=sha(ckpt),
        config_sha256=sha(config),
        strict_state_load=True,
        hf_revision=metadata[0],
    )


def run(kind: str) -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    admission()
    regs = registrations()
    rows = validate_manifest(read(MANIFEST))
    directory = OUT / f"garl_{kind}"
    clocks = {}
    coverage = []
    for row in rows:
        path = _validate_row(row)
        if str(path) not in clocks:
            with h5py.File(path, "r") as h:
                clocks[str(path)] = np.asarray(h["blackflys/right/ts"], np.int64)
        t = timing(row, clocks[str(path)])
        reg = regs[row["query_id"]]
        coverage.append(
            dict(
                query_id=row["query_id"],
                sequence_id=row["sequence_id"],
                temporal=t,
                registration_accepted=reg["accepted"],
                available=bool(t["available"] and reg["accepted"]),
                reason=reg.get("reason", "") if not reg["accepted"] else t["reason"],
            )
        )
    write(
        directory / "AVAILABILITY_SEALED.json",
        dict(
            manifest_sha256=sha(MANIFEST),
            registration_sha256=sha(REG / "REGISTRATION_SEALED.json"),
            rows=coverage,
            labels_read=False,
        ),
    )
    t0 = time.perf_counter()
    model, binding = load(kind)
    load_ms = (time.perf_counter() - t0) * 1000
    binding.update(
        input_scope=(
            "FCWD estimated event-assisted registration; "
            "native normalization/crops; not official extrinsics"
        ),
        availability_sha256=sha(directory / "AVAILABILITY_SEALED.json"),
        adapter_sha256=sha(__file__),
        device="cpu",
        threads=1,
        optimizer_updates=0,
    )
    if (directory / "BINDING.json").exists():
        assert read(directory / "BINDING.json") == binding
    write(directory / "BINDING.json", binding)
    write(
        directory / "LOAD_COST.json",
        dict(load_ms=load_ms, resources=admission(), concurrent_V15=True),
    )
    for index, (row, cov) in enumerate(zip(rows, coverage, strict=True)):
        output = directory / "queries" / f"{index:04d}.json"
        if output.exists():
            assert read(output)["binding_sha256"] == sha(directory / "BINDING.json")
            continue
        admission()
        result = dict(
            query_id=row["query_id"],
            sequence_id=row["sequence_id"],
            anchor_us=row["anchor_us"],
            available=cov["available"],
            reason=cov["reason"],
            ttc=None,
            heights=None,
            binding_sha256=sha(directory / "BINDING.json"),
        )
        if cov["available"]:
            started = time.perf_counter()
            sensor, cost = prepare(row, regs[row["query_id"]], cov["temporal"], kind == "full")
            if sensor is None:
                result.update(available=False, reason=cost["reason"])
            else:
                prepared = time.perf_counter()
                with torch.inference_mode():
                    heights, _ = model.forward_test(torch.from_numpy(sensor[None]))
                heights = heights.numpy()
                assert heights.shape == (1, 2) and np.isfinite(heights).all()
                ttc = float(release_ttc(heights, float(model.dT))[0])
                result.update(
                    ttc=ttc if np.isfinite(ttc) else None,
                    heights=heights[0].tolist(),
                    nonfinite_prediction=not np.isfinite(ttc),
                    preprocess=cost,
                    inference_ms=(time.perf_counter() - prepared) * 1000,
                    total_ms=(time.perf_counter() - started) * 1000,
                )
                del sensor, heights
        write(output, result)
        if index % 20 == 0:
            print(kind, index + 1, "/", len(rows), flush=True)
    seal: dict = dict(
        kind=kind,
        population=len(rows),
        query_sha256={p.name: sha(p) for p in sorted((directory / "queries").glob("*.json"))},
        binding_sha256=sha(directory / "BINDING.json"),
        labels_read=False,
        optimizer_updates=0,
    )
    assert len(seal["query_sha256"]) == 630
    write(directory / "PREDICTIONS_SEALED.json", seal)
    del model
    gc.collect()
    print(kind, "SEALED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("visual", "full"), required=True)
    run(parser.parse_args().kind)
