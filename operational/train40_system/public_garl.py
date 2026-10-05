"""Admit the published event-only Garl checkpoint without accessing any holdout."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch

from operational.efficient_context.common import atomic_json, digest
from operational.train40_system.data_audit import OUTPUT


def run(output: Path, code_root: Path, raw_root: Path) -> None:
    """Strict-load every published parameter and execute one TRAIN sensor-only input on CPU."""
    import pandas as pd
    import yaml

    from e_jepa_ttc.efficient_context.garl_input import inference_record
    from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool

    directory = output / "public_garl"
    path = directory / "paper_event_only_lhr.pth"
    expected = "fcaf9be47e2dafc6f73c6c3ebd102595ae06119dcae78aea698a42627b2b4fef"
    if digest(path) != expected:
        raise ValueError("Published Garl checkpoint bytes changed")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    config_path = directory / "configs/ablation/event_lhr.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for name in ("pretrained_ckpt_event", "pretrained_ckpt_rgb"):
        config["model"].pop(name, None)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(code_root.resolve()))
    constructor = importlib.import_module("garl_ttc.models.ttc_network").TTCNetwork
    model = constructor(config, is_train=False).float().eval().requires_grad_(False)
    weights = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    with np.load(output / "TRAIN40_INDEX.npz", allow_pickle=False) as stored:
        sequence = str(stored["sequences"][0])
        windows = stored["windows_us"][0, 1:]
        selected = stored["endpoint_indices"][0]
    record = pd.read_parquet(output / "TRAIN40_ROWS.parquet", columns=["boxes_xyxy"]).iloc[0]
    boxes = np.asarray(record["boxes_xyxy"].tolist())[selected]
    pool = ReaderPool()
    try:
        sensor = inference_record(
            {"sequence_id": sequence, "event_windows_us": windows, "boxes_xyxy": boxes},
            pool,
            raw_root / "data/train",
        )
        with torch.inference_mode():
            heights, _ = model.forward_test(sensor[None])
        if heights.shape != (1, 2) or not torch.isfinite(heights).all():
            raise ValueError("Published Garl output contract failed")
        atomic_json(
            output / "PUBLIC_GARL_REAL_INPUT_ADMISSION.json",
            {
                "status": "PASSED",
                "checkpoint_sha256": expected,
                "config_sha256": digest(config_path),
                "source_files": [
                    {"path": str(code_root / relative), "sha256": digest(code_root / relative)}
                    for relative in ("garl_ttc/models/ttc_network.py", "garl_ttc/models/resnet.py")
                ],
                "state_keys_loaded_strictly": len(weights),
                "parameters": sum(p.numel() for p in model.parameters()),
                "input_shape": list(sensor.shape),
                "dtype": "float32",
                "device": "cpu",
                "TRAIN_ordinal": 0,
                "sequence_id": sequence,
                "predicted_visible_heights": heights.numpy().tolist(),
                "optimizer_updates": 0,
                "holdout_targets_read": False,
                "exact_public_checkpoint_training_sequence_manifest_verified": False,
                "not_a_generalization_metric": True,
            },
        )
    finally:
        pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve(), args.code_root.resolve(), args.raw_root.resolve())
