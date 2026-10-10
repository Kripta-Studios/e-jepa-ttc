"""Train a small A5-only feature student on TRAIN40, with disjoint sequence folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from operational.efficient_context.common import digest
from operational.train40_system.durable_io import atomic_json

INPUTS = (0, 1, 2, 3, 4, 8)
TARGETS = (5, 6, 7, 9, 10)


class FeatureStudent(nn.Module):
    """Predict C2F/PAIR coordinates from A5, then rebuild exact algebraic differences."""

    mean: Tensor
    scale: Tensor

    def __init__(self, mean: Tensor, scale: Tensor) -> None:
        super().__init__()
        if mean.shape != (17,) or scale.shape != (17,) or not bool((scale > 0).all()):
            raise ValueError("positive PHASE17 normalizer required")
        self.register_buffer("mean", mean.float().clone())
        self.register_buffer("scale", scale.float().clone())
        self.net = nn.Sequential(
            nn.Linear(6, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 5)
        )

    def forward(self, raw: Tensor) -> Tensor:
        """Consume only A5 columns; never read teacher-only coordinates at inference."""
        x = ((raw - self.mean) / self.scale)[..., list(INPUTS)]
        inferred = self.net(x) * self.scale[list(TARGETS)] + self.mean[list(TARGETS)]
        columns = list(raw.unbind(-1))
        for index, value in zip(TARGETS, inferred.unbind(-1), strict=True):
            columns[index] = value
        columns[11:14] = [
            columns[10] - columns[8],
            columns[10] - columns[9],
            columns[9] - columns[8],
        ]
        columns[14:17] = [value.abs() for value in columns[11:14]]
        return torch.stack(columns, -1)


def load_student(path: Path, *, int8_cpu: bool = False) -> nn.Module:
    """Restore a versioned student; optional actual dynamic INT8 Linear kernels on CPU."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["schema"] != "a5_feature_student_v1":
        raise ValueError("incompatible distilled student")
    model = FeatureStudent(payload["mean"], payload["scale"])
    model.load_state_dict(payload["state"], strict=True)
    model.eval()
    if int8_cpu:
        from torch.ao.quantization import quantize_dynamic

        model.net = quantize_dynamic(model.net, {nn.Linear}, dtype=torch.qint8)
    return model


def train(campaign: Path, output: Path, *, updates: int = 600, seed: int = 7) -> None:
    """Fixed CPU pilot; holdout tests student imitation, not unseen teacher generalization."""
    if updates < 1:
        raise ValueError("positive update count required")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    begin = time.perf_counter()
    manifest = json.loads((campaign / "H8_FEATURE_MANIFEST.json").read_text(encoding="utf-8"))
    feature_path = campaign / "H8_FEATURES.npz"
    expected = next(x["sha256"] for x in manifest["files"] if x["path"] == feature_path.name)
    if digest(feature_path) != expected:
        raise ValueError("teacher feature digest changed")
    with np.load(feature_path, allow_pickle=False) as stored:
        features, history = stored["features"], stored["history"]
    with np.load(campaign / "TRAIN40_INDEX.npz", allow_pickle=False) as stored:
        sequences = stored["sequences"].astype(str)
    if len(sequences) != len(history):
        raise ValueError("sequence labels and feature rows are not aligned")
    ordered = sorted(set(sequences), key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    heldout = set(ordered[: max(1, len(ordered) // 5)])
    val_mask = np.isin(sequences, list(heldout))
    train_ids, val_ids = np.unique(history[~val_mask]), np.unique(history[val_mask])
    train_ids, val_ids = train_ids[train_ids >= 0], val_ids[val_ids >= 0]
    if not len(train_ids) or not len(val_ids) or np.intersect1d(train_ids, val_ids).size:
        raise ValueError("nonempty disjoint observation folds required")
    training = torch.from_numpy(features[train_ids])
    validation = torch.from_numpy(features[val_ids])
    mean, scale = training.mean(0), training.std(0).clamp_min(1e-5)
    model = FeatureStudent(mean, scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    receipt = {
        "schema": "a5_feature_student_v1",
        "status": "RUNNING",
        "seed": seed,
        "requested_updates": updates,
        "device": "cpu",
        "gpu_seconds": 0,
        "teacher_features_sha256": expected,
        "index_sha256": digest(campaign / "TRAIN40_INDEX.npz"),
        "source_sha256": digest(Path(__file__)),
        "validation_sequences": sorted(heldout),
        "training_sequences": sorted(set(sequences) - heldout),
        "training_observations": len(training),
        "validation_observations": len(validation),
        "selection": "fixed updates; no DEV32, FCWD or test12 selection",
        "scope": "student imitation holdout; teacher producers were trained on all TRAIN40",
    }
    atomic_json(output / "RESULT.json", receipt)
    losses = []
    model.train()
    for step in range(updates):
        raw = training[torch.randint(len(training), (512,))]
        prediction = model(raw)
        error = (prediction[:, list(TARGETS)] - raw[:, list(TARGETS)]) / scale[list(TARGETS)]
        loss = torch.nn.functional.smooth_l1_loss(error, torch.zeros_like(error))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            losses.append({"step": step + 1, "loss": float(loss.detach())})
    model.eval()
    torch.save(
        {
            "schema": "a5_feature_student_v1",
            "mean": mean,
            "scale": scale,
            "state": model.state_dict(),
            "seed": seed,
            "updates": updates,
        },
        output / "student.pt",
    )
    quantized = load_student(output / "student.pt", int8_cpu=True)
    result = {}
    with torch.inference_mode():
        for name, candidate in (("fp32", model), ("int8_cpu", quantized)):
            errors = []
            for raw in validation.split(2048):
                errors.append((candidate(raw)[:, list(TARGETS)] - raw[:, list(TARGETS)]).abs())
            result[name] = {"feature_mae": torch.cat(errors).mean(0).tolist()}
            sample = validation[:1]
            for _ in range(10):
                candidate(sample)
            times = []
            for _ in range(100):
                started = time.perf_counter()
                candidate(sample)
                times.append((time.perf_counter() - started) * 1000)
            result[name].update(
                p50_ms=float(np.median(times)), p95_ms=float(np.quantile(times, 0.95))
            )
    atomic_json(
        output / "RESULT.json",
        {
            **receipt,
            "status": "COMPLETE_PILOT",
            "optimizer_updates": updates,
            "loss_samples": losses,
            "validation": result,
            "wall_seconds": time.perf_counter() - begin,
            "student_sha256": digest(output / "student.pt"),
        },
    )


def main() -> None:
    """Train only from the frozen TRAIN40 feature cache, without reserving GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    train(args.campaign, args.output, updates=args.updates, seed=args.seed)


if __name__ == "__main__":
    main()
