"""Bounded vector extraction with frozen A5; no optimizer or selection."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Any

# ruff: noqa: ANN401 -- bridge to the native heterogeneous producer batch
import numpy as np
import torch

from e_jepa_ttc.data.object_event_v4 import BoxGeometryTargets, box_geometry_targets
from e_jepa_ttc.distillation.dinov3_relational import local_cosine_relation_maps
from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig, causal_scale_ttc_loss
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from e_jepa_ttc.range_readout.readout import VectorReadout
from operational.rgb_port.infer_experts import Fragments, delta_matrix
from operational.rgb_port.train_producers import make_event_producer_source

from .common import OWNER, ROOT, RUN, freeze, read, resources, sha, utc, write


def parent() -> tuple[CausalScaleTTC, CausalScaleTTCLossConfig]:
    """Load only the admitted final matched A5, never TRAIN40 weights."""
    path = OWNER / "fits/E_A5_MATCHED/checkpoint_last.pt"
    if sha(path) != read(RUN / "ADMISSION.json")["parents"]["E_A5_MATCHED"]:
        raise ValueError("A5 identity changed")
    value = torch.load(path, map_location="cpu", weights_only=False)
    model = CausalScaleTTC(CausalScaleTTCConfig(**value["identity"]["model_config"]))
    model.load_state_dict(value["model_state_dict"], strict=True)
    config = read(ROOT / "configs/rgb_port/producers.json")["producers"]["E_A5_MATCHED"][
        "loss_config"
    ]
    return model.float().eval().requires_grad_(False), CausalScaleTTCLossConfig(**config)


def capture(model: CausalScaleTTC, batch: Any, *, labels: bool) -> tuple[dict, Any]:
    """Cache label-free vector inputs; training targets remain in separate keys."""
    captured = []
    hook = model.residual.register_forward_pre_hook(lambda _m, args: captured.append(args))
    delta = delta_matrix(batch.delta_t_s, batch.events.shape[1])
    try:
        out = model(batch.events.float(), delta, return_dense_features=labels)
    finally:
        hook.remove()
    if len(captured) != 1:
        raise ValueError("Unexpected residual route")
    args = captured[0]
    values = {
        "geometry_tokens": out.geometry_tokens,
        "forward_transport": args[2],
        "reverse_transport": args[3],
        "analytic": out.analytic_log_height_ratio,
        "delta": delta,
        "sensor_support": out.sensor_support,
        "visible_height": out.visible_height_normalized,
        "height_normalized": out.diagnostics["raw_foreground_height_normalized"],
        "width_normalized": out.visible_width_normalized,
        "centroid_x_normalized": out.diagnostics["foreground_centroid_x"],
        "centroid_y_normalized": out.diagnostics["foreground_centroid_y"],
        "foreground_fraction": out.diagnostics["foreground_fraction"],
        "raw_foreground_ratio": out.diagnostics["raw_foreground_log_height_ratio"],
        "raw_foreground_width_ratio": out.diagnostics["raw_foreground_log_width_ratio"],
        "log_height_correction": out.diagnostics["log_height_correction"],
        "transport_raw_features": out.transport_raw_features,
    }
    if labels:
        valid = torch.isfinite(batch.target_ttc_s) & (batch.target_ttc_s != 0)
        endpoints = torch.ones(batch.boxes_xyxy.shape[:2], dtype=torch.bool, device=delta.device)
        endpoints[:, 0] = False
        geometry = box_geometry_targets(
            batch.boxes_xyxy,
            height=batch.events.shape[-2],
            width=batch.events.shape[-1],
            endpoint_valid=endpoints & valid[:, None],
        )
        values.update({"target_" + f.name: getattr(geometry, f.name) for f in fields(geometry)})
        values["target_ttc"] = batch.target_ttc_s
        rel = local_cosine_relation_maps(out.endpoint_dense_features[:, -2:])
        mask = rel.valid & batch.dinov3_relation_valid.bool()
        difference = (rel.values.float() - batch.dinov3_relation_targets.float()).abs()
        values["dino_numerator"] = torch.where(mask, difference, 0).flatten(1).sum(1)
        values["dino_count"] = mask.flatten(1).sum(1)
    return {k: v.detach().clone() for k, v in values.items()}, out


def loss(
    model: VectorReadout, values: dict, config: CausalScaleTTCLossConfig
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Recompute every residual-dependent native component and global top-k."""
    output = model({k: v for k, v in values.items() if not k.startswith(("target_", "dino_"))})
    geometry = BoxGeometryTargets(
        **{f.name: values["target_" + f.name] for f in fields(BoxGeometryTargets)}
    )
    base = causal_scale_ttc_loss(
        output,
        target_ttc_seconds=values["target_ttc"],
        delta_t_s=values["delta"],
        risk_thresholds_s=model.config.risk_thresholds_s,
        target_valid=torch.isfinite(values["target_ttc"]) & (values["target_ttc"] != 0),
        target_geometry=geometry,
        config=config,
    )
    dino = values["dino_numerator"].sum() / values["dino_count"].sum().clamp_min(1)
    return base.total + 8 * dino, {**base.components, "dinov3_relational_raw": dino}


def extract(role: str, device: str, *, max_shards: int | None = None) -> None:
    """Resume CPU/GPU FP32 extraction in 128-row shards, checking resources."""
    if role not in ("P", "H", "V"):
        raise ValueError("Unknown role")
    if not (RUN / "qa/READOUT_PARITY.json").exists():
        raise ValueError("Raw/cache parity is required before full extraction")
    folder = RUN / "vector_cache" / role
    if device == "cuda":
        if read(RUN / "qa/DEVICE_PARITY.json")["status"] != "PASS":
            raise ValueError("GPU extraction needs device parity")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    source_root = Path(
        read(OWNER / "fits/E_A5_MATCHED/RUN_PROVENANCE.json")["source_identity"][
            "event_source_root"
        ]
    )
    source = make_event_producer_source(
        OWNER / f"{role}_MANIFEST.json", event_source_root=source_root
    )
    source._inputs.limit = 384 * 1024**2
    model, _ = parent()
    model.to(device)
    binding = {
        "role": role,
        "manifest_sha256": sha(OWNER / f"{role}_MANIFEST.json"),
        "parent_sha256": sha(OWNER / "fits/E_A5_MATCHED/checkpoint_last.pt"),
        "device": device,
        "precision": "float32",
        "shard_size": 128,
        "capture_source": sha(Path(__file__)),
        "query_order": "native_manifest",
        "targets_cached": role == "P",
    }
    store = Fragments(folder, binding)
    started = __import__("time").perf_counter()
    executed = 0
    try:
        for begin in range(0, source.population_size, 128):
            part = begin // 128
            if store.read(part) is not None:
                continue
            snapshot = resources(RUN, starting=True)
            snapshot["extraction_device"] = device
            snapshot["gpu_used_by_v15"] = device == "cuda"
            if device == "cuda":
                for process in snapshot["processes"]:
                    command = " ".join(process["cmdline"]).lower()
                    if any(
                        x in command
                        for x in ("train_producer", "resident", "rgb_port.run", "r1_resume")
                    ):
                        raise InterruptedError("Foreign scientific owner became active")
            write(folder / "RESOURCES.json", snapshot)
            if not snapshot["allowed"] or (RUN / "PAUSE").exists():
                raise InterruptedError("Vector extraction resource pause")
            slices = []
            for index in range(begin, min(begin + 128, source.population_size), 8):
                ids = list(range(index, min(index + 8, begin + 128, source.population_size)))
                batch = source.batch(ids, "event").to(torch.device(device))
                with torch.no_grad():
                    values, _ = capture(model, batch, labels=role == "P")
                slices.append({k: v.cpu().numpy() for k, v in values.items()})
                del batch, values
            arrays = {k: np.concatenate([s[k] for s in slices]) for k in slices[0]}
            arrays["sample_token"] = np.asarray(
                source._tokens[begin : begin + len(arrays["delta"])]
            )
            for k, v in arrays.items():
                if v.dtype.kind == "f" and not np.isfinite(v).all():
                    raise FloatingPointError(f"Nonfinite cache {role}/{part}/{k}")
            store.write(part, arrays)
            executed += 1
            write(
                folder / "PROGRESS.json",
                {
                    "completed_rows": begin + len(arrays["delta"]),
                    "total_rows": source.population_size,
                    "utc": utc(),
                    "elapsed_this_call_s": __import__("time").perf_counter() - started,
                },
            )
            if max_shards is not None and executed >= max_shards:
                return
        freeze(
            folder / "COMPLETE.json",
            {
                "status": "COMPLETE",
                "rows": source.population_size,
                "binding": binding,
                "optimizer_updates": 0,
            },
        )
    finally:
        source.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=["P", "H", "V"])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-shards", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    extract(args.role, args.device, max_shards=args.max_shards)


if __name__ == "__main__":
    main()
