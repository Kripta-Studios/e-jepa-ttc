"""Frozen matched A5 vertical features, without another encoder or target inputs."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from operational.range_transfer.admission import revalidate
from operational.range_transfer.common import freeze, read, resources, sha, write
from operational.rgb_port.infer_experts import delta_matrix
from operational.rgb_port.train_producers import EventProducerSource, make_event_producer_source

from .run import OLD, OWNER, ROOT, RUN, gpu_charge


def native_parent() -> CausalScaleTTC:
    path = OWNER / "fits/E_A5_MATCHED/checkpoint_last.pt"
    if sha(path) != read(OLD / "ADMISSION.json")["parents"]["E_A5_MATCHED"]:
        raise ValueError("Wrong matched A5 parent")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = CausalScaleTTC(CausalScaleTTCConfig(**payload["identity"]["model_config"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if model.temporal_channel_gate is not None or model.transport_encoder is not None:
        raise ValueError("Registered A5 encoder route changed")
    return model.float().eval().requires_grad_(False)


def source_for(role: str) -> EventProducerSource:
    source_root = Path(
        read(OWNER / "fits/E_A5_MATCHED/RUN_PROVENANCE.json")["source_identity"][
            "event_source_root"
        ]
    )
    source = make_event_producer_source(
        OWNER / f"{role}_MANIFEST.json", event_source_root=source_root
    )
    source._inputs.limit = 384 * 1024**2
    return source


def source_binding() -> dict:
    paths = [
        Path(__file__),
        ROOT / "docs/representation_audit_20261012/PROTOCOL_G_SPATIAL_GEOMETRY.md",
        ROOT / "src/e_jepa_ttc/models/causal_scale_ttc.py",
        ROOT / "operational/rgb_port/train_producers.py",
    ]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def pool(dense: torch.Tensor, batch: int, steps: int) -> torch.Tensor:
    if dense.shape != (batch * steps, 64, 32, 32):
        raise ValueError("Unregistered encoder map shape")
    return functional.adaptive_avg_pool2d(dense, (8, 1)).flatten(1).reshape(batch, steps, 512)


def qa() -> None:
    revalidate(read(OLD / "ADMISSION.json"))
    source = source_for("P")
    try:
        batch = source.batch([0, 1, 2, 3], "event")
        model = native_parent()
        b, t, c, h, w = batch.events.shape
        with torch.inference_mode():
            direct = model.encoder.features(batch.events.reshape(b * t, c, h, w).float())
            whole = model(
                batch.events.float(), delta_matrix(batch.delta_t_s, t), return_dense_features=True
            )
            if whole.endpoint_dense_features is None:
                raise ValueError("No native map")
            if not torch.equal(
                direct.reshape_as(whole.endpoint_dense_features), whole.endpoint_dense_features
            ):
                raise AssertionError("Direct encoder differs from full parent forward")
            expected = pool(direct, b, t)
            with gpu_charge("spatial_cache_QA", reserve=120):
                model.to("cuda")
                actual = pool(
                    model.encoder.features(batch.events.to("cuda").reshape(b * t, c, h, w).float()),
                    b,
                    t,
                ).cpu()
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        freeze(
            RUN / "qa/SPATIAL_CACHE_PASS.json",
            {
                "status": "PASS",
                "sources": source_binding(),
                "role": "P",
                "queries": 4,
                "native_dense_exact": True,
                "cpu_gpu_atol": 1e-5,
                "cpu_gpu_rtol": 1e-5,
                "cpu_gpu_max_abs": float((actual - expected).abs().max()),
                "optimizer_updates": 0,
            },
        )
    finally:
        source.close()


def extract(role: str) -> None:
    if role not in ("P", "H", "V"):
        raise ValueError("Unknown role")
    if (
        role == "V"
        and read(RUN / "SPATIAL_REPLICAS_DECISION.json")["status"] != "HEAD_INTEGRATION_ADMITTED"
    ):
        raise ValueError("V extraction requires admitted parents")
    if read(RUN / "qa/SPATIAL_CACHE_PASS.json")["sources"] != source_binding():
        raise ValueError("Spatial cache parity required")
    revalidate(read(OLD / "ADMISSION.json"))
    source = source_for(role)
    folder = RUN / "spatial_cache" / role
    folder.mkdir(parents=True, exist_ok=True)
    binding = {
        "role": role,
        "manifest_sha256": sha(OWNER / f"{role}_MANIFEST.json"),
        "parent_sha256": read(OLD / "ADMISSION.json")["parents"]["E_A5_MATCHED"],
        "sources": source_binding(),
        "precision": "float32_TF32off",
        "device": "cuda",
        "input": "raw admitted event tensors only",
        "targets_cached": False,
    }
    freeze(folder / "BINDING.json", binding)
    model = native_parent()
    try:
        for part in range(math.ceil(source.population_size / 128)):
            path = folder / f"part_{part:06d}.npz"
            if path.with_suffix(".json").exists():
                if sha(path) != read(path.with_suffix(".json"))["sha256"]:
                    raise ValueError("Spatial cache changed")
                continue
            snapshot = resources(RUN, starting=True)
            if not snapshot["allowed"] or (RUN / "PAUSE").exists():
                raise InterruptedError("Spatial extraction resource pause")
            begin = part * 128
            count = min(128, source.population_size - begin)
            tokens = np.asarray(source._tokens[begin : begin + count])
            original_path = OLD / f"vector_cache/{role}/part_{part:06d}.npz"
            if sha(original_path) != read(original_path.with_suffix(".json"))["sha256"]:
                raise ValueError("Original vector cache changed")
            with np.load(original_path, allow_pickle=False) as arc:
                if not np.array_equal(tokens, arc["sample_token"]):
                    raise ValueError("Spatial/original cohort order differs")
            with gpu_charge(f"spatial_extract_{role}_{part}", reserve=120):
                model.to("cuda")
                pieces = []
                for start in range(begin, begin + count, 8):
                    batch = source.batch(list(range(start, min(start + 8, begin + count))), "event")
                    b, t, c, h, w = batch.events.shape
                    with torch.inference_mode():
                        dense = model.encoder.features(
                            batch.events.to("cuda").reshape(b * t, c, h, w).float()
                        )
                        pieces.append(pool(dense, b, t).cpu().numpy())
                values = np.concatenate(pieces)
                if not np.isfinite(values).all():
                    raise FloatingPointError("Nonfinite spatial features")
                temporary = path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, sample_token=tokens, height_features=values)
                temporary.replace(path)
                write(
                    path.with_suffix(".json"),
                    {
                        "sha256": sha(path),
                        "rows": count,
                        "original_vector_sha256": sha(original_path),
                    },
                )
            write(
                folder / "PROGRESS.json",
                {"rows": begin + count, "total": source.population_size, "part": part},
            )
            if part % 10 == 0:
                print(role, begin + count, source.population_size, flush=True)
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
    parser.add_argument("stage", choices=("qa", "P", "H", "V"))
    args = parser.parse_args()
    lock = RUN / "PROCESS.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.stage == "qa":
            qa()
        else:
            extract(args.stage)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
