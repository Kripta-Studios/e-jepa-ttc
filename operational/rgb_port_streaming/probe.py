"""Short synthetic GPU parity probe using current event checkpoint weights.

This verifies runtime mechanics, not TTC accuracy or end-to-end latency.
The PAIR head is deterministic random initialization until its endpoint exists.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch

from operational.rgb_port.accounting import atomic_write_json, read_json_shared, sha256_file
from operational.rgb_port.recipe import canonical_sha256, resolved_recipe
from operational.rgb_port.train_heads import _model
from operational.rgb_port.train_producers import build_producer

from .runtime import NativeExperts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    models = []
    parents = {}
    protected = {}
    for name in ("E_A5_MATCHED", "E_C2F_MATCHED"):
        fit = args.run / "fits" / name
        receipt = read_json_shared(fit / "CHECKPOINT_RECEIPT.json")
        path = Path(receipt["checkpoint_path"])
        if sha256_file(path) != receipt["checkpoint_sha256"]:
            raise ValueError("Checkpoint bytes differ from native receipt")
        parents[name] = receipt["checkpoint_sha256"]
        for member in (
            path,
            fit / "CHECKPOINT_RECEIPT.json",
            fit / "CHECKPOINT_POINTER.json",
            fit / "UPDATE_JOURNAL.json",
        ):
            protected[member] = sha256_file(member)
        manifest = args.run / "P_MANIFEST.json"
        recipe = resolved_recipe(
            Path("configs/rgb_port/producers.json"),
            fit_id=name,
            producer_population=read_json_shared(manifest)["population_size"],
            role_manifest_sha256=sha256_file(manifest),
        )
        model = build_producer(recipe)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        models.append(model.cuda().eval())
        del payload
    torch.manual_seed(7)
    pair = _model("PAIR_E_MATCHED").cuda().eval()
    sensor = torch.rand(2, 3, 12, 128, 128, device="cuda")
    delta = torch.full((2, 2), 0.1, device="cuda")
    results = {}
    reference = None
    for name, options in [
        ("native_fp32", {"vectorized": False}),
        ("vectorized_fp32", {}),
        ("vectorized_graph_fp32", {"graphs": True}),
        ("vectorized_encoder_bf16", {"encoder_precision": "bf16"}),
    ]:
        runner = NativeExperts(
            models[0],
            models[1],
            pair,
            modality="event",
            parent_sha256=canonical_sha256(parents),
            **options,
        )
        begin = time.perf_counter()
        actual = runner(sensor, delta)
        torch.cuda.synchronize()
        cold = (time.perf_counter() - begin) * 1000
        if reference is None:
            reference = actual.clone()
        difference = float((actual - reference).abs().max())
        if "bf16" not in name:
            torch.testing.assert_close(actual, reference, atol=2e-4, rtol=2e-4)
        retained = actual.clone()
        for _ in range(3):
            runner(sensor, delta)
        torch.testing.assert_close(actual, retained, atol=0, rtol=0)
        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            runner(sensor, delta)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - tick) * 1000)
        results[name] = {
            "cold_ms": cold,
            "median_batch2_ms": float(np.median(times)),
            "max_feature_abs_difference": difference,
            "finite": bool(torch.isfinite(actual).all()),
        }
        print(name, results[name], flush=True)
        del runner
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()
    if any(sha256_file(p) != digest for p, digest in protected.items()):
        raise RuntimeError("Protected checkpoint state changed during inference probe")
    atomic_write_json(
        args.output,
        {
            "status": "PASS",
            "parents": parents,
            "results": results,
            "scope": "synthetic event inputs; random PAIR; not TTC accuracy or end-to-end latency",
            "optimizer_updates": 0,
            "protected_unchanged": True,
            "gpu": torch.cuda.get_device_name(),
        },
    )


if __name__ == "__main__":
    main()
