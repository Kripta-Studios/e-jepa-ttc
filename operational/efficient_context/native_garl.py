"""Native pinned Garl hardware admission; no scientific optimizer before freeze."""

from __future__ import annotations

import argparse
import gc
import importlib
import sys
from collections.abc import Callable
from pathlib import Path

from .common import ROOT, Campaign, atomic_json, digest, read


def native_config(c: Campaign) -> dict:
    """Disable every external eAP checkpoint without changing the event-only recipe."""
    import yaml

    path = Path(c.local["garl_code_candidate"]) / "configs/ablation/event_lhr.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("pretrained_ckpt_rgb", "pretrained_ckpt_event"):
        config["model"].pop(key, None)
    config["training_settings"]["resume"] = False
    config["training_settings"]["num_threads"] = 0
    return config


def model_class(c: Campaign) -> Callable:
    """Import only the pinned model code, with no repository writes or downloads."""
    sys.dont_write_bytecode = True
    sys.path.insert(0, c.local["garl_code_candidate"])
    return importlib.import_module("garl_ttc.models.ttc_network").TTCNetwork


def profile(c: Campaign) -> dict:
    """Profile128 then fixed descending microbatches; maintain effective batch128."""
    import psutil
    import torch

    c.require_resources()
    if (c.out / "WRITER.lock").exists():
        owner = read(c.out / "WRITER.lock")
        if psutil.pid_exists(owner["pid"]):
            raise InterruptedError("another heavy campaign writer is active")
    admission = read(c.out / "garl/ADMISSION.json")
    if admission["reasons"]:
        return admission
    if not torch.cuda.is_available():
        result = {"status": "BLOCKED_DEPENDENCY", "reason": "CUDA_UNAVAILABLE"}
        atomic_json(c.out / "garl/MICROBATCH_PROFILE.json", result)
        return result
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    ctor = model_class(c)
    attempts = []
    selected = None
    for batch in (128, 64, 32, 16, 8):
        c.require_resources()
        model = optimizer = x = heights = target = losses = loss = None
        try:
            torch.manual_seed(7)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = ctor(native_config(c), is_train=True).float().cuda().train()
            parameters = sum(p.numel() for p in model.parameters())
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0, foreach=False)
            x = torch.randn(batch, 40, 128, 128, device="cuda")
            target = torch.ones(batch, device="cuda") * 2
            heights = torch.ones(batch, 2, device="cuda") * 30
            _, _, losses, _ = model.forward_train(
                x, target, visible_height_target=heights, epoch_idx=1
            )
            loss = torch.stack(list(losses.values())).sum()
            if not torch.isfinite(loss):
                raise ArithmeticError("synthetic native source loss not finite")
            loss.backward()
            # Reserve Adam's persistent states without any optimizer update.
            states = [torch.empty_like(p) for p in model.parameters() for _ in range(2)]
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            peak = torch.cuda.max_memory_allocated()
            admitted = c.check() and free >= 1024**3
            attempts.append(
                {
                    "microbatch": batch,
                    "peak_vram_bytes": peak,
                    "gpu_free_bytes": free,
                    "gpu_total_bytes": total,
                    "parameters": parameters,
                    "reserved_adam_state_bytes": sum(v.numel() * v.element_size() for v in states),
                    "admitted": admitted,
                    "optimizer_updates": 0,
                }
            )
            del states
            if admitted:
                selected = batch
                break
        except torch.cuda.OutOfMemoryError as error:
            attempts.append(
                {
                    "microbatch": batch,
                    "admitted": False,
                    "reason": "CUDA_OOM",
                    "error": str(error),
                    "optimizer_updates": 0,
                }
            )
        finally:
            del model, optimizer, x, heights, target, losses, loss
            gc.collect()
            torch.cuda.empty_cache()
    result = {
        "status": "ADMITTED" if selected else "BLOCKED_DEPENDENCY",
        "selected_microbatch": selected,
        "effective_batch": 128,
        "gradient_accumulation": 128 // selected if selected else None,
        "batchnorm_equivalent_to_batch128": selected == 128,
        "hardware_adaptation": selected != 128,
        "attempts": attempts,
        "optimizer_updates": 0,
        "external_pretrained_eap_checkpoint_loaded": False,
        "upstream_commit": c.config["garl_upstream_commit"],
        "upstream_config_blob": c.config["garl_config_blob"],
        "implementation_sha256": digest(Path(__file__)),
    }
    atomic_json(c.out / "garl/MICROBATCH_PROFILE.json", result)
    return result


def main() -> int:
    """Run actual hardware profiling in its own compatible interpreter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("profile",))
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/campaign/efficient_context_v1.json"
    )
    args = parser.parse_args()
    result = profile(Campaign(args.protocol))
    print(result["status"], result.get("selected_microbatch"), flush=True)
    return 0 if result["status"] == "ADMITTED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
