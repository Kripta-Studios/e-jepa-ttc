"""Pre-extraction CUDA FP32/CPU cached readout parity; never optimizer updates."""

import copy
from pathlib import Path

import torch

from e_jepa_ttc.range_readout.operators import ResidualSpec
from e_jepa_ttc.range_readout.readout import VectorReadout, wrap
from operational.rgb_port.train_producers import _producer_loss, make_event_producer_source

from .cache_readout import capture, loss, parent
from .common import OWNER, RUN, read, resources, utc, write


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    snapshot = resources(RUN, starting=True)
    if not snapshot["allowed"]:
        raise InterruptedError("Resource guard")
    for p in snapshot["processes"]:
        command = " ".join(p["cmdline"]).lower()
        if any(x in command for x in ("train_producer", "resident", "rgb_port.run", "r1_resume")):
            raise RuntimeError("Foreign scientific worker active; preserve its ownership")
    root = Path(
        read(OWNER / "fits/E_A5_MATCHED/RUN_PROVENANCE.json")["source_identity"][
            "event_source_root"
        ]
    )
    source = make_event_producer_source(OWNER / "P_MANIFEST.json", event_source_root=root)
    source._inputs.limit = 384 * 1024**2
    batch = source.batch([0, 1, 2, 3], "event").to(torch.device("cuda"))
    original, cfg = parent()
    model = wrap(original, ResidualSpec(0.20), reset=False).cuda()
    model.eval()
    with torch.no_grad():
        values, raw = capture(model, batch, labels=True)
    raw_loss, _ = _producer_loss(model, batch, cfg)
    raw_grad = torch.autograd.grad(raw_loss, tuple(model.residual.scorer.parameters()))
    cached_model = VectorReadout(copy.deepcopy(model).cpu())
    cpu = {k: v.cpu() for k, v in values.items()}
    cached = cached_model({k: v for k, v in cpu.items() if not k.startswith(("target_", "dino_"))})
    differences = {}
    for key in (
        "ttc_mean_seconds",
        "ttc_log_variance",
        "inverse_ttc_mean",
        "collision_logits",
        "pair_tokens",
    ):
        a, b = getattr(raw, key).cpu(), getattr(cached, key)
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
        differences[key] = float((a - b).detach().abs().max())
    cached_loss, _ = loss(cached_model, cpu, cfg)
    cached_grad = torch.autograd.grad(
        cached_loss, tuple(cached_model.model.residual.scorer.parameters())
    )
    torch.testing.assert_close(raw_loss.cpu(), cached_loss, atol=2e-5, rtol=2e-5)
    for a, b in zip(raw_grad, cached_grad, strict=True):
        torch.testing.assert_close(a.cpu(), b, atol=2e-5, rtol=2e-5)
    differences["gradient"] = max(
        float((a.cpu() - b).abs().max()) for a, b in zip(raw_grad, cached_grad, strict=True)
    )
    result = {
        "status": "PASS",
        "utc": utc(),
        "device": torch.cuda.get_device_name(),
        "extraction_precision": "cuda_float32_tf32_false",
        "training_precision": "cpu_float32",
        "optimizer_updates": 0,
        "atol": 2e-5,
        "rtol": 2e-5,
        "max_abs_errors": differences,
        "resource_snapshot": snapshot,
        "V13_state": read(OWNER / "RGB_PORT_STATE.json")["campaign_status"],
        "R1_state": read(OWNER / "r1_continuation_retry_02/STATE.json")["status"],
        "scope": "P-only device parity; prospective GPU extraction replaces CPU after timing pilot",
    }
    write(RUN / "qa/DEVICE_PARITY.json", result)
    source.close()
    print(result)


if __name__ == "__main__":
    main()
