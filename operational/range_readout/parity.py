"""Pre-fit P-only forward/gradient and frozen-input identity checks; zero updates."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import cast

import torch

from e_jepa_ttc.range_readout.operators import ResidualSpec
from e_jepa_ttc.range_readout.readout import EqualSlopeResidual, VectorReadout, wrap
from e_jepa_ttc.rgb_port.features import pair133
from operational.rgb_port.train_producers import _producer_loss, make_event_producer_source

from .cache_readout import capture, loss, parent
from .common import OWNER, ROOT, RUN, read, sha, write


def main() -> None:
    """Compare raw and cached paths with identical dropout RNG before any R fit."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source_root = Path(
        read(OWNER / "fits/E_A5_MATCHED/RUN_PROVENANCE.json")["source_identity"][
            "event_source_root"
        ]
    )
    source = make_event_producer_source(OWNER / "P_MANIFEST.json", event_source_root=source_root)
    source._inputs.limit = 384 * 1024**2
    model, cfg = parent()
    batch = source.batch([0, 1, 2, 3], "event")
    model = wrap(model, ResidualSpec(0.05), reset=False)
    checks = {}
    max_errors = {}
    for bound in (0.05, 0.20):
        cast(EqualSlopeResidual, model.residual).spec = ResidualSpec(bound)
        model.residual.eval()
        with torch.no_grad():
            values, raw = capture(model, batch, labels=True)
            cached = VectorReadout(model)(
                {k: v for k, v in values.items() if not k.startswith(("target_", "dino_"))}
            )
        names = (
            "ttc_mean_seconds",
            "ttc_log_variance",
            "inverse_ttc_mean",
            "inverse_ttc_log_variance",
            "collision_logits",
            "known_mask",
            "log_height_ratio",
            "pair_log_height_ratio",
            "residual_log_height_ratio",
            "pair_ttc_seconds",
            "pair_tokens",
            "sensor_support",
        )
        for name in names:
            left, right = getattr(raw, name), getattr(cached, name)
            torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-5)
            max_errors[f"{bound}/{name}"] = float((left.float() - right.float()).abs().max())
        model.residual.scorer.train()
        torch.manual_seed(781)
        raw_loss, _ = _producer_loss(model, batch, cfg)
        parameters = tuple(model.residual.scorer.parameters())
        raw_grad = torch.autograd.grad(raw_loss, parameters)
        torch.manual_seed(781)
        cached_loss, _ = loss(VectorReadout(model), values, cfg)
        cached_grad = torch.autograd.grad(cached_loss, parameters)
        torch.testing.assert_close(raw_loss, cached_loss, atol=2e-5, rtol=2e-5)
        for a, b in zip(raw_grad, cached_grad, strict=True):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
        max_errors[f"{bound}/loss"] = float((raw_loss - cached_loss).detach().abs())
        max_errors[f"{bound}/gradient"] = max(
            float((a - b).abs().max()) for a, b in zip(raw_grad, cached_grad, strict=True)
        )
        checks[f"{bound}/raw_cache_gradient"] = True
    # Change only scorer; pair tokens/support/133 input remain bit-identical.
    model.residual.eval()
    with torch.no_grad():
        first = VectorReadout(model)(
            {k: v for k, v in values.items() if not k.startswith(("target_", "dino_"))}
        )
        cast(torch.nn.Linear, model.residual.scorer[-1]).weight.add_(0.2)
        second = VectorReadout(model)(
            {k: v for k, v in values.items() if not k.startswith(("target_", "dino_"))}
        )
    assert torch.equal(first.pair_tokens, second.pair_tokens)
    assert torch.equal(first.sensor_support, second.sensor_support)
    support = first.sensor_support
    supports = torch.stack((support[:, -1], torch.minimum(support[:, -2], support[:, -1])), -1)
    assert torch.equal(
        pair133(first.pair_tokens[:, -1], values["delta"][:, -1], supports),
        pair133(second.pair_tokens[:, -1], values["delta"][:, -1], supports),
    )
    assert first.known_mask.shape == (len(batch.target_ttc_s),)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("residual.scorer.") for n in trainable)
    checks["pair133_invariant"] = checks["known_query_shape"] = checks["scorer_only_trainable"] = (
        True
    )
    # Initial analytic output is identical after the prospectively paired reset.
    a, _ = parent()
    b = copy.deepcopy(a)
    a = wrap(a, ResidualSpec(0.05), reset=True)
    b = wrap(b, ResidualSpec(0.20), reset=True)
    with torch.no_grad():
        torch.testing.assert_close(
            VectorReadout(a)(
                {k: v for k, v in values.items() if not k.startswith(("target_", "dino_"))}
            ).ttc_mean_seconds,
            VectorReadout(b)(
                {k: v for k, v in values.items() if not k.startswith(("target_", "dino_"))}
            ).ttc_mean_seconds,
            atol=0,
            rtol=0,
        )
    checks["equal_reset_output"] = True
    result = {
        "status": "PASS",
        "role": "P",
        "sample_tokens": batch.sample_tokens,
        "optimizer_updates": 0,
        "device": "cpu",
        "precision": "float32",
        "atol": 2e-5,
        "rtol": 2e-5,
        "checks": checks,
        "max_abs_errors": max_errors,
        "sources": {
            p: sha(ROOT / p)
            for p in (
                "src/e_jepa_ttc/range_readout/readout.py",
                "operational/range_readout/cache_readout.py",
                "operational/range_readout/parity.py",
            )
        },
    }
    write(RUN / "qa/READOUT_PARITY.json", result)
    source.close()
    print(result)


if __name__ == "__main__":
    main()
