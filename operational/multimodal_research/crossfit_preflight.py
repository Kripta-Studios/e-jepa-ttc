"""Check native temporal loss and gradients against the new grouped-context entrypoint."""

from __future__ import annotations

import copy
from pathlib import Path

import torch

from operational.multimodal_validation.common import sha, write
from operational.rgb_port.train_heads import SealedCache, _loss, _model

from .common import OUT
from .crossfit_context import context_loss
from .crossfit_data import DEST


def main() -> None:
    torch.set_num_threads(1)
    rows = []
    for modality, fit_id in (("E", "E_CTX_MATCHED"), ("R", "R_CTX")):
        cache = SealedCache(DEST / f"fold0/{modality}/train.npz", fit_id)
        ids = torch.linspace(0, cache.population - 1, 128).long()
        torch.manual_seed(7)
        reference = _model(fit_id)
        actual = copy.deepcopy(reference)
        tensors = {
            key: torch.from_numpy(cache.values[key])
            for key in ("features", "timing", "valid", "expert_phase", "target_phase", "mass")
        }
        expected_loss = _loss(reference, fit_id, cache, ids, torch.device("cpu"))
        actual_loss = context_loss(actual, tensors, ids, cache.population)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        expected_loss.backward()
        actual_loss.backward()
        count = 0
        for (name, a), (other, b) in zip(
            reference.named_parameters(), actual.named_parameters(), strict=True
        ):
            assert name == other and a.grad is not None and b.grad is not None
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
            count += a.numel()
        rows.append(
            dict(
                modality=modality,
                loss=float(actual_loss.detach()),
                gradient_parameters=count,
                loss_exact=True,
                gradients_exact=True,
            )
        )
    write(
        OUT / "GROUP_CROSSFIT_NATIVE_PARITY.json",
        dict(
            rows=rows,
            optimizer_updates=0,
            device="CPU FP32",
            source_sha256=sha(__file__),
            context_source_sha256=sha(Path(__file__).with_name("crossfit_context.py")),
            meaning="Native temporal loss and gradients preserved on the same fold data",
        ),
    )
    print(rows, flush=True)


if __name__ == "__main__":
    main()
