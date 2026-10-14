"""Export the actual synthetic fusion QA heads with their reconstructed H transforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.rgb_port.normalization import fit_normalizer
from operational.range_transfer.common import freeze, read, sha, utc

from .common import RUN
from .fusion_engine_qa import fixture
from .fusion_export import export_head
from .fusion_features import FeatureIdentity
from .fusion_model import RGBAnchorComplement


def inputs() -> tuple:
    """Independently reconstruct the synthetic QA inputs and normalization provenance."""
    generator = np.random.default_rng(20261014)
    raw = [generator.normal(size=(8, 6)).astype(np.float32) for _ in range(2)]
    mask = np.ones(8, np.bool_)
    mask[[1, 4, 6]] = False
    raw[0][~mask] = np.nan
    tokens = np.array([f"synthetic-{i}" for i in range(8)])
    identities = [
        FeatureIdentity(modality, "a" * 64, "b" * 64, "c" * 64) for modality in ("rgb", "event")
    ]
    norms = []
    bank = fixture()
    for values, identity, available in zip(
        raw, identities, (mask, np.ones(8, np.bool_)), strict=True
    ):
        normalizer = fit_normalizer(
            values[available],
            tokens[available].tolist(),
            fit_role="H",
            expected_role="H",
            producer_sha256=identity.normalizer_producer_sha256(),
            schema_sha256=identity.schema_sha256,
            modality=identity.modality,
        )
        normalized = np.zeros_like(values)
        normalized[available] = normalizer.transform(values[available])
        np.testing.assert_array_equal(
            normalized, bank.inputs[f"{identity.modality}_features"].numpy()
        )
        norms.append(normalizer)
    examples = []
    for indices in (np.arange(8), np.flatnonzero(mask), np.flatnonzero(~mask)):
        examples.append(
            (
                torch.from_numpy(raw[0][indices].copy()),
                torch.from_numpy(raw[1][indices].copy()),
                bank.inputs["rgb_ttc"][indices].clone(),
                bank.inputs["event_ttc"][indices].clone(),
                bank.inputs["rgb_available"][indices].clone(),
            )
        )
    return norms, identities, examples


def run() -> dict:
    """Verify the existing ten-update heads without performing any new updates."""
    qa = RUN / "qa/fusion_engine/cpu"
    proof = read(qa / "PASS.json")
    if proof["status"] != "PASS" or not all(
        all(checks.values()) for checks in proof["checks"].values()
    ):
        raise ValueError("Require the existing full-state fusion QA proof")
    norms, identities, examples = inputs()
    out = RUN / "verification/fusion_normalized_export_v1"
    results = {}
    for mode in ("TRUE", "ZERO"):
        directory = qa / mode / "continuous"
        receipt = read(directory / "CHECKPOINT_RECEIPT.json")
        checkpoint = directory / "checkpoint_last.pt"
        if (
            receipt["status"] != "COMPLETE"
            or receipt["completed_updates"] != 10
            or sha(checkpoint) != receipt["checkpoint_sha256"]
        ):
            raise ValueError("Synthetic QA endpoint changed")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        recipe = payload["identity"]["recipe"]
        if recipe["mode"] != mode or payload["completed_updates"] != 10:
            raise ValueError("Synthetic QA payload identity changed")
        model = RGBAnchorComplement(
            6, 6, hidden=recipe["hidden"], residual_scale=recipe["residual_scale"], mode=mode
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        results[mode] = export_head(
            out / mode,
            model,
            *norms,
            *identities,
            examples,
            binding={
                "scope": "synthetic QA head only; no scientific fusion candidate",
                "checkpoint_sha256": sha(checkpoint),
                "qa_pass_sha256": sha(qa / "PASS.json"),
                "preflight_source_sha256": sha(Path(__file__)),
                "qa_source_sha256": sha(Path(__file__).with_name("fusion_engine_qa.py")),
                "H_normalization_reconstruction_exact_against_QA_bank": True,
            },
        )
    assert not torch.cuda.is_initialized()
    result = {
        "status": "PASS",
        "utc": utc(),
        "exports": results,
        "optimizer_updates": 0,
        "cuda_initialized": False,
        "scientific_export_complete": False,
        "scope": "Actual synthetic QA weights; H normalizers reconstructed exactly",
        "retained_failed_test_logs": {
            name: sha(RUN / "verification" / name)
            for name in (
                "FUSION_NORMALIZED_EXPORT_TESTS_20261014.log",
                "FUSION_NORMALIZED_EXPORT_TESTS_FIXED_20261014.log",
            )
        },
    }
    freeze(out / "PASS.json", result)
    return result


if __name__ == "__main__":
    print(run())
