"""Zero-optimizer tests for RGB-PORT cache and durable accounting contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from e_jepa_ttc.rgb_port.features import RGB_PHASE17_SHA256
from operational.rgb_port.prepare_features import (
    ENDPOINT_IDS,
    build_history,
    build_pair_cache,
    fit_h_normalizer,
)
from operational.rgb_port.train_heads import SealedCache
from operational.rgb_port.train_producers import ProducerCheckpoint


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pair_cache_uses_frozen_parent_and_exact_support_order(tmp_path: Path) -> None:
    checkpoint = tmp_path / "parent.pt"
    checkpoint.write_bytes(b"parent")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "fit_id": "E_A5_MATCHED",
                "status": "COMPLETE",
                "scientific_endpoint": True,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": _sha(checkpoint),
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "source.npz"
    np.savez(
        source,
        token128=np.zeros((2, 128), np.float32),
        delta_t_s=np.asarray([0.1, 0.2], np.float32),
        support=np.asarray([[0.8, 0.2], [0.1, 0.7]], np.float32),
        target_phase=np.zeros(2, np.float32),
        mass=np.full(2, 0.5, np.float32),
        sequence_id=np.asarray(["g0", "g1"]),
        sample_token=np.asarray(["s0", "s1"]),
        group_id=np.asarray(["g0", "g1"]),
        role=np.asarray(["P", "P"]),
    )
    output = tmp_path / "pair.npz"
    build_pair_cache(source, output, modality="event", producer_receipt=receipt)
    cache = SealedCache(output, "PAIR_E_MATCHED")
    np.testing.assert_allclose(cache.values["features"][:, -2:], [[0.2, 0.2], [0.7, 0.1]])


def test_h_normalizer_counts_unique_observations_only(tmp_path: Path) -> None:
    observations = tmp_path / "observations.npz"
    values = np.stack((np.arange(17), np.arange(17) + 2), 0).astype(np.float32)
    np.savez(
        observations,
        observation_id=np.asarray(["o0", "o1"]),
        features=values,
        anchor_us=np.asarray([1, 2], np.int64),
        available_us=np.asarray([1, 2], np.int64),
        role=np.asarray(["H", "H"]),
        schema_sha256=np.asarray("x"),
        endpoint_freeze_sha256=np.asarray("f" * 64),
    )
    output = tmp_path / "normalizer.json"
    result = fit_h_normalizer(observations, output, modality="event")
    assert result["fit_role"] == "H"
    np.testing.assert_allclose(result["mean"], np.arange(17) + 1)


def test_journal_charges_pending_and_uncheckpointed_recovery(tmp_path: Path) -> None:
    identity = "a" * 64
    frozen = {"identity_sha256": identity, "fit_id": "E_CTX_MATCHED"}
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    generator = torch.Generator().manual_seed(7)
    state = ProducerCheckpoint(tmp_path, frozen, 2500)
    state.save(model, optimizer, scheduler, generator, {"losses": []}, status="READY")
    state.completed = 3
    state.durable = 0
    state.pending = True
    state._write_journal()
    recovered = ProducerCheckpoint(tmp_path, frozen, 2500)
    recovered.restore(model, optimizer, scheduler, generator)
    assert recovered.recovery_upper == 4
    assert recovered.pending is False


def test_history_builder_accepts_one_genuine_t2_cold_observation(tmp_path: Path) -> None:
    endpoint_items = {}
    for fit_id in ENDPOINT_IDS:
        checkpoint = tmp_path / f"{fit_id}.pt"
        checkpoint.write_bytes(fit_id.encode())
        endpoint_items[fit_id] = {
            "status": "COMPLETE",
            "scientific_endpoint": True,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha(checkpoint),
        }
    freeze = tmp_path / "endpoints.json"
    freeze.write_text(json.dumps({"endpoints": endpoint_items}), encoding="utf-8")
    observations = tmp_path / "observations.npz"
    np.savez(
        observations,
        observation_id=np.asarray(["t2-current"]),
        features=np.arange(17, dtype=np.float32)[None],
        anchor_us=np.asarray([100], np.int64),
        available_us=np.asarray([100], np.int64),
        role=np.asarray(["H"]),
        schema_sha256=np.asarray(RGB_PHASE17_SHA256),
        endpoint_freeze_sha256=np.asarray(_sha(freeze)),
    )
    normalizer = tmp_path / "normalizer.json"
    fit_h_normalizer(observations, normalizer, modality="rgb")
    history = tmp_path / "history.npz"
    np.savez(
        history,
        query_id=np.asarray(["q0"]),
        observation_id=np.asarray([["t2-current"]]),
        timing=np.zeros((1, 1, 4), np.float32),
        valid=np.ones((1, 1), np.bool_),
        current_expert_phase=np.asarray([[0.1, 0.2, 0.3]], np.float32),
        target_phase=np.asarray([0.2], np.float32),
        mass=np.asarray([1.0], np.float32),
        sample_token=np.asarray(["s0"]),
        group_id=np.asarray(["g0"]),
        role=np.asarray(["H"]),
    )
    output = tmp_path / "head.npz"
    build_history(
        observations, history, normalizer, output, role="H", modality="rgb", freeze=freeze
    )
    cache = SealedCache(output, "R_CTX")
    assert cache.values["features"].shape == (1, 1, 17)
    assert bool(cache.values["valid"][0, 0])
