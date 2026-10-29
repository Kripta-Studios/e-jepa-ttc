from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from operational.independent_refit.fitter import (
    EngineeringSmokeConfig,
    FoundationIdentity,
    bind_foundation_identity,
    require_real_fit_admission,
    state_tensor_sha256,
    verify_engineering_smoke_checkpoint,
    verify_historical_sources,
    write_engineering_smoke_checkpoint,
)
from operational.independent_refit.population import canonical_sha256

torch = importlib.import_module("torch")


def _identity() -> FoundationIdentity:
    return FoundationIdentity(
        rotation=0,
        train_role="producer",
        train_group_ids=tuple(f"group_{index:02d}" for index in range(16)),
        train_token_set_sha256="a" * 64,
        fold_contract_sha256="b" * 64,
        raw_history_receipt_sha256="c" * 64,
        teacher_cache_manifest_sha256="d" * 64,
        teacher_role_receipt_sha256="e" * 64,
        source_population_sha256="f" * 64,
    )


@pytest.mark.parametrize("modality", ["event", "rgb"])
def test_engineering_smoke_checkpoint_has_internal_identity(
    tmp_path: Path, modality: str
) -> None:
    checkpoint = tmp_path / f"{modality}.pt"
    result = write_engineering_smoke_checkpoint(
        checkpoint,
        config=EngineeringSmokeConfig(modality=modality),  # type: ignore[arg-type]
        identity=_identity(),
    )
    assert result["status"] == "PASS_ENGINEERING_SMOKE_NO_TRAINING"
    assert result["optimizer_updates"] == 0
    assert verify_engineering_smoke_checkpoint(checkpoint) == result


def test_checkpoint_rejects_internal_tensor_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "event.pt"
    write_engineering_smoke_checkpoint(
        checkpoint,
        config=EngineeringSmokeConfig(modality="event"),
        identity=_identity(),
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    name = next(iter(payload["model_state"]))
    payload["model_state"][name].view(-1)[0] += 1.0
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="internal identity"):
        verify_engineering_smoke_checkpoint(checkpoint)


def test_checkpoint_rejects_rehashed_arbitrary_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "event.pt"
    write_engineering_smoke_checkpoint(
        checkpoint,
        config=EngineeringSmokeConfig(modality="event"),
        identity=_identity(),
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    name = next(iter(payload["model_state"]))
    payload["model_state"][name].view(-1)[0] += 1.0
    identity = payload["identity"]
    identity["state_tensor_sha256"] = state_tensor_sha256(payload["model_state"])
    identity.pop("checkpoint_identity_sha256")
    identity["checkpoint_identity_sha256"] = canonical_sha256(identity)
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="internal identity"):
        verify_engineering_smoke_checkpoint(checkpoint)


def test_real_fit_remains_blocked_without_physical_h5_admission(tmp_path: Path) -> None:
    receipt = tmp_path / "RAW_HISTORY.json"
    receipt.write_text(
        json.dumps(
            {
                "training_admitted": False,
                "stream_bounds_checked": False,
                "stream_h5_bounds_receipt": None,
                "stream_h5_hash_receipt": None,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="blocked by raw H5 admission"):
        require_real_fit_admission(receipt)


def test_historical_architecture_sources_are_exact() -> None:
    assert set(verify_historical_sources()) == {"model", "data"}


def test_producer_identity_requires_exact_16_groups() -> None:
    with pytest.raises(ValueError, match="16 sorted groups"):
        FoundationIdentity(
            **{
                **_identity().__dict__,
                "train_group_ids": ("one",),
            }
        )


def test_binding_uses_verified_producer_and_blocked_raw_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fold = tmp_path / "population/FOLD_CONTRACT.json"
    fold.parent.mkdir()
    fold.write_text("{}", encoding="utf-8")
    (fold.parent / "population.parquet").write_bytes(b"population")
    raw = tmp_path / "RAW_HISTORY.json"
    raw.write_text(
        json.dumps(
            {
                "observation_packets": 31,
                "raw_observation_span_us": 620_000,
                "packet_endpoints": 31,
                "endpoint_span_us": 600_000,
                "packet_intervals": 30,
                "step_us": 20_000,
                "training_admitted": False,
                "stream_bounds_checked": False,
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text("{}", encoding="utf-8")
    role_receipt = tmp_path / "ROLE.json"
    rows = pd.DataFrame(
        {
            "sample_token": [f"token_{index}" for index in range(16)],
            "group_id": [f"group_{index:02d}" for index in range(16)],
        }
    )

    def load_teacher_role(*_args: object, **_kwargs: object) -> SimpleNamespace:
        role_receipt.write_text("{}", encoding="utf-8")
        return SimpleNamespace(allowed_tokens=frozenset(rows["sample_token"]))

    modules = {
        "operational.independent_refit.raw_history": SimpleNamespace(
            verify_raw_history_index=lambda _path: {"status": "PASS"}
        ),
        "operational.independent_refit.lineage": SimpleNamespace(
            load_rotation_role=lambda *_args, **_kwargs: rows
        ),
        "operational.independent_refit.teacher_cache": SimpleNamespace(
            load_teacher_role=load_teacher_role
        ),
    }
    original = importlib.import_module
    monkeypatch.setattr(
        "operational.independent_refit.fitter.importlib.import_module",
        lambda name: modules[name] if name in modules else original(name),
    )
    identity = bind_foundation_identity(
        fold_contract_path=fold,
        raw_history_receipt_path=raw,
        teacher_cache_manifest_path=manifest,
        teacher_cache_manifest_sha256="0" * 64,
        teacher_role_receipt_path=role_receipt,
        rotation=0,
    )
    assert identity.train_role == "producer"
    assert len(identity.train_group_ids) == 16
