"""Focused regressions for the public eAP causal-scale runner."""

from __future__ import annotations

import torch

from scripts.train_causal_scale_eap_screen import (
    _load_grouped_development_contract,
    _reset_peak_memory_stats,
)

GROUPED_PROTOCOL = (
    "configs/protocol/scientific_recovery_v5_train_only_grouped_dev.json"
)
GROUPED_PROTOCOL_FILE_SHA256 = (
    "be48917ae52d1c77d046318bd9ed284a32e8b16258257203fff439332b547874"
)
GROUPED_PROTOCOL_ARTIFACT_SHA256 = (
    "f09c688fb4991714abc9d645dda787cb27f1e02a2d1857312ce3e45519bd7a63"
)


def _grouped_reference(*, fold: int = 0) -> dict[str, object]:
    return {
        "development_protocol": {
            "path": GROUPED_PROTOCOL,
            "file_sha256": GROUPED_PROTOCOL_FILE_SHA256,
            "artifact_sha256": GROUPED_PROTOCOL_ARTIFACT_SHA256,
            "fold": fold,
        }
    }


def test_grouped_development_contract_loads_frozen_fold() -> None:
    contract = _load_grouped_development_contract(_grouped_reference(fold=1))

    assert contract["fold_index"] == 1
    assert len(contract["train_sequences"]) == 6
    assert len(contract["dev_sequences"]) == 3
    assert contract["train_sequences"].isdisjoint(contract["dev_sequences"])
    assert contract["fold"]["train_rows"] == 5461
    assert contract["fold"]["dev_rows"] == 2731


def test_grouped_development_contract_rejects_stale_file_hash() -> None:
    reference = _grouped_reference()
    reference["development_protocol"]["file_sha256"] = "0" * 64

    try:
        _load_grouped_development_contract(reference)
    except ValueError as error:
        assert "file SHA256" in str(error)
    else:
        raise AssertionError("stale grouped-development reference must fail closed")


def test_grouped_development_contract_rejects_unknown_fold() -> None:
    try:
        _load_grouped_development_contract(_grouped_reference(fold=3))
    except ValueError as error:
        assert "fold 3 is unavailable" in str(error)
    else:
        raise AssertionError("unknown grouped-development fold must fail closed")


def test_peak_memory_reset_selects_cuda_and_uses_default_api(monkeypatch) -> None:
    selected: list[torch.device] = []
    resets = 0

    def record_device(device: torch.device) -> None:
        selected.append(device)

    def record_reset() -> None:
        nonlocal resets
        resets += 1

    monkeypatch.setattr(torch.cuda, "set_device", record_device)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", record_reset)

    device = torch.device("cuda", 0)
    _reset_peak_memory_stats(device)

    assert selected == [device]
    assert resets == 1


def test_peak_memory_reset_is_noop_on_cpu(monkeypatch) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "set_device",
        lambda _device: (_ for _ in ()).throw(AssertionError("must not select CUDA")),
    )
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda: (_ for _ in ()).throw(AssertionError("must not reset CUDA")),
    )

    _reset_peak_memory_stats(torch.device("cpu"))
