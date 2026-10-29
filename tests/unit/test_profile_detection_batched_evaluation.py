"""CPU-only contracts for the deferred B1/B2/B4 CUDA profiler."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from scripts.profile_detection_batched_evaluation import (
    _main,
    _write_failure_receipt,
    batch_ranges,
    begin_cuda_mode_boundary,
    compare_decoded,
    compare_nested,
    slice_batch,
    validate_panel,
    validate_runtime_settings,
)


def test_panel_requires_explicit_selection_16_or_32() -> None:
    selection = {f"t{index}" for index in range(40)}
    assert len(validate_panel([f"t{index}" for index in range(16)], selection)) == 16
    with pytest.raises(ValueError, match="16 or 32"):
        validate_panel(["t0"] * 16, selection)
    with pytest.raises(ValueError, match="subset"):
        validate_panel([*[f"t{index}" for index in range(15)], "other"], selection)


def test_batch_ranges_never_pad_or_drop() -> None:
    assert [list(item) for item in batch_ranges(16, 4)] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ]
    with pytest.raises(ValueError, match="divide exactly"):
        batch_ranges(15, 4)


def test_runtime_seed_and_threads_are_strict() -> None:
    assert validate_runtime_settings(7, 4) == (7, 4)
    for seed, threads in ((True, 4), (-1, 4), (7, False), (7, 0)):
        with pytest.raises(ValueError):
            validate_runtime_settings(seed, threads)


def test_slice_batch_preserves_all_nested_heads() -> None:
    output = {
        "pred_logits": torch.arange(12).reshape(2, 2, 3),
        "pred_velocity": torch.arange(8).reshape(2, 2, 2),
        "aux_outputs": [{"pred_depth": torch.arange(4).reshape(2, 2)}],
    }
    sliced = slice_batch(output, 1, 2)
    assert sliced["pred_logits"].shape == (1, 2, 3)  # type: ignore[index,union-attr]
    assert sliced["pred_velocity"].tolist() == [[[4, 5], [6, 7]]]  # type: ignore[index,union-attr]
    assert sliced["aux_outputs"][0]["pred_depth"].tolist() == [[2, 3]]  # type: ignore[index]


def test_tensor_comparison_records_exact_and_tolerated_differences() -> None:
    reference = {"head": torch.tensor([[1.0, 2.0]])}
    exact = compare_nested(reference, {"head": reference["head"].clone()})
    assert exact["exact_tensor_count"] == exact["tensor_count"] == 1
    close = compare_nested(reference, {"head": torch.tensor([[1.0 + 5e-7, 2.0]])})
    assert close["within_tolerance"] and close["exact_tensor_count"] == 0
    far = compare_nested(reference, {"head": torch.tensor([[1.1, 2.0]])})
    assert isinstance(far["max_abs"], float)
    assert not far["within_tolerance"] and far["max_abs"] > 0.09


def test_decoded_comparison_keeps_categories_exact() -> None:
    reference = [{"class_name": "car", "velocity_mps": [1.0, 2.0]}]
    assert compare_decoded(reference, reference)["within_tolerance"]
    changed_class = [{"class_name": "truck", "velocity_mps": [1.0, 2.0]}]
    assert not compare_decoded(reference, changed_class)["within_tolerance"]
    close_velocity = [{"class_name": "car", "velocity_mps": [1.0000005, 2.0]}]
    assert compare_decoded(reference, close_velocity)["within_tolerance"]


def test_existing_output_is_refused_before_checkpoint_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "receipt.json"
    output.write_text("immutable", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile",
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--expected-checkpoint-sha256",
            "0" * 64,
            "--tokens",
            str(tmp_path / "missing.json"),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(FileExistsError, match="already exists"):
        _main()
    assert output.read_text(encoding="utf-8") == "immutable"


def test_checkpoint_identity_is_checked_before_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "selected.pt"
    checkpoint.write_bytes(b"not a torch checkpoint")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA was queried"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile",
            "--checkpoint",
            str(checkpoint),
            "--expected-checkpoint-sha256",
            "0" * 64,
            "--tokens",
            str(tmp_path / "tokens.json"),
            "--output",
            str(tmp_path / "receipt.json"),
        ],
    )
    with pytest.raises(ValueError, match="Checkpoint SHA-256 differs"):
        _main()


def test_failure_receipt_does_not_initialize_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "failure.json"
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA was queried"))
    _write_failure_receipt(
        RuntimeError("synthetic failure"),
        ["profile", "--output", str(output), "--expected-checkpoint-sha256", "0" * 64],
    )
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "FAIL"
    assert receipt["retry_or_batch_downshift_performed"] is False
    assert receipt["resources_at_failure"] == {"cuda_initialized": False}


def test_mode_boundary_cleans_then_resets_peaks(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda: calls.append("reset_peaks")
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 11)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 17)
    boundary = begin_cuda_mode_boundary()
    assert calls == ["synchronize", "empty_cache", "synchronize", "reset_peaks"]
    assert boundary["baseline_allocated_bytes"] == 11
    assert boundary["baseline_reserved_bytes"] == 17
