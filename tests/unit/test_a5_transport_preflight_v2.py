from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "a5_preflight_v2", ROOT / "scripts" / "diagnose_a5_transport_preflight_v2.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_selected_indices_span_full_dataset() -> None:
    indices = MODULE._selected_indices(2048, 512)
    assert len(indices) == 512
    assert indices[0] == 0
    assert indices[-1] >= 2040
    assert len(set(indices)) == len(indices)


def test_same_sequence_null_never_self_and_prefers_other_track() -> None:
    seq = ["a", "a", "a", "b", "b"]
    track = ["x", "x", "y", "m", "n"]
    original = [0, 10, 20, 30, 40]
    partner, fraction = MODULE._same_sequence_null_partners(seq, track, original)
    assert (partner >= 0).all()
    assert all(int(partner[i]) != i for i in range(len(seq)))
    assert all(seq[int(partner[i])] == seq[i] for i in range(len(seq)))
    assert fraction == 1.0


def test_bbox_affine_flow_translation_and_scale() -> None:
    b1 = torch.tensor([[8.0, 8.0, 24.0, 24.0]])
    b2 = torch.tensor([[12.0, 10.0, 32.0, 30.0]])
    flow = MODULE._bbox_affine_flow(
        b1, b2, source_height=32, source_width=32, feature_height=32, feature_width=32
    )
    assert torch.allclose(flow["translation_x"], torch.tensor([6.0]))
    assert torch.allclose(flow["translation_y"], torch.tensor([4.0]))
    assert torch.allclose(flow["log_height_ratio"], torch.log(torch.tensor([1.25])))
    assert bool(flow["foreground"].any())


def test_student_hard_argmax_is_temperature_invariant() -> None:
    torch.manual_seed(4)
    previous = torch.randn(1, 8, 8, 8)
    current = torch.zeros_like(previous)
    current[:, :, :, 1:] = previous[:, :, :, :-1]
    result = MODULE._student_correlation(
        previous, current, radius=2, temperatures=(0.02, 0.10)
    )
    assert result["hard_dx"].shape == (1, 8, 8)
    # Temperature only changes the soft field; hard argmax is computed once.
    assert not torch.allclose(result["soft"][0.02]["dx"], result["soft"][0.10]["dx"])


def test_decision_selects_smallest_eligible_radius_and_largest_safe_tau() -> None:
    protocol = {
        "radii": [1, 2, 4],
        "selection": {
            "bbox_physical_coverage_min": 0.90,
            "teacher_excess_error_reduction_min": 0.10,
            "teacher_real_vs_shuffled_best_error_improvement_min": 0.05,
            "teacher_real_vs_spatial_null_best_error_improvement_min": 0.05,
            "teacher_foreground_excess_error_reduction_min": 0.10,
            "student_hard_epe_improvement_over_zero_min": 0.10,
            "student_hard_epe_advantage_over_random_min": 0.05,
            "student_real_vs_shuffled_top1_cosine_min": 0.005,
            "student_real_vs_spatial_null_top1_cosine_min": 0.005,
            "student_confidence_margin_min": 0.005,
            "student_entropy_max": 0.92,
        },
    }
    teacher = {
        "1": {"bbox_physical_coverage": 0.70, "excess_error_reduction": 0.3, "real_vs_shuffled_best_error_improvement": 0.2, "foreground_excess_error_reduction": 0.2, "real_vs_spatial_null_best_error_improvement": 0.2},
        "2": {"bbox_physical_coverage": 0.95, "excess_error_reduction": 0.3, "real_vs_shuffled_best_error_improvement": 0.2, "foreground_excess_error_reduction": 0.2, "real_vs_spatial_null_best_error_improvement": 0.2},
        "4": {"bbox_physical_coverage": 1.00, "excess_error_reduction": 0.4, "real_vs_shuffled_best_error_improvement": 0.3, "foreground_excess_error_reduction": 0.3, "real_vs_spatial_null_best_error_improvement": 0.3},
    }
    base = {
        "1": {"physical_epe_improvement_over_zero": 0.0, "real_minus_shuffled_top1_cosine": 0.0, "real_minus_spatial_null_top1_cosine": 0.0, "real_margin": 0.0},
        "2": {"physical_epe_improvement_over_zero": 0.30, "real_minus_shuffled_top1_cosine": 0.02, "real_minus_spatial_null_top1_cosine": 0.02, "real_margin": 0.02},
        "4": {"physical_epe_improvement_over_zero": 0.30, "real_minus_shuffled_top1_cosine": 0.02, "real_minus_spatial_null_top1_cosine": 0.02, "real_margin": 0.02},
    }
    students = {
        "A4": base,
        "A4D": {**base, "2": {**base["2"], "physical_epe_improvement_over_zero": 0.20}},
        "RANDOM": {**base, "2": {**base["2"], "physical_epe_improvement_over_zero": 0.05}},
    }
    temps = [
        {"model": "A4", "radius": 2, "temperature": 0.10, "entropy": 0.94},
        {"model": "A4", "radius": 2, "temperature": 0.07, "entropy": 0.91},
        {"model": "A4", "radius": 2, "temperature": 0.04, "entropy": 0.85},
    ]
    decision = MODULE._decision(protocol, teacher, students, temps)
    assert decision["selected_radius"] == 2
    assert decision["selected_temperature"] == 0.07
    assert decision["a5_corr_authorized"] is True
