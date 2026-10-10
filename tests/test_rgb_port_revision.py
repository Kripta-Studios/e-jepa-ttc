"""Regression contracts for the audited RGB-PORT amendment (no optimizer steps)."""

import numpy as np
import pytest
import torch

from operational.rgb_port_revision.loss import reduce_parts
from operational.rgb_port_revision.outputs import interval_from_phase


def test_tail_is_selected_once_over_effective_batch_with_equal_gradients():
    x = torch.tensor([1.0] + [0.01] * 31, requires_grad=True)
    whole = reduce_parts([({"log_ratio_tail": x.mean()}, {}, x)], 0.1)
    pieces = reduce_parts([({"log_ratio_tail": y.mean()}, {}, y) for y in (x[:1], x[1:])], 0.1)
    assert whole["log_ratio_tail"].item() == pytest.approx(0.2575)
    torch.testing.assert_close(whole["log_ratio_tail"], pieces["log_ratio_tail"])
    g1 = torch.autograd.grad(whole["log_ratio_tail"], x, retain_graph=True)[0]
    g2 = torch.autograd.grad(pieces["log_ratio_tail"], x)[0]
    torch.testing.assert_close(g1, g2)


def test_component_means_use_valid_counts_not_sample_counts():
    a, b = torch.tensor(2.0, requires_grad=True), torch.tensor(8.0, requires_grad=True)
    result = reduce_parts(
        [
            (
                {"foreground_center": a},
                {"foreground_center": 1},
                a.reshape(1)[:0],
            ),
            ({"foreground_center": b}, {"foreground_center": 3}, b.reshape(1)[:0]),
        ],
        0.1,
    )
    assert result["foreground_center"].item() == pytest.approx(6.5)
    result["foreground_center"].backward()
    assert a.grad.item() == pytest.approx(0.25)
    assert b.grad.item() == pytest.approx(0.75)


def test_ttc_interval_reverses_levels_and_marks_discontinuity():
    lo, hi, status = interval_from_phase(np.array([0.1, -0.3, -0.1]), np.array([0.3, -0.1, 0.1]))
    assert np.all(lo[:2] <= hi[:2])
    assert np.isnan(lo[2]) and np.isnan(hi[2])
    assert status.tolist() == [
        "SAME_BRANCH_INTERVAL",
        "SAME_BRANCH_INTERVAL",
        "CROSSES_ZERO_UNAVAILABLE",
    ]


def test_report_uses_measured_nested_profile_values(tmp_path):
    import json
    from pathlib import Path

    from operational.rgb_port.report import build_report

    run = tmp_path / "run"
    (run / "profiles").mkdir(parents=True)
    config = tmp_path / "execution.json"
    config.write_text(
        json.dumps({"schema": "rgb_port_execution_v1", "tasks": []}), encoding="utf-8"
    )
    (run / "profiles/ROUTE_COSTS.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "profiles": {"RAW_TEST": {"warm_ms": {"median": 12.5, "p95": 19.0}}},
            }
        ),
        encoding="utf-8",
    )
    result = build_report(run=run, config_path=config, output=tmp_path / "report")
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "mediana=12.5" in report and "p95=19" in report


def _model(fit_id="R_C2F"):
    from pathlib import Path

    from operational.rgb_port.recipe import resolved_recipe
    from operational.rgb_port.train_producers import build_producer

    recipe = resolved_recipe(
        Path("configs/rgb_port/producers.json"),
        fit_id=fit_id,
        producer_population=32,
        role_manifest_sha256="a" * 64,
    )
    return build_producer(recipe), recipe


def test_rgb_router_receives_luma_gradient_not_green_blue_channels():
    from e_jepa_ttc.rgb_port.features import raw_rgb_statistics

    model, _ = _model()
    seen = []
    model.transport_router.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].detach())
    )
    rgb = torch.rand(1, 3, 3, 64, 64)
    with torch.no_grad():
        model(rgb, torch.full((1, 2), 0.1))
    torch.testing.assert_close(
        seen[0][:, :2], raw_rgb_statistics(rgb[:, 1:].flatten(0, 1)), rtol=0, atol=0
    )


def _batch(model, count, steps):
    from types import SimpleNamespace

    from e_jepa_ttc.distillation.dinov3_relational import local_cosine_relation_maps

    events = torch.rand(count, steps, 3, 16, 16)
    delta = torch.full((count, steps - 1), 0.1)
    with torch.no_grad():
        dense = model(events, delta, return_dense_features=True).endpoint_dense_features
        relations = local_cosine_relation_maps(dense[:, -2:])
    return SimpleNamespace(
        events=events,
        delta_t_s=delta,
        target_ttc_s=torch.linspace(1.0, 3.0, count),
        boxes_xyxy=torch.tensor([2.0, 2.0, 14.0, 14.0]).expand(count, steps, 4).clone(),
        dinov3_relation_targets=torch.zeros_like(relations.values),
        dinov3_relation_valid=relations.valid,
    )


def test_real_loss_and_gradients_are_invariant_to_homogeneous_partition():
    from types import SimpleNamespace

    from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig
    from operational.rgb_port.train_producers import _producer_loss
    from operational.rgb_port_revision.loss import EffectiveBatch, effective_loss

    model, recipe = _model("R_A5")
    model.eval()  # Partition equivalence isolates reductions from random dropout draws.
    batch = _batch(model, 3, 3)
    config = CausalScaleTTCLossConfig(**recipe.loss_config)
    full, _ = _producer_loss(model, batch, config)
    params = tuple(model.parameters())
    full_grad = torch.autograd.grad(full, params, allow_unused=True)
    parts = [
        SimpleNamespace(**{name: value[part] for name, value in vars(batch).items()})
        for part in (slice(0, 1), slice(1, 3))
    ]
    split, _ = effective_loss(model, EffectiveBatch(parts), config)
    split_grad = torch.autograd.grad(split, params, allow_unused=True)
    torch.testing.assert_close(full, split, rtol=2e-5, atol=2e-6)
    for left, right in zip(full_grad, split_grad, strict=True):
        if left is None:
            assert right is None
        else:
            torch.testing.assert_close(left, right, rtol=2e-3, atol=2e-5)


def test_mixed_t2_t3_real_loss_backpropagates_without_padding():
    from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig
    from operational.rgb_port_revision.loss import EffectiveBatch, effective_loss

    model, recipe = _model("R_A5")
    total, components = effective_loss(
        model,
        EffectiveBatch([_batch(model, 1, 2), _batch(model, 2, 3)]),
        CausalScaleTTCLossConfig(**recipe.loss_config),
    )
    assert torch.isfinite(total)
    assert all(torch.isfinite(value) for value in components.values())
    total.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_transfer_uses_real_sensor_triplets_and_never_future_frames():
    from operational.rgb_port_revision.transfer_inputs import EVENT_LAGS_US, sensor_triplets

    np.testing.assert_array_equal(EVENT_LAGS_US, [300000, 200000, 100000, 0])
    clock = np.arange(0, 2_000_001, 100_000)
    groups = sensor_triplets(clock, 1_000_000)
    assert all(len(group) == 3 for group in groups)
    assert all(clock[group[-1]] <= 1_000_000 and clock[group[0]] >= 350_000 for group in groups)
    assert len(sensor_triplets(clock, 100_000)[0]) == 2
    assert sensor_triplets(clock, 50_000) == []
    fast = np.arange(0, 2_000_001, 50_000)
    groups = sensor_triplets(fast, 1_000_000)
    assert all(np.array_equal(np.diff(fast[group]), [100_000, 100_000]) for group in groups)


def test_cpu_profile_has_no_fabricated_cuda_measurement():
    from operational.rgb_port.profile import profile_callable

    value = profile_callable(lambda: None, scope="test", device="cpu", warm_iterations=2)
    assert value["vram_peak_allocated_bytes"] is None
    assert "not_OS_cache_flush" in value["cold_definition"]


def test_migration_accepts_only_exact_archived_before_and_reviewed_after(tmp_path, monkeypatch):
    from operational.rgb_port.accounting import atomic_write_json, sha256_file
    from operational.rgb_port.recipe import canonical_sha256
    from operational.rgb_port_revision import migration

    monkeypatch.setattr(migration, "ROOT", tmp_path)
    run = tmp_path / "run"
    archive = run / "audit_fixes_20261009/before/code.py"
    archive.parent.mkdir(parents=True)
    archive.write_text("old", encoding="utf-8")
    path = tmp_path / "code.py"
    path.write_text("corrected", encoding="utf-8")
    before = sha256_file(archive)
    atomic_write_json(run / "SOURCE_FREEZE.json", {"files": {str(path): before}})
    document = {
        "schema": "rgb_port_direct_code_migration_v1",
        "geometry_precision": "bf16_unchanged",
        "event_objective_unchanged": True,
        "historical_freezes": {},
        "files": {"code.py": {"before_sha256": before, "after_sha256": sha256_file(path)}},
    }
    document["identity_sha256"] = canonical_sha256(document)
    atomic_write_json(run / migration.MANIFEST, document)
    assert migration.source_matches(path, before, run)
    assert not migration.source_matches(path, "0" * 64, run)
    path.write_text("unreviewed", encoding="utf-8")
    with pytest.raises(ValueError, match="Unreviewed"):
        migration.source_matches(path, before, run)


def test_existing_opt_in_technical_qa_binding_is_kept_without_allowing_science_bypass(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from operational.rgb_port.accounting import atomic_write_json
    from operational.rgb_port_revision.loss import EffectiveSource
    from operational.rgb_port_revision.migration import bind_runtime

    freeze = {
        "schema": "rgb_port_technical_source_freeze_v1",
        "producer_source_identity": {"technical_subset_only": True},
    }
    atomic_write_json(tmp_path / "SOURCE_FREEZE.json", freeze)
    monkeypatch.delenv("RGB_PORT_TECHNICAL_RESUME_QA", raising=False)
    with pytest.raises(ValueError, match="Scientific continuation"):
        bind_runtime(tmp_path, "R_A5", rgb=True)
    monkeypatch.setenv("RGB_PORT_TECHNICAL_RESUME_QA", "1")
    digest = bind_runtime(tmp_path, "R_A5", rgb=True)
    assert len(digest) == 64
    source = SimpleNamespace(identity={}, population_size=1, frame_counts=[2])
    wrapped = EffectiveSource(source, SimpleNamespace(microbatch_size=1), digest)
    assert wrapped.population_size == 1 and wrapped.input_span_us == ()
