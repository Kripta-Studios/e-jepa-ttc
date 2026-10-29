"""Fail-closed gates for the foreground R0 -> R1 campaign queue."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/run_detection_campaign_queue.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_detection_campaign_queue", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def binding(module, path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": module.sha256_file(path)}


def make_summary(
    module,
    output: Path,
    config,
    *,
    status: str,
    start: int,
    final: int,
    resume_sha256: str | None,
) -> dict:
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True)
    last = checkpoints / "last.pt"
    selected = checkpoints / "selected.pt"
    last.write_bytes(f"last-{start}-{final}".encode())
    selected.write_bytes(f"selected-{start}-{final}".encode())
    summary = {
        "schema": module.RUN_SCHEMA,
        "status": status,
        "final_update": final,
        "start_update": start,
        "config_sha256": config.sha256,
        "best_selection_metric": 0.125,
        "last_checkpoint_sha256": module.sha256_file(last),
        "checkpoint_path": str(last.resolve()),
        "resumed_from_sha256": resume_sha256,
        "selected_checkpoint": {
            "status": "AVAILABLE",
            "path": str(selected.resolve()),
            "sha256": module.sha256_file(selected),
            "best_metric": 0.125,
        },
        "parent_run": (
            {"resume_checkpoint_sha256": resume_sha256} if resume_sha256 is not None else None
        ),
        "parent_run_chain": [] if resume_sha256 is None else [{"status": "BUDGET_STOPPED"}],
        "evaluation_protocol_sha256": "a" * 64,
        "full_selection_count": 2,
        "dataset_manifest_hash": config.data.expected_manifest_sha256,
        "split_hash": config.data.expected_split_sha256,
        "full_selection_sha256": config.data.expected_full_selection_sha256,
        "resource_admission": {
            "status": "PASS",
            "available_ram_bytes": 9 * 2**30,
            "available_vram_bytes": 9 * 2**30,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return summary


def test_exact_resume_summary_gate_accepts_only_bound_checkpoint(tmp_path: Path):
    module = load_module()
    config = module.load_campaign_config(
        ROOT / "configs/detection_campaign/rgb_r0_d4096_seed7.yaml"
    )
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"parent")
    resume_sha256 = module.sha256_file(parent)
    output = tmp_path / "r0_8192"
    make_summary(
        module,
        output,
        config,
        status="COMPLETED",
        start=1024,
        final=8192,
        resume_sha256=resume_sha256,
    )
    verified = module.verify_summary(
        output,
        config,
        expected_status="COMPLETED",
        expected_final_update=8192,
        expected_start_update=1024,
        resume_sha256=resume_sha256,
    )
    assert verified["last_checkpoint_sha256"] == module.sha256_file(
        output / "checkpoints/last.pt"
    )
    with pytest.raises(ValueError, match="checkpoint identity"):
        module.verify_summary(
            output,
            config,
            expected_status="COMPLETED",
            expected_final_update=8192,
            expected_start_update=1024,
            resume_sha256="f" * 64,
        )


def test_missing_full_selection_artifact_blocks_r1(tmp_path: Path):
    module = load_module()
    summary = {"evaluation_protocol_sha256": "a" * 64, "full_selection_count": 2}
    with pytest.raises(FileNotFoundError):
        module.verify_full_evaluations(tmp_path, summary)


def test_full_selection_fixture_requires_schema_gt_and_protocol(tmp_path: Path):
    module = load_module()
    protocol = "a" * 64
    payload = {
        "metrics_v2": {
            "schema": module.EVALUATION_SCHEMA,
            "protocol_sha256": protocol,
            "sample_count": 1,
            "population": {"ground_truth_sha256_by_token": {"frame": "b" * 64}},
            "dev_AP101_supported_classes": 0.25,
        },
        "predictions": {"frame": []},
        "ground_truth_v2": {"frame": {"protocol_sha256": protocol}},
        "decode_counts": {"frame": {}},
    }
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    for update in (4096, 8192):
        (evaluation / f"full_selection_update_{update:06d}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    verified = module.verify_full_evaluations(
        tmp_path, {"evaluation_protocol_sha256": protocol, "full_selection_count": 1}
    )
    assert verified["4096"]["ground_truth_sha256_by_token"] == {"frame": "b" * 64}
    payload["ground_truth_v2"]["frame"]["protocol_sha256"] = "c" * 64
    (evaluation / "full_selection_update_008192.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="GT protocol"):
        module.verify_full_evaluations(
            tmp_path, {"evaluation_protocol_sha256": protocol, "full_selection_count": 1}
        )


def test_r0_child_failure_stops_without_r1_and_writes_durable_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = load_module()
    producer = tmp_path / "fake_producer.py"
    producer.write_text("raise SystemExit(99)\n", encoding="utf-8")
    source = tmp_path / "source.receipt"
    data = tmp_path / "data.receipt"
    source.write_text("source", encoding="utf-8")
    data.write_text("data", encoding="utf-8")
    r0_config = ROOT / "configs/detection_campaign/rgb_r0_d4096_seed7.yaml"
    r1_config = ROOT / "configs/detection_campaign/rgb_r1_d4096_seed7.yaml"
    plan = {
        "schema": module.SCHEMA,
        "python": binding(module, Path(sys.executable)),
        "producer": binding(module, producer),
        "gpu": "0",
        "r0": {
            "config": binding(module, r0_config),
            "output_1024": str(tmp_path / "r0_1024"),
            "output_8192": str(tmp_path / "r0_8192"),
        },
        "r1": {
            "config": binding(module, r1_config),
            "output": str(tmp_path / "r1_8192"),
        },
        "source_bindings": [binding(module, source)],
        "data_bindings": [binding(module, data)],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    calls = []

    def fail_child(**kwargs):
        calls.append(kwargs["name"])
        raise RuntimeError("synthetic R0 failure")

    monkeypatch.setattr(module, "run_child", fail_child)
    monkeypatch.setattr(module, "resource_preflight", lambda config, gpu: {"status": "PASS"})
    queue_output = tmp_path / "queue"
    with pytest.raises(RuntimeError, match="synthetic R0 failure"):
        module.run_queue(plan_path, module.sha256_file(plan_path), queue_output)
    assert calls == ["r0_1024"]
    assert not (tmp_path / "r1_8192").exists()
    receipt = json.loads((queue_output / "queue_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "FAILED"
    assert receipt["automatic_retries"] == 0
    assert receipt["active_segment"] == "r0_1024"


def test_recipe_pair_allows_only_preregistered_scheduler_change():
    module = load_module()
    r0 = module.load_campaign_config(ROOT / "configs/detection_campaign/rgb_r0_d4096_seed7.yaml")
    r1 = module.load_campaign_config(ROOT / "configs/detection_campaign/rgb_r1_d4096_seed7.yaml")
    module.verify_recipe_pair(r0, r1)


def test_r0_r1_population_gate_rejects_different_gt_or_protocol():
    module = load_module()
    base = {
        update: {
            "protocol_sha256": "a" * 64,
            "ground_truth_sha256_by_token": {"frame": "b" * 64},
        }
        for update in ("4096", "8192")
    }
    same = json.loads(json.dumps(base))
    module.verify_population_same(base, same)
    same["8192"]["ground_truth_sha256_by_token"]["frame"] = "c" * 64
    with pytest.raises(ValueError, match="population or protocol"):
        module.verify_population_same(base, same)


def test_output_containment_is_rejected(tmp_path: Path):
    module = load_module()
    with pytest.raises(ValueError, match="contain"):
        module.verify_disjoint_outputs((tmp_path / "queue", tmp_path / "queue/r0"))
