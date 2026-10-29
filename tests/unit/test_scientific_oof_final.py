# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from operational.final_refit import scientific_oof_final as final
from operational.final_refit import scientific_oof_pipeline as pipeline
from operational.final_refit import test12 as test12_module


def _json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_oof_heads_have_no_fake_checkpoint_parents() -> None:
    declared = final.StageRecipe(
        stage_id="E_CTX_MATCHED",
        kind="temporal_head",
        updates=1,
        batch_size=1,
        weight_decay=0.0,
        depends_on=("PAIR_E_MATCHED",),
        fresh_initialization=True,
        cache_schema="cache",
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(final, "stage", lambda _recipe, _stage: declared)
        result = final._oof_stage(cast(final.FinalRefitRecipe, SimpleNamespace()), "E_CTX_MATCHED")
    assert result.depends_on == ()


def test_final_receipt_requires_endpoints_admissions_and_fragments(tmp_path: Path) -> None:
    run = tmp_path / "run"
    stages = (
        "PAIR_E_MATCHED",
        "PAIR_R",
        "E_CTX_MATCHED",
        "R_CTX",
        "PHASE_ONLY_EVENT_S7_TRAIN40",
    )
    for name in stages:
        _json(
            run / "final" / "fits" / name / "CHECKPOINT_RECEIPT.json",
            {"status": "COMPLETE", "completed_updates": 7},
        )
    paused = final._write_final_receipt(run, "ignored", {}, 100)
    assert paused["status"] == "RUNNING"
    _json(run / "rotations/0/fits/E_A5/ENDPOINT_ADMISSION.json", {"status": "PASS"})
    fragment = run / "rotations/0/generation/E_CTX/fragment.npz"
    fragment.parent.mkdir(parents=True, exist_ok=True)
    fragment.write_bytes(b"fragment")
    running = final._write_final_receipt(run, "ignored", {}, 100)
    artifact = run / "final/artifact.bin"
    artifact.write_bytes(b"done")
    with pytest.raises(ValueError, match="exact seven deliverables"):
        final._complete_final_receipt(run, {"artifact": artifact})
    assert running["status"] == "RUNNING"
    assert running["stage_id"] == "FINAL"
    assert running["scientific_endpoint"] is False
    assert running["all_head_endpoints_complete"] is True
    assert running["all_oof_inputs_complete"] is False
    assert running["completed_updates"] == running["durable_updates"] == 35
    assert running["admissions_sha256"] and running["fragment_files_sha256"]
    assert len(running["fragment_sha256"]) == 64


class _Predictor:
    def __init__(self) -> None:
        self.closed = False

    def predict(self, query: Mapping[str, Any], inputs: Mapping[str, Any]) -> float:
        del inputs
        return float(query["value"])

    def close(self) -> None:
        self.closed = True


def test_scientific_test12_is_seal_bound_resumable_and_keeps_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    seal = run / "final/MODEL_SEAL.json"
    _json(seal, {"unused": True})
    paths = {
        name: tmp_path / name
        for name in (
            "test12_inputs",
            "official_test_inputs",
            "model_input_freeze",
            "event_input_freeze",
        )
    }
    for path in paths.values():
        path.write_bytes(b"bound")
    monkeypatch.setattr(final, "_config", lambda _run: paths)
    monkeypatch.setattr(final, "_scientific_seal", lambda _path: ({}, "a" * 64))
    monkeypatch.setattr(pipeline, "make_guard", lambda _run: lambda: None)
    monkeypatch.setattr(
        final,
        "validate_test12_inputs",
        lambda *_args, **_kwargs: {
            "inputs_sha256": "b" * 64,
            "canonical_tokens": ["s:1", "s:2"],
            "tokens": ["s:1", "s:2"],
            "fallback_count": 1,
            "inputs": {
                "physical_aliases": {"s:1": "s:1", "s:2": "s:2"},
                "queries": [
                    {"sample_token": "s:1", "rgb_history": ["r"], "value": 1.5},
                    {"sample_token": "s:2", "rgb_history": [], "value": -2.0},
                ],
            },
        },
    )
    predictor = _Predictor()
    predictions = run / "final/test12/predictions.csv"
    receipt = run / "final/test12/receipt.json"
    result = final.run_scientific_test12(
        run=run,
        predictor=predictor,
        predictions_output=predictions,
        receipt_output=receipt,
    )
    assert predictor.closed is True
    assert result["computed_under_seal"] is True
    assert result["private_targets_read"] is False
    assert result["fallback_count"] == 1
    text = predictions.read_text(encoding="utf-8")
    assert "s:1,1.5,True,PHASE_ONLY" in text
    assert "s:2,-2.0,False,E_CTX" in text


def test_scientific_seal_rejects_incomplete_deployment_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(final, "_config", lambda _run: {"recipe": recipe_path})
    monkeypatch.setattr(final, "load_recipe", lambda _path: SimpleNamespace(recipe_sha256="f" * 64))
    with pytest.raises(ValueError, match="exact full-fit deployment graph"):
        final.seal_scientific_final(tmp_path, deployment_dependencies={"E_A5": "a" * 64})


def test_existing_package_must_equal_current_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    official = tmp_path / "test.parquet"
    official.write_bytes(b"official")
    monkeypatch.setattr(final, "OFFICIAL_TEST_QUERIES", 2)
    monkeypatch.setattr(final, "_config", lambda _run: {"official_test_inputs": official})
    monkeypatch.setattr(final, "_scientific_seal", lambda _path: ({}, "a" * 64))
    monkeypatch.setattr(
        final.pd,
        "read_parquet",
        lambda *_args, **_kwargs: final.pd.DataFrame({"sample_token": ["a", "b"]}),
    )
    predictions = run / "final/test12/predictions.csv"
    predictions.parent.mkdir(parents=True)
    final.pd.DataFrame({"sample_token": ["a", "b"], "prediction": [1.0, 2.0]}).to_csv(
        predictions, index=False
    )
    prediction_receipt = run / "final/test12/PREDICTION_RECEIPT.json"
    _json(
        prediction_receipt,
        {
            "status": "COMPLETE",
            "model_seal_sha256": "a" * 64,
            "predictions_sha256": final.sha256(predictions),
        },
    )
    _json(run / "final/MODEL_SEAL.json", {})
    package = run / "final/test12/PHASE_ONLY_S7_TRAIN40_SCIENTIFIC_OOF.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr(
            "submission.json",
            json.dumps(
                {
                    "meta": {"format": "garlttc_prediction_v1"},
                    "results": {"a": {"ttc": 9.0}, "b": {"ttc": 9.0}},
                }
            ),
        )
    _json(
        package.with_suffix(".json"),
        {
            "schema": "scientific_oof_test12_package_v1",
            "status": "PACKAGED_NOT_SUBMITTED",
            "candidate": final.self_candidate(),
            "model_seal_sha256": "a" * 64,
            "prediction_receipt_sha256": final.sha256(prediction_receipt),
            "population": 2,
            "zip_sha256": final.sha256(package),
            "private_targets_read": False,
            "external_submission": False,
        },
    )
    with pytest.raises(ValueError, match="content differs"):
        final.package_scientific_predictions(run)


def test_scientific_predictor_keeps_rgb_event_order_and_event_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = cast(Any, object.__new__(final.ScientificOOFFinalPredictor))
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    class Head:
        def __call__(self, features: torch.Tensor, width: torch.Tensor) -> dict[str, torch.Tensor]:
            calls.append((features.clone(), width.clone()))
            return {"ttc": torch.tensor([4.0])}

    predictor.models = {
        "E_CTX_MATCHED": "event",
        "R_CTX": "rgb",
        "PHASE_ONLY_EVENT_S7_TRAIN40": Head(),
    }
    monkeypatch.setattr(
        final.ScientificOOFFinalPredictor,
        "_context",
        lambda *_args, **_kwargs: (torch.zeros(1),) * 4,
    )

    def frozen(
        model: object, *_args: object
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if model == "rgb":
            return torch.ones(1, 170), torch.tensor([[0.2, 0.3]]), {}
        return (
            torch.full((1, 170), 2.0),
            torch.zeros(1, 2),
            {"point_phase": torch.tensor([0.5])},
        )

    monkeypatch.setattr(test12_module, "frozen_context_state", frozen)
    multimodal = predictor.predict(
        {"query_time_us": 1, "event_history": ["e"], "rgb_history": ["r"]}, {}
    )
    assert multimodal == 4.0
    assert torch.equal(calls[0][0][:, :170], torch.ones(1, 170))
    assert torch.equal(calls[0][0][:, 170:], torch.full((1, 170), 2.0))
    assert torch.equal(calls[0][1], torch.tensor([[0.2, 0.3]]))
    calls.clear()
    fallback = predictor.predict(
        {"query_time_us": 1, "event_history": ["e"], "rgb_history": []}, {}
    )
    assert fallback > 0
    assert calls == []
