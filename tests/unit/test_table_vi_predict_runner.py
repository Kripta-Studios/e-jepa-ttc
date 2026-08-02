from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from e_jepa_ttc.data.garl_input_contract import (
    EVENT_CHANNEL_NAMES,
    INPUT_SCHEMA_VERSION,
    NORMALIZATION_ID,
)
from scripts.evaluate_garl_evttc_table_vi import main as evaluate_main
from scripts.evaluate_garl_evttc_table_vi import predict_preflight
from scripts.predict_garl_evttc_table_vi import predict, run_label_free_inference


def _write(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _row() -> dict[str, object]:
    return {
        "sequence_id": "table-vi-seq-1",
        "sample_token": "sample-1",
        "track_id": "track-1",
        "timestamp_us": 1000,
        "predicted_ttc_s": 1.25,
        "modalities": ["events", "rgb", "boxes"],
    }


def test_predict_emits_label_free_payload(tmp_path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "predictions.json"
    _write(source, {"predictions": [_row()]})

    result = predict(source, output, protocol_id="garl_evttc_table_vi_v1")

    assert result["target_labels_opened"] is False
    assert result["selection_uses_evttc"] is False
    assert result["sample_count"] == 1
    assert result["predictions"][0]["predicted_ttc_s"] == 1.25
    assert "ttc_s" not in json.loads(output.read_text(encoding="utf-8"))["predictions"][0]


def test_predict_rejects_nested_target_and_duplicate_identity(tmp_path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "predictions.json"
    row = _row()
    _write(source, {"predictions": [{**row, "model_input": {"target_ttc_s": 1.0}}]})
    with pytest.raises(ValueError, match="forbidden target field"):
        predict(source, output, protocol_id="garl_evttc_table_vi_v1")

    _write(source, {"predictions": [row, row]})
    with pytest.raises(ValueError, match="Duplicate prediction identity"):
        predict(source, output, protocol_id="garl_evttc_table_vi_v1")


class _MeanModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.mean(dim=1)


def _inference_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    shard = tmp_path / "inputs.npz"
    np.savez(
        shard,
        sequence_id=np.asarray(["seq-a", "seq-a", "seq-b"]),
        sample_token=np.asarray(["token-a0", "token-a1", "token-b0"]),
        track_id=np.asarray(["track-a", "track-a", "track-b"]),
        timestamp_us=np.asarray([1000, 2000, 3000], dtype=np.int64),
        inputs=np.asarray([[3.0], [5.0], [7.0]], dtype=np.float32),
    )
    manifest = tmp_path / "inputs.json"
    _write(
        manifest,
        {
            "artifact_type": "garl_evttc_table_vi_label_free_inputs_v1",
            "token_schema": ["sequence_id", "sample_token", "track_id", "timestamp_us"],
            "input_schema": {
                "version": INPUT_SCHEMA_VERSION,
                "event_roi_shape": [2, 20, 128, 128],
                "channel_names": list(EVENT_CHANNEL_NAMES),
                "normalization": NORMALIZATION_ID,
            },
            "shards": [{"path": shard.name, "sample_count": 3}],
        },
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "model:",
                "  architecture: torchscript",
                "  input_key: inputs",
                "inference:",
                f"  input_manifest: {manifest.name}",
                "  batch_size: 2",
                "  device: cpu",
            ]
        ),
        encoding="utf-8",
    )
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        "\n".join(
            [
                "protocol_id: garl_evttc_table_vi_test_v1",
                "predict_score_separation: true",
                "labels_used_by_predict: false",
                "selection_uses_evttc: false",
                "bbox_protocol: P0_oracle_bbox_roi",
                "token_schema: [sequence_id, sample_token, track_id, timestamp_us]",
                "sequences:",
                "  - local_sequence_id: seq-a",
                "    sample_count: 2",
                "  - local_sequence_id: seq-b",
                "    sample_count: 1",
            ]
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "model.ts"
    traced = torch.jit.trace(_MeanModel(), torch.ones(1, 1))
    with checkpoint.open("wb") as handle:
        torch.jit.save(traced, handle)
    return checkpoint, config, protocol, manifest


def test_checkpoint_pipeline_runs_without_targets_and_applies_explicit_normalization(
    tmp_path: Path,
) -> None:
    checkpoint, config, protocol, _manifest = _inference_fixture(tmp_path)
    normalization = tmp_path / "normalization.json"
    _write(normalization, {"inputs": {"mean": 1.0, "std": 2.0}})
    output = tmp_path / "predictions.json"

    result = run_label_free_inference(
        checkpoint,
        config,
        protocol,
        output,
        normalization_path=normalization,
    )

    assert result["target_labels_opened"] is False
    assert result["training_updates_on_target_dataset"] == 0
    assert result["coverage_by_sequence"] == {"seq-a": 2, "seq-b": 1}
    assert result["normalization_explicit"] is True
    assert result["bbox_protocol"] == "P0_oracle_bbox_roi"
    assert [row["predicted_ttc_s"] for row in result["predictions"]] == [1.0, 2.0, 3.0]
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert all("target_ttc_s" not in row for row in persisted["predictions"])


@pytest.mark.parametrize("forbidden_key", ["target_ttc_s", "category_index", "foreground_mask"])
def test_checkpoint_pipeline_rejects_privileged_arrays_before_loading_checkpoint(
    tmp_path: Path, forbidden_key: str
) -> None:
    checkpoint, config, protocol, manifest = _inference_fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    shard = tmp_path / payload["shards"][0]["path"]
    with np.load(shard, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    arrays[forbidden_key] = np.ones(3, dtype=np.float32)
    np.savez(shard, **arrays)
    checkpoint.unlink()

    with pytest.raises(ValueError, match="forbidden target field"):
        run_label_free_inference(checkpoint, config, protocol, tmp_path / "out.json")


def test_checkpoint_pipeline_rejects_coverage_and_token_schema(tmp_path: Path) -> None:
    checkpoint, config, protocol, manifest = _inference_fixture(tmp_path)
    protocol.write_text(
        protocol.read_text(encoding="utf-8").replace("sample_count: 2", "sample_count: 3"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Protocol coverage mismatch"):
        run_label_free_inference(checkpoint, config, protocol, tmp_path / "coverage.json")

    protocol.write_text(
        protocol.read_text(encoding="utf-8").replace(
            "token_schema: [sequence_id, sample_token, track_id, timestamp_us]",
            "token_schema: [sample_token]",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="token_schema must be exactly"):
        run_label_free_inference(checkpoint, config, protocol, tmp_path / "schema.json")

    # The manifest itself is also guarded, independently of the protocol.
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["token_schema"] = ["sample_token"]
    _write(manifest, manifest_payload)
    protocol.write_text(
        protocol.read_text(encoding="utf-8")
        .replace("sample_count: 3", "sample_count: 2")
        .replace(
            "token_schema: [sample_token]",
            "token_schema: [sequence_id, sample_token, track_id, timestamp_us]",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="token_schema must be exactly"):
        run_label_free_inference(checkpoint, config, protocol, tmp_path / "manifest-schema.json")


def test_checkpoint_pipeline_requires_explicit_scientific_declarations(tmp_path: Path) -> None:
    checkpoint, config, protocol, _manifest = _inference_fixture(tmp_path)
    protocol_text = protocol.read_text(encoding="utf-8")
    protocol.write_text(
        protocol_text.replace("selection_uses_evttc: false", "selection_uses_evttc: true"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="selection_uses_evttc=false"):
        run_label_free_inference(checkpoint, config, protocol, tmp_path / "selection.json")

    protocol.write_text(
        protocol_text.replace("bbox_protocol: P0_oracle_bbox_roi\n", ""),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="bbox_protocol"):
        run_label_free_inference(checkpoint, config, protocol, tmp_path / "bbox.json")


def _evaluation_protocol(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "protocol_id: garl_evttc_table_vi_test_v1",
                "zero_shot_completed: false",
                "predict_score_separation: true",
                "labels_used_by_predict: false",
                "selection_uses_evttc: false",
                "token_schema: [sequence_id, sample_token, track_id, timestamp_us]",
            ]
        ),
        encoding="utf-8",
    )


def test_plan_predict_alias_fails_before_opening_checkpoint_or_manifests(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.yaml"
    _evaluation_protocol(protocol)

    with pytest.raises(ValueError, match="preflight-only.*real inference config"):
        predict_preflight(
            tmp_path / "missing-checkpoint.pt",
            protocol,
            [tmp_path / "missing-manifest.yaml"],
            tmp_path / "predictions.json",
        )


def test_plan_score_alias_requires_explicit_targets(tmp_path: Path, capsys) -> None:
    protocol = tmp_path / "protocol.yaml"
    _evaluation_protocol(protocol)

    with pytest.raises(SystemExit) as raised:
        evaluate_main(
            [
                "score",
                "--predictions",
                str(tmp_path / "predictions.json"),
                "--protocol",
                str(protocol),
                "--output",
                str(tmp_path / "metrics.json"),
            ]
        )

    assert raised.value.code == 2
    assert "requires explicit --targets" in capsys.readouterr().err


def test_legacy_flat_score_cli_remains_compatible(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.json"
    targets = tmp_path / "targets.json"
    output = tmp_path / "metrics.json"
    identity = {
        "sequence_id": "seq-a",
        "sample_token": "token-a",
        "track_id": "track-a",
        "timestamp_us": 1000,
    }
    _write(predictions, {"predictions": [{**identity, "predicted_ttc_s": 1.0}]})
    _write(targets, {"targets": [{**identity, "target_ttc_s": 1.0}]})

    exit_code = evaluate_main(
        [
            "--predictions",
            str(predictions),
            "--targets",
            str(targets),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert output.is_file()
