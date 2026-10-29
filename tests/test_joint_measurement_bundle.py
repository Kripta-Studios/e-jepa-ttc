"""Bundle integrity rejects omitted, extra, altered and unsafe archive contents."""

import json
import shutil
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch

from operational.joint_measurement.bundle import (
    MANIFEST,
    safe_name,
    verify_archive,
    write_archive,
)
from operational.joint_measurement.replay import file_hash, replay, verify_extraction
from operational.range_transfer.analyze import metrics


def fixture_bundle(tmp_path):
    data = tmp_path / "weights.bin"
    data.write_bytes(bytes(range(256)) * 100)
    document = tmp_path / "status.json"
    document.write_text('{"status": "RESEARCH_ONGOING"}', encoding="utf-8")
    output = tmp_path / "result.zip"
    files = {"campaign/weights.bin": data, "campaign/status.json": document}
    result = write_archive(output, files, {"scientific_claim": False})
    return output, result


def test_bundle_exact_payload_and_independent_reopen(tmp_path):
    output, created = fixture_bundle(tmp_path)
    verified = verify_archive(output)
    assert created == verified
    assert verified["status"] == "PASS"
    assert verified["payload_members"] == 2
    assert verified["context"]["scientific_claim"] is False
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read(MANIFEST))
        assert manifest["files"]["campaign/weights.bin"]["bytes"] == 25600


@pytest.mark.parametrize("fault", ["missing", "extra", "changed"])
def test_bundle_detects_membership_and_content_changes(tmp_path, fault):
    original, _ = fixture_bundle(tmp_path)
    target = tmp_path / "broken.zip"
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(target, "w") as output:
        for name in source.namelist():
            if fault == "missing" and name == "campaign/weights.bin":
                continue
            content = source.read(name)
            if fault == "changed" and name == "campaign/weights.bin":
                content = b"modified" + content[8:]
            output.writestr(name, content)
        if fault == "extra":
            output.writestr("campaign/undeclared.txt", "extra")
    with pytest.raises(ValueError, match="differs"):
        verify_archive(target)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/absolute", "a\\b", "a/../b"])
def test_bundle_rejects_unsafe_names(name):
    with pytest.raises(ValueError, match="Unsafe"):
        safe_name(name)


def test_bundle_refuses_windows_case_alias_and_existing_destination(tmp_path):
    output, _ = fixture_bundle(tmp_path)
    source = tmp_path / "weights.bin"
    with pytest.raises(FileExistsError):
        write_archive(output, {"one": source}, {})
    with pytest.raises(ValueError, match="Duplicate"):
        write_archive(tmp_path / "case.zip", {"File": source, "file": source}, {})


def test_independent_extraction_and_altered_copy(tmp_path):
    original, _ = fixture_bundle(tmp_path)
    independent = tmp_path / "independent"
    with zipfile.ZipFile(original) as archive:
        archive.extractall(independent)
    assert verify_extraction(independent)["payload_members"] == 2
    (independent / "campaign/weights.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Extracted file differs"):
        verify_extraction(independent)


def test_replay_included_predictions_and_real_torchscript_bytes(tmp_path):
    campaign = tmp_path / "input/campaign"
    predictions = campaign / "predictions/P/fixture"
    export = campaign / "exports/fixture"
    predictions.mkdir(parents=True)
    export.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "sample_token": ["a", "b", "c", "d"],
            "group_id": ["g"] * 4,
            "target_ttc": [0.7, 4.0, 8.0, -3.0],
            "prediction_ttc": [0.8, 4.5, 7.0, -2.5],
        }
    )
    frame.to_parquet(predictions / "predictions.parquet", index=False)

    def write(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    write(
        predictions / "METRICS.json",
        metrics(frame.target_ttc.to_numpy(), frame.prediction_ttc.to_numpy(), frame.group_id),
    )
    write(
        predictions / "COMPLETE.json",
        {"status": "COMPLETE", "rows": 4, "sha256": file_hash(predictions / "predictions.parquet")},
    )

    class Fixture(torch.nn.Module):
        def forward(self, events, delta):
            return events.mean((1, 2, 3, 4)) + delta.reshape(-1)

    inputs = (torch.ones(1, 2, 3, 4, 4), torch.tensor([[0.25]]))
    with (export / "model.torchscript.pt").open("wb") as stream:
        torch.jit.save(torch.jit.trace(Fixture(), inputs), stream)
    np.savez(export / "example_input.npz", events=inputs[0].numpy(), delta_t_s=inputs[1].numpy())
    write(export / "example_output.json", {"signed_ttc_seconds": [1.25]})
    write(export / "BINDING.json", {"atol_seconds": 1e-5, "rtol": 1e-5})
    write(
        export / "COMPLETE.json",
        {"status": "COMPLETE", "files": {p.name: file_hash(p) for p in export.iterdir()}},
    )
    # Later campaign branches retain their own artifacts, even for equal fit names.
    shutil.copytree(predictions, campaign / "rgb/predictions/P/fixture")
    shutil.copytree(export, campaign / "rgb/exports/fixture")
    rgb_export = campaign / "rgb/exports/fixture"
    np.savez(rgb_export / "example_input.npz", rgb=inputs[0].numpy(), delta_t_s=inputs[1].numpy())
    write(
        rgb_export / "BINDING.json",
        {
            "atol_seconds": 1e-5,
            "rtol": 1e-5,
            "pixel_input_key": "rgb",
        },
    )
    write(
        rgb_export / "COMPLETE.json",
        {
            "status": "COMPLETE",
            "files": {
                p.name: file_hash(p) for p in rgb_export.iterdir() if p.name != "COMPLETE.json"
            },
        },
    )
    files = {
        p.relative_to(tmp_path / "input").as_posix(): p for p in campaign.rglob("*") if p.is_file()
    }
    archive = tmp_path / "portable.zip"
    write_archive(archive, files, {"fixture": True, "optimizer_updates": 0})
    detached = tmp_path / "detached"
    with zipfile.ZipFile(archive) as stream:
        stream.extractall(detached)
    result = replay(detached)
    assert result["rows_recalculated"] == 8
    assert result["portable_exports"]["fixture"]["bit_exact"]
    assert result["portable_exports"]["rgb/exports/fixture"]["bit_exact"]
    assert set(result["prediction_tables"]) == {"P/fixture", "rgb/predictions/P/fixture"}
    assert result["optimizer_updates"] == 0
