from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from operational.sota_eval import garl_parity


def test_canonical_hash_is_format_independent() -> None:
    assert garl_parity.canonical_sha256({"b": 2, "a": 1}) == garl_parity.canonical_sha256(
        {"a": 1, "b": 2}
    )


def test_evttc_diagnostic_counts_non_millisecond_windows(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "queries": [
                    {"windows_us": [[0, 1], [100_001, 200_001], [200_000, 300_000]]},
                    {"windows_us": [[0, 1], [400_000, 500_000], [500_000, 600_000]]},
                ]
            }
        ),
        encoding="utf-8",
    )
    result = garl_parity._evttc_diagnostic(path)
    assert result["endpoint_windows"] == 4
    assert result["non_millisecond_aligned_endpoint_windows"] == 1
    assert result["millisecond_floor_quantization_us"]["start_shift_range"] == [-1, 0]
    assert result["millisecond_floor_quantization_us"]["duration_change_range"] == [0, 0]
    assert result["event_pixel_diff"]["eap_upstream"] == 5
    assert result["event_pixel_diff"]["evttc_transfer"] == 0


def test_release_height_order_and_sign_are_native() -> None:
    from operational.train40_system.garl_predictions import release_ttc

    heights = np.asarray([[4.0, 5.0], [5.0, 4.0]], dtype=np.float32)
    result = release_ttc(heights, 0.1)
    np.testing.assert_allclose(result, [0.5, -0.4], rtol=1e-6, atol=0)


def test_run_rejects_preprocessing_drift(monkeypatch, tmp_path: Path) -> None:
    row = {"sequence_id": "s", "sample_token": "q"}
    monkeypatch.setattr(garl_parity, "_source_hashes", lambda *_: {"source": "a" * 64})
    monkeypatch.setattr(garl_parity, "_select_rows", lambda *args: [row, row, row])
    monkeypatch.setattr(
        garl_parity,
        "_reference_sensor",
        lambda *args: (np.zeros((46, 128, 128), np.float32), {}),
    )
    changed = np.zeros((46, 128, 128), np.float32)
    changed[0, 0, 0] = 1
    monkeypatch.setattr(
        garl_parity,
        "_local_sensor",
        lambda *args: (changed, changed[6:], {}),
    )
    monkeypatch.setattr(garl_parity, "_evttc_diagnostic", lambda _: {})
    monkeypatch.setattr(garl_parity, "digest", lambda _: "b" * 64)
    campaign = tmp_path / "campaign"
    full = tmp_path / "full"
    campaign.mkdir()
    full.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    args = argparse.Namespace(
        campaign=campaign,
        eap_root=tmp_path,
        code_root=tmp_path,
        full_public=full,
        evttc_manifest=manifest,
        output=tmp_path / "out",
        samples=3,
        run_models=False,
    )
    report = garl_parity.run(args)
    assert report["status"] == "FAILED"
    assert report["preprocessing_max_abs"] == 1.0
    assert (args.output / "SHA256.json").is_file()


def test_bind_existing_adds_local_lineage_without_model_rerun(
    monkeypatch, tmp_path: Path
) -> None:
    report = {
        "status": "PASSED",
        "samples": 3,
        "preprocessing_max_abs": 0.0,
        "records": [
            {"bit_exact": True, "reference_sensor_sha256": str(index) * 64}
            for index in range(1, 4)
        ],
        "models": {"full": {"strict_state_load": True}},
        "identities": {
            "event_checkpoint_sha256": "a" * 64,
            "full_checkpoint_sha256": "b" * 64,
        },
    }
    (tmp_path / "RESULT.json").write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "REPORT.md").write_text("passed\n", encoding="utf-8")
    monkeypatch.setattr(
        garl_parity, "_local_source_hashes", lambda: {"helper.py": "c" * 64}
    )
    binding = garl_parity.bind_existing(tmp_path)
    rebound = json.loads((tmp_path / "RESULT.json").read_text(encoding="utf-8"))
    assert rebound["identities"]["local_executed_helpers"] == {"helper.py": "c" * 64}
    assert binding["models_rerun_for_binding_only"] is False
    assert binding["passed_evidence"]["all_records_bit_exact"] is True
    assert set(json.loads((tmp_path / "SHA256.json").read_text())) == {
        "RESULT.json",
        "REPORT.md",
        "AUDIT_BINDING.json",
    }
