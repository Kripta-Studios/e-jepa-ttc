"""Fixed baseline loading requires independent identities and exact published bytes."""

import json

import numpy as np
import pytest
import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.simplex_t.fixed_baseline_inputs import load_fixed_baselines
from e_jepa_ttc.simplex_t.phase import phase_to_ttc


@pytest.fixture
def saved(tmp_path):
    tokens, sequences = np.array(["q0", "q1"]), np.array(["s", "s"])
    history = np.array([[-1] * 7 + [0], list(range(8))], np.int64)
    phase = np.array([0.02, 0.03], np.float32)
    payload = {"sample_token": tokens, "sequence_id": sequences, "history": history}
    for name in ("CURRENT_MEDIAN", "EWMA_0P3S_H8"):
        payload[f"{name}_prediction_phase"] = phase
        payload[f"{name}_prediction_ttc_s"] = phase_to_ttc(torch.from_numpy(phase)).numpy()
    file = tmp_path / "outer_dev.npz"
    np.savez(file, **payload)
    report = {
        "status": "FIXED_BASELINES_ONE_FOLD_COMPLETE_NOT_SCORED",
        "optimizer_updates": 0,
        "scores_computed": False,
        "roles": {
            "inner_oof": {},
            "outer_dev": {
                "path": file.name,
                "queries": 2,
                "sha256": compute_file_hash(str(file)),
                "source_sha256": "a" * 64,
            },
        },
    }
    manifest = tmp_path / "BASELINES.json"
    manifest.write_text(json.dumps(report), encoding="utf-8")
    args = dict(
        manifest=manifest,
        manifest_sha256=compute_file_hash(str(manifest)),
        role="outer_dev",
        source_sha256="a" * 64,
        tokens=tokens.copy(),
        sequences=sequences.copy(),
        history=history.copy(),
        validate_prerequisites=lambda: None,
    )
    return args, payload, report, file


def test_verified_payload_retains_exact_phase_and_ttc(saved):
    args, payload, _, _ = saved
    actual = load_fixed_baselines(**args)
    for key in payload:
        np.testing.assert_array_equal(actual[key], payload[key])


@pytest.mark.parametrize("fault", ["source", "tokens", "history", "bytes", "role"])
def test_mismatched_inputs_or_mutation_rejected(saved, fault):
    args, _, _, file = saved
    if fault == "source":
        args["source_sha256"] = "b" * 64
    elif fault == "tokens":
        args["tokens"] = args["tokens"][::-1]
    elif fault == "history":
        args["history"][0, -1] = 1
    elif fault == "role":
        args["role"] = "confirmation"
    else:
        file.write_bytes(b"corrupt fixture")
    with pytest.raises(ValueError):
        load_fixed_baselines(**args)


def test_inconsistent_ttc_rejected_even_with_matching_file_hashes(saved):
    args, payload, report, file = saved
    payload["CURRENT_MEDIAN_prediction_ttc_s"] = payload["CURRENT_MEDIAN_prediction_ttc_s"] + 1
    np.savez(file, **payload)
    report["roles"]["outer_dev"]["sha256"] = compute_file_hash(str(file))
    args["manifest"].write_text(json.dumps(report), encoding="utf-8")
    args["manifest_sha256"] = compute_file_hash(str(args["manifest"]))
    with pytest.raises(ValueError, match="emitted TTC"):
        load_fixed_baselines(**args)


def test_explicit_float64_emission_matches_head_export(saved):
    args, payload, report, file = saved
    for name in ("CURRENT_MEDIAN", "EWMA_0P3S_H8"):
        phase = payload[f"{name}_prediction_phase"].astype(np.float64)
        payload[f"{name}_prediction_ttc_s"] = phase_to_ttc(torch.from_numpy(phase)).numpy()
    np.savez(file, **payload)
    report["ttc_emission_dtype"] = "float64"
    report["roles"]["outer_dev"]["sha256"] = compute_file_hash(str(file))
    args["manifest"].write_text(json.dumps(report), encoding="utf-8")
    args["manifest_sha256"] = compute_file_hash(str(args["manifest"]))
    result = load_fixed_baselines(**args)
    assert result["CURRENT_MEDIAN_prediction_ttc_s"].dtype == np.float64
