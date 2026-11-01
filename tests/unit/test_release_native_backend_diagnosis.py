"""Platform diagnostics cannot bypass scientific gates or consume unauthenticated histories."""

import hashlib
import io
import json
import tarfile

import numpy as np
import pytest

from operational.final_refit.test12 import TRUSTED_INPUTS_SHA256
from scripts.release_native_backend_diagnosis import admit, archived_contexts


def admission_inputs():
    return (
        {"outcome": "native_cuda_profile_parity_failed_candidate_not_run"},
        {
            "device": "cuda",
            "profile_queries": 2,
            "targets_read": False,
            "candidate_evaluated": False,
        },
        {"historical_parity": "FAIL", "candidate_executed": False},
    )


def test_terminal_original_only_failure_admits_diagnosis():
    admit(*admission_inputs())


@pytest.mark.parametrize(
    "index,key,value",
    [
        (0, "outcome", "cadence_not_admitted"),
        (1, "device", "cpu"),
        (1, "targets_read", True),
        (2, "candidate_executed", True),
    ],
)
def test_diagnosis_cannot_bypass_population_or_original_only_scope(index, key, value):
    values = admission_inputs()
    values[index][key] = value
    with pytest.raises(ValueError, match="terminal original-only"):
        admit(*values)


def archived_fixture(tmp_path, corruption=None):
    query = {
        "sample_token": "q",
        "event_history": ["e"],
        "rgb_history": ["r"],
        "query_time_us": 100,
        "sequence_id": "s",
        "track_id": 1,
    }
    buffer = io.BytesIO()
    np.savez(buffer, allow_pickle=False, **{"E_CTX_MATCHED.input_0": np.array([42], np.float32)})
    payload = buffer.getvalue()
    metadata = {
        **query,
        "private_targets_read": False,
        "run_binding_sha256": "binding",
        "npz_sha256": hashlib.sha256(payload).hexdigest(),
    }
    if corruption == "checksum":
        metadata["npz_sha256"] = "0" * 64
    elif corruption == "history":
        metadata["query_time_us"] += 1
    elif corruption == "targets":
        metadata["private_targets_read"] = True
    admission = {
        "model_seal_sha256": "model",
        "inputs_sha256": TRUSTED_INPUTS_SHA256,
        "run_binding_sha256": "binding",
    }
    path = tmp_path / "archive.tar"
    with tarfile.open(path, "w") as archive:
        for name, content in {
            "run/s.admission.json": json.dumps(admission).encode(),
            "run/diagnostics/q.json": json.dumps(metadata).encode(),
            "run/diagnostics/q.npz": payload,
        }.items():
            item = tarfile.TarInfo(name)
            item.size = len(content)
            archive.addfile(item, io.BytesIO(content))
    return (
        [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}],
        {"q": query},
        {"model_seal_sha256": "model"},
    )


def test_historical_context_is_bound_to_exact_query_and_payload(tmp_path):
    value = archived_contexts(*archived_fixture(tmp_path))
    np.testing.assert_array_equal(value["q"]["E_CTX_MATCHED.input_0"], np.array([42], np.float32))


@pytest.mark.parametrize("corruption", ["checksum", "history", "targets"])
def test_tampered_or_target_exposed_archive_is_rejected(tmp_path, corruption):
    with pytest.raises(ValueError):
        archived_contexts(*archived_fixture(tmp_path, corruption))
