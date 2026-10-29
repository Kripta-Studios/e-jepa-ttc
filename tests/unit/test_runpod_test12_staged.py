"""Focused safety checks for the staged public-input adapter."""

import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from scripts import runpod_test12_staged as staged


def test_raw_eviction_requires_verified_lineage():
    receipt = {
        "schema": "test12_raw_source_verified_v1",
        "status": "SHA256_VERIFIED",
        "relative_path": "data/test/seq/events.h5",
        "sha256": "a" * 64,
        "size_bytes": 100,
        "revision": staged.MEDIA_REVISION,
        "inputs_sha256": "inputs",
        "input_freeze_sha256": "freeze",
        "storage_state": "EVICTED_AFTER_TENSOR_VERIFICATION",
    }
    pin = {"path": receipt["relative_path"], "sha256": "a" * 64, "bytes": 100}
    assert staged.verify_raw_receipt(receipt, "seq", "inputs", "freeze", pin) == "a" * 64
    for key, value in (
        ("status", "DOWNLOADED"),
        ("input_freeze_sha256", "wrong"),
        ("sha256", "g" * 64),
        ("revision", "wrong-revision"),
        ("relative_path", "data/train/seq/events.h5"),
    ):
        altered = {**receipt, key: value}
        with pytest.raises(ValueError):
            staged.verify_raw_receipt(altered, "seq", "inputs", "freeze", pin)


def test_rgb_allows_mtime_relocation_but_not_byte_or_revision_change():
    frozen = {"sha256": "a" * 64, "size_bytes": 10, "revision": "revision", "mtime_ns": 1}
    receipt = {
        **frozen,
        "schema": "ejepa_test12_rgb_shard_receipt_v1",
        "status": "PASS",
        "relative_path": "data/test/seq/rgb.tar",
        "inputs_sha256": "inputs",
        "actual_mtime_ns": 99,
    }
    staged.verify_rgb_receipt(receipt, "data/test/seq/rgb.tar", frozen, "inputs")
    for key, value in (("sha256", "b" * 64), ("size_bytes", 11), ("revision", "other")):
        with pytest.raises(ValueError):
            staged.verify_rgb_receipt(
                {**receipt, key: value}, "data/test/seq/rgb.tar", frozen, "inputs"
            )


def test_containment_rejects_traversal(tmp_path):
    with pytest.raises(ValueError):
        staged.contained(tmp_path, "../outside.json")
    assert staged.contained(tmp_path, "data/test/a") == tmp_path / "data/test/a"


def test_frozen_validator_ast_retains_scientific_gates():
    # Compile the real frozen function without importing the GPU runtime.
    source = Path("operational/final_refit/test12.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "validate_test12_inputs"
    )
    namespace = {
        name: "hash"
        for name in (
            "TRUSTED_INPUTS_SHA256",
            "TRUSTED_OFFICIAL_TEST_INPUTS_SHA256",
            "TRUSTED_MODEL_INPUT_FREEZE_SHA256",
            "TRUSTED_EVENT_INPUT_FREEZE_SHA256",
        )
    }
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            "operational/final_refit/test12.py",
            "exec",
            flags=__import__("__future__").annotations.compiler_flag,
        ),
        namespace,
    )
    clone = staged.structural_validator(namespace["validate_test12_inputs"])
    names = clone.__code__.co_names
    assert "_history_is_causal" in names
    assert "_contains_private_key" in names
    assert "OFFICIAL_TEST_QUERIES" in names
    assert "read_parquet" in names
    assert "sha256" in names  # original four file/provenance hashes remain
    assert "stat" not in names  # physical checks are done by admit_sequence


def test_sequence_admission_exact_tensor_and_rgb_sha(tmp_path):
    events, rgb = tmp_path / "events", tmp_path / "rgb"
    tensor = events / "tensors/query_0.npz"
    tensor.parent.mkdir(parents=True)
    np.savez_compressed(tensor, valid=np.array([True]), events=np.ones((1, 3, 2, 2)))
    tensor_sha = staged.digest(tensor)
    raw = events / "raw_receipts/seq.json"
    staged.atomic(
        raw,
        {
            "schema": "test12_raw_source_verified_v1",
            "status": "SHA256_VERIFIED",
            "relative_path": "data/test/seq/events.h5",
            "sha256": "a" * 64,
            "size_bytes": 100,
            "revision": staged.MEDIA_REVISION,
            "inputs_sha256": "inputs",
            "input_freeze_sha256": "freeze",
            "storage_state": "EVICTED_AFTER_TENSOR_VERIFICATION",
        },
    )
    sidecar = tensor.with_suffix(".json")
    staged.atomic(
        sidecar,
        {
            "sha256": tensor_sha,
            "sample_token": "token",
            "input_freeze_sha256": "freeze",
            "inputs_sha256": "inputs",
            "raw_relative_path": "data/test/seq/events.h5",
            "raw_sha256": "a" * 64,
        },
    )
    staged.atomic(
        events / "sequences/seq.json",
        {
            "status": "COMPLETE",
            "raw_receipt_path": str(raw),
            "tensors": {
                "token": {"path": str(tensor), "sha256": tensor_sha, "receipt_path": str(sidecar)}
            },
        },
    )
    shard = rgb / "data/test/seq/rgb_shards/0.tar"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"fake shard")
    relative = "data/test/seq/rgb_shards/0.tar"
    frozen = {
        "sha256": staged.digest(shard),
        "size_bytes": shard.stat().st_size,
        "revision": "rev",
        "mtime_ns": 1,
    }
    rgb_receipt = rgb / "receipts/seq/0.json"
    staged.atomic(
        rgb_receipt,
        {
            **frozen,
            "schema": "ejepa_test12_rgb_shard_receipt_v1",
            "status": "PASS",
            "relative_path": relative,
            "inputs_sha256": "inputs",
        },
    )
    staged.atomic(
        rgb / "sequences/seq.json",
        {"status": "READY", "source_receipts": {relative: str(rgb_receipt)}},
    )
    validated = {
        "inputs_sha256": "inputs",
        "frozen_validator_source_sha256": "validator",
        "media_inventory": {
            "data/test/seq/events.h5": {
                "path": "data/test/seq/events.h5",
                "sha256": "a" * 64,
                "bytes": 100,
            }
        },
        "input_provenance": {"event_input_freeze_sha256": "freeze"},
        "inputs": {
            "queries": [{"sequence_id": "seq"}],
            "media_root": "historical-root",
            "event_observations": {
                "token": {
                    "sequence_id": "seq",
                    "current_tensor_sha256": tensor_sha,
                    "current_tensor_path": "historical-path",
                }
            },
            "rgb_sources": {relative: frozen},
        },
    }
    original = copy.deepcopy(validated)
    relocated, admission = staged.admit_sequence(validated, "seq", events, rgb)
    assert validated == original
    assert relocated["event_observations"]["token"]["current_tensor_path"] == str(tensor)
    assert admission["status"] == "PASS"
    staged.assert_rgb_unchanged(admission)
    shard.write_bytes(b"changed!!!")
    with pytest.raises(ValueError):
        staged.admit_sequence(validated, "seq", events, rgb)
    with pytest.raises(ValueError):
        staged.assert_rgb_unchanged(admission)


def test_diagnostic_hooks_preserve_exact_outputs():
    from types import SimpleNamespace

    import torch

    class Head(torch.nn.Module):
        def forward(self, value):
            return {"point_phase": value * 2, "q10": value - 1}

    names = ("E_CTX_MATCHED", "R_CTX", "PHASE_ONLY_EVENT_S7_TRAIN40")
    predictor = SimpleNamespace(models={name: Head() for name in names})
    value = torch.tensor([0.5])
    expected = predictor.models[names[0]](value)
    captured = {}
    handles = staged.attach_diagnostics(predictor, captured)
    try:
        output = predictor.models[names[0]](value)
    finally:
        for handle in handles:
            handle.remove()
    assert all(torch.equal(expected[key], output[key]) for key in expected)
    assert torch.equal(captured["E_CTX_MATCHED.input_0"], value)
    assert torch.equal(captured["E_CTX_MATCHED.output_q10"], expected["q10"])


def test_durable_diagnostic_rejects_changed_bytes_or_prediction(tmp_path):
    npz, receipt = staged.diagnostic_paths(tmp_path, "token")
    staged.save_npz(npz, {"phase": np.array([0.1], dtype=np.float32)})
    staged.atomic(
        receipt,
        {
            "schema": "test12_staged_query_diagnostic_v1",
            "sample_token": "token",
            "run_binding_sha256": "binding",
            "prediction": 1.5,
            "npz_sha256": staged.digest(npz),
        },
    )
    assert staged.verify_diagnostic(tmp_path, "token", "binding", 1.5) == [npz, receipt]
    with pytest.raises(ValueError):
        staged.verify_diagnostic(tmp_path, "token", "binding", 1.6)
    staged.save_npz(npz, {"phase": np.array([0.2], dtype=np.float32)})
    with pytest.raises(ValueError):
        staged.verify_diagnostic(tmp_path, "token", "binding", 1.5)


def test_native_known_boundary_preserves_scalar_mask_and_exact_pair_input():
    import torch

    from e_jepa_ttc.rgb_port.features import ProducerObservation, pair133
    from operational.rgb_port_streaming.contracts import NativeObservation

    support = torch.tensor([[0.1, 0.5, 0.8], [0.9, 0.2, 0.3]])
    known = torch.tensor([True, False])
    token = torch.arange(256, dtype=torch.float32).reshape(2, 128)
    output = {
        "token128": token,
        "prediction_ttc": torch.tensor([1.0, 2.0]),
        "point_phase": torch.tensor([0.1, 0.2]),
        "flow": torch.zeros(2),
        "margin": torch.ones(2),
        "log_variance": torch.zeros(2),
        "support": support,
        "known": known,
    }
    with pytest.raises(ValueError, match="known mask"):
        ProducerObservation.from_output(output)
    observed = NativeObservation.from_output(output)
    assert observed.known is known
    assert observed.support is support
    dt = torch.tensor([0.05, 0.1])
    reference = pair133(
        token, dt, torch.stack((support[:, -1], torch.minimum(support[:, -2], support[:, -1])), -1)
    )
    assert torch.equal(observed.pair_input(dt), reference)
    with pytest.raises(ValueError, match="known mask"):
        NativeObservation.from_output({**output, "known": known.float()})
    with pytest.raises(ValueError, match="support"):
        NativeObservation.from_output({**output, "support": support + 2})


def test_complete_without_consumed_reconciles_without_raw_or_inference(tmp_path):
    from types import SimpleNamespace

    directory, rgb_root = tmp_path / "outputs", tmp_path / "evicted_rgb"
    admission = directory / "seq.admission.json"
    staged.atomic(
        admission,
        {
            "schema": "test12_staged_sequence_admission_v1",
            "status": "PASS",
            "sequence_id": "seq",
            "run_binding_sha256": "binding",
            "private_targets_read": False,
            "external_submission": False,
            "rgb_sources": {
                "data/test/seq/a.tar": {"receipt_path": "archived_source_receipt.json"}
            },
        },
    )
    features = directory / "seq.features.npz"
    staged.save_npz(features, {"x": np.ones(1)})
    fragment = directory / "fragment.json"
    staged.atomic(fragment, {"prediction": 1.5})

    class Fragments:
        def read(self, token):
            assert token == "token"
            return staged.read(fragment)["prediction"]

        def _path(self, token):
            assert token == "token"
            return fragment

    diagnostic, metadata = staged.diagnostic_paths(directory, "token")
    staged.save_npz(diagnostic, {"phase": np.ones(1)})
    staged.atomic(
        metadata,
        {
            "schema": "test12_staged_query_diagnostic_v1",
            "sample_token": "token",
            "run_binding_sha256": "binding",
            "prediction": 1.5,
            "npz_sha256": staged.digest(diagnostic),
        },
    )
    staged.atomic(
        directory / "seq.complete.json",
        {
            "schema": "test12_staged_sequence_complete_v1",
            "status": "COMPLETE",
            "sequence_id": "seq",
            "run_binding_sha256": "binding",
            "private_targets_read": False,
            "external_submission": False,
            "admission_sha256": staged.digest(admission),
            "canonical_tokens": ["token"],
            "predictions_sha256": staged.canonical({"token": 1.5}),
            "feature_cache_sha256": staged.digest(features),
        },
    )
    final = SimpleNamespace(canonical_sha256=staged.canonical)
    staged.ensure_consumed(final, directory, "seq", ["token"], Fragments(), "binding", rgb_root)
    consumed = rgb_root / "consumed/seq.json"
    first = consumed.read_bytes()
    assert len(staged.read(consumed)["outputs"]) == 6
    # Repeat after simulated crash: marker regeneration is byte-identical.
    consumed.unlink()
    staged.ensure_consumed(final, directory, "seq", ["token"], Fragments(), "binding", rgb_root)
    assert consumed.read_bytes() == first
    fragment.write_text('{"prediction":2.5}', encoding="utf-8")
    with pytest.raises(ValueError):
        staged.ensure_consumed(final, directory, "seq", ["token"], Fragments(), "binding", rgb_root)
