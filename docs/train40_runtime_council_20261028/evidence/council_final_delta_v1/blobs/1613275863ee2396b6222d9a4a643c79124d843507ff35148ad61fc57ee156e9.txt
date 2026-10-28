"""CPU checks of native reduction attribution and immutable tensor provenance."""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

from e_jepa_ttc.models import local_transport
from operational.train40_runtime.compiled_transport import TransportPairTensorCore
from scripts.diagnose_train40_transport_summaries import _save_tensors, clone_preserving_strides
from scripts.diagnose_train40_weighted_mean import (
    load_bound_tensor_archive,
    make_native_mean_physical,
    make_weighted_mean_probe,
    native_weighted_mean_operator,
    verify_scientific_sources,
)

ProbeInputs = tuple[Tensor, Tensor, Tensor, Tensor]


def _inputs(seed: int) -> ProbeInputs:
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(2, 4, 8, 8, generator=generator),
        torch.randn(2, 4, 8, 8, generator=generator),
        torch.rand(2, 1, 8, 8, generator=generator),
        torch.rand(2, 1, 8, 8, generator=generator),
    )


@pytest.mark.parametrize("radius", [1, 2])
@pytest.mark.parametrize("deterministic_sampling", [False, True])
def test_native_mean_partition_preserves_original_cpu_for_changed_inputs(
    radius: int, deterministic_sampling: bool
) -> None:
    probes = make_weighted_mean_probe(
        radius, 0.02, deterministic_sampling=deterministic_sampling, backend="eager"
    )
    assert set(probes) == {
        "original",
        "compiled_summary_native_sampler",
        "native_weighted_mean",
    }
    original = TransportPairTensorCore(
        radius,
        0.02,
        deterministic_sampling=deterministic_sampling,
        reuse_reverse=False,
        reuse_samples=False,
    ).eval()
    inputs = _inputs(201)
    changed: ProbeInputs = (
        inputs[0].flip(-1),
        inputs[1] * 0.7,
        inputs[2] * 0.3,
        inputs[3].flip(-2),
    )
    zero: ProbeInputs = (
        torch.zeros_like(inputs[0]),
        torch.zeros_like(inputs[1]),
        torch.zeros_like(inputs[2]),
        torch.zeros_like(inputs[3]),
    )
    with torch.inference_mode():
        for values in (inputs, changed, zero):
            expected = original(*values)
            for name, probe in probes.items():
                actual = probe(*values)
                assert len(actual) == 2
                assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True)), name


@pytest.mark.parametrize("zero_weights", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_native_weighted_mean_matches_complete_original_reduction_and_schema(
    zero_weights: bool, strided: bool
) -> None:
    generator = torch.Generator().manual_seed(202)
    values = torch.randn(2, 8, 8, generator=generator)
    weights = torch.rand(2, 8, 8, generator=generator)
    if strided:
        values = values[:, ::2, 1::2].transpose(-1, -2)
        weights = weights[:, ::2, 1::2].transpose(-1, -2)
    if zero_weights:
        weights.zero_()
    operation = native_weighted_mean_operator()
    with torch.inference_mode():
        expected = local_transport._weighted_mean(values, weights)
        actual = operation(values, weights)
        assert actual.shape == (2,)
        assert actual.dtype == torch.float32
        assert torch.equal(actual, expected)
        assert actual.untyped_storage().data_ptr() not in {
            values.untyped_storage().data_ptr(),
            weights.untyped_storage().data_ptr(),
        }
        checked = torch.library.opcheck(
            operation, (values, weights), test_utils=("test_schema", "test_faketensor")
        )
        assert checked == {"test_schema": "SUCCESS", "test_faketensor": "SUCCESS"}


def test_private_native_mean_functions_leave_original_globals_and_identities_intact() -> None:
    names = (
        "_weighted_mean",
        "_weighted_slope",
        "_summarise_one",
        "_sample_reverse_flow",
        "transport_physical_features",
    )
    originals = {name: getattr(local_transport, name) for name in names}
    original_globals = dict(local_transport.transport_physical_features.__globals__)
    original_source = inspect.getsource(local_transport.transport_physical_features)
    private, binding = make_native_mean_physical()
    assert private is not originals["transport_physical_features"]
    assert private.__code__ is originals["transport_physical_features"].__code__
    assert private.__module__ in sys.modules
    assert private.__globals__ is vars(sys.modules[private.__module__])
    assert private.__globals__["_weighted_mean"] is not originals["_weighted_mean"]
    assert private.__globals__["_sample_reverse_flow"] is not originals["_sample_reverse_flow"]
    assert binding["private_module"] == private.__module__
    assert binding["function_source_sha256"]
    assert all(len(digest) == 64 for digest in binding["function_source_sha256"].values())
    assert binding["substitutions"] == {
        "mean": "ORIGINAL_ATEN_WEIGHTED_MEAN",
        "sampler": "ORIGINAL_ATEN_COMPLETE_REVERSE_SAMPLE",
    }
    assert all(getattr(local_transport, name) is value for name, value in originals.items())
    assert local_transport.transport_physical_features.__globals__ == original_globals
    assert inspect.getsource(local_transport.transport_physical_features) == original_source


def _archive(directory: Path) -> tuple[dict[str, Any], dict[str, Tensor]]:
    directory.mkdir(parents=True, exist_ok=True)
    storage = torch.arange(2 * 6 * 8, dtype=torch.float32).reshape(2, 6, 8)
    values = {
        "previous": storage[:, 1::2, ::2].transpose(-1, -2),
        "valid": (storage[:, 1::2, ::2] % 2 == 0).transpose(-1, -2),
    }
    return _save_tensors(directory / "inputs.pt", values), values


def test_bound_archive_recovers_exact_values_dtype_and_strides(tmp_path: Path) -> None:
    binding, expected = _archive(tmp_path)
    actual = load_bound_tensor_archive(tmp_path, binding)
    assert set(actual) == set(expected)
    for name, value in actual.items():
        assert torch.equal(value, expected[name])
        assert value.dtype == expected[name].dtype
        assert value.shape == expected[name].shape
        assert value.stride() == expected[name].stride()


def test_bound_archive_rejects_corrupt_file_bytes_before_loading(tmp_path: Path) -> None:
    binding, _ = _archive(tmp_path)
    path = tmp_path / binding["path"]
    payload = bytearray(path.read_bytes())
    payload[len(payload) // 2] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        load_bound_tensor_archive(tmp_path, binding)


@pytest.mark.parametrize("field", ["logical_bytes_sha256", "shape", "stride", "dtype"])
def test_bound_archive_rejects_tensor_metadata_mismatch(tmp_path: Path, field: str) -> None:
    binding, _ = _archive(tmp_path)
    changed = deepcopy(binding)
    metadata = changed["tensors"]["previous"]
    if field == "logical_bytes_sha256":
        metadata[field] = "0" * 64
    elif field == "shape":
        metadata[field][0] += 1
    elif field == "stride":
        metadata[field][0] += 1
    else:
        metadata[field] = "torch.float64"
    with pytest.raises(ValueError):
        load_bound_tensor_archive(tmp_path, changed)


def test_bound_archive_rejects_changed_tensor_even_with_new_container_hash(tmp_path: Path) -> None:
    binding, values = _archive(tmp_path)
    path = tmp_path / binding["path"]
    altered = {name: clone_preserving_strides(value) for name, value in values.items()}
    altered["previous"][0, 0, 0] += 1.0
    torch.save(altered, path)
    binding["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    binding["bytes"] = path.stat().st_size
    with pytest.raises(ValueError):
        load_bound_tensor_archive(tmp_path, binding)


@pytest.mark.parametrize("escape", ["relative", "absolute"])
def test_bound_archive_rejects_paths_outside_declared_directory(
    tmp_path: Path, escape: str
) -> None:
    outside_binding, _ = _archive(tmp_path)
    directory = tmp_path / "evidence"
    directory.mkdir()
    outside_binding["path"] = (
        "../inputs.pt" if escape == "relative" else str((tmp_path / "inputs.pt").resolve())
    )
    with pytest.raises(ValueError):
        load_bound_tensor_archive(directory, outside_binding)


def _scientific_sources(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Path], str]:
    relative = "src/e_jepa_ttc/models/local_transport.py"
    current = tmp_path / "current" / relative
    current.parent.mkdir(parents=True)
    current.write_bytes(b"# immutable scientific fixture\n")
    digest = hashlib.sha256(current.read_bytes()).hexdigest()
    archived = str((tmp_path / "old_checkout" / relative).resolve())
    return {archived: {"sha256": digest}}, {relative: current}, digest


def test_scientific_source_hashes_match_across_checkout_locations(tmp_path: Path) -> None:
    snapshot, current, digest = _scientific_sources(tmp_path)
    validated = verify_scientific_sources(snapshot, current)
    assert isinstance(validated, dict) and validated
    assert digest in json.dumps(validated, sort_keys=True)


@pytest.mark.parametrize("damage", ["changed_source", "changed_binding", "missing_binding"])
def test_scientific_source_hashes_reject_drift_and_missing_evidence(
    tmp_path: Path, damage: str
) -> None:
    snapshot, current, _ = _scientific_sources(tmp_path)
    if damage == "changed_source":
        next(iter(current.values())).write_bytes(b"# changed scientific fixture\n")
    elif damage == "changed_binding":
        next(iter(snapshot.values()))["sha256"] = "0" * 64
    else:
        snapshot.clear()
    with pytest.raises(ValueError):
        verify_scientific_sources(snapshot, current)
