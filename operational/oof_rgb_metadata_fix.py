"""Narrow, hash-bound repair of RGB observation metadata equality.

Keep the frozen scientific files intact. Replace exactly one comparison in the
materializer's AST; all model, data, cache and ancestry operations stay original.
"""

from __future__ import annotations

# ruff: noqa: ANN401 -- preserve the frozen materializer's heterogeneous metadata API.
import ast
import inspect
import types
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from operational.final_refit.recipe import sha256

ORIGINAL_SHA256 = "aaf47f9f3c211b5e46ebdb67a4a980347c6e59524ff2b84ee4a85a0d1ce268c5"


def metadata_equal(left: Any, right: Any) -> bool:
    """Compare nested sensor metadata exactly, without broadcasting or tolerance."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and (
                all(metadata_equal(a, b) for a, b in zip(left.flat, right.flat, strict=True))
                if left.dtype.hasobject
                else bool(np.array_equal(left, right))
            )
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and left.keys() == right.keys()
            and all(metadata_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(metadata_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    result = left == right
    if not isinstance(result, (bool, np.bool_)):
        raise TypeError("Unsupported observation metadata equality result")
    return bool(result)


def observation_metadata_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Compare sensor identity; clocks and tokens belong to each requesting query.

    The unmodified materializer retains query_rows and derives each history's
    clocks from that query's time and the observation's anchor/availability.
    decode_query does not consume these three query-reference fields.
    """
    query_fields = {"sample_token", "query_time_us", "query_age_us"}
    return left.keys() == right.keys() and metadata_equal(
        {k: v for k, v in left.items() if k not in query_fields},
        {k: v for k, v in right.items() if k not in query_fields},
    )


def patched_materializer() -> Any:
    """Compile the frozen function with only its broken dictionary comparison repaired."""
    from operational.final_refit import materialize

    path = Path(materialize.__file__)
    if sha256(path) != ORIGINAL_SHA256:
        raise ValueError("RGB metadata repair requires the audited original materializer")
    tree = ast.parse(inspect.getsource(materialize.materialize_rgb_observations))
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.dump(node.test) == ast.dump(ast.parse("old != enriched", mode="eval").body)
    ]
    if len(matches) != 1:
        raise ValueError("Expected exactly one frozen observation comparison")
    matches[0].test = ast.copy_location(
        ast.parse("not _metadata_equal(old, enriched)", mode="eval").body,
        matches[0].test,
    )
    ast.fix_missing_locations(tree)
    namespace = dict(vars(materialize), _metadata_equal=observation_metadata_equal)
    exec(compile(tree, str(path) + "[metadata-equality-repair]", "exec"), namespace)
    return namespace["materialize_rgb_observations"]


@contextmanager
def repaired_rgb_metadata() -> Iterator[None]:
    """Patch both consumers for this process, then restore them even on failure."""
    from operational.final_refit import features, materialize, scientific_oof_features

    original = materialize.materialize_rgb_observations
    alias = scientific_oof_features.materialize_rgb_observations
    expected = scientific_oof_features._expected_role_receipt
    feature_path = Path(scientific_oof_features.__file__)
    if sha256(feature_path) != "088d17c4029622f76001a379bf4f39035cec4c03f2dba3b048ea2723d0882f81":
        raise ValueError("Cache receipt repair requires the audited original features module")
    # Match the writer and validate_cache_receipt's original JSON encoding.
    # Scope this to the expected-receipt function; other scientific hashes stay unchanged.
    repaired_expected = types.FunctionType(
        expected.__code__,
        dict(expected.__globals__, canonical_sha256=features._canonical_sha256),
        expected.__name__,
        expected.__defaults__,
        expected.__closure__,
    )
    repaired_expected.__kwdefaults__ = expected.__kwdefaults__
    replacement = patched_materializer()
    materialize.materialize_rgb_observations = replacement
    scientific_oof_features.materialize_rgb_observations = replacement
    scientific_oof_features._expected_role_receipt = repaired_expected
    try:
        yield
    finally:
        materialize.materialize_rgb_observations = original
        scientific_oof_features.materialize_rgb_observations = alias
        scientific_oof_features._expected_role_receipt = expected
