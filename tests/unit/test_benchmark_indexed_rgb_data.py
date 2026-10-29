"""Evidence encoding used by indexed-reader benchmark receipts."""

import sys
from pathlib import Path

import pytest

from scripts.benchmark_indexed_rgb_data import structural_sha256
from scripts.benchmark_indexed_rgb_data_v2 import main as benchmark_v2_main


def test_structural_hash_is_domain_and_boundary_separated() -> None:
    assert structural_sha256([1, 23]) != structural_sha256([12, 3])
    assert structural_sha256(["ab", "c"]) != structural_sha256(["a", "bc"])
    assert structural_sha256([1, 2]) != structural_sha256((1, 2))
    assert structural_sha256({"a": [1, 2]}) != structural_sha256({"a": [12]})
    assert structural_sha256({"a": 1, "b": 2}) == structural_sha256({"b": 2, "a": 1})


def test_v2_refuses_existing_output_before_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "receipt.json"
    output.write_text("immutable", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output", str(output)])
    with pytest.raises(FileExistsError, match="already exists"):
        benchmark_v2_main()
    assert output.read_text(encoding="utf-8") == "immutable"
