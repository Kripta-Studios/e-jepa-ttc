from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from operational.runpod_migration.paths import mapped_path_class
from scripts.runpod_event_directory_routing_v1 import compositional_path_factory


class Mapping:
    def __init__(self, primary: Path, roots: tuple[Path, ...]) -> None:
        self.primary = SimpleNamespace(destination=primary)
        self.roots = roots

    def translate(self, value: Any) -> str:
        return str(value)


class QueueMap:
    def __init__(self, mapping: Mapping) -> None:
        self.mapping = mapping

    def translate(self, value: Any) -> str:
        return self.mapping.translate(value)


def test_split_directory_routes_each_child_and_preserves_path_type(tmp_path: Path) -> None:
    primary, authority, supplement = (
        tmp_path / "primary",
        tmp_path / "authority",
        tmp_path / "supplement",
    )
    relative = Path("workspace/v12/artifacts/train40")
    for root in (primary, authority, supplement):
        (root / relative).mkdir(parents=True)
    (authority / relative / "H8_FEATURE_MANIFEST.json").write_text("{}")
    (supplement / relative / "H8_FEATURES.npz").write_bytes(b"npz")
    (primary / relative / "TRAIN40_INDEX.npz").write_bytes(b"index")
    path_type = compositional_path_factory(
        mapped_path_class, QueueMap(Mapping(primary, (authority, supplement)))
    )
    logical = path_type(primary / relative)
    assert isinstance(logical, Path)
    assert logical / "H8_FEATURE_MANIFEST.json" == authority / relative / "H8_FEATURE_MANIFEST.json"
    assert logical.joinpath("H8_FEATURES.npz") == supplement / relative / "H8_FEATURES.npz"
    assert logical / "TRAIN40_INDEX.npz" == primary / relative / "TRAIN40_INDEX.npz"


@pytest.mark.parametrize(
    "name",
    [
        "H8_FEATURES.npz",
        "H8_FEATURE_MANIFEST.json",
        "TRAIN40_INDEX.npz",
        "DATA_AUDIT.json",
        "RAW_PLAN.json",
        "PREPARE_FREEZE.json",
    ],
)
def test_required_history_child_closure(tmp_path: Path, name: str) -> None:
    primary, supplemental = tmp_path / "primary", tmp_path / "supplement"
    relative = Path("workspace/v12/artifacts/train40")
    (supplemental / relative).mkdir(parents=True)
    expected = supplemental / relative / name
    expected.write_bytes(b"x")
    path_type = compositional_path_factory(
        mapped_path_class, Mapping(primary, (supplemental,))
    )
    assert path_type(primary / relative) / name == expected


def test_ambiguous_file_is_rejected(tmp_path: Path) -> None:
    primary, one, two = tmp_path / "primary", tmp_path / "one", tmp_path / "two"
    relative = Path("workspace/v12/artifacts/train40/H8_FEATURES.npz")
    for root in (one, two):
        (root / relative).parent.mkdir(parents=True)
        (root / relative).write_bytes(b"same")
    path_type = compositional_path_factory(mapped_path_class, Mapping(primary, (one, two)))
    with pytest.raises(ValueError, match="Ambiguous"):
        path_type(primary / relative.parent) / relative.name


def test_unrelated_paths_are_unchanged(tmp_path: Path) -> None:
    primary, supplement = tmp_path / "primary", tmp_path / "supplement"
    path_type = compositional_path_factory(mapped_path_class, Mapping(primary, (supplement,)))
    unrelated = tmp_path / "unrelated" / "file.json"
    assert path_type(unrelated.parent) / unrelated.name == unrelated
