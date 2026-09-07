"""Frozen source-to-history binding; synthetic metadata only."""

from types import SimpleNamespace

import pytest

from e_jepa_ttc.simplex_t.frozen_history import frozen_history_pools


@pytest.mark.parametrize("fault", ["none", "missing", "duplicate", "changed", "fold"])
def test_frozen_history_pool_correspondence(tmp_path, fault):
    files, folders = [], {}
    for name in ("D0", "D1", "DENSE_OLD"):
        root = tmp_path / name
        root.mkdir()
        folders[name] = root
        for filename in ("INDEX_MANIFEST.json", "DEDUP_MANIFEST.json"):
            (root / filename).write_text("{}")
            files.append(dict(root="work", relative_path=f"{name}/{filename}", sha256="a" * 64))
        for fold in range(3):
            (root / f"outer{fold}.npz").write_bytes(b"fixture")

    def folds(name):
        return {
            fold: SimpleNamespace(
                index_manifest=folders[name] / "INDEX_MANIFEST.json",
                index_manifest_sha256="a" * 64,
                dedup=folders[name] / f"outer{fold}.npz",
            )
            for fold in range(3)
        }

    source = SimpleNamespace(
        index_root=folders["D0"],
        dedup_root=folders["D0"],
        expansion_folds=folds("D1"),
        dense_folds=folds("DENSE_OLD"),
    )
    if fault == "missing":
        files.pop()
    elif fault == "duplicate":
        files.append(files[0])
    elif fault == "changed":
        source.expansion_folds[0].index_manifest_sha256 = "b" * 64
    elif fault == "fold":
        source.dense_folds.pop(2)
    record = dict(files=files, source_contract=dict(availability=dict(d1=True, density=True)))
    if fault != "none":
        with pytest.raises(ValueError):
            frozen_history_pools(source, record, roots={"work": tmp_path})
        return
    result = frozen_history_pools(source, record, roots={"work": tmp_path})
    assert set(result) == {"D0", "D1", "DENSE_OLD"}
    assert all(pin.index_sha256 == "a" * 64 for pin in result.values())
