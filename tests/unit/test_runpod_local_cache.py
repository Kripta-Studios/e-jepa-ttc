"""Local immutable-cache staging and selective routing contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from operational.runpod_migration.local_cache import (
    GIB,
    prove_rgb_route,
    readiness_barrier,
    route_rgb_source,
    stage_bundle,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(root: Path) -> tuple[Path, Path]:
    rgb = root / "cache/rgb/shard.npz"
    teacher = root / "cache/teacher/shard.npz"
    rgb.parent.mkdir(parents=True)
    teacher.parent.mkdir(parents=True)
    rgb.write_bytes(b"immutable-rgb")
    teacher.write_bytes(b"immutable-teacher")
    files = [
        {
            "relative": str(path.relative_to(root)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
        for path in (rgb, teacher)
    ]
    (root / "BUNDLE_MANIFEST.json").write_text(
        json.dumps({"profile": "rgb_pilot", "files": files}), encoding="utf-8"
    )
    return rgb, teacher


def test_stage_readiness_and_selective_routing_preserve_identity(tmp_path: Path) -> None:
    logical, stage = tmp_path / "logical", tmp_path / "local-stage"
    logical.mkdir()
    rgb, teacher = _bundle(logical)
    receipt = stage_bundle(logical, stage, reserve_bytes=4 * GIB)
    ready = Path(receipt["readiness_path"])
    assert readiness_barrier(logical, stage, ready)["status"] == "READY"

    rgb_shard = {"path": str(rgb), "sha256": _sha(rgb), "tokens": ["x"]}
    prepared = SimpleNamespace(_lookup={0: (rgb_shard, 0)}, _loaded={}, _loaded_bytes=0)
    dataset = SimpleNamespace(prepared_cache=prepared)
    teacher_cache = SimpleNamespace(
        shards=({"path": str(teacher), "sha256": _sha(teacher)},),
        _loaded={},
        _sizes={},
        retained_bytes=0,
    )
    identity = {"schema": "final_refit_fold_source_v1", "identity_sha256": "a" * 64}
    source = SimpleNamespace(
        identity=identity,
        base=SimpleNamespace(datasets={"P": dataset}, teacher=teacher_cache),
    )
    routing = route_rgb_source(source, logical_root=logical, stage_root=stage, readiness_path=ready)
    assert routing["status"] == "READY_FOR_NUMERIC_PROOF"
    assert source.identity == identity
    assert Path(rgb_shard["path"]).is_relative_to(stage.resolve())
    assert Path(teacher_cache.shards[0]["path"]).is_relative_to(stage.resolve())


def test_numeric_proof_reads_both_routes_without_rng_or_optimizer(tmp_path: Path) -> None:
    logical, stage = tmp_path / "logical", tmp_path / "local-stage"
    logical.mkdir()
    rgb, teacher = _bundle(logical)
    receipt = stage_bundle(logical, stage, reserve_bytes=4 * GIB)
    rgb_shard = {"path": str(rgb), "sha256": _sha(rgb)}
    source = SimpleNamespace(
        identity={"identity_sha256": "c" * 64},
        base=SimpleNamespace(
            datasets={
                "P": SimpleNamespace(
                    prepared_cache=SimpleNamespace(
                        _lookup={0: (rgb_shard, 0)}, _loaded={}, _loaded_bytes=0
                    )
                )
            },
            teacher=SimpleNamespace(
                shards=({"path": str(teacher), "sha256": _sha(teacher)},),
                _loaded={},
                _sizes={},
                retained_bytes=0,
            ),
        ),
    )

    def physical_identity(value):
        path = Path(value.base.datasets["P"].prepared_cache._lookup[0][0]["path"])
        return {"payload_sha256": _sha(path)}

    proof = prove_rgb_route(
        source,
        logical_root=logical,
        stage_root=stage,
        readiness_path=Path(receipt["readiness_path"]),
        identity_function=physical_identity,
    )
    assert proof["status"] == "PASS"
    assert proof["optimizer_updates"] == 0


def test_readiness_fails_closed_after_staged_corruption(tmp_path: Path) -> None:
    logical, stage = tmp_path / "logical", tmp_path / "local-stage"
    logical.mkdir()
    _bundle(logical)
    receipt = stage_bundle(logical, stage, reserve_bytes=4 * GIB)
    damaged = stage / "cache/rgb/shard.npz"
    damaged.chmod(0o644)
    damaged.write_bytes(b"changed")
    with pytest.raises(ValueError):
        readiness_barrier(logical, stage, Path(receipt["readiness_path"]))


def test_router_rejects_shards_outside_logical_bundle(tmp_path: Path) -> None:
    logical, stage = tmp_path / "logical", tmp_path / "local-stage"
    logical.mkdir()
    _, teacher = _bundle(logical)
    receipt = stage_bundle(logical, stage, reserve_bytes=4 * GIB)
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"outside")
    shard = {"path": str(outside), "sha256": _sha(outside)}
    source = SimpleNamespace(
        identity={"identity_sha256": "b" * 64},
        base=SimpleNamespace(
            datasets={
                "P": SimpleNamespace(prepared_cache=SimpleNamespace(_lookup={0: (shard, 0)}))
            },
            teacher=SimpleNamespace(shards=({"path": str(teacher), "sha256": _sha(teacher)},)),
        ),
    )
    with pytest.raises(ValueError, match="outside"):
        route_rgb_source(
            source,
            logical_root=logical,
            stage_root=stage,
            readiness_path=Path(receipt["readiness_path"]),
        )
