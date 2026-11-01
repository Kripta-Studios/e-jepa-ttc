"""Native augmentations join only by sealed identity and remain inside fit groups."""

import hashlib
import json

import numpy as np
import pytest

from scripts.train_phase_native_context_fusion import (
    augmentation_weights,
    fold_masks,
    join_augmentations,
    validate_expansion,
    validate_source_closure,
    validate_variant_identity,
    verify_sealed_artifacts,
)


def population():
    return {
        "token": np.array(["a", "b", "c", "d"]),
        "group": np.array(["g1", "g1", "g2", "g3"]),
        "family": np.array(["A", "A", "B", "C"]),
        "native": np.ones((4, 3), np.float32),
        "geometry": np.arange(1, 5, dtype=np.float32),
        "quality": np.ones((4, 20), np.float32),
        "target": np.arange(2, 6, dtype=np.float32),
    }


def augmentation():
    return {
        "token": np.array(["a", "a", "a", "c", "c", "c", "d", "d", "d"]),
        "group": np.array(["g1"] * 3 + ["g2"] * 3 + ["g3"] * 3),
        "keep": np.array([1, 2, 3] * 3),
        "native": np.full((9, 3), 2, np.float32),
        "status": np.repeat("fused", 9),
        "reason": np.repeat("available", 9),
    }


def test_join_preserves_geometry_quality_target_and_original_inputs():
    clean, aug = population(), augmentation()
    before = {k: v.copy() for k, v in clean.items()}
    joined, indices = join_augmentations(clean, aug)
    np.testing.assert_array_equal(indices, [0, 0, 0, 2, 2, 2, 3, 3, 3])
    for key in ("geometry", "quality", "target", "group", "family"):
        np.testing.assert_array_equal(joined[key], clean[key][indices])
    np.testing.assert_array_equal(joined["native"], aug["native"])
    for key in clean:
        np.testing.assert_array_equal(clean[key], before[key])


@pytest.mark.parametrize("fault", ["missing", "duplicate", "swapped_group", "population_duplicate"])
def test_identity_join_fails_closed(fault):
    clean, aug = population(), augmentation()
    if fault == "missing":
        aug["token"][0] = "z"
    elif fault == "duplicate":
        aug["keep"][1] = 1
    elif fault == "swapped_group":
        aug["group"][0] = "g2"
    else:
        clean["token"][1] = "a"
    with pytest.raises(ValueError):
        join_augmentations(clean, aug)


def test_missing_variant_and_wrong_keep_rejected():
    aug = augmentation()
    for candidate in (
        {k: v[:-1] for k, v in aug.items()},
        {**aug, "keep": np.array([1, 2, 4] * 3)},
    ):
        with pytest.raises(ValueError):
            validate_variant_identity(candidate)


def test_family_query_weights_do_not_overweight_more_admitted_variants():
    tokens = np.array(["a", "a", "a", "b", "c", "c"])
    families = np.array(["A", "A", "A", "A", "B", "B"])
    weights = augmentation_weights(tokens, families)
    assert weights.sum() == pytest.approx(1)
    assert weights[families == "A"].sum() == pytest.approx(0.5)
    assert weights[families == "B"].sum() == pytest.approx(0.5)
    assert weights[tokens == "a"].sum() == pytest.approx(0.25)
    assert weights[tokens == "b"].sum() == pytest.approx(0.25)
    np.testing.assert_allclose(weights[-2:], [0.25, 0.25])
    with pytest.raises(ValueError):
        augmentation_weights(np.array(["same", "same"]), np.array(["A", "B"]))


def test_fold_masks_exclude_all_variants_of_held_out_family():
    clean = population()
    joined, indices = join_augmentations(clean, augmentation())
    train, test, aug_train, aug_test = fold_masks(clean, joined, indices, "B")
    assert not np.any(aug_train & aug_test)
    assert set(joined["family"][aug_train]) == {"A", "C"}
    assert set(joined["family"][aug_test]) == {"B"}
    assert set(clean["group"][train]).isdisjoint(clean["group"][test])
    clean["group"][3] = "g2"
    with pytest.raises(ValueError):
        fold_masks(clean, joined, indices, "B")


def sealed_files(tmp_path):
    artifacts = {}
    for name in ("FREEZE.json", "CLEAN.npz", "AUGMENTATIONS.npz"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        artifacts[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema": "phase_native_context_augmentations_v1",
        "status": "COMPLETE",
        "queries": 149,
        "augmentations": 447,
        "clean_exact": True,
        "disk_roundtrip_exact": True,
        "targets_read": False,
        "optimizer_updates": 0,
        "native_order": ["F0", "RGB", "event"],
        "artifacts": artifacts,
        "freeze_sha256": artifacts["FREEZE.json"],
    }


@pytest.mark.parametrize(
    "fault", ["hash", "missing", "clean_parity", "disk_parity", "order", "escape"]
)
def test_seal_rejects_changed_missing_or_unverified_evidence(tmp_path, fault):
    receipt = sealed_files(tmp_path)
    verify_sealed_artifacts(tmp_path, receipt)
    if fault == "hash":
        (tmp_path / "AUGMENTATIONS.npz").write_bytes(b"changed")
    elif fault == "missing":
        receipt["artifacts"].pop("CLEAN.npz")
    elif fault == "clean_parity":
        receipt["clean_exact"] = False
    elif fault == "disk_parity":
        receipt["disk_roundtrip_exact"] = False
    elif fault == "order":
        receipt["native_order"] = ["F0", "event", "RGB"]
    else:
        receipt["artifacts"]["../outside.npz"] = "wrong"
    with pytest.raises(ValueError):
        verify_sealed_artifacts(tmp_path, receipt)


def test_no_population_payload_or_target_join_before_seal_validation(monkeypatch, tmp_path):
    from scripts import train_phase_native_context_fusion as runner

    def reject(*_args):
        raise ValueError("seal not valid")

    def forbidden(*_args):
        pytest.fail("Population/targets opened before seal admission")

    monkeypatch.setattr(runner, "validate_expansion", reject)
    monkeypatch.setattr(runner, "load_arrays", forbidden)
    with pytest.raises(ValueError, match="seal not valid"):
        runner.load_training_data(tmp_path / "population", tmp_path / "config", tmp_path)


def test_complete_synthetic_seal_admission_without_fitting(tmp_path):
    """Exercise the complete149-query receipt chain using unavailable toy observations."""
    from operational.phase_observation_cache.integrity import seal_observations

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf8")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def canonical(value):
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    groups, tokens, families, chosen = [], [], [], []
    for index in range(24):
        n = 116 if index == 0 else 105
        group = f"road{index:02d}"
        selected = {n * q // 4 for q in (1, 2, 3)}
        for i in range(n):
            groups.append(group)
            tokens.append(f"{group}:{i}")
            families.append(f"family{index % 7}")
            if i in selected:
                chosen.append(len(tokens) - 1)
    for i in range(77):
        chosen.append(len(tokens))
        groups.append("Slider-500")
        tokens.append(f"Slider-500:{i}")
        families.append("Slider")
    tokens, groups, families = map(np.array, (tokens, groups, families))
    population_path, config_path = tmp_path / "population.npz", tmp_path / "original.json"
    np.savez(population_path, token=tokens, group=groups, family=families)
    write(
        config_path,
        {
            "fit_groups": sorted(set(groups)),
            "excluded_road": ["excluded"],
            "excluded_slider": ["Slider-750", "Slider-1000"],
        },
    )
    cache = tmp_path / "cache"
    archive = cache / "executed_source/scripts/toy_capture.py"
    archive.parent.mkdir(parents=True)
    archive.write_text("# Toy evidence for schema validation only\n")
    sources = {"scripts/toy_capture.py": sha(archive)}
    freeze = {
        "inputs": {str(population_path): sha(population_path), str(config_path): sha(config_path)},
        "source_hashes": sources,
        "upstream": {},
        "selected_tokens": tokens[chosen].tolist(),
        "selected_groups": groups[chosen].tolist(),
    }
    write(cache / "FREEZE.json", freeze)
    for token, group in zip(tokens[chosen], groups[chosen], strict=True):
        dest = cache / "queries" / hashlib.sha256(str(token).encode()).hexdigest()[:24]
        dest.mkdir(parents=True)
        query = {"anchor_us": 1000000}
        binding = {
            "token": str(token),
            "group": str(group),
            "query_us": 1000000,
            "query_sha256": canonical(query),
            "calibration_sha256": "toy",
            "model_sha256": canonical({}),
            "source_sha256": canonical(sources),
        }
        write(dest / "QUERY.json", query)
        write(dest / "OBSERVATIONS.json", {"event": [], "rgb": []})
        np.savez(
            dest / "OBSERVATIONS.npz",
            **{
                f"{m}.{name}": np.empty((0, width), np.float32)
                for m in ("event", "rgb")
                for name, width in (("raw17", 17), ("expert3", 3))
            },
        )
        seal_observations(dest, binding)
        write(dest / "PARITY.json", {"clean_exact": True, "disk_roundtrip_exact": True})
        np.savez(dest / "NATIVE.npz", native=np.full((4, 3), np.nan, np.float32))
        write(
            dest / "COMPLETE.json",
            {
                "binding": binding,
                "clean_exact": True,
                "disk_roundtrip_exact": True,
                "status": "unavailable",
                "reason": "missing_event",
                "artifacts": {p.name: sha(p) for p in dest.iterdir()},
            },
        )
    clean = {
        "token": tokens[chosen],
        "group": groups[chosen],
        "native": np.full((149, 3), np.nan, np.float32),
        "status": np.repeat("unavailable", 149),
        "reason": np.repeat("missing_event", 149),
    }
    aug = {k: np.repeat(v, 3, axis=0) for k, v in clean.items()}
    aug["keep"] = np.tile([1, 2, 3], 149)
    np.savez(cache / "CLEAN.npz", **clean)
    np.savez(cache / "AUGMENTATIONS.npz", **aug)
    receipt = {
        "schema": "phase_native_context_augmentations_v1",
        "status": "COMPLETE",
        "queries": 149,
        "augmentations": 447,
        "clean_exact": True,
        "disk_roundtrip_exact": True,
        "targets_read": False,
        "optimizer_updates": 0,
        "native_order": ["F0", "RGB", "event"],
        "freeze_sha256": sha(cache / "FREEZE.json"),
        "inputs": freeze["inputs"],
        "artifacts": {str(p.relative_to(cache)): sha(p) for p in cache.rglob("*") if p.is_file()},
    }
    write(cache / "COMPLETE.json", receipt)
    actual, admission = validate_expansion(cache, population_path, config_path)
    assert admission["variants"] == 447 and admission["queries"] == 149
    assert np.isnan(actual["native"]).all()
    # Even if an aggregate file and its outer digest are resealed, row-level sources disagree.
    changed = {k: v.copy() for k, v in aug.items()}
    changed["group"][0] = "road01"
    np.savez(cache / "AUGMENTATIONS.npz", **changed)
    receipt["artifacts"]["AUGMENTATIONS.npz"] = sha(cache / "AUGMENTATIONS.npz")
    write(cache / "COMPLETE.json", receipt)
    with pytest.raises(ValueError):
        validate_expansion(cache, population_path, config_path)


def test_source_closure_blocks_missing_receipt(tmp_path):
    cache, closure_dir = tmp_path / "cache", tmp_path / "closure"
    cache.mkdir()
    closure_dir.mkdir()
    for name in ("COMPLETE.json", "FREEZE.json"):
        (cache / name).write_text("{}")
    closure = closure_dir / "SOURCE_CLOSURE.json"
    with pytest.raises(ValueError):
        validate_source_closure(closure, cache)


def test_valid_source_closure_and_later_dependency_change(tmp_path, monkeypatch):
    from scripts import train_phase_native_context_fusion as runner

    cache, closure_dir = tmp_path / "cache", tmp_path / "closure"
    cache.mkdir()
    closure_dir.mkdir()

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    (cache / "COMPLETE.json").write_text(json.dumps({"artifacts": {"file": "digest"}}))
    (cache / "FREEZE.json").write_text("{}")
    dependency = tmp_path / "dependency.py"
    dependency.write_text("# exact source\n")
    name = str(dependency)
    dependencies = {name: sha(dependency)}
    snapshot = {"dependency_sha256": dependencies, "archived_dependency_sha256": dependencies}
    snapshot_path = closure_dir / "SNAPSHOT.json"
    snapshot_path.write_text(json.dumps(snapshot))
    payload = {
        "schema": "phase_native_source_closure_v1",
        "status": "PASS",
        "snapshot_to_completion_unchanged": True,
        "frozen_numerical_sources_match": True,
        "labels_opened": False,
        "producer_calls": 0,
        "extraction_root": str(cache.resolve()),
        "expanded_complete_sha256": sha(cache / "COMPLETE.json"),
        "expanded_freeze_sha256": sha(cache / "FREEZE.json"),
        "snapshot_sha256": sha(snapshot_path),
        "dependency_sha256": dependencies,
        "original_artifacts_verified": 1,
        "dependency_equivalence": {
            name: {
                "current_sha256": dependencies[name],
                "frozen_sha256": dependencies[name],
                "equivalence": "byte_identical",
            }
        },
        "audit_source_sha256": sha(runner.Path("scripts/audit_phase_native_source_closure.py")),
    }
    path = closure_dir / "SOURCE_CLOSURE.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(runner, "DEPENDENCIES", (name,))

    def verify(_root, expected):
        if sha(dependency) != expected[name]:
            raise ValueError("Dependency changed")

    monkeypatch.setattr(runner, "verify_snapshot", verify)
    validate_source_closure(path, cache)
    for key, wrong in (
        ("status", "FAIL"),
        ("snapshot_sha256", "wrong"),
        ("original_artifacts_verified", 2),
    ):
        changed = {**payload, key: wrong}
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            validate_source_closure(path, cache)
    path.write_text(json.dumps(payload))
    dependency.write_text("# changed\n")
    with pytest.raises(ValueError, match="Dependency changed"):
        validate_source_closure(path, cache)


def test_unavailable_native_rows_keep_geometry_and_original_admission():
    from scripts.train_phase_native_context_fusion import admitted

    clean, aug = population(), augmentation()
    aug["native"][:] = np.nan
    aug["status"] = np.repeat("unavailable", len(aug["token"]))
    joined, indices = join_augmentations(clean, aug)
    assert admitted(joined).all()
    assert np.isnan(joined["native"]).all()
    np.testing.assert_array_equal(joined["geometry"], clean["geometry"][indices])
