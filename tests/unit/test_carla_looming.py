from __future__ import annotations

from pathlib import Path

import numpy as np

from e_jepa_ttc.data.carla_looming import (
    CARLA_LOOMING_EVENT_DTYPE,
    CarlaLoomingMetadata,
    CarlaLoomingSequence,
    build_carla_window_sample,
    carla_window_references_ms,
    create_carla_looming_splits,
    load_carla_looming_metadata,
    read_carla_event_window,
    read_carla_looming_manifest,
    scan_carla_looming_root,
    write_carla_looming_manifest,
    write_carla_looming_splits,
)
from e_jepa_ttc.training.carla_jepa import (
    CarlaJEPATrainerConfig,
    inspect_carla_jepa_pairs,
)


def _write_sequence(
    root: Path,
    index: int,
    *,
    collision_type: str,
    timestamps_ms: list[int],
    t_end_ms: int = 120,
) -> None:
    directory = root / f"example_{index}"
    directory.mkdir(parents=True)
    events = np.zeros(len(timestamps_ms), dtype=CARLA_LOOMING_EVENT_DTYPE)
    events["t"] = timestamps_ms
    events["x"] = np.arange(len(events), dtype=np.uint16) + 1
    events["y"] = np.arange(len(events), dtype=np.uint16) + 2
    events["p"] = np.arange(len(events), dtype=np.uint16) % 2
    np.save(directory / "events.npy", events, allow_pickle=False)
    collision = collision_type in {"cars", "pedestrian", "pedestrians"}
    diameter = np.asarray(2.5) if collision else np.asarray(None, dtype=object)
    np.savez(
        directory / "sim_data.npz",
        coll_type=np.asarray(collision_type),
        t_end=np.asarray(t_end_ms),
        dt=np.asarray(10.0),
        vel=np.asarray(5.0),
        diameter_object=diameter,
    )


def test_metadata_never_requires_pickle_for_negative_sequences(tmp_path: Path) -> None:
    _write_sequence(
        tmp_path,
        0,
        collision_type="none",
        timestamps_ms=[0, 10, 20, 30],
    )

    metadata = load_carla_looming_metadata(tmp_path / "example_0" / "sim_data.npz")

    assert metadata.collision is False
    assert metadata.object_diameter is None
    assert metadata.relative_velocity_mps == 5.0


def test_scan_window_and_censored_target_contract(tmp_path: Path) -> None:
    timestamps = list(range(0, 120, 10))
    _write_sequence(tmp_path, 0, collision_type="cars", timestamps_ms=timestamps)
    _write_sequence(tmp_path, 1, collision_type="none_with_crossing", timestamps_ms=timestamps)
    _write_sequence(
        tmp_path,
        2,
        collision_type="cars",
        timestamps_ms=[],
        t_end_ms=-30,
    )

    sequences = scan_carla_looming_root(
        tmp_path,
        context_ms=20,
        group_size=2,
        full_event_validation=True,
    )

    assert len(sequences) == 3
    assert [sequence.valid for sequence in sequences] == [True, True, False]
    assert "empty_events" in sequences[2].issues
    assert "nonpositive_t_end" in sequences[2].issues
    window = read_carla_event_window(
        tmp_path,
        sequences[0],
        start_us=10_000,
        end_us=30_000,
    )
    assert window.t_us.tolist() == [10_000, 20_000]
    assert window.polarity.tolist() == [1, -1]

    positive = build_carla_window_sample(
        tmp_path,
        sequences[0],
        reference_ms=50,
        context_ms=20,
        horizons_ms=(10,),
        future_window_ms=20,
    )
    negative = build_carla_window_sample(
        tmp_path,
        sequences[1],
        reference_ms=50,
        context_ms=20,
    )
    assert positive.ttc_seconds == 0.07
    assert positive.collision_within is not None
    assert positive.collision_within[0.5] is True
    assert positive.future_events[10].t_us.tolist() == [60_000, 70_000]
    assert negative.ttc_seconds is None
    assert negative.metadata["ttc_censored"] is True
    assert not any(negative.collision_within.values())


def test_window_references_are_bounded_and_deterministic(tmp_path: Path) -> None:
    _write_sequence(
        tmp_path,
        4,
        collision_type="pedestrians",
        timestamps_ms=list(range(0, 1000, 10)),
        t_end_ms=1000,
    )
    sequence = scan_carla_looming_root(tmp_path, context_ms=100)[0]

    first = carla_window_references_ms(
        sequence,
        context_ms=100,
        stride_ms=10,
        minimum_positive_ttc_s=0.1,
        max_windows=8,
    )
    second = carla_window_references_ms(
        sequence,
        context_ms=100,
        stride_ms=10,
        minimum_positive_ttc_s=0.1,
        max_windows=8,
    )

    assert np.array_equal(first, second)
    assert first.shape == (8,)
    assert first[0] >= 100
    assert first[-1] <= 900


def _manifest_sequence(index: int, collision_type: str) -> CarlaLoomingSequence:
    metadata = CarlaLoomingMetadata(
        collision_type=collision_type,
        collision_type_raw=collision_type,
        collision=collision_type in {"car", "pedestrian"},
        t_end_ms=4000,
        dt_ms=10.0,
        relative_velocity_mps=5.0,
        object_diameter=2.0,
    )
    return CarlaLoomingSequence(
        sequence_id=f"example_{index}",
        example_index=index,
        relative_dir=f"example_{index}",
        events_filename="events.npy",
        metadata_filename="sim_data.npz",
        metadata=metadata,
        num_events=100 + index,
        first_event_ms=0,
        last_event_ms=3990,
        valid=True,
        issues=(),
        split_group=f"generation_block_{index // 2:04d}",
    )


def test_manifest_roundtrip_and_grouped_splits_are_disjoint(tmp_path: Path) -> None:
    labels = ("car", "pedestrian", "none", "none_with_traffic")
    sequences = [_manifest_sequence(index, labels[index % len(labels)]) for index in range(18)]
    manifest_path = tmp_path / "manifest.json"
    split_path = tmp_path / "split.json"

    manifest = write_carla_looming_manifest(manifest_path, sequences, root_hint="portable/root")
    split = write_carla_looming_splits(
        split_path,
        manifest_path=manifest_path,
        sequences=sequences,
        seed=7,
        folds=3,
    )

    assert manifest["security"]["numpy_allow_pickle"] is False
    assert len(read_carla_looming_manifest(manifest_path)) == len(sequences)
    assignments = {role: set(ids) for role, ids in split["assignments"].items()}
    groups = {role: set(ids) for role, ids in split["groups"].items()}
    assert all(assignments.values())
    assert all(groups.values())
    assert assignments["train"].isdisjoint(assignments["validation"])
    assert assignments["train"].isdisjoint(assignments["test"])
    assert assignments["validation"].isdisjoint(assignments["test"])
    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])
    assert set().union(*assignments.values()) == {sequence.sequence_id for sequence in sequences}


def test_signed_split_survives_manifest_line_ending_conversion(tmp_path: Path) -> None:
    labels = ("car", "pedestrian", "none", "none_with_traffic")
    sequences = [_manifest_sequence(index, labels[index % len(labels)]) for index in range(18)]
    manifest_path = tmp_path / "manifest.json"
    split_path = tmp_path / "split.json"
    write_carla_looming_manifest(manifest_path, sequences, root_hint="portable/root")
    split = write_carla_looming_splits(
        split_path,
        manifest_path=manifest_path,
        sequences=sequences,
        seed=7,
        folds=3,
    )
    manifest_path.write_bytes(manifest_path.read_bytes().replace(b"\n", b"\r\n"))

    inspection = inspect_carla_jepa_pairs(
        root=tmp_path,
        manifest_path=manifest_path,
        split_path=split_path,
        config=CarlaJEPATrainerConfig(
            context_ms=100,
            stride_ms=100,
            horizons_ms=(50,),
            future_window_ms=100,
            max_windows_per_sequence=1,
        ),
    )

    assert split["format_version"] == 2
    assert split["manifest_artifact_sha256"]
    assert all(
        inspection["roles"][role]["pair_count"] > 0
        for role in ("train", "validation", "test")
    )


def test_split_builder_is_reproducible() -> None:
    labels = ("car", "pedestrian", "none", "none_with_crossing")
    sequences = [_manifest_sequence(index, labels[index % len(labels)]) for index in range(30)]

    first = create_carla_looming_splits(sequences, seed=42, folds=5)
    second = create_carla_looming_splits(sequences, seed=42, folds=5)

    assert first == second
