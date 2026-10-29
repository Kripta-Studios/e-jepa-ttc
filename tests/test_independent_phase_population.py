from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from operational.independent_refit import lineage, population


@pytest.fixture(scope="module")
def independent_contract(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("independent_population")
    split = root / "split.json"
    garl_data = root / "garl_data.parquet"
    annotations = root / "annotations.parquet"
    eap = root / "eap.parquet"
    teacher = root / "model.safetensors"
    for path in (split, garl_data, annotations, eap, teacher):
        path.write_text(path.name, encoding="utf-8")

    train_groups = {f"train_{index:02d}" for index in range(24)}
    split_roles = {
        "train": train_groups,
        "selection": {f"selection_{index:02d}" for index in range(8)},
        "calibration": {f"calibration_{index:02d}" for index in range(8)},
    }
    counts = [2211] * 23 + [2194]
    records: list[dict[str, object]] = []
    public_ordinal = 0
    for group, count in zip(sorted(train_groups), counts, strict=True):
        for local in range(count):
            records.append(
                {
                    "public_source_ordinal": public_ordinal,
                    "fit_ordinal": public_ordinal,
                    "sample_token": f"{group}_track_{local:06d}",
                    "group_id": group,
                    "sequence_id": group,
                    "track_id": f"{group}_track",
                    "public_track_id": f"{group}_track",
                    "timestamp_us": local * 100_000,
                }
            )
            public_ordinal += 1
    frame = pd.DataFrame.from_records(records, columns=population.IDENTITY_COLUMNS)
    assert len(frame) == 53_047

    population._verify_split = lambda _path: ({"seed": 7}, split_roles)
    population._load_raw_identity = lambda *_args: (
        frame,
        dict(population.EXPECTED_COUNTS),
        {
            "history_slots_checked": 0,
            "future_history_entries": 0,
            "cross_sequence_history_entries": 0,
            "annotation_values_read": False,
            "sensor_payloads_read": False,
            "historical_train40_learned_artifacts_read": False,
        },
    )
    lineage._verify_split = population._verify_split
    lineage._load_raw_identity = population._load_raw_identity
    population.EXPECTED_EAP_SHA256 = population.file_sha256(eap)
    teacher_sha = population.file_sha256(teacher)
    population.TEACHER_BYTES = teacher.stat().st_size
    population.TEACHER_FILE_SHA256 = teacher_sha
    lineage.TEACHER_BYTES = teacher.stat().st_size
    lineage.TEACHER_FILE_SHA256 = teacher_sha
    output = root / "output"
    population.build_independent_population(
        split_path=split,
        garl_data_path=garl_data,
        garl_annotations_path=annotations,
        eap_data_path=eap,
        teacher_weights_path=teacher,
        output_dir=output,
    )
    return output / "FOLD_CONTRACT.json"


def test_exact_oof_coverage_and_role_frequency(independent_contract: Path) -> None:
    result = lineage.verify_independent_contract(independent_contract)
    assert result["queries"] == 53_047
    assert result["groups"] == 24
    assert result["rotations"] == 6


def test_balancing_is_deterministic_and_target_independent() -> None:
    counts = {f"g{index:02d}": 100 + (index % 7) for index in range(24)}
    arbitrary_targets_a = {group: float(index) for index, group in enumerate(counts)}
    arbitrary_targets_b = {group: -value for group, value in arbitrary_targets_a.items()}
    first = population.balanced_group_blocks(counts, seed=7)
    second = population.balanced_group_blocks(dict(reversed(list(counts.items()))), seed=7)
    assert first == second
    assert arbitrary_targets_a != arbitrary_targets_b
    assert sorted(group for block in first for group in block) == sorted(counts)
    assert {len(block) for block in first} == {4}


def test_rejects_duplicate_query_bytes(independent_contract: Path, tmp_path: Path) -> None:
    root = independent_contract.parent
    role_path = root / "folds/rotation_0/producer.parquet"
    backup = tmp_path / "producer.parquet"
    backup.write_bytes(role_path.read_bytes())
    frame = pd.read_parquet(role_path)
    frame.loc[1, "sample_token"] = frame.loc[0, "sample_token"]
    frame.to_parquet(role_path, index=False)
    try:
        with pytest.raises(ValueError, match="rows changed"):
            lineage.verify_independent_contract(independent_contract)
    finally:
        role_path.write_bytes(backup.read_bytes())


def test_rejects_forged_group_count(independent_contract: Path) -> None:
    root = independent_contract.parent
    manifest_path = root / "folds/rotation_0/MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["roles"]["producer"]["groups"] = 15
    with pytest.raises(ValueError, match="identity differs"):
        lineage._role_rows(manifest_path, manifest, "producer")


def test_rejects_nonexistent_parent_receipt(independent_contract: Path) -> None:
    with pytest.raises(FileNotFoundError):
        lineage.validate_fit_parents(
            independent_contract,
            rotation=0,
            fit_role="producer",
            component_id="E_A5",
            parents=[
                {
                    "receipt_path": "X:/does/not/exist.json",
                    "receipt_sha256": "b" * 64,
                }
            ],
        )


def test_physical_teacher_and_local_fits_fail_closed(independent_contract: Path) -> None:
    teacher_receipt = independent_contract.parent / "TEACHER_PARENT.json"
    lineage.validate_fit_parents(
        independent_contract,
        rotation=0,
        fit_role="producer",
        component_id="E_A5",
        parents=[
            {
                "receipt_path": str(teacher_receipt),
                "receipt_sha256": population.file_sha256(teacher_receipt),
            }
        ],
    )

    producer = lineage.load_rotation_role(independent_contract, rotation=0, role="producer")
    generation = lineage.load_rotation_role(independent_contract, rotation=0, role="generation")
    references = []
    for stage in ("E_A5", "E_C2F", "PAIR_E"):
        artifact = independent_contract.parent / f"{stage}.bin"
        artifact.write_bytes(stage.encode())
        receipt = {
            "schema": "independent_phase_fit_parent_v1",
            "status": "COMPLETE_VERIFIED_INDEPENDENT_FIT",
            "kind": "independent_phase_fit",
            "artifact_path": str(artifact.resolve()),
            "artifact_sha256": population.file_sha256(artifact),
            "fold_contract_sha256": population.file_sha256(independent_contract),
            "rotation": 0,
            "stage_id": stage,
            "train_role": "producer",
            "train_group_ids": sorted(producer["group_id"].astype(str).unique()),
            "train_token_set_sha256": population.identity_sha256(
                producer["sample_token"].astype(str).tolist(), ordered=False
            ),
            "generation_group_ids": sorted(generation["group_id"].astype(str).unique()),
            "generation_excluded": True,
        }
        receipt["receipt_sha256"] = population.canonical_sha256(receipt)
        receipt_path = independent_contract.parent / f"{stage}.json"
        receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
        references.append(
            {
                "receipt_path": str(receipt_path),
                "receipt_sha256": population.file_sha256(receipt_path),
            }
        )
    with pytest.raises(RuntimeError, match="fitter emits an admitted"):
        lineage.validate_fit_parents(
            independent_contract,
            rotation=0,
            fit_role="refiner",
            component_id="E_CTX",
            parents=references,
        )


def test_rejects_unknown_source_hash(independent_contract: Path) -> None:
    source = independent_contract.parent / "SOURCE_RECEIPT.json"
    value = json.loads(source.read_text(encoding="utf-8"))
    first_source = Path(next(iter(value["sources"].values()))["path"])
    original = first_source.read_bytes()
    first_source.write_bytes(original + b"changed")
    try:
        with pytest.raises(ValueError, match="Raw source changed"):
            lineage.verify_independent_contract(independent_contract)
    finally:
        first_source.write_bytes(original)


def test_generation_targets_are_never_admitted(independent_contract: Path) -> None:
    with pytest.raises(ValueError, match="own producer/refiner"):
        lineage.join_supervised_targets(
            independent_contract,
            rotation=0,
            fit_role="generation",
            annotations_path=independent_contract.parent / "unused.parquet",
        )
