from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pandas as pd
import pytest

from operational.independent_refit import h5_admission
from operational.independent_refit.population import canonical_sha256, file_sha256
from operational.independent_refit.raw_history import ROLE_KEYS, SCHEMA


def _event_file(path: Path, *, offset: int = 0, broken_index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamps = np.array([0, 100_000, 620_000], dtype=np.int64) + offset
    with h5py.File(path, "w") as handle:
        events = handle.create_group("events")
        for key, value in {
            "x": np.array([1, 2, 3], dtype=np.int16),
            "y": np.array([4, 5, 6], dtype=np.int16),
            "t": timestamps,
            "p": np.array([0, 1, 0], dtype=np.uint8),
        }.items():
            events.create_dataset(key, data=value)
        index = np.searchsorted(timestamps, np.arange(622, dtype=np.int64) * 1000)
        if broken_index:
            index[20], index[21] = 2, 1
        handle.create_dataset("ms_to_idx", data=index)


@pytest.fixture()
def raw_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root, eap = tmp_path / "raw", tmp_path / "eap"
    root.mkdir()
    sequences = [f"s{index:02d}" for index in range(24)]
    rows = []
    for ordinal, sequence in enumerate(sequences):
        rows.append(
            {
                "fit_ordinal": ordinal,
                "sample_token": f"{sequence}-q",
                "sequence_id": sequence,
                "events_path": f"data/train/{sequence}/events.h5",
                "observation_source_start_us": 0,
                "event_history_anchor_us": 620_000,
                "observation_windows_us": [
                    [20_000 * index, 20_000 + 20_000 * index] for index in range(31)
                ],
            }
        )
        _event_file(eap / f"data/train/{sequence}/events.h5")
    public_manifest = tmp_path / "eap_public.json"
    public_manifest.write_text(
        json.dumps(
            {
                "api_sha256": "fixture",
                "files": {
                    f"data/train/{sequence}/events.h5": {
                        "sha256": file_sha256(eap / f"data/train/{sequence}/events.h5"),
                        "size_bytes": (eap / f"data/train/{sequence}/events.h5").stat().st_size,
                    }
                    for sequence in sequences
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(h5_admission, "PUBLIC_MANIFEST_PATH", public_manifest)
    pd.DataFrame(rows).to_parquet(root / "raw_history_index.parquet", index=False)
    roles = {}
    blocks = [set(sequences[index * 4 : index * 4 + 4]) for index in range(6)]
    for rotation in range(6):
        role_groups = {
            "refiner": blocks[(rotation + 1) % 6],
            "producer": set(sequences) - blocks[rotation] - blocks[(rotation + 1) % 6],
        }
        for role, groups in role_groups.items():
            key = f"rotation_{rotation}_{role}"
            path = root / "roles" / f"{key}.parquet"
            path.parent.mkdir(exist_ok=True)
            selected = [row for row in rows if row["sequence_id"] in groups]
            pd.DataFrame(
                {
                    "rotation": rotation,
                    "rotation_role": role,
                    "fit_ordinal": [row["fit_ordinal"] for row in selected],
                    "sample_token": [row["sample_token"] for row in selected],
                    "sequence_id": [row["sequence_id"] for row in selected],
                }
            ).to_parquet(path, index=False)
            roles[key] = {
                "path": path.relative_to(root).as_posix(),
                "sha256": file_sha256(path),
            }
    assert set(roles) == set(ROLE_KEYS)
    receipt_path = root / "RECEIPT.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "training_admitted": False,
                "index_path": "raw_history_index.parquet",
                "index_sha256": file_sha256(root / "raw_history_index.parquet"),
                "fold_contract_path": str(root / "FOLD_CONTRACT.json"),
                "roles": roles,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        h5_admission,
        "verify_raw_history_index",
        lambda _path: {"status": "PASS", "queries": 53_047, "groups": 24, "roles": 12},
    )
    (root / "FOLD_CONTRACT.json").write_text("{}", encoding="utf-8")
    by_sequence = {row["sequence_id"]: row for row in rows}
    monkeypatch.setattr(
        h5_admission,
        "_load_rotation_role_unverified",
        lambda _fold, *, rotation, role: pd.DataFrame(
            {
                "rotation": rotation,
                "rotation_role": role,
                "fit_ordinal": [by_sequence[s]["fit_ordinal"] for s in sorted(blocks[rotation])],
                "sample_token": [by_sequence[s]["sample_token"] for s in sorted(blocks[rotation])],
                "sequence_id": sorted(blocks[rotation]),
            }
        ),
    )
    return receipt_path, eap


def test_inventory_is_stat_only_and_training_closed(
    raw_contract: tuple[Path, Path], tmp_path: Path
) -> None:
    raw, eap = raw_contract
    receipt = h5_admission.build_h5_inventory(raw, eap, tmp_path / "inventory")
    assert (
        receipt["file_count"],
        receipt["h5_files_opened"],
        receipt["full_hashes_computed"],
        receipt["training_admitted"],
    ) == (24, False, False, False)
    assert all(len(item["roles"]) == 5 for item in receipt["files"])
    with pytest.raises(FileExistsError, match="overwrite"):
        h5_admission.build_h5_inventory(raw, eap, tmp_path / "inventory")


def test_full_admission_and_substituted_file_fail_closed(
    raw_contract: tuple[Path, Path], tmp_path: Path
) -> None:
    raw, eap = raw_contract
    inventory = tmp_path / "inventory"
    preflight, admission = tmp_path / "preflight", tmp_path / "admission"
    h5_admission.build_h5_inventory(raw, eap, inventory)
    assert (
        h5_admission.build_h5_metadata_preflight(inventory / "INVENTORY.json", preflight)["status"]
        == "PASS_ALL_HISTORY_VALID_NOT_HASHED"
    )
    assert (
        h5_admission.build_h5_admission(preflight / "PREFLIGHT.json", admission)[
            "training_admitted"
        ]
        is True
    )
    with (eap / "data/train/s00/events.h5").open("ab") as stream:
        stream.write(b"substituted")
    with pytest.raises(ValueError, match="inventory file|file/hash"):
        h5_admission.verify_h5_admission(admission / "ADMISSION.json")


@pytest.mark.parametrize(("offset", "broken_index"), [(10_000_000, False), (0, True)])
def test_rejects_timebase_shift_and_broken_ms_index(
    tmp_path: Path, offset: int, broken_index: bool
) -> None:
    path = tmp_path / "events.h5"
    _event_file(path, offset=offset, broken_index=broken_index)
    with pytest.raises(ValueError, match="bounds/index"):
        h5_admission._inspect_h5(path, 0, 620_000)


def test_metadata_preflight_blocks_uncovered_queries(
    raw_contract: tuple[Path, Path], tmp_path: Path
) -> None:
    raw, eap = raw_contract
    _event_file(eap / "data/train/s00/events.h5", offset=1_000_000)
    inventory, preflight = tmp_path / "inventory", tmp_path / "preflight"
    h5_admission.build_h5_inventory(raw, eap, inventory)
    receipt = h5_admission.build_h5_metadata_preflight(inventory / "INVENTORY.json", preflight)
    assert receipt["status"] == "PASS_WITH_INVALID_HISTORY_ROWS_EXCLUDED_NOT_HASHED"
    assert receipt["blocked_queries"] == 1
    mask = pd.read_parquet(preflight / "history_admissibility.parquet")
    history_valid = cast(pd.Series, mask["history_valid"])
    invalid_reason = cast(pd.Series, mask.loc[~history_valid, "invalid_reason"])
    assert history_valid.tolist().count(True) == 23
    assert set(invalid_reason.astype(str).tolist()) == {
        "NO_AUTHORITATIVE_RECORDING_START_BEFORE_FIRST_EVENT"
    }


def test_rejects_rehashed_inventory_admission_claim(
    raw_contract: tuple[Path, Path], tmp_path: Path
) -> None:
    raw, eap = raw_contract
    output = tmp_path / "inventory"
    h5_admission.build_h5_inventory(raw, eap, output)
    path = output / "INVENTORY.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["training_admitted"] = True
    payload = dict(receipt)
    payload.pop("receipt_sha256")
    receipt["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="inventory receipt"):
        h5_admission.verify_h5_inventory(path)


def test_rejects_raw_receipt_substitution(raw_contract: tuple[Path, Path], tmp_path: Path) -> None:
    raw, eap = raw_contract
    output = tmp_path / "inventory"
    receipt = h5_admission.build_h5_inventory(raw, eap, output)
    raw.write_text("{}", encoding="utf-8")
    assert file_sha256(raw) != receipt["raw_history_receipt_sha256"]
    with pytest.raises(ValueError, match="Raw-history receipt changed"):
        h5_admission.verify_h5_inventory(output / "INVENTORY.json")


def test_preflight_counts_overlapping_invalid_causes_once(
    raw_contract: tuple[Path, Path], tmp_path: Path
) -> None:
    raw, eap = raw_contract
    with h5py.File(eap / "data/train/s00/events.h5", "r+") as handle:
        handle.attrs["t_start_us"] = 100_000
        handle.attrs["t_end_us"] = 500_000
    public_path = h5_admission.PUBLIC_MANIFEST_PATH
    public = json.loads(public_path.read_text(encoding="utf-8"))
    changed = eap / "data/train/s00/events.h5"
    public["files"]["data/train/s00/events.h5"].update(
        sha256=file_sha256(changed), size_bytes=changed.stat().st_size
    )
    public_path.write_text(json.dumps(public), encoding="utf-8")
    inventory, preflight = tmp_path / "inventory", tmp_path / "preflight"
    h5_admission.build_h5_inventory(raw, eap, inventory)
    receipt = h5_admission.build_h5_metadata_preflight(inventory / "INVENTORY.json", preflight)
    assert receipt["blocked_queries"] == 1
    assert receipt["admissible_queries"] == 23


def test_admission_resume_skips_completed_public_digest(
    raw_contract: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, eap = raw_contract
    inventory, preflight = tmp_path / "inventory", tmp_path / "preflight"
    output, resume = tmp_path / "admission", tmp_path / ".admission.resume"
    h5_admission.build_h5_inventory(raw, eap, inventory)
    h5_admission.build_h5_metadata_preflight(inventory / "INVENTORY.json", preflight)
    original = h5_admission._sha256_stream
    calls: list[str] = []

    def interrupt_second(path: Path) -> str:
        calls.append(path.name + ":" + path.parent.name)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return original(path)

    monkeypatch.setattr(h5_admission, "_sha256_stream", interrupt_second)
    with pytest.raises(RuntimeError, match="interrupted"):
        h5_admission.build_h5_admission(preflight / "PREFLIGHT.json", output, resume_dir=resume)
    first_calls = calls.count("events.h5:s00")
    monkeypatch.setattr(h5_admission, "_sha256_stream", original)
    h5_admission.build_h5_admission(preflight / "PREFLIGHT.json", output, resume_dir=resume)
    assert first_calls == 1
    assert not resume.exists()
    assert (output / "parts/00_s00.json").is_file()


def test_admission_rejects_digest_not_in_public_release(
    raw_contract: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, eap = raw_contract
    inventory, preflight = tmp_path / "inventory", tmp_path / "preflight"
    h5_admission.build_h5_inventory(raw, eap, inventory)
    h5_admission.build_h5_metadata_preflight(inventory / "INVENTORY.json", preflight)
    monkeypatch.setattr(h5_admission, "_sha256_stream", lambda _path: "0" * 64)
    with pytest.raises(ValueError, match="public eAP release"):
        h5_admission.build_h5_admission(preflight / "PREFLIGHT.json", tmp_path / "admission")
