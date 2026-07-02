"""Official EvTTC bbox/ROI protocol coverage checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from e_jepa_ttc.data.types import DatasetSequence


@dataclass(frozen=True)
class OfficialEvTTCSequenceSpec:
    """One sequence required by the published EvTTC bbox/ROI comparison table."""

    name: str
    family: str
    speed_bucket: str | None
    speed_variant: str | None
    group: str
    required_assets: tuple[str, ...] = ("event_hdf5", "gt_ttc", "bbox_roi")


OFFICIAL_EVTTC_REAL_WORLD_BBOX_ROI_SEQUENCES: tuple[OfficialEvTTCSequenceSpec, ...] = (
    OfficialEvTTCSequenceSpec("CCRs-1-low-100%", "CCRs-1", "low", "low-100", "real_world"),
    OfficialEvTTCSequenceSpec(
        "CCRs-1-medium-100%",
        "CCRs-1",
        "medium",
        "medium-100",
        "real_world",
    ),
    OfficialEvTTCSequenceSpec("CCRs-1-high-100%", "CCRs-1", "high", "high-100", "real_world"),
    OfficialEvTTCSequenceSpec("CCRs-2-low-100%", "CCRs-2", "low", "low-100", "real_world"),
    OfficialEvTTCSequenceSpec(
        "CCRs-2-medium-100%",
        "CCRs-2",
        "medium",
        "medium-100",
        "real_world",
    ),
    OfficialEvTTCSequenceSpec("CCRs-2-high-100%", "CCRs-2", "high", "high-100", "real_world"),
    OfficialEvTTCSequenceSpec("CCRm-low-100%", "CCRm", "low", "low-100", "real_world"),
    OfficialEvTTCSequenceSpec("CCRm-medium-100%", "CCRm", "medium", "medium-100", "real_world"),
)

OFFICIAL_EVTTC_SLIDER_SEQUENCES: tuple[OfficialEvTTCSequenceSpec, ...] = (
    OfficialEvTTCSequenceSpec("Slider-750", "Slider", None, "750", "slider"),
    OfficialEvTTCSequenceSpec("Slider-1000", "Slider", None, "1000", "slider"),
)


def _sequence_path_tokens(sequence: DatasetSequence) -> set[str]:
    raw_parts = sequence.extra.get("relative_parts")
    if isinstance(raw_parts, list) and all(isinstance(part, str) for part in raw_parts):
        parts = raw_parts
    else:
        parts = list(Path(sequence.local_path).parts)
    return {part.lower().replace("%", "") for part in parts}


def _matches_spec(sequence: DatasetSequence, spec: OfficialEvTTCSequenceSpec) -> bool:
    if sequence.scenario_family != spec.family:
        return False
    if spec.speed_bucket is not None and sequence.speed_bucket != spec.speed_bucket:
        return False
    if spec.speed_variant is not None:
        tokens = _sequence_path_tokens(sequence)
        if spec.speed_variant.lower().replace("%", "") not in tokens:
            return False
    return True


def _asset_path_exists(sequence: DatasetSequence, field_name: str) -> bool:
    path = sequence.resolve(field_name)
    return bool(path is not None and path.exists())


def _label_count(sequence: DatasetSequence) -> int:
    raw_count = sequence.extra.get("label_count")
    if isinstance(raw_count, int):
        return raw_count
    label_dir = sequence.resolve("label_dir")
    if label_dir is None or not label_dir.exists():
        return 0
    return len(list(label_dir.glob("*.json")))


def _asset_presence(sequence: DatasetSequence | None) -> dict[str, Any]:
    if sequence is None:
        return {
            "event_hdf5": False,
            "gt_ttc": False,
            "bbox_roi": False,
            "gt_hdf5": False,
            "label_count": 0,
        }

    label_count = _label_count(sequence)
    label_dir = sequence.resolve("label_dir")
    return {
        "event_hdf5": _asset_path_exists(sequence, "event_hdf5"),
        "gt_ttc": _asset_path_exists(sequence, "ttc_csv"),
        "bbox_roi": bool(label_dir is not None and label_dir.exists() and label_count > 0),
        "gt_hdf5": _asset_path_exists(sequence, "gt_hdf5"),
        "label_count": label_count,
    }


def _asset_score(sequence: DatasetSequence) -> int:
    assets = _asset_presence(sequence)
    return sum(int(bool(assets[name])) for name in ("event_hdf5", "gt_ttc", "bbox_roi"))


def _find_matching_sequence(
    sequences: list[DatasetSequence],
    spec: OfficialEvTTCSequenceSpec,
) -> DatasetSequence | None:
    candidates = [sequence for sequence in sequences if _matches_spec(sequence, spec)]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda sequence: (-_asset_score(sequence), sequence.sequence_id),
    )[0]


def _coverage_percent(complete_count: int, total_count: int) -> float:
    if total_count == 0:
        return 0.0
    return round(100.0 * complete_count / total_count, 2)


def evaluate_official_evttc_coverage(
    sequences: list[DatasetSequence],
    *,
    include_slider: bool = True,
) -> dict[str, Any]:
    """Evaluate whether local assets cover the official EvTTC bbox/ROI table."""

    specs = list(OFFICIAL_EVTTC_REAL_WORLD_BBOX_ROI_SEQUENCES)
    if include_slider:
        specs.extend(OFFICIAL_EVTTC_SLIDER_SEQUENCES)

    rows: list[dict[str, Any]] = []
    for spec in specs:
        matched = _find_matching_sequence(sequences, spec)
        assets = _asset_presence(matched)
        missing_assets = [name for name in spec.required_assets if not bool(assets[name])]
        complete = matched is not None and not missing_assets
        rows.append(
            {
                "name": spec.name,
                "group": spec.group,
                "spec": asdict(spec),
                "matched_sequence_id": matched.sequence_id if matched is not None else None,
                "local_path": matched.local_path if matched is not None else None,
                "assets": assets,
                "missing_assets": missing_assets,
                "complete": complete,
                "status": "complete"
                if complete
                else ("missing_sequence" if matched is None else "incomplete_assets"),
            }
        )

    real_world_rows = [row for row in rows if row["group"] == "real_world"]
    complete_real_world = [row for row in real_world_rows if row["complete"]]
    complete_all = [row for row in rows if row["complete"]]
    missing_real_world = [row["name"] for row in real_world_rows if not row["complete"]]
    missing_all = [row["name"] for row in rows if not row["complete"]]
    real_world_complete = len(complete_real_world) == len(real_world_rows)
    table_v_complete = len(complete_all) == len(rows)

    blockers: list[str] = []
    if not real_world_complete:
        blockers.append("missing official real-world CCRs1/CCRs2/CCRm bbox/ROI assets")
    if include_slider and not table_v_complete:
        blockers.append("missing slider rows required for full EvTTC Table V reproduction")
    blockers.append(
        "official CMax/STRTTC/ETTCM baseline wrappers still need reproduced runtime parity"
    )

    return {
        "protocol": "evttc-table-v-bbox-roi-asset-coverage",
        "include_slider": include_slider,
        "scanned_sequence_count": len(sequences),
        "official_real_world_required_sequence_count": len(real_world_rows),
        "official_real_world_complete_sequence_count": len(complete_real_world),
        "official_real_world_coverage_percent": _coverage_percent(
            len(complete_real_world),
            len(real_world_rows),
        ),
        "official_table_v_required_sequence_count": len(rows),
        "official_table_v_complete_sequence_count": len(complete_all),
        "official_table_v_coverage_percent": _coverage_percent(len(complete_all), len(rows)),
        "official_real_world_asset_coverage_complete": real_world_complete,
        "official_table_v_asset_coverage_complete": table_v_complete,
        "official_sota_claim_allowed": False,
        "official_sota_claim_blockers": blockers,
        "missing_real_world_sequences": missing_real_world,
        "missing_table_v_sequences": missing_all,
        "rows": rows,
    }
