"""Versioned, nonofficial eAP development evaluation contract."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from e_jepa_ttc.detection.development_metrics import CLASSES

SCHEMA = "phase_ttc_detection_evaluation_protocol_v2"
PROFILE = "eap_local_dev_xy_ap101_population_v2"
DATASET_RELEASE = "NAIL-HNU/eAP-dataset@a814a946e60d37f7242358af5f344542177e50c5"
COORDINATE_FRAME = "ego: X forward, Y left, Z up"
CENTER_UNITS = "metres"
SIZE_ORDER = "length,width,height"
VISIBILITY_RULE = (
    "provisional pinhole: camera-center depth is within center_depth_range_m and the "
    "near-plane-clipped cuboid projection rectangle intersects the image; projected center "
    "and minimum pixel size are not eligibility filters"
)
PREDICTION_DOMAIN_RULE = (
    "retain top-K by score/index per frame, then apply the same pinhole depth and "
    "image-intersection rule as ground truth"
)
MATCHING_RULE = "confidence-ranked one-to-one strict ego-XY center distance"
AP_INTERPOLATION = "101-point precision envelope without recall/precision clipping"
UNSUPPORTED_CLASS_POLICY = "AP is null; supported macro excludes it; all-eight macro uses zero"


@dataclass(frozen=True)
class EvaluationProtocol:
    """Immutable definition of one local detection evaluation population and scorer."""

    schema: str
    profile: str
    dataset_release: str
    classes: tuple[str, ...]
    coordinate_frame: str
    center_units: str
    size_order: str
    center_depth_range_m: tuple[float, float]
    visibility_rule: str
    prediction_domain_rule: str
    matching_rule: str
    distance_thresholds_m: tuple[float, ...]
    diagnostic_threshold_m: float
    ap_interpolation: str
    unsupported_class_policy: str
    max_predictions_per_frame: int
    official_eap_score: bool = False

    @classmethod
    def default(cls) -> EvaluationProtocol:
        """Return the frozen local v2 profile used before an official scorer is available."""
        return cls(
            schema=SCHEMA,
            profile=PROFILE,
            dataset_release=DATASET_RELEASE,
            classes=tuple(CLASSES),
            coordinate_frame=COORDINATE_FRAME,
            center_units=CENTER_UNITS,
            size_order=SIZE_ORDER,
            center_depth_range_m=(0.1, 60.0),
            visibility_rule=VISIBILITY_RULE,
            prediction_domain_rule=PREDICTION_DOMAIN_RULE,
            matching_rule=MATCHING_RULE,
            distance_thresholds_m=(0.5, 1.0, 2.0, 4.0),
            diagnostic_threshold_m=2.0,
            ap_interpolation=AP_INTERPOLATION,
            unsupported_class_policy=UNSUPPORTED_CLASS_POLICY,
            max_predictions_per_frame=50,
        )

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError("Unsupported evaluation protocol schema")
        semantics = (
            (self.profile, PROFILE),
            (self.dataset_release, DATASET_RELEASE),
            (self.coordinate_frame, COORDINATE_FRAME),
            (self.center_units, CENTER_UNITS),
            (self.size_order, SIZE_ORDER),
            (self.visibility_rule, VISIBILITY_RULE),
            (self.prediction_domain_rule, PREDICTION_DOMAIN_RULE),
            (self.matching_rule, MATCHING_RULE),
            (self.ap_interpolation, AP_INTERPOLATION),
            (self.unsupported_class_policy, UNSUPPORTED_CLASS_POLICY),
        )
        if any(actual != expected for actual, expected in semantics):
            raise ValueError("Protocol semantics differ from the implemented v2 scorer")
        if not isinstance(self.classes, tuple) or self.classes != tuple(CLASSES):
            raise ValueError(
                "Protocol classes must be the immutable ordered public eight-class taxonomy"
            )
        if not isinstance(self.center_depth_range_m, tuple) or len(self.center_depth_range_m) != 2:
            raise ValueError("center_depth_range_m must be an immutable pair")
        if not isinstance(self.distance_thresholds_m, tuple):
            raise ValueError("distance_thresholds_m must be an immutable tuple")
        low, high = self.center_depth_range_m
        numeric = (
            *self.center_depth_range_m,
            *self.distance_thresholds_m,
            self.diagnostic_threshold_m,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in numeric
        ):
            raise ValueError("Protocol numeric fields must be finite numbers")
        if not 0 < low < high:
            raise ValueError("Depth range must satisfy 0 < low < high")
        if (
            not self.distance_thresholds_m
            or len(set(self.distance_thresholds_m)) != len(self.distance_thresholds_m)
            or min(self.distance_thresholds_m) <= 0
            or self.diagnostic_threshold_m <= 0
            or self.diagnostic_threshold_m not in self.distance_thresholds_m
        ):
            raise ValueError(
                "Matching thresholds must be distinct, positive and include diagnostics"
            )
        if (
            isinstance(self.max_predictions_per_frame, bool)
            or not isinstance(self.max_predictions_per_frame, int)
            or self.max_predictions_per_frame < 1
        ):
            raise ValueError("max_predictions_per_frame must be a positive integer")
        if self.official_eap_score is not False:
            raise ValueError("This local protocol must not claim to be the official eAP scorer")

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible protocol value."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvaluationProtocol:
        """Validate and restore a protocol from a JSON-compatible mapping."""
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise ValueError("Protocol fields differ from the v2 schema")
        converted = dict(value)
        for name in ("classes", "center_depth_range_m", "distance_thresholds_m"):
            raw = converted[name]
            if not isinstance(raw, (list, tuple)):
                raise ValueError(f"{name} must be an array")
            converted[name] = tuple(raw)
        return cls(**converted)

    @property
    def sha256(self) -> str:
        """Hash canonical UTF-8 JSON so equivalent serialized protocols share identity."""
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
