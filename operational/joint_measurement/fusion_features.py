"""Admission and H-only normalization for newly extracted frozen-parent features.

No parent or feature recipe is selected here. Callers must bind extraction code,
endpoint bytes, ordered feature semantics and the cohort before using this API.
Sensor clocks remain separate: equal queries do not imply equal frame spacing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from e_jepa_ttc.rgb_port.normalization import FrozenNormalizer, fit_normalizer


@dataclass(frozen=True)
class FeatureIdentity:
    """Values verified against durable extraction receipts by the cache loader."""

    modality: str
    endpoint_sha256: str
    schema_sha256: str
    extractor_sha256: str

    def validate(self) -> None:
        """Reject missing identity rather than accepting a matching tensor width."""
        if self.modality not in {"rgb", "event"}:
            raise ValueError("Unknown feature modality")
        for value in (self.endpoint_sha256, self.schema_sha256, self.extractor_sha256):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Feature identity requires lowercase SHA-256 digests")

    def normalizer_producer_sha256(self) -> str:
        """Bind statistics to both the endpoint and the code that extracted features."""
        self.validate()
        value = (
            f"v18_fusion_features_v1:{self.modality}:{self.endpoint_sha256}:"
            f"{self.schema_sha256}:{self.extractor_sha256}"
        )
        return hashlib.sha256(value.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class FeatureBlock:
    """Label-free observations in exact manifest order, including missing RGB rows."""

    identity: FeatureIdentity
    role: str
    manifest_sha256: str
    sample_tokens: np.ndarray
    features: np.ndarray
    prediction_ttc: np.ndarray
    available: np.ndarray
    sensor_pair_us: np.ndarray

    def validate(
        self,
        *,
        expected_identity: FeatureIdentity,
        role: str,
        manifest_sha256: str,
        sample_tokens: np.ndarray,
        feature_count: int,
    ) -> None:
        """Check producer genealogy, cohort order, coverage and native sensor clocks."""
        self.identity.validate()
        if self.identity != expected_identity:
            raise ValueError("Feature producer/schema/extractor identity changed")
        if role not in {"P", "H", "V"} or self.role != role:
            raise ValueError("Feature population role changed")
        if (
            self.manifest_sha256 != manifest_sha256
            or len(manifest_sha256) != 64
            or any(c not in "0123456789abcdef" for c in manifest_sha256)
        ):
            raise ValueError("Feature cohort manifest changed")
        count = len(sample_tokens)
        if (
            count < 1
            or sample_tokens.shape != (count,)
            or sample_tokens.dtype.kind not in {"U", "S"}
            or len(np.unique(sample_tokens)) != count
            or self.sample_tokens.dtype != sample_tokens.dtype
            or not np.array_equal(self.sample_tokens, sample_tokens)
        ):
            raise ValueError("Feature queries are duplicate, reordered or incomplete")
        if (
            feature_count < 1
            or self.features.shape != (count, feature_count)
            or self.features.dtype != np.float32
            or self.prediction_ttc.shape != (count,)
            or self.prediction_ttc.dtype != np.float32
            or self.available.shape != (count,)
            or self.available.dtype != np.bool_
            or self.sensor_pair_us.shape != (count, 2)
            or self.sensor_pair_us.dtype != np.int64
        ):
            raise ValueError("Feature arrays violate FP32, boolean or clock schema")
        if self.identity.modality == "event" and not self.available.all():
            raise ValueError("Event fallback must cover every query")
        selected = self.available
        if (
            not np.isfinite(self.features[selected]).all()
            or not np.isfinite(self.prediction_ttc[selected]).all()
            or np.any(self.prediction_ttc[selected] == 0)
        ):
            raise ValueError("Available features and signed TTC must be valid")
        clocks = self.sensor_pair_us[selected]
        if np.any(clocks[:, 1] <= clocks[:, 0]):
            raise ValueError("Each modality requires its own chronological sensor pair")

    def fit_h_normalizer(self) -> FrozenNormalizer:
        """Fit only available H observations; never substitute missing RGB with zero."""
        self.validate(
            expected_identity=self.identity,
            role="H",
            manifest_sha256=self.manifest_sha256,
            sample_tokens=self.sample_tokens,
            feature_count=self.features.shape[-1],
        )
        if not self.available.any():
            raise ValueError("No available H observations for normalization")
        return fit_normalizer(
            self.features[self.available],
            self.sample_tokens[self.available].astype(str).tolist(),
            fit_role=self.role,
            expected_role="H",
            producer_sha256=self.identity.normalizer_producer_sha256(),
            schema_sha256=self.identity.schema_sha256,
            modality=self.identity.modality,
        )

    def normalized(self, normalizer: FrozenNormalizer) -> np.ndarray:
        """Apply the admitted H transform; keep missing rows masked and harmless."""
        normalizer.validate_endpoint(
            modality=self.identity.modality,
            fit_role="H",
            schema_sha256=self.identity.schema_sha256,
            producer_sha256=self.identity.normalizer_producer_sha256(),
        )
        result = np.zeros_like(self.features)
        result[self.available] = normalizer.transform(self.features[self.available])
        return result


def load_feature_cache(
    path: Path,
    *,
    expected_sha256: str,
    expected_identity: FeatureIdentity,
    role: str,
    manifest_sha256: str,
    sample_tokens: np.ndarray,
    feature_count: int,
) -> FeatureBlock:
    """Load a lossless NPZ against independently admitted bytes and cohort identity.

    Expected values must come from the frozen recipe and extraction receipt, not
    be copied from this archive. Hash and decode use the same read-only handle.
    The feature cache has no targets, masses, metric results or pickle objects.
    Call once at admission; retain the returned arrays for batches thereafter.
    """
    fields = {"sample_tokens", "features", "prediction_ttc", "available", "sensor_pair_us"}
    with path.open("rb") as stream:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
        if checksum.hexdigest() != expected_sha256:
            raise ValueError("Fusion feature cache bytes differ from the admitted receipt")
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as archive:
            if set(archive.files) != fields | {"metadata"} or len(archive.files) != 6:
                raise ValueError("Fusion feature cache contains missing or unregistered arrays")
            metadata_array = archive["metadata"]
            if metadata_array.shape != () or metadata_array.dtype.kind != "U":
                raise ValueError("Fusion cache metadata must be a scalar JSON string")
            metadata = json.loads(str(metadata_array.item()))
            if set(metadata) != {"schema", "identity", "role", "manifest_sha256"} or (
                metadata["schema"] != "v18_fusion_feature_cache_v1"
            ):
                raise ValueError("Unsupported fusion cache metadata")
            arrays = {name: archive[name] for name in fields}
    result = FeatureBlock(
        identity=FeatureIdentity(**metadata["identity"]),
        role=metadata["role"],
        manifest_sha256=metadata["manifest_sha256"],
        **arrays,
    )
    result.validate(
        expected_identity=expected_identity,
        role=role,
        manifest_sha256=manifest_sha256,
        sample_tokens=sample_tokens,
        feature_count=feature_count,
    )
    return result
