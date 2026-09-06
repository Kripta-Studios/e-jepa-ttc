"""Verified, role-separated model inputs and supervision for frozen expert tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.risk_geometry_v10 import object_digest, verify
from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase, strict_macro_mass
from e_jepa_ttc.training.stage62_local_field import cross_track_derangement_indices


def array_hash(a: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


@dataclass(frozen=True)
class ModelInputs:
    features: np.ndarray
    phases: np.ndarray
    expert_ttc: np.ndarray


@dataclass(frozen=True)
class Supervision:
    target_phase: np.ndarray
    global_mass: np.ndarray


@dataclass(frozen=True)
class FrozenTable:
    inputs: ModelInputs
    supervision: Supervision
    metadata: pd.DataFrame
    record: dict[str, Any]
    seal: str

    def validate(self, role: str) -> None:
        if object_digest(self.record) != self.seal or self.record["role"] != role:
            raise ValueError("forged table role or manifest")
        for r in self.record["sources"]:
            verify(r)
        physical_metadata = pd.read_csv(self.record["sources"][1]["path"])
        if not physical_metadata.role.eq(role).all():
            raise ValueError("physical producer role cannot be relabeled")
        for key, value in self.array_values().items():
            if array_hash(value) != self.record["arrays"][key]:
                raise ValueError(f"mutated model/supervision array: {key}")
        if (
            object_digest(self.metadata.to_json(orient="split", double_precision=15))
            != self.record["metadata_hash"]
        ):
            raise ValueError("mutated table metadata")

    def array_values(self) -> dict[str, np.ndarray]:
        return dict(
            features=self.inputs.features,
            phases=self.inputs.phases,
            expert_ttc=self.inputs.expert_ttc,
            target_phase=self.supervision.target_phase,
            global_mass=self.supervision.global_mass,
        )


def load_table(record: dict[str, Any], *, role: str) -> FrozenTable:
    """Load exactly indexed numeric fields and bind role independently of frame columns."""
    if record["role"] != role:
        raise ValueError("requested role disagrees with frozen index")
    arrays_path, metadata_path = [verify(r) for r in record["sources"]]
    with np.load(arrays_path, allow_pickle=False) as z:
        x, phases, expert = (z[k].copy() for k in ("features17", "expert_phase", "expert_ttc"))
        target, mass = (z[k].copy() for k in ("target_phase", "global_mass"))
    metadata = pd.read_csv(metadata_path)
    if x.shape != (len(metadata), 17) or any(
        a.dtype != np.float64 or not np.isfinite(a).all() for a in (x, phases, expert, target, mass)
    ):
        raise ValueError("frozen table dtype/shape/finite contract")
    if metadata.sample_token.duplicated().any() or not metadata.role.eq(role).all():
        raise ValueError("duplicate token or metadata role mismatch")
    if not np.array_equal(phases, benchmark_phase(expert)) or not np.allclose(
        target, benchmark_phase(metadata.target_ttc.to_numpy()), rtol=0, atol=1e-14
    ):
        raise ValueError("phase/expert/target identity mismatch")
    if not np.allclose(
        mass,
        strict_macro_mass(metadata.target_ttc.to_numpy(), metadata.sequence_id.to_numpy()),
        rtol=0,
        atol=1e-14,
    ):
        raise ValueError("global mass mismatch")
    table = FrozenTable(
        ModelInputs(x, phases, expert),
        Supervision(target, mass),
        metadata,
        record,
        object_digest(record),
    )
    table.validate(role)
    return table


def create_record(arrays_path: Path, metadata_path: Path, outer: int, role: str) -> dict[str, Any]:
    """Freeze a verified forensic export as a consumer record; no fit here."""
    from e_jepa_ttc.artifacts.risk_geometry_v10 import binding

    with np.load(arrays_path, allow_pickle=False) as z:
        arrays = {
            k: array_hash(z[v])
            for k, v in dict(
                features="features17",
                phases="expert_phase",
                expert_ttc="expert_ttc",
                target_phase="target_phase",
                global_mass="global_mass",
            ).items()
        }
    meta = pd.read_csv(metadata_path)
    return dict(
        outer_fold=outer,
        role=role,
        sources=[binding(arrays_path), binding(metadata_path)],
        arrays=arrays,
        metadata_hash=object_digest(meta.to_json(orient="split", double_precision=15)),
        sequences=sorted(meta.sequence_id.unique().tolist()),
    )


def pair_permutation(table: FrozenTable, seed: int) -> np.ndarray:
    return cross_track_derangement_indices(
        sequence_ids=table.metadata.sequence_id.astype(str).tolist(),
        track_ids=table.metadata.track_id.astype(str).tolist(),
        seed=seed,
    )


def arm_inputs(
    table: FrozenTable, arm: str, donor: np.ndarray, geometry: np.ndarray | None = None
) -> ModelInputs:
    """Move a PAIR action and every derived field together; geometry stays bundled."""
    x = table.inputs.features.copy()
    p = table.inputs.phases.copy()
    t = table.inputs.expert_ttc.copy()
    if "PAIRPERM" in arm:
        p[:, 2] = p[donor, 2]
        t[:, 2] = t[donor, 2]
        x[:, 10] = p[:, 2]
        x[:, 11] = p[:, 2] - p[:, 0]
        x[:, 12] = p[:, 2] - p[:, 1]
        x[:, 14] = np.abs(x[:, 11])
        x[:, 15] = np.abs(x[:, 12])
    if geometry is not None:
        g = geometry[donor] if "SHUFFLE" in arm else geometry
        x = np.concatenate((x, g), axis=1)
    return ModelInputs(x, p, t)


def feature_mask(arm: str, dim: int) -> np.ndarray:
    mask = np.ones(dim, dtype=np.float64)
    if "8GATE" in arm:
        mask[8:17] = 0
    if "GZERO" in arm:
        mask[17:] = 0
    if "GQUALITY" in arm:
        mask[17:] = 0
        mask[np.array([6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23]) + 17] = 1
    return mask
