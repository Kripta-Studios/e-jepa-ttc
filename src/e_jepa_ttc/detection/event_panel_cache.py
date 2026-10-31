"""Read a completed, hash-bound event panel without opening annotations or raw streams."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from e_jepa_ttc.detection.eap_rgb_data import safe_dataset_path, sha256_file
from e_jepa_ttc.detection.event_fusion import EventFusionObservables


def seal_event_panel(root: str | Path) -> dict[str, Any]:
    """Bind causal receipts and tensor files after the entire preparation has completed."""
    root = Path(root).resolve()
    status = json.loads((root / "STATUS.json").read_text())
    contract_sha = sha256_file(root / "CONTRACT.json")
    index_sha = sha256_file(root / "INDEX.json")
    if (
        status.get("status") != "COMPLETE"
        or status.get("contract_sha256") != contract_sha
        or status.get("index_sha256") != index_sha
    ):
        raise ValueError("Cannot seal an incomplete or changed event panel")
    rows = json.loads((root / "INDEX.json").read_text())
    sealed = []
    for row in rows:
        path = safe_dataset_path(root, row["path"])
        receipt_path = path.with_suffix(".json")
        receipt = json.loads(receipt_path.read_text())
        if (
            sha256_file(path) != row["sha256"]
            or receipt["file_sha256"] != row["sha256"]
            or receipt["identity"]["sample_token"] != row["sample_token"]
            or receipt["identity"]["role"] != row["role"]
            or receipt["identity"]["contract_sha256"] != contract_sha
        ):
            raise ValueError("Cannot seal inconsistent event identity or bytes")
        sealed.append({**row, "receipt_sha256": sha256_file(receipt_path)})
    destination = root / "SEALED_INDEX.json"
    text = json.dumps(sealed, indent=2, allow_nan=False)
    if destination.exists() and destination.read_text() != text:
        raise ValueError("Existing event seal differs")
    if not destination.exists():
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(destination)
    admission = {
        "schema": "eap_causal_event_panel_admission_v1",
        "status": "PASS",
        "source_index_sha256": index_sha,
        "sealed_index_sha256": sha256_file(destination),
        "contract_sha256": contract_sha,
        "frames": len(rows),
    }
    target = root / "ADMISSION.json"
    if target.exists() and json.loads(target.read_text()) != admission:
        raise ValueError("Existing event admission differs")
    target.write_text(json.dumps(admission, indent=2), encoding="utf-8")
    return admission


@dataclass(frozen=True)
class CachedEventBatch:
    """Current-frame sensor inputs, bound in exactly the RGB batch order."""

    tokens: tuple[str, ...]
    voxel: torch.Tensor
    mask: torch.Tensor
    observables: EventFusionObservables

    def model_kwargs(self) -> dict[str, Any]:
        """Return only sensor inputs; the caller chooses the experimental control."""
        return {
            "event_tokens": self.tokens,
            "event_voxel": self.voxel,
            "event_mask": self.mask,
            "event_observables": self.observables,
        }


class CausalEventPanelCache:
    """Require an externally pinned contract and index before any training reads."""

    def __init__(
        self, root: str | Path, *, contract_sha256: str, index_sha256: str, manifest_sha256: str
    ) -> None:
        self.root = Path(root).resolve()
        if sha256_file(self.root / "CONTRACT.json") != contract_sha256:
            raise ValueError("Event cache contract hash differs")
        if sha256_file(self.root / "SEALED_INDEX.json") != index_sha256:
            raise ValueError("Event cache index hash differs")
        status = json.loads((self.root / "STATUS.json").read_text())
        admission = json.loads((self.root / "ADMISSION.json").read_text())
        if (
            status.get("status") != "COMPLETE"
            or status.get("contract_sha256") != contract_sha256
            or status.get("index_sha256") != sha256_file(self.root / "INDEX.json")
            or admission.get("source_index_sha256") != status.get("index_sha256")
            or admission.get("sealed_index_sha256") != index_sha256
            or admission.get("status") != "PASS"
        ):
            raise ValueError("Event panel preparation is not complete and bound")
        self.contract_sha256 = contract_sha256
        self.contract = json.loads((self.root / "CONTRACT.json").read_text())
        if (
            self.contract.get("schema") != "eap_causal_event_panel_v1"
            or self.contract.get("manifest_sha256") != manifest_sha256
        ):
            raise ValueError("Event panel population contract differs")
        self.config = self.contract["config"]
        if (
            self.config["width"],
            self.config["height"],
            self.config["window_ms"],
            self.config["bins_per_polarity"],
            self.config["normalize"],
        ) != (160, 96, 50, 5, True):
            raise ValueError("Unsupported event representation contract")
        self.rows: dict[str, dict[str, Any]] = {}
        entries = json.loads((self.root / "SEALED_INDEX.json").read_text())
        ownership = {
            token: role for role, tokens in self.contract["populations"].items() for token in tokens
        }
        for row in entries:
            token = row["sample_token"]
            if token in self.rows or ownership.get(token) != row["role"]:
                raise ValueError("Duplicate event token or invalid role ownership")
            self.rows[token] = row
        if set(self.rows) != set(ownership) or len(self.rows) != status["frames"]:
            raise ValueError("Event cache population is incomplete")

    def load_batch(
        self, requests: Sequence[tuple[str, int]], *, role: str, device: torch.device
    ) -> CachedEventBatch:
        """Verify token, role, exposure cutoff and exact voxel bytes on every read."""
        if role not in ("train", "selection") or not requests:
            raise ValueError("A nonempty train or selection event batch is required")
        tokens = tuple(token for token, _ in requests)
        if len(tokens) != len(set(tokens)):
            raise ValueError("Duplicate current identities in event batch")
        voxels, counts = [], []
        for token, cutoff in requests:
            row = self.rows.get(token)
            if row is None or row["role"] != role:
                raise ValueError("Event token is outside the requested role")
            path = safe_dataset_path(self.root, row["path"])
            receipt_path = path.with_suffix(".json")
            if sha256_file(receipt_path) != row["receipt_sha256"]:
                raise ValueError("Event causal receipt hash differs")
            receipt = json.loads(receipt_path.read_text())
            identity = receipt["identity"]
            representation = receipt["representation"]
            if (
                identity["sample_token"] != token
                or identity["role"] != role
                or identity["contract_sha256"] != self.contract_sha256
                or identity["rgb_exposure_end_timestamp_us"] != cutoff
                or representation["window_end_us_exclusive"] != cutoff
                or representation["window_start_us_inclusive"] != cutoff - 50_000
                or representation["uses_future_events"] is not False
                or representation["config"] != self.config
            ):
                raise ValueError("Event identity or causal window differs from RGB request")
            if sha256_file(path) != row["sha256"] or receipt["file_sha256"] != row["sha256"]:
                raise ValueError("Event tensor file hash differs")
            with np.load(path, allow_pickle=False) as arrays:
                if arrays.files != ["voxel"]:
                    raise ValueError("Event cache contains unexpected arrays")
                voxel = arrays["voxel"]
            if voxel.shape != (10, 96, 160) or voxel.dtype != np.float32:
                raise ValueError("Event voxel shape or dtype differs")
            if not np.isfinite(voxel).all():
                raise ValueError("Event voxel contains nonfinite values")
            if (
                hashlib.sha256(np.asarray(voxel, dtype="<f4").tobytes()).hexdigest()
                != (representation["voxel_sha256_float32_le"])
            ):
                raise ValueError("Event voxel content hash differs")
            count = representation["event_count"]
            if type(count) is not int or count < 0:
                raise ValueError("Event count must be a nonnegative integer")
            voxels.append(torch.from_numpy(voxel))
            counts.append(count)
        return CachedEventBatch(
            tokens,
            torch.stack(voxels).to(device),
            torch.ones(len(tokens), dtype=torch.bool, device=device),
            EventFusionObservables(
                torch.tensor(counts, dtype=torch.float32, device=device),
                torch.full((len(tokens),), 50_000.0, device=device),
            ),
        )
