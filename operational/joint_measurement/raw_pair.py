"""Native raw-event preprocessing followed by the admitted P2 suffix, without labels."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import torch

from operational.range_transfer.common import read, sha, utc, write
from operational.rgb_port.train_producers import EventProducerSource
from operational.rgb_port_revision.raw_events import RawEventSource

from .common import ROOT, RUN, owner, require_resources
from .data import role_rows


class RawPairSource:
    """Preserve historical common-ROI preprocessing, including its T3 computation cost."""

    def __init__(self, role: str) -> None:
        if role not in {"P", "H", "V"}:
            raise ValueError("Unknown raw sensor role")
        self.role = role
        manifest_path = owner() / f"{role}_MANIFEST.json"
        admission = read(RUN / "ADMISSION.json")
        if sha(manifest_path) != admission["files"][str(manifest_path)]:
            raise ValueError("Raw role manifest changed after admission")
        manifest = read(manifest_path)
        if sha(Path(manifest["rows_path"])) != manifest["rows_sha256"]:
            raise ValueError("Raw role metadata changed")
        provenance = read(owner() / "fits/E_A5_MATCHED/RUN_PROVENANCE.json")
        source_root = Path(provenance["source_identity"]["event_source_root"])
        self.prepared = EventProducerSource(manifest_path, source_root)
        self.raw = RawEventSource(self.prepared, Path(manifest["eap_root"]))
        self.rows = role_rows(role)
        self.identity = {
            "role": role,
            "manifest_sha256": sha(manifest_path),
            "raw_identity": self.raw.identity,
            "prepared_identity": dict(self.prepared.identity),
            "source_hashes": {
                name: sha(ROOT / name)
                for name in (
                    "operational/joint_measurement/raw_pair.py",
                    "operational/rgb_port_revision/raw_events.py",
                    "src/e_jepa_ttc/simplex_t/cached_event_reader.py",
                    "src/e_jepa_ttc/data/garlttc_lhr_cache.py",
                    "operational/train40_system/prepare.py",
                )
            },
            "scope": (
                "HDF5 events through historical common-ROI FP32 preprocessing, then last two "
                "windows. Includes redundant third-window preprocessing where present. "
                "No teacher, prepared voxels or TTC labels enter raw batch output."
            ),
        }

    def batch(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return only P2 pixels and their real final-pair separation."""
        batch = self.raw.batch(indices, "event")
        return batch.events[:, -2:], batch.delta_t_s.reshape(len(indices), 1)

    def close(self) -> None:
        """Close only this reader's own bounded handles."""
        self.raw.close()
        self.prepared.close()


def verify(role: str) -> dict:
    """Check one label-blind query per acquisition group against admitted native tensors."""
    require_resources(starting=True)
    torch.set_num_threads(1)
    source = RawPairSource(role)
    checks = []
    try:
        for group, frame in source.rows.groupby("group_id", sort=True):
            require_resources()
            index = min(
                frame.index.tolist(),
                key=lambda i: hashlib.sha256(
                    str(source.rows.iloc[i].sample_token).encode()
                ).hexdigest(),
            )
            prepared = source.prepared.batch([index], "event")
            start = time.perf_counter()
            events, delta = source.batch([index])
            duration = time.perf_counter() - start
            if not torch.equal(events, prepared.events[:, -2:]) or not torch.equal(
                delta.reshape(-1), prepared.delta_t_s.reshape(-1)
            ):
                raise AssertionError(f"Raw P2 parity differs for {group}/{index}")
            checks.append(
                {
                    "group_id": str(group),
                    "role_index": index,
                    "sample_token": str(source.rows.iloc[index].sample_token),
                    "events_exact": True,
                    "delta_exact": True,
                    "raw_preprocessing_seconds_under_concurrent_load": duration,
                }
            )
            print(f"RAW_P2_PARITY {role} {group} PASS", flush=True)
        result = {
            "status": "PASS",
            "identity": source.identity,
            "checks": checks,
            "utc": utc(),
            "optimizer_updates": 0,
            "scope": "Selected queries only, not a full population replay or latency benchmark",
        }
        write(RUN / "verification/raw_pair" / f"{role}.json", result)
        return result
    finally:
        source.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=["P", "H", "V"])
    print(verify(parser.parse_args().role))
