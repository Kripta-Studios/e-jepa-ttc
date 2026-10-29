"""Lossless, random-access P2 sensor packs; never frozen neural features."""

from __future__ import annotations

import argparse
import io
import os
import time
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch
from operational.range_transfer.common import digest, freeze, read, sha, utc, write
from operational.rgb_port.train_producers import EventProducerSource

from .common import RUN, admit, config, owner, require_resources


def cache_root(role: str) -> Path:
    """Resolve a declared sensor role, with no cross-role fallback."""
    if role not in {"P", "H", "V"}:
        raise ValueError("Invalid role")
    return Path(config()["cache_root"]) / "sensor" / role


def role_rows(role: str) -> pd.DataFrame:
    """Read immutable role metadata and maintain its exact row order."""
    manifest = read(owner() / f"{role}_MANIFEST.json")
    rows = pd.read_parquet(manifest["rows_path"])
    if len(rows) != manifest["population_size"] or set(rows.role) != {role}:
        raise ValueError("Role metadata mismatch")
    return rows


def prepare(role: str, *, max_packs: int | None = None) -> dict:
    """Repack exact native FP32 suffixes and training targets with durable receipts."""
    admission = read(RUN / "ADMISSION.json")
    rows = role_rows(role)
    parent_provenance = read(owner() / "fits/E_A5_MATCHED/RUN_PROVENANCE.json")
    source_root = Path(parent_provenance["source_identity"]["event_source_root"])
    source = EventProducerSource(owner() / f"{role}_MANIFEST.json", source_root)
    folder = cache_root(role)
    folder.mkdir(parents=True, exist_ok=True)
    binding = {
        "schema": "v18_exact_native_p2_sensor_v1", "role": role,
        "manifest_sha256": sha(owner() / f"{role}_MANIFEST.json"),
        "source_identity": dict(source.identity), "admission": digest(admission),
        "source_code_sha256": sha(Path(__file__)),
        "events": "last two common-ROI FP32 windows, no quantization",
        "teacher": "P-only historical last pair; separate target arrays",
        "pack_size": 256,
    }
    freeze(folder / "BINDING.json", binding)
    done = 0
    started = time.perf_counter()
    try:
        for start in range(0, len(rows), 256):
            stop = min(start + 256, len(rows))
            pack = folder / f"pack_{start // 256:05d}.zip"
            receipt_path = pack.with_suffix(".json")
            if receipt_path.exists():
                receipt = read(receipt_path)
                if receipt["sha256"] != sha(pack) or receipt["binding"] != digest(binding):
                    raise ValueError("Sensor pack changed")
                continue
            # Worst-case space for one pack plus remaining uncompressed P2 rows.
            # Reserve pack-at-a-time; a disk guard pause never changes the cohort.
            require_resources(disk_reserve=(stop - start) * 2_000_000 + 1_000_000_000)
            pending = pack.with_suffix(".pending")
            original_indices = source._event_indices[start:stop]
            positions = {index: start + offset for offset, index in enumerate(original_indices)}
            source_receipts = {}
            with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=1, allowZip64=True) as output:
                for number in sorted({i // 32 for i in original_indices}):
                    path = source._inputs.directory / f"shard_{number:05d}.npz"
                    receipt = read(path.with_suffix(".json"))
                    expected = receipt.get("sha256", receipt.get("output_sha256"))
                    actual = sha(path)
                    if expected != actual:
                        raise ValueError(f"Source shard hash mismatch: {path}")
                    source_receipts[str(path)] = actual
                    with np.load(path, allow_pickle=False) as archive:
                        values = {k: archive[k] for k in archive.files}
                    teacher = {}
                    if role == "P":
                        teacher_path = source_root / "teacher" / path.name
                        teacher_receipt = read(teacher_path.with_suffix(".json"))
                        teacher_sha = sha(teacher_path)
                        if teacher_sha != teacher_receipt.get("sha256", teacher_receipt.get("output_sha256")):
                            raise ValueError("Teacher shard hash mismatch")
                        source_receipts[str(teacher_path)] = teacher_sha
                        with np.load(teacher_path, allow_pickle=False) as archive:
                            teacher = {k: archive[k] for k in archive.files}
                        if not np.array_equal(values["tokens"], teacher["tokens"]):
                            raise ValueError("Teacher token alignment changed")
                    for offset, ordinal in enumerate(values["ordinals"]):
                        index = positions.get(int(ordinal))
                        if index is None:
                            continue
                        token = str(values["tokens"][offset])
                        if token != str(rows.iloc[index].sample_token):
                            raise ValueError("Native token alignment changed")
                        events = values["events"][offset, -2:]
                        if events.dtype != np.float32 or events.shape != (2, 12, 128, 128):
                            raise ValueError("Native P2 tensor contract changed")
                        if not np.isfinite(events).all():
                            raise ValueError("Nonfinite sensor input")
                        target = values["target_ttc"][offset]
                        if np.float32(rows.iloc[index].target_ttc) != target:
                            raise ValueError("Query-anchor target mismatch")
                        record = {k: values[k][offset] for k in (
                            "delta", "target_ttc", "visible_heights", "square", "motion"
                        )}
                        record.update(events=events, boxes=values["boxes"][offset, -2:],
                                      token=np.asarray(token), source_ordinal=np.asarray(ordinal))
                        if role == "P":
                            record.update({k: teacher[k][offset] for k in (
                                "relation_targets", "relation_valid"
                            )})
                        stream = io.BytesIO()
                        np.savez(stream, **record)
                        raw = stream.getvalue()
                        # Immediate exact replay verifies serialization, not a model result.
                        with np.load(io.BytesIO(raw), allow_pickle=False) as check:
                            if not all(np.array_equal(check[k], v) for k, v in record.items()):
                                raise ValueError("Lossless repack failed")
                        output.writestr(f"{index:06d}.npz", raw)
                    del values, teacher
            with zipfile.ZipFile(pending) as check:
                if len(check.namelist()) != stop - start or check.testzip() is not None:
                    raise ValueError("Incomplete sensor pack")
            os.replace(pending, pack)
            write(receipt_path, {"start": start, "stop": stop, "binding": digest(binding),
                                 "sha256": sha(pack), "bytes": pack.stat().st_size,
                                 "source_shards": source_receipts, "exact_serialization": True})
            done += 1
            write(RUN / f"PREPARE_{role}_PROGRESS.json",
                  {"utc": utc(), "completed_rows": stop, "total_rows": len(rows),
                   "elapsed_s_this_call": time.perf_counter() - started})
            print(f"PREPARE {role} {stop}/{len(rows)}", flush=True)
            if max_packs is not None and done >= max_packs:
                return {"status": "PARTIAL", "completed_rows": stop}
        receipts = [read(folder / f"pack_{i:05d}.json") for i in range((len(rows) + 255) // 256)]
        result = {"status": "COMPLETE", "rows": len(rows), "binding": digest(binding),
                  "packs": len(receipts), "bytes": sum(r["bytes"] for r in receipts),
                  "pack_receipts_hash": digest(receipts)}
        freeze(folder / "COMPLETE.json", result)
        return result
    finally:
        source.close()


class SensorDataset:
    """Load exact random rows; labels never enter the returned inference mapping."""

    def __init__(self, role: str, *, require_complete: bool = True) -> None:
        self.role = role
        self.folder = cache_root(role)
        self.rows = role_rows(role)
        self.population = len(self.rows)
        if require_complete and read(self.folder / "COMPLETE.json")["rows"] != self.population:
            raise ValueError("Incomplete sensor population")
        self.packs: OrderedDict[int, zipfile.ZipFile] = OrderedDict()
        self.verified: set[int] = set()

    def _row(self, index: int) -> dict[str, np.ndarray]:
        number = index // 256
        path = self.folder / f"pack_{number:05d}.zip"
        if number not in self.verified:
            if sha(path) != read(path.with_suffix(".json"))["sha256"]:
                raise ValueError("Sensor pack changed before consumption")
            self.verified.add(number)
        if number not in self.packs:
            if len(self.packs) >= 16:
                self.packs.popitem(last=False)[1].close()
            self.packs[number] = zipfile.ZipFile(path)
        self.packs.move_to_end(number)
        with np.load(io.BytesIO(self.packs[number].read(f"{index:06d}.npz")), allow_pickle=False) as a:
            result = {k: a[k] for k in a.files}
        if str(result["token"]) != str(self.rows.iloc[index].sample_token):
            raise ValueError("Sensor row token mismatch")
        return result

    def batch(self, indices: list[int]) -> ObjectEventV4Batch:
        """Collate P2 and separate supervision, preserving duplicate random draws."""
        records = [self._row(i) for i in indices]

        def tensor(key: str) -> torch.Tensor:
            return torch.from_numpy(np.stack([r[key] for r in records]))

        meta = self.rows.iloc[indices]
        return ObjectEventV4Batch(
            events=tensor("events"), delta_t_s=tensor("delta"),
            observable_motion=tensor("motion"), visible_heights_px=tensor("visible_heights"),
            target_ttc_s=tensor("target_ttc"), boxes_xyxy=tensor("boxes"),
            common_square_xyxy=tensor("square"), sequence_ids=meta.sequence_id.astype(str).tolist(),
            sample_tokens=meta.sample_token.astype(str).tolist(), track_ids=meta.track_id.astype(str).tolist(),
            dinov3_relation_targets=tensor("relation_targets") if self.role == "P" else None,
            dinov3_relation_valid=tensor("relation_valid") if self.role == "P" else None,
        )

    def close(self) -> None:
        """Release only own read-only archive handles."""
        for pack in self.packs.values():
            pack.close()
        self.packs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=["P", "H", "V"])
    parser.add_argument("--max-packs", type=int)
    args = parser.parse_args()
    admit()
    print(prepare(args.role, max_packs=args.max_packs))
