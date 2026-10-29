"""Explicit optional storage adapter; scientific trainer and recipes remain intact."""

from __future__ import annotations

from operational.range_transfer.common import digest, read, sha

from .data import SensorDataset, cache_root
from .mapped_data import MappedRows, mapped_root


def admit_mapped() -> MappedRows:
    """Verify the full new cache once before entering GPU scopes in this process."""
    folder = mapped_root()
    complete = read(folder / "COMPLETE.json")
    binding = read(folder / "BINDING.json")
    source = read(cache_root("P") / "COMPLETE.json")
    if (
        complete["status"] != "COMPLETE"
        or complete["rows"] != source["rows"]
        or complete["binding"] != digest(binding)
        or binding["source_complete_sha256"] != sha(cache_root("P") / "COMPLETE.json")
        or binding["source_pack_receipts_hash"] != source["pack_receipts_hash"]
        or binding["role"] != "P"
        or not complete["all_fields_exact"]
    ):
        raise ValueError("Uncompressed cache does not preserve the admitted P population")
    receipts = []
    for number in range(complete["packs"]):
        receipt = read(folder / f"pack_{number:05d}/COMPLETE.json")
        pack_binding = receipt["binding"]
        if (
            pack_binding["campaign"] != digest(binding)
            or pack_binding["source_receipt_sha256"]
            != sha(cache_root("P") / f"pack_{number:05d}.json")
            or pack_binding["start"] != number * 256
            or pack_binding["stop"] != min((number + 1) * 256, complete["rows"])
            or not receipt["all_fields_exact"]
        ):
            raise ValueError("Uncompressed pack ancestry or row positions changed")
        receipts.append(receipt)
    if (
        len(receipts) != (complete["rows"] + 255) // 256
        or digest(receipts) != complete["pack_receipts_hash"]
    ):
        raise ValueError("Uncompressed pack receipt manifest changed")
    reader = MappedRows(folder, complete["rows"])
    reader.admit_all()
    return reader


class MappedSensorDataset(SensorDataset):
    """Reuse exact native collation and metadata; replace only row byte transport."""

    def __init__(self, role: str, *, admission: MappedRows) -> None:
        if role != "P":
            raise ValueError("Only the P sensor cache has this storage adapter")
        super().__init__(role)
        if (
            admission.population != self.population
            or len(admission.verified) != (self.population + 255) // 256
        ):
            raise ValueError("Admit all uncompressed sensor packs before constructing a dataset")
        self.reader = admission

    def _row(self, index: int) -> dict:
        result = self.reader.row(index)
        if str(result["token"]) != str(self.rows.iloc[index].sample_token):
            raise ValueError("Mapped sensor row token mismatch")
        return result

    def close(self) -> None:
        """Close this scope's maps while preserving process-level byte admission."""
        self.reader.close()
        super().close()
