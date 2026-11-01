"""Read only ancestry receipts to distinguish OOF context caches from deployment histories."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_campaign.data import read


def main() -> None:
    """Verify the known archive and preserve exact sidecars without reading target arrays."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--input-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze = read(args.input_freeze)
    name = "final_generated_fd84565b0df06d79.tar"
    archive = args.archive_root / name
    digest = file_sha(archive)
    if digest != freeze["archives"][name]:
        raise ValueError("Archive differs from the previously verified PAIR input source")
    records = []
    suffixes = (
        "/cache/E_CTX_MATCHED.json",
        "/cache/R_CTX.json",
        "/oof/PHASE17_RAW_EVENT.json",
        "/oof/PHASE17_RAW_RGB.json",
    )
    with tarfile.open(archive) as handle:
        members = handle.getmembers()
        for suffix in suffixes:
            selected = [m for m in members if m.isfile() and m.name.endswith(suffix)]
            if len(selected) != 1:
                raise ValueError("Missing or ambiguous history receipt")
            stream = handle.extractfile(selected[0])
            if stream is None:
                raise ValueError("History receipt is not a regular file")
            payload = stream.read()
            receipt = json.loads(payload)
            records.append(
                {
                    "member": selected[0].name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "receipt": receipt,
                }
            )
    atomic_receipt(
        args.output,
        {
            "status": "INVENTORIED",
            "archive": str(archive),
            "archive_sha256": digest,
            "input_freeze_sha256": file_sha(args.input_freeze),
            "source_sha256": file_sha(Path(__file__)),
            "targets_read": False,
            "npz_arrays_read": [],
            "producer_calls": 0,
            "records": records,
            "scope": "OOF context ancestry inventory, not deployment-history preservation",
        },
    )
    print({r["member"]: r["receipt"].get("parents") for r in records})


if __name__ == "__main__":
    main()
