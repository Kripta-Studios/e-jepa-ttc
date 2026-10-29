"""Inspect original-role ZIP directories without downloading media or labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from fetch_simplex_t_original_annotations import RangeReader

from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.simplex_t.coordination import verified_ack


def main() -> None:
    """Record every member name, preserving the downloaded archive identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    config = json.loads(Path("configs/experiment/simplex_t_coordination.json").read_text())
    ack = verified_ack(
        Path(paths["shared_coordination"]) / config["ack_filename"], config["ack_sha256"]
    )
    if args.output.exists():
        raise FileExistsError("preserve previous inventory")
    inventories = []
    for sequence in sorted(ack["interfaces"]["role_manifest"]["roles"]["original"]):
        receipt_path = args.receipts / sequence / "DOWNLOAD_RECEIPT.json"
        receipt_bytes = receipt_path.read_bytes()
        receipt = json.loads(receipt_bytes)
        remote = RangeReader(receipt["source_url"])
        if (remote.etag, remote.size) != (receipt["archive_etag"], receipt["archive_bytes"]):
            raise ValueError("archive changed since annotation download")
        with zipfile.ZipFile(remote) as archive:
            members = [
                {"name": item.filename, "bytes": item.file_size, "crc32": item.CRC}
                for item in archive.infolist()
            ]
        record = {
            "sequence": sequence,
            "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "archive_etag": remote.etag,
            "metadata_range_bytes": remote.transferred,
            "members": members,
        }
        inventories.append(record)
        print(
            json.dumps(
                {
                    "sequence": sequence,
                    "member_count": len(members),
                    "metadata_range_bytes": remote.transferred,
                }
            ),
            flush=True,
        )
    write_new_json(
        args.output,
        {"status": "completed", "member_payloads_read": False, "inventories": inventories},
    )


if __name__ == "__main__":
    main()
