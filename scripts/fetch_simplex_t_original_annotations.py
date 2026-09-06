"""Download only original-role annotation ZIP members; never execute pickle data."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

import requests

from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.simplex_t.coordination import verified_ack


class RangeReader(io.RawIOBase):
    """Bound remote reads and pin ZIP identity across HTTP range requests."""

    def __init__(self, url: str) -> None:
        self.url, self.position, self.transferred = url, 0, 0
        with requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=25) as response:
            response.raise_for_status()
            if response.status_code != 206:
                raise ValueError("server must support selective range download")
            self.size = int(response.headers["Content-Range"].split("/")[-1])
            self.etag = response.headers["ETag"]

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if not 0 <= self.position <= self.size:
            raise ValueError("range outside archive")
        return self.position

    def read(self, size: int = -1) -> bytes:
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size == 0:
            return b""
        if self.transferred + size > 64 * 1024**2:
            raise ValueError("selective download exceeds 64 MiB network budget")
        start, end = self.position, self.position + size - 1
        with requests.get(
            self.url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=25
        ) as response:
            response.raise_for_status()
            if (
                response.status_code != 206
                or response.headers.get("ETag") != self.etag
                or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}"
            ):
                raise ValueError("changed archive or ignored range")
            data = response.raw.read(size + 1)
            if len(data) != size:
                raise ValueError("truncated or excessive range")
        self.position += size
        self.transferred += size
        return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    config = json.loads(Path("configs/experiment/simplex_t_coordination.json").read_text())
    ack = verified_ack(
        Path(paths["shared_coordination"]) / config["ack_filename"], config["ack_sha256"]
    )
    if args.sequence not in ack["interfaces"]["role_manifest"]["roles"]["original"]:
        raise ValueError("this download audit permits original groups only")
    if args.output.exists():
        raise FileExistsError("preserve existing download evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output.parent).free - 512 * 1024**2 < 40_000_000_000:
        raise ValueError("40 GB free after bounded download reservation required")
    catalog_url = "https://nail-hnu.github.io/eAP_dataset/assets/data/release_catalog.json"
    response = requests.get(catalog_url, timeout=25)
    response.raise_for_status()
    assets = [item for item in response.json()["assets"] if item["asset_id"] == args.sequence]
    if len(assets) != 1:
        raise ValueError("missing or ambiguous official catalog asset")
    url = assets[0]["dropbox_url"].replace("dl=0", "dl=1")
    remote = RangeReader(url)
    with zipfile.ZipFile(remote) as archive:
        selected = [
            entry
            for entry in archive.infolist()
            if PurePosixPath(entry.filename).name in {"annotations.pkl", "frames.pkl"}
        ]
        if len(selected) != 2 or {PurePosixPath(e.filename).name for e in selected} != {
            "annotations.pkl",
            "frames.pkl",
        }:
            raise ValueError("ZIP does not contain an unambiguous annotation/frame pair")
        if sum(e.file_size for e in selected) > 256 * 1024**2:
            raise ValueError("annotation extraction exceeds 256 MiB bound")
        args.output.mkdir()
        receipts = []
        for entry in selected:
            data = archive.read(entry)  # CRC checked; pickle is NOT deserialized.
            target = args.output / PurePosixPath(entry.filename).name
            with target.open("xb") as stream:
                stream.write(data)
            receipts.append(
                {
                    "zip_member": entry.filename,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    write_new_json(
        args.output / "DOWNLOAD_RECEIPT.json",
        {
            "sequence": args.sequence,
            "catalog_url": catalog_url,
            "catalog_sha256": hashlib.sha256(response.content).hexdigest(),
            "source_url": url,
            "archive_etag": remote.etag,
            "archive_bytes": remote.size,
            "downloaded_range_bytes": remote.transferred,
            "members": receipts,
            "pickle_deserialized": False,
            "history_validated": False,
            "authorization": "Explicit user request to find and download original annotations",
        },
    )
    print(
        json.dumps(
            {"sequence": args.sequence, "range_bytes": remote.transferred, "members": receipts}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
