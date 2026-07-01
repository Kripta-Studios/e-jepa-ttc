import json
from pathlib import Path

import pytest
import yaml

from e_jepa_ttc.data.downloads import (
    build_download_plan,
    build_gdown_command,
    download_gdown_listing,
    google_uc_to_usercontent_url,
)


def test_build_download_plan_filters_sequence_and_kind(tmp_path: Path) -> None:
    manifest = {
        "sequences": [
            {
                "sequence_id": "seq-a",
                "local_dir": "datasets/seq-a",
                "assets": {
                    "hdf5": {
                        "kind": "file",
                        "output": "data.hdf5",
                        "url": "https://example.com/a",
                    },
                    "bbox_segmentation": {
                        "kind": "folder",
                        "output": "bbox",
                        "url": "https://example.com/folder-a",
                    },
                },
            },
            {
                "sequence_id": "seq-b",
                "local_dir": "datasets/seq-b",
                "assets": {
                    "hdf5": {
                        "kind": "file",
                        "output": "data.hdf5",
                        "url": "https://example.com/b",
                    },
                },
            },
        ]
    }
    manifest_path = tmp_path / "downloads.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    plan = build_download_plan(
        manifest_path=manifest_path,
        root=tmp_path,
        sequences=("seq-a",),
        kinds=("bbox_segmentation",),
    )

    assert plan == [
        {
            "sequence_id": "seq-a",
            "asset": "bbox_segmentation",
            "kind": "folder",
            "url": "https://example.com/folder-a",
            "output": (tmp_path / "datasets/seq-a/bbox").as_posix(),
            "size_gb": None,
        }
    ]


def test_build_gdown_command_supports_quiet_resume_and_folders() -> None:
    command = build_gdown_command(
        {
            "kind": "folder",
            "url": "https://example.com/folder-a",
            "output": "datasets/seq-a/bbox",
        },
        python="python",
        quiet=True,
        resume=True,
    )

    assert command == [
        "python",
        "-m",
        "gdown",
        "-q",
        "--continue",
        "--folder",
        "https://example.com/folder-a",
        "-O",
        "datasets/seq-a/bbox",
    ]


def test_google_uc_to_usercontent_url() -> None:
    assert google_uc_to_usercontent_url("https://drive.google.com/uc?id=file-123") == (
        "https://drive.usercontent.google.com/download?id=file-123&export=download"
    )


def test_download_gdown_listing_skips_existing_and_downloads_missing(tmp_path, monkeypatch) -> None:
    listing_path = tmp_path / "listing.json"
    listing_path.write_text(
        json.dumps(
            [
                {"url": "https://drive.google.com/uc?id=existing-id", "path": "0001.json"},
                {"url": "https://drive.google.com/uc?id=missing-id", "path": "nested/0002.json"},
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "0001.json").write_text("{}", encoding="utf-8")
    calls = []

    def fake_urlretrieve(url: str, output_path: Path) -> None:
        calls.append((url, output_path))
        output_path.write_text('{"ok": true}', encoding="utf-8")

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)

    records = download_gdown_listing(listing_path=listing_path, output_dir=output_dir)

    assert [record["status"] for record in records] == ["skipped", "downloaded"]
    assert calls == [
        (
            "https://drive.usercontent.google.com/download?id=missing-id&export=download",
            output_dir / "nested/0002.json",
        )
    ]


def test_download_gdown_listing_retries_transient_download_errors(tmp_path, monkeypatch) -> None:
    listing_path = tmp_path / "listing.json"
    listing_path.write_text(
        json.dumps([{"url": "https://drive.google.com/uc?id=file-id", "path": "0001.json"}]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    calls = []

    def fake_urlretrieve(url: str, output_path: Path) -> None:
        calls.append(url)
        if len(calls) == 1:
            output_path.write_text("", encoding="utf-8")
            raise RuntimeError("temporary disconnect")
        output_path.write_text('{"ok": true}', encoding="utf-8")

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)
    monkeypatch.setattr("time.sleep", lambda _: None)

    records = download_gdown_listing(
        listing_path=listing_path,
        output_dir=output_dir,
        retries=1,
    )

    assert records[0]["status"] == "downloaded"
    assert len(calls) == 2
    assert (output_dir / "0001.json").read_text(encoding="utf-8") == '{"ok": true}'


def test_download_gdown_listing_rejects_unsafe_paths(tmp_path: Path) -> None:
    listing_path = tmp_path / "listing.json"
    listing_path.write_text(
        json.dumps([{"url": "https://drive.google.com/uc?id=file-id", "path": "../escape.json"}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsafe listing path"):
        download_gdown_listing(listing_path=listing_path, output_dir=tmp_path / "out")
