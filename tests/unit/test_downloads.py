from pathlib import Path

import yaml

from e_jepa_ttc.data.downloads import build_download_plan


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
