import json
from pathlib import Path

from e_jepa_ttc.data.annotations import parse_isat_label


def test_parse_isat_label_uses_largest_segmentation(tmp_path: Path) -> None:
    path = tmp_path / "0001.json"
    path.write_text(
        json.dumps(
            {
                "objects": [
                    {"category": "small", "segmentation": [[0, 0], [1, 0], [1, 1], [0, 1]]},
                    {"category": "car", "segmentation": [[10, 20], [30, 20], [30, 50], [10, 50]]},
                ]
            }
        ),
        encoding="utf-8",
    )

    parsed = parse_isat_label(path)

    assert parsed is not None
    category, bbox = parsed
    assert category == "car"
    assert bbox == (10.0, 20.0, 30.0, 50.0)
