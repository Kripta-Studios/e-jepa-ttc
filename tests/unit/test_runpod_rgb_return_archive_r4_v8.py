from __future__ import annotations

import json
from pathlib import Path

import pytest

from operational.runpod_migration import rgb_return_archive as frozen
from operational.runpod_migration import rgb_return_archive_r4_v8 as overlay


@pytest.fixture(autouse=True)
def isolated_authority(tmp_path: Path) -> None:
    """Build the authority chain using small independent fixture files."""
    global CONFIG, LAYOUT, COMPOSED, LAYOUT_AUTHORITY
    CONFIG = tmp_path / "config.json"
    config = {"schema": "ejepa_rgb_media_scope_config_v1", "scopes": []}
    CONFIG.write_text(json.dumps(config))
    LAYOUT = tmp_path / "layout.json"
    layout = {
        "schema": "ejepa_rgb_compositional_media_layout_v8",
        "status": "PASS",
        "original_config_path": str(CONFIG),
        "original_config_sha256": frozen.sha256(CONFIG),
        "scopes": [],
    }
    LAYOUT.write_text(json.dumps(layout))
    LAYOUT_AUTHORITY = frozen._canonical_sha256({k: v for k, v in layout.items() if k != "scopes"})
    COMPOSED = frozen._canonical_sha256(
        {
            "original_authority_sha256": frozen._canonical_sha256(
                {k: v for k, v in config.items() if k != "scopes"}
            ),
            "layout_authority_sha256": LAYOUT_AUTHORITY,
        }
    )


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "r4_rgb_downstream_v1"
    run.mkdir()
    (run / "RGB_OPERATIONAL_RECOVERY_V8.json").write_text(
        json.dumps(
            {
                "schema": "ejepa_rgb_compositional_context_v8",
                "status": "PASS",
                "layout_sha256": frozen.sha256(LAYOUT),
                "layout_authority_sha256": LAYOUT_AUTHORITY,
                "original_config_sha256": frozen.sha256(CONFIG),
            }
        ),
        encoding="utf-8",
    )
    (run / "RGB_OPERATIONAL_RECOVERY_V6.json").write_text(
        json.dumps({"media_authority_sha256": COMPOSED}), encoding="utf-8"
    )
    return run


def test_composed_authority_resolves_exact_config(tmp_path: Path) -> None:
    run = _run(tmp_path)
    path, config = overlay._composed_v8_config_source(run, [CONFIG, LAYOUT], COMPOSED)
    assert path == CONFIG.resolve()
    assert config["schema"] == "ejepa_rgb_media_scope_config_v1"
    with pytest.raises(ValueError, match="genealogy changed"):
        overlay._composed_v8_config_source(run, [CONFIG, LAYOUT], "0" * 64)


def test_composed_authority_rejects_missing_layout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="layout SHA"):
        overlay._composed_v8_config_source(_run(tmp_path), [CONFIG], COMPOSED)
