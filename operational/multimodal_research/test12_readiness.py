"""Inventory label-free test12 inputs using the unchanged native RGB timeline builder."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from operational.multimodal_validation.common import V12, read, sha, write
from operational.rgb_port import prepare

from .common import OUT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-root", type=Path, required=True)
    args = parser.parse_args()
    source = V12 / "artifacts/sota_evidence_20261010/sources/test_inputs.parquet"
    previous = V12 / "artifacts/garlttc_submission_20261010/INPUT_FREEZE.json"
    old = read(previous)
    assert sha(source) == old["input_sha256"]
    # Explicit projection excludes targets even if the source schema changes.
    metadata = pd.read_parquet(source, columns=list(prepare.LABEL_FREE_COLUMNS))
    assert len(metadata) == 6762 and not metadata.sample_token.duplicated().any()
    assert set(metadata.sample_token) == {r["sample_token"] for r in old["rows"]}
    roles = dict.fromkeys(metadata.sequence_id.unique(), "TEST12_INPUT_AUDIT")
    assert len(roles) == 12
    indexed = prepare.build_label_blind_index(metadata, roles)
    assert set(indexed.sample_token) == set(metadata.sample_token)
    # The generic function's descriptive TRAIN40 strings must not imply train membership.
    indexed["calibration_id"] = "eap_public_test_rgb_bbox_space_v1"
    indexed["foreground_source"] = "public_test_bbox_geometry"
    shards: set[str] = set()
    for observations in indexed.observation_shards:
        for triplet in observations:
            shards.update(str(value) for value in triplet)
    rows = []
    root = args.media_root.resolve(strict=True)
    for relative in sorted(shards):
        path = (root / relative).resolve()
        assert path.is_relative_to(root)
        assert relative.replace("\\", "/").startswith("data/test/")
        rows.append(
            dict(
                path=relative,
                present=path.is_file(),
                bytes=path.stat().st_size if path.is_file() else None,
            )
        )
    destination = OUT / "test12_readiness"
    destination.mkdir(exist_ok=True)
    frame_path = destination / "LABEL_FREE_RGB_INDEX.parquet"
    prepare.atomic_parquet(frame_path, indexed)
    (destination / "REQUIRED_RGB_SHARDS.txt").write_text(
        "\n".join(r["path"] for r in rows) + "\n", encoding="utf-8"
    )
    write(
        destination / "READINESS.json",
        dict(
            utc=datetime.now(UTC).isoformat(),
            status="INPUT_INVENTORY_ONLY",
            source_sha256=sha(source),
            source_columns=list(prepare.LABEL_FREE_COLUMNS),
            previous_input_freeze_sha256=sha(previous),
            queries=len(indexed),
            sequences=len(roles),
            rgb_temporal_metadata_available=int(indexed.rgb_available.sum()),
            observation_counts={
                str(k): int(v)
                for k, v in indexed.distinct_observations.value_counts().sort_index().items()
            },
            shards=rows,
            missing_shards=sum(not r["present"] for r in rows),
            index_sha256=sha(frame_path),
            native_builder_sha256=sha(Path(prepare.__file__)),
            audit_source_sha256=sha(__file__),
            targets_read=False,
            predictions_generated=0,
            optimizer_updates=0,
            candidate_selected=False,
            external_submission=False,
            limitations=[
                "Temporal metadata does not establish that RGB bytes exist or decode correctly.",
                "Native event history must use real same-track observations as in TRAIN40; "
                "regular Dev32/FCWD histories or H8 history cannot be assumed equivalent.",
                "No candidate freeze or official score is implied by this input inventory.",
            ],
        ),
    )
    print(
        {
            "queries": len(indexed),
            "shards": len(rows),
            "missing_shards": sum(not r["present"] for r in rows),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
