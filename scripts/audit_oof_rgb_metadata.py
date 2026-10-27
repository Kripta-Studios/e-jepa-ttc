"""Read-only audit of shared observation identities in every public TRAIN40 query."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from operational.final_refit.recipe import sha256
from operational.final_refit.rgb import load_all_train_source
from operational.final_refit.scientific_authority import load_config
from operational.oof_rgb_metadata_fix import metadata_equal, observation_metadata_equal
from operational.rgb_port.accounting import atomic_write_json


def main() -> None:
    """Reject any actual sensor/group conflict before admitting the repair."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.run)
    source = load_all_train_source(Path(config["rgb_source"]), Path(config["teacher"]))
    seen: dict[str, dict[str, Any]] = {}
    differences: dict[str, int] = {}
    duplicates = 0
    try:
        for index in range(source.population_size):
            row = source.row(index)
            for item in source.history(index):
                value = {
                    **item,
                    "historical_role": str(row["historical_role"]),
                    "group_id": str(row["group_id"]),
                }
                old = seen.setdefault(str(item["observation_id"]), value)
                if old is value:
                    continue
                duplicates += 1
                if not observation_metadata_equal(old, value):
                    raise ValueError(f"Conflicting sensor metadata: {item['observation_id']}")
                for key in old.keys() | value.keys():
                    if not metadata_equal(old[key], value[key]):
                        differences[key] = differences.get(key, 0) + 1
        load_config(args.run)
        result = {
            "status": "PASS",
            "queries": source.population_size,
            "unique_observations": len(seen),
            "duplicates": duplicates,
            "differing_fields": differences,
            "scientific_optimizer_updates": 0,
            "technical_optimizer_updates": 0,
            "scientific_config_sha256": sha256(args.run / "SCIENTIFIC_CONFIG.json"),
            "audit_source_sha256": sha256(Path(__file__)),
        }
        atomic_write_json(args.run / "RGB_METADATA_REAL_AUDIT.json", result)
        print(result)
    finally:
        source.close()


if __name__ == "__main__":
    main()
