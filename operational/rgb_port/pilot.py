"""Run only the bounded RGB cache pilot from an immutable role manifest."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from operational.rgb_port.prepare import run_pilot


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--role-manifest", type=Path, required=True)
    result.add_argument("--cache-root", type=Path, required=True)
    result.add_argument("--count", type=int, required=True)
    result.add_argument("--workers", type=int, default=1)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    manifest_path = args.role_manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        raise ValueError("pilot requires a COMPLETE immutable role manifest")
    receipt = run_pilot(
        Path(manifest["rows_path"]).resolve(strict=True),
        Path(manifest["eap_root"]).resolve(strict=True),
        args.cache_root,
        args.count,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "status",
                    "requested",
                    "completed",
                    "workers",
                    "elapsed_s",
                    "queries_per_second",
                    "compressed_bytes",
                    "estimated_full_compressed_bytes",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
