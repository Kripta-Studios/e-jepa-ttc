"""Resume the exact archived Q1 runner without changing its sealed source identity."""

from __future__ import annotations

import argparse
import runpy

from .common import OUT, read, sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("visual", "full"), required=True)
    args = parser.parse_args()
    source = OUT / "executed_sources/operational/multimodal_validation/garl.py"
    expected = read(OUT / f"garl_{args.kind}/BINDING.json")["adapter_sha256"]
    assert sha(source) == expected
    runpy.run_path(
        str(source),
        run_name="__main__",
        init_globals={"__package__": "operational.multimodal_validation"},
    )


if __name__ == "__main__":
    main()
