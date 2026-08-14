#!/usr/bin/env python
"""Materialize a train-only Scientific Recovery V8 temporal cache.

The command fails closed for sealed evaluation paths.  A caller may supply
explicit frozen row identities, or omit them and derive the exact paired A5 /
Garl train-only OOF universe declared by the signed protocol.  It never reads
public validation, private test, EvTTC test, or CodaBench data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.data.scientific_recovery_v8_cache import (  # noqa: E402
    ScientificRecoveryV8CacheConfig,
    materialize_scientific_recovery_v8_cache,
)


def _default_root(*, argument: Path | None, env_name: str, pinned: str, label: str) -> Path:
    """Resolve explicit, environment, then existing local provenance roots."""

    if argument is not None:
        return argument.resolve()
    environment = os.environ.get(env_name)
    if environment:
        return Path(environment).resolve()
    local = Path(pinned)
    if local.is_dir():
        return local.resolve()
    raise ValueError(
        f"{label} root is required: pass the CLI flag or set {env_name}. "
        f"The pinned local default {pinned!r} does not exist."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eap-root", type=Path, help="Public eAP train root.")
    parser.add_argument("--garlttc-root", type=Path, help="Official GarlTTC train root.")
    parser.add_argument(
        "--selection-metadata", type=Path, help="Optional explicit frozen V8 row identities."
    )
    parser.add_argument(
        "--protocol", type=Path, required=True, help="Frozen V8 temporal protocol JSON."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--representation", choices=("timevol20", "exp6"), required=True)
    parser.add_argument("--steps", type=int, choices=(2, 3), default=3)
    parser.add_argument("--roi-size", type=int, default=128)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = ScientificRecoveryV8CacheConfig(
        representation=args.representation,
        steps=args.steps,
        roi_size=args.roi_size,
        shard_size=args.shard_size,
        storage_dtype=args.storage_dtype,
    )
    try:
        manifest = materialize_scientific_recovery_v8_cache(
            eap_root=_default_root(
                argument=args.eap_root, env_name="EAP_ROOT", pinned="E:/eAP_dataset", label="eAP"
            ),
            garlttc_root=_default_root(
                argument=args.garlttc_root,
                env_name="GARLTTC_ROOT",
                pinned="E:/GarlTTC_dataset",
                label="GarlTTC",
            ),
            selection_metadata_path=(
                args.selection_metadata.resolve() if args.selection_metadata is not None else None
            ),
            protocol_path=args.protocol.resolve(),
            output_dir=args.output_dir.resolve(),
            config=config,
            resume=args.resume,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"V8 cache build failed closed: {type(error).__name__}: {error}\n")
    print(json.dumps({"status": "completed", "manifest": manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
