"""Bind the additive overlap implementation to exact input and gradient evidence."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

from operational.rgb_port.accounting import read_json_shared, sha256_file
from operational.rgb_port.recipe import canonical_sha256
from operational.rgb_port_rgb_loader.contracts import NAME as PARENT_NAME
from operational.rgb_port_rgb_loader.contracts import validate as validate_parent

ROOT = Path(__file__).resolve().parents[2]
NAME = "RGB_OVERLAP_FREEZE.json"
DEPTH = 2
DECODER_WORKERS = 4
DECODER_PENDING_BYTES = 128 * 1024**2
COMPUTE_ENABLED = False


def payload(run: Path, qa: Path) -> dict[str, Any]:
    """Require CPU batch parity and GPU loss/gradient parity before activation."""
    validate_parent(run)
    quality = read_json_shared(qa)
    for key in ("exact_batch_parity", "exact_gpu_gradient_parity", "cursor_rng_unchanged"):
        if quality.get(key) is not True:
            raise ValueError(f"RGB overlap admission requires {key}")
    if quality.get("status") != "PASSED":
        raise ValueError("RGB overlap QA did not pass")
    if quality.get("decoder_workers") != DECODER_WORKERS:
        raise ValueError("RGB overlap QA used a different decoder worker count")
    if quality.get("geometry_cache_enabled") is not COMPUTE_ENABLED:
        raise ValueError("RGB overlap QA used a different compute admission")
    value = {
        "schema": "rgb_port_rgb_overlap_freeze_v1",
        "qa": {"path": str(qa.resolve(strict=True)), "sha256": sha256_file(qa)},
        "parent_sha256": sha256_file(run / PARENT_NAME),
        "fit_ids": ["R_A5", "R_C2F"],
        "prefetch_depth": DEPTH,
        "decoder_workers": DECODER_WORKERS,
        "decoder_pending_bytes": DECODER_PENDING_BYTES,
        "geometry_cache_enabled": COMPUTE_ENABLED,
        "geometry_precision": "bf16_unchanged",
        "objective_changed": False,
        "files": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in [
                *sorted(Path(__file__).parent.glob("*.py")),
                *sorted((ROOT / "tests").glob("test_rgb_port_rgb_overlap*.py")),
            ]
        },
    }
    return {**value, "identity_sha256": canonical_sha256(value)}


def validate(run: Path, *, committed: bool = True) -> dict[str, Any]:
    """Retain all historical guards and reject uncommitted implementation bytes."""
    value = read_json_shared(run / NAME)
    if value != payload(run, Path(value["qa"]["path"])):
        raise ValueError("RGB overlap implementation or parent admission changed")
    if committed:
        for relative, expected in value["files"].items():
            data = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=ROOT)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"RGB overlap differs from committed bytes: {relative}")
    return value
