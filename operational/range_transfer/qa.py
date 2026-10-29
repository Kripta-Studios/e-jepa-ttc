"""Accounted 10 versus 5+restore+5 synthetic continuity test of the real runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from e_jepa_ttc.simplex_t.phase import ttc_to_phase

from .cache import HeadCache
from .common import read, sha, write
from .train_heads import accounting, fit


def same(a: object, b: object) -> bool:
    """Exact nested state comparison, including optimizer and RNG arrays."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return (
            isinstance(b, (list, tuple))
            and type(a) is type(b)
            and len(a) == len(b)
            and all(same(x, y) for x, y in zip(a, b, strict=True))
        )
    return bool(a == b)


def synthetic(path: Path) -> None:
    """Fixed synthetic input, never mixed with H or scientific results."""
    rng = np.random.default_rng(7)
    targets = np.tile([0.5, 1.5, 2.5, 4.5, 7.0, 9.0, -2.0, -7.0], 8).astype(np.float32)
    phase = ttc_to_phase(torch.from_numpy(targets)).numpy()
    np.savez(
        path,
        features=rng.normal(size=(64, 4, 17)).astype(np.float32),
        timing=np.zeros((64, 4, 4), np.float32),
        valid=np.ones((64, 4), bool),
        expert_phase=np.repeat(phase[:, None], 3, axis=1) * np.float32(0.8),
        target_phase=phase,
        mass=np.full(64, 1 / 64, np.float32),
        sample_token=np.array([f"synthetic_{i}" for i in range(64)]),
        group_id=np.array([f"synthetic_group_{i // 16}" for i in range(64)]),
        role=np.full(64, "H"),
    )


def run_qa(run: Path) -> dict[str, Any]:
    """Execute exactly twenty new synthetic updates once; continuation is idempotent."""
    result_path = run / "qa/RESUME_TEST.json"
    if result_path.exists():
        result = read(result_path)
        for path, expected in result["checkpoints"].items():
            if sha(Path(path)) != expected:
                raise ValueError("Synthetic QA checkpoint changed")
        return result
    root = run / "qa"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "SYNTHETIC.npz"
    if not path.exists():
        synthetic(path)
    cache = HeadCache(path)
    binding = {"status": "SYNTHETIC_ONLY", "cache_sha256": sha(path), "files": {}}
    continuous = fit(
        cache, run, binding, mass_id=0, loss_id=0, seed=7, technical_name="continuous10"
    )
    if continuous["status"] != "COMPLETE":
        return continuous
    partial = fit(
        cache,
        run,
        binding,
        mass_id=0,
        loss_id=0,
        seed=7,
        max_updates_this_call=5,
        technical_name="split5_restore5",
    )
    if partial["status"] == "PAUSED_RESOURCE":
        return partial
    restored = fit(
        cache,
        run,
        binding,
        mass_id=0,
        loss_id=0,
        seed=7,
        max_updates_this_call=5,
        technical_name="split5_restore5",
    )
    if restored["status"] != "COMPLETE":
        return restored
    a = torch.load(continuous["checkpoint_path"], map_location="cpu", weights_only=False)
    b = torch.load(restored["checkpoint_path"], map_location="cpu", weights_only=False)
    fields = (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "sampler_generator_state",
        "torch_rng_state",
        "numpy_random_state",
        "python_random_state",
        "completed_updates",
        "accumulation_index",
    )
    checks = {field: same(a[field], b[field]) for field in fields}
    checks["draws"] = a["cursor"]["draw_hashes"] == b["cursor"]["draw_hashes"]
    checks["losses"] = same(a["cursor"]["losses"], b["cursor"]["losses"])
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "scope": "CPU_FP32_GRU160_batch128_synthetic_only_no_scientific_result",
        "optimizer_updates": 20,
        "checkpoints": {
            r["checkpoint_path"]: r["checkpoint_sha256"] for r in (continuous, restored)
        },
    }
    write(result_path, result)
    accounting(run)
    if not all(checks.values()):
        raise AssertionError("Synthetic continuity failed")
    return result
