"""Compare official raw inputs with cached inputs through the unchanged Garl model.

This audit is intentionally small and read-only with respect to source and data.
It validates the quantized cache on real samples, then measures the resulting
drift at the official network output.  It is separate from the synthetic
checkpoint-to-replica audit and does not claim private-test or paper parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from e_jepa_ttc.data.garl_release_cache import GarlReleaseCacheDataset  # noqa: E402
from e_jepa_ttc.data.garlttc_eap import load_garlttc_train_index  # noqa: E402
from scripts.audit_garl_release_cache_parity import _release_input  # noqa: E402
from scripts.train_garl_release_cache import _cache_input, _load_config  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.from_numpy(np.asarray(value))


def _official_ttc(raw: torch.Tensor, delta_t_s: float) -> torch.Tensor:
    values = raw.reshape(len(raw), -1)
    return delta_t_s / (1.0 - values[:, 0] / values[:, 1])


def _raw_input(expected: Mapping[str, object], variant: str) -> torch.Tensor:
    event = _as_tensor(expected["event_roi"]).unsqueeze(0).to(dtype=torch.float32)
    rgb = _as_tensor(expected["rgb_pair"]).unsqueeze(0).to(dtype=torch.float32)
    batch: dict[str, torch.Tensor] = {"event_roi": event, "rgb_pair": rgb}
    return _cache_input(batch, variant)


def _model_output(model: torch.nn.Module, value: torch.Tensor) -> torch.Tensor:
    output = model(value)
    raw = output[0] if isinstance(output, tuple) else output
    if not isinstance(raw, torch.Tensor):
        raise TypeError("Official TTCNetwork returned a non-tensor output.")
    return raw.detach().to(dtype=torch.float32, device="cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eap-root", type=Path, required=True)
    parser.add_argument("--garlttc-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=("event_only", "visual_only", "rgbe_late_fusion"),
        required=True,
    )
    parser.add_argument("--samples-per-split", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples_per_split <= 0:
        raise ValueError("--samples-per-split must be positive.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    release_root = args.release_root.resolve()
    sys.path.insert(0, str(release_root))
    from garl_ttc.datasets.event_representation import (  # type: ignore[reportMissingImports]
        get_timevolume_roi_np,
    )
    from garl_ttc.datasets.ttc_dataset import (  # type: ignore[reportMissingImports]
        get_target_roi_from_feature_torch,
    )
    from garl_ttc.models import TTCNetwork  # type: ignore[reportMissingImports]
    from garl_ttc.utils.events import (  # type: ignore[reportMissingImports]
        extract_from_h5_by_timewindow,
    )

    config, config_path = _load_config(release_root, args.variant)
    torch.manual_seed(7)
    model = TTCNetwork(config, is_train=False).eval().to(device)
    delta_t_s = float(cast(Any, model).dT)

    split_payload = json.loads(args.split.read_text(encoding="utf-8"))
    assignments = split_payload["assignments"]
    sequences = sorted(
        str(sequence) for role in ("train", "validation") for sequence in assignments[role]
    )
    index = load_garlttc_train_index(args.garlttc_root, sequences)
    rows = cast(Any, index.merged).to_dict("records")
    row_by_token = {str(row["sample_token"]): row for row in rows}

    input_stats: dict[str, dict[str, float | int]] = {
        "event_roi": {"max_abs": 0.0, "mean_abs_sum": 0.0, "count": 0},
        "rgb_pair": {"max_abs": 0.0, "mean_abs_sum": 0.0, "count": 0},
    }
    output_stats = {"raw_height_max_abs": 0.0, "ttc_max_abs": 0.0}
    checked = 0
    failures: list[dict[str, object]] = []
    with torch.inference_mode():
        for split in ("train", "validation"):
            dataset = GarlReleaseCacheDataset(args.manifest, split=split)
            for dataset_index in range(min(len(dataset), args.samples_per_split)):
                cached = dataset[dataset_index]
                token = str(cached["sample_token"])
                row = row_by_token[token]
                expected = _release_input(
                    row,
                    eap_root=args.eap_root,
                    extract_from_h5_by_timewindow=extract_from_h5_by_timewindow,
                    get_timevolume_roi_np=get_timevolume_roi_np,
                    get_target_roi_from_feature_torch=get_target_roi_from_feature_torch,
                )
                for name in ("event_roi", "rgb_pair"):
                    reference = _as_tensor(expected[name]).to(dtype=torch.float32).numpy()
                    actual = _as_tensor(cached[name]).to(dtype=torch.float32).numpy()
                    difference = np.abs(actual - reference)
                    stats = input_stats[name]
                    stats["max_abs"] = max(float(stats["max_abs"]), float(difference.max()))
                    stats["mean_abs_sum"] += float(difference.mean())
                    stats["count"] += 1

                raw_input = _raw_input(expected, args.variant).to(device)
                cached_batch = {
                    key: _as_tensor(value).unsqueeze(0)
                    for key, value in cached.items()
                    if key in {"event_roi", "rgb_pair"}
                }
                cached_input = _cache_input(cached_batch, args.variant).to(device)
                raw_output = _model_output(model, raw_input)
                cached_output = _model_output(model, cached_input)
                raw_difference = torch.abs(raw_output - cached_output)
                raw_ttc = _official_ttc(raw_output, delta_t_s)
                cached_ttc = _official_ttc(cached_output, delta_t_s)
                ttc_difference = torch.abs(raw_ttc - cached_ttc)
                raw_height_max_abs = float(raw_difference.max())
                ttc_max_abs = float(ttc_difference.max())
                output_stats["raw_height_max_abs"] = max(
                    output_stats["raw_height_max_abs"], raw_height_max_abs
                )
                output_stats["ttc_max_abs"] = max(output_stats["ttc_max_abs"], ttc_max_abs)
                if not torch.isfinite(raw_output).all() or not torch.isfinite(cached_output).all():
                    failures.append({"sample_token": token, "field": "model_output_nonfinite"})
                checked += 1

    for stats in input_stats.values():
        stats["mean_abs"] = stats["mean_abs_sum"] / max(int(stats["count"]), 1)
        del stats["mean_abs_sum"]
    finite_outputs = all(np.isfinite(value) for value in output_stats.values())
    payload: dict[str, object] = {
        "artifact_type": "garl_cache_model_input_parity_v1",
        "status": "pass" if not failures and finite_outputs else "fail",
        "manifest": args.manifest.resolve().as_posix(),
        "manifest_sha256": _sha256(args.manifest.resolve()),
        "release_root": release_root.as_posix(),
        "release_config": config_path.as_posix(),
        "release_config_sha256": _sha256(config_path),
        "variant": args.variant,
        "device": str(device),
        "checked_samples": checked,
        "input_stats": input_stats,
        "output_stats": output_stats,
        "input_tolerances": {
            "event_roi_max_abs": 1.0 / 65535.0 + 1e-5,
            "rgb_pair_max_abs": 2e-3,
        },
        "failures": failures[:100],
        "failure_count": len(failures),
        "raw_release_vs_quantized_cache": True,
        "bbox_protocol": "P0_oracle_bbox_roi",
        "private_test_or_paper_parity_claim": False,
        "negative_result_preserved": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
