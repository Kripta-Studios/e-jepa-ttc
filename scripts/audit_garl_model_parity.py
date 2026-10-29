"""Compare the official Garl checkpoint with the local source-audited adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.models.garl_ttc_replica import GarlTTCConfig, GarlTTCReplica  # noqa: E402


def _local_key(key: str) -> str | None:
    for source, target in (
        ("backbone_rgb.", "rgb_encoder."),
        ("backbone_event.", "event_encoder."),
    ):
        if key.startswith(source):
            return target + key[len(source) :]
    if key.startswith("final_layer."):
        return "height_head.height_regressor." + key[len("final_layer.") :]
    if key.startswith("middle_layer."):
        return key
    return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(args.release_root.resolve()))
    from garl_ttc.config import load_config
    from garl_ttc.models.ttc_network import TTCNetwork

    config = load_config(args.release_root / "configs" / "garl_ttc_eventdecoder.yaml")
    config["model"].pop("pretrained_ckpt_rgb", None)
    config["model"].pop("pretrained_ckpt_event", None)
    official = TTCNetwork(config, is_train=False).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("Official checkpoint must be a state-dict mapping.")
    official.load_state_dict(checkpoint, strict=True)

    local = GarlTTCReplica(
        GarlTTCConfig(
            event_channels=40,
            modality="rgbe",
            fusion="late",
            objective="height_ratio",
            foreground_supervision=False,
        )
    ).eval()
    local_keys = set(local.state_dict())
    mapped: dict[str, torch.Tensor] = {}
    ignored_keys: list[str] = []
    for key, value in checkpoint.items():
        target = _local_key(str(key))
        if target is not None and target in local_keys:
            mapped[target] = value
        elif str(key).startswith(("backbone_rgb.", "backbone_event.")):
            ignored_keys.append(str(key))
    local.load_state_dict(mapped, strict=True)

    torch.manual_seed(7)
    data = torch.randn(1, 46, 128, 128)
    elapsed = torch.full((1,), 0.1)
    mean = data.new_tensor((0.485, 0.456, 0.406)).view(1, 1, 3, 1, 1)
    std = data.new_tensor((0.229, 0.224, 0.225)).view(1, 1, 3, 1, 1)
    local_rgb_pair = data[:, :6].reshape(1, 2, 3, 128, 128) * std + mean
    with torch.inference_mode():
        official_raw, _ = official(data)
        local_output = local(
            data[:, 6:],
            elapsed,
            rgb_pair=local_rgb_pair,
        )
    raw_error = float(torch.max(torch.abs(official_raw - local_output.predicted_heights)).item())
    official_ttc = elapsed / (1.0 - official_raw[:, 0] / official_raw[:, 1])
    ttc_error = float(torch.max(torch.abs(official_ttc - local_output.ttc_seconds)).item())
    return {
        "artifact_type": "garl_model_parity_v1",
        "release_root": args.release_root.as_posix(),
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_sha256": args.checkpoint_sha256,
        "input_shape": list(data.shape),
        "mapped_parameter_count": len(mapped),
        "ignored_checkpoint_key_count": len(ignored_keys),
        "raw_height_max_abs": raw_error,
        "ttc_max_abs": ttc_error,
        "tolerance": args.tolerance,
        "status": "pass" if raw_error <= args.tolerance and ttc_error <= args.tolerance else "fail",
        "adapter_mode": "local_resnet50_source_audited_state_mapping",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=5e-5)
    args = parser.parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
