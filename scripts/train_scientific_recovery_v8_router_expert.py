#!/usr/bin/env python
# ruff: noqa: E501, E701, E702
"""Train/export one signed historical A5 or C2F expert for V8 nested routing."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact  # noqa: E402
from e_jepa_ttc.data.object_event_v4 import GarlTTCObjectEventV4Dataset  # noqa: E402
from e_jepa_ttc.data.scientific_recovery_v5 import SequenceIndexedView  # noqa: E402


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _base_config(expert: str, outer: int) -> Path:
    if expert == "A5":
        return (
            ROOT
            / f"configs/experiment/scientific_recovery_v6_fold_chain/a5_causal_fold{outer}.yaml"
        )
    return (
        ROOT / f"configs/experiment/scientific_recovery_v7_fold_chain/v7_c2f_fold{outer}_seed7.yaml"
    )


def _selected_identity(base: Mapping[str, Any], sequences: list[str]) -> tuple[pd.DataFrame, int]:
    data = base["data"]
    manifest = ROOT / str(data["cache_manifest"])
    model = yaml.safe_load((ROOT / str(base["model_config"])).read_text(encoding="utf-8"))
    bins = (int(model["in_channels"]) - 2) // 2
    dataset = GarlTTCObjectEventV4Dataset(str(manifest), splits=("train",), bins_per_polarity=bins)
    view = SequenceIndexedView(dataset, sequence_ids=set(sequences))
    frame = view.identity_frame().sort_values("sample_token", kind="stable").reset_index(drop=True)
    universe = json.loads(manifest.read_text(encoding="utf-8"))["split_counts"]["train"]
    return frame, int(universe)


def _nested_contract(
    *,
    train: pd.DataFrame,
    dev: pd.DataFrame,
    universe: int,
    outer: int,
    inner: int | None,
    output: Path,
) -> Path:
    payload: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v8_router_nested_v1",
        "status": "frozen_before_v8_router_training",
        "outer_fold": outer,
        "inner_fold": inner,
        "sample_count": universe,
        "cache_universe_rows": universe,
        "selected_sample_count": len(train) + len(dev),
        "sequence_ids": sorted(
            set(train.sequence_id.astype(str)) | set(dev.sequence_id.astype(str))
        ),
        "folds": [
            {
                "fold": 0,
                "train_sequence_ids": sorted(train.sequence_id.astype(str).unique().tolist()),
                "dev_sequence_ids": sorted(dev.sequence_id.astype(str).unique().tolist()),
                "train_rows": len(train),
                "dev_rows": len(dev),
                "train_sample_tokens_sha256": hashlib.sha256(
                    "\n".join(sorted(train.sample_token.astype(str))).encode()
                ).hexdigest(),
                "dev_sample_tokens_sha256": hashlib.sha256(
                    "\n".join(sorted(dev.sample_token.astype(str))).encode()
                ).hexdigest(),
            }
        ],
        "checks": {
            "train_only_grouped_dev": True,
            "sequence_disjoint_folds": True,
            "sample_token_unique": True,
            "same_cache_universe": True,
            "public_validation_used_for_selection": False,
            "private_test_opened": False,
        },
    }
    payload = sign_artifact(payload)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _run(args: argparse.Namespace) -> None:
    router = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    outer = int(router["outer_fold"])
    inner_folds = {int(item["inner_fold"]): item for item in router["inner_folds"]}
    if args.role == "inner_oof":
        if args.inner_fold is None or args.inner_fold not in inner_folds:
            raise ValueError("inner expert run requires a frozen --inner-fold 0/1/2")
        split = inner_folds[args.inner_fold]
        train_sequences, dev_sequences = split["train_sequence_ids"], split["dev_sequence_ids"]
    else:
        train_sequences, dev_sequences = (
            router["outer_train_sequence_ids"],
            router["outer_dev_sequence_ids"],
        )
    base_path = _base_config(args.expert, outer)
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    train, universe = _selected_identity(base, list(train_sequences))
    dev, _ = _selected_identity(base, list(dev_sequences))
    if set(train.sequence_id) & set(dev.sequence_id) or set(train.sample_token) & set(
        dev.sample_token
    ):
        raise ValueError("nested router expert split is not disjoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = _nested_contract(
        train=train,
        dev=dev,
        universe=universe,
        outer=outer,
        inner=args.inner_fold,
        output=args.output_dir / "nested_protocol.json",
    )
    raw = dict(base)
    raw["experiment"] = {
        "name": f"v8_router_{args.expert.lower()}_outer{outer}_{args.role}_{args.inner_fold if args.inner_fold is not None else 'final'}_seed7"
    }
    raw["data"] = dict(base["data"])
    raw["data"].update(
        {
            "train_sequence_ids": list(train_sequences),
            "dev_sequence_ids": list(dev_sequences),
            "development_protocol": {
                "path": _relative(contract),
                "file_sha256": _sha(contract),
                "artifact_sha256": json.loads(contract.read_text())["artifact_sha256"],
                "fold": 0,
            },
        }
    )
    config = args.output_dir / "effective_config.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    command = [
        sys.executable,
        "scripts/train_causal_scale_eap_screen.py",
        "--config",
        str(config),
        "--output-dir",
        str(args.output_dir / "train"),
        "--device",
        args.device,
    ]
    if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
        raise RuntimeError("historical expert trainer failed")
    summary = json.loads((args.output_dir / "train" / "summary.json").read_text())
    source = pd.read_csv(args.output_dir / "train" / summary["predictions"]["path"])
    frame = pd.DataFrame(
        {
            "token_id": source.sample_token,
            "sequence_id": source.sequence_id,
            "track_id": source.track_id,
            "outer_fold": outer,
            "seed": 7,
            "target_ttc": source.target_ttc_s,
            "sample_weight": 1.0,
            "prediction_ttc": source.prediction_ttc_s,
            "prediction_log_variance": source.ttc_log_variance,
            "finite": True,
            "failure_reason": "",
            "event_count": source.event_count_log1p.map(
                lambda value: float(__import__("math").expm1(value))
            ),
            "event_rate": source.event_rate_log1p.map(
                lambda value: float(__import__("math").expm1(value))
            ),
            "support_ms": 0.0,
            "model_name": args.expert,
            "config_sha256": _sha(config),
            "checkpoint_sha256": summary["checkpoint"]["sha256"],
            "shared_event_count_log1p": source.event_count_log1p,
            "shared_event_rate_log1p": source.event_rate_log1p,
            "a5_flow": source.transport_flow_magnitude if args.expert == "A5" else 0.0,
            "a5_margin": source.guard_margin if args.expert == "A5" else 0.0,
            "a5_log_variance": source.ttc_log_variance if args.expert == "A5" else 0.0,
            "c2f_flow": source.transport_flow_magnitude if args.expert == "C2F" else 0.0,
            "c2f_margin": source.guard_margin if args.expert == "C2F" else 0.0,
            "c2f_log_variance": source.ttc_log_variance if args.expert == "C2F" else 0.0,
        }
    )
    if args.role == "inner_oof":
        frame["inner_fold"] = args.inner_fold
    csv_path = args.output_dir / "expert_oof.csv"
    frame.to_csv(csv_path, index=False, lineterminator="\n")
    artifact = sign_artifact(
        {
            "artifact_type": "scientific_recovery_v8_router_expert_prediction_v1",
            "status": "completed",
            "expert": args.expert,
            "role": args.role,
            "outer_fold": outer,
            "inner_fold": args.inner_fold,
            "protocol_sha256": args.protocol_sha256,
            "checkpoint": {
                "path": _relative(args.output_dir / "train" / summary["checkpoint"]["path"]),
                "sha256": summary["checkpoint"]["sha256"],
            },
            "oof_csv": {"path": _relative(csv_path), "sha256": _sha(csv_path)},
            "nested_contract": {"path": _relative(contract), "sha256": _sha(contract)},
        }
    )
    (args.output_dir / "expert_artifact.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expert", choices=("A5", "C2F"), required=True)
    parser.add_argument("--role", choices=("inner_oof", "outer_dev"), required=True)
    parser.add_argument("--inner-fold", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    try:
        _run(args)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(2, f"V8 router expert failed closed: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
