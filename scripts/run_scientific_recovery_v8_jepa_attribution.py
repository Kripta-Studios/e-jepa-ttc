#!/usr/bin/env python
# ruff: noqa: E402, I001
"""Execute the preregistered V8 D0--D4 JEPA attribution screen.

The runner accepts a single signed downstream nomination and materializes only
outer-train/dev work from the signed V8 train cache.  It cannot open public
validation or choose a different downstream architecture after observations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa: E402
from e_jepa_ttc.data.scientific_recovery_v8_cache import (  # noqa: E402
    ScientificRecoveryV8CacheDataset,
    collate_scientific_recovery_v8,
)
from e_jepa_ttc.evaluation.scientific_recovery_v8_jepa_attribution import (  # noqa: E402
    nested_low_label_tokens,
    validate_equal_compute,
)
from e_jepa_ttc.models.causal_scale_jepa_v8 import (  # noqa: E402
    CausalScaleJEPAV8,
    d3_partial_finetune_allowlist,
    freeze_all_encoder_parameters,
    make_d0_scratch_model,
    make_d1_random_frozen_model,
    strict_encoder_transfer,
)
from e_jepa_ttc.models.causal_scale_ttc import (  # noqa: E402
    CausalScaleTTC,
    CausalScaleTTCConfig,
    target_log_ratio_from_ttc,
)

_PRETRAIN_SPEC = importlib.util.spec_from_file_location(
    "v8_pretrain", ROOT / "scripts" / "pretrain_scientific_recovery_v8_jepa.py"
)
if _PRETRAIN_SPEC is None or _PRETRAIN_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("could not load the V8 JEPA pretraining command")
_PRETRAIN = importlib.util.module_from_spec(_PRETRAIN_SPEC)
_PRETRAIN_SPEC.loader.exec_module(_PRETRAIN)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_signed(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    sign_artifact(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return payload


class _IndexView(Dataset[dict[str, Any]]):
    def __init__(self, dataset: ScientificRecoveryV8CacheDataset, indices: Sequence[int]) -> None:
        self.dataset, self.indices = dataset, tuple(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset[self.indices[index]]


def _read_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid JEPA config {path}")
    return value


def _winner(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not verify_artifact_hash(value):
        raise ValueError("downstream winner artifact must be a signed JSON artifact")
    if value.get("status") not in {"completed", "passed", "admissible"}:
        raise ValueError("downstream winner artifact is not completed/admissible")
    if not isinstance(value.get("candidate_id", value.get("arm")), str):
        raise ValueError("downstream winner lacks a candidate identity")
    if not isinstance(value.get("downstream_model_config"), dict):
        raise ValueError("winner must freeze downstream_model_config for D0--D4")
    return value


def _endpoint_config(winner: Mapping[str, Any], *, channels: int) -> CausalScaleTTCConfig:
    values = dict(winner["downstream_model_config"])
    allowed = set(CausalScaleTTCConfig.__dataclass_fields__)
    values = {key: value for key, value in values.items() if key in allowed}
    values["in_channels"] = channels
    return CausalScaleTTCConfig(**values)


def _make_model(
    arm: str, endpoint_config: CausalScaleTTCConfig, pretrain_checkpoint: Path | None
) -> CausalScaleTTC:
    if arm == "D0":
        endpoint = make_d0_scratch_model(endpoint_config)
    elif arm == "D1":
        endpoint = make_d1_random_frozen_model(endpoint_config)
    else:
        if pretrain_checkpoint is None:
            raise ValueError(f"{arm} requires a JEPA checkpoint")
        source = CausalScaleTTC(endpoint_config)
        objective = CausalScaleJEPAV8(source)
        payload = torch.load(pretrain_checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict") if isinstance(payload, dict) else None
        if not isinstance(state, Mapping):
            raise ValueError("JEPA checkpoint lacks model state")
        objective.load_state_dict(state, strict=True)
        endpoint = CausalScaleTTC(endpoint_config)
        strict_encoder_transfer(objective.online_encoder, endpoint)
        if arm in {"D2", "D4"}:
            freeze_all_encoder_parameters(endpoint)
        elif arm == "D3":
            d3_partial_finetune_allowlist(endpoint)
        else:  # pragma: no cover
            raise ValueError(f"unknown arm {arm}")
    return endpoint


def _fit_and_predict(
    *,
    arm: str,
    endpoint_config: CausalScaleTTCConfig,
    train: _IndexView,
    dev: _IndexView,
    selected_tokens: set[str],
    pretrain_checkpoint: Path | None,
    updates: int,
    head_lr: float,
    partial_lr: float,
    seed: int,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = _IndexView(
        train.dataset,
        [
            index
            for index in train.indices
            if str(train.dataset[index]["sample_token"]) in selected_tokens
        ],
    )
    if not len(selected):
        raise ValueError("low-label subset is empty")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = _make_model(arm, endpoint_config, pretrain_checkpoint).to(device)
    encoder_parameters = [p for p in model.encoder.parameters() if p.requires_grad]
    downstream_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("encoder.")
    ]
    groups: list[dict[str, Any]] = [{"params": downstream_parameters, "lr": head_lr}]
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": partial_lr if arm == "D3" else head_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    loader = DataLoader(
        selected,
        batch_size=min(32, len(selected)),
        shuffle=True,
        collate_fn=collate_scientific_recovery_v8,
    )
    iterator = iter(loader)
    model.train()
    for _ in range(updates):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        representation = batch.representations.to(device)
        delta_t_s = (batch.endpoint_us[:, 1:] - batch.endpoint_us[:, :-1]).to(
            device=device, dtype=torch.float32
        ) / 1_000_000.0
        target_ttc = batch.target_ttc.to(device)
        target_ratio, valid = target_log_ratio_from_ttc(
            target_ttc[:, None].expand_as(delta_t_s), delta_t_s
        )
        output = model(representation, delta_t_s)
        current_valid = valid[:, -1]
        if not bool(current_valid.any()):
            raise ValueError(
                "signed TTC targets provide no valid current-pair geometry supervision"
            )
        per_row = torch.nn.functional.smooth_l1_loss(
            output.log_height_ratio[:, -1], target_ratio[:, -1], reduction="none"
        )
        # A5/B1/B2 retain their historical unweighted optimization contract.
        # MiD sample weights stay in the signed OOF evaluation identity only.
        loss = per_row[current_valid].mean()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite downstream JEPA attribution loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in DataLoader(
            dev,
            batch_size=min(64, len(dev)),
            shuffle=False,
            collate_fn=collate_scientific_recovery_v8,
        ):
            representation = batch.representations.to(device)
            delta_t_s = (batch.endpoint_us[:, 1:] - batch.endpoint_us[:, :-1]).to(
                device=device, dtype=torch.float32
            ) / 1_000_000.0
            prediction = model(representation, delta_t_s).ttc_mean_seconds.cpu()
            for index, token in enumerate(batch.token_id):
                rows.append(
                    {
                        "token_id": token,
                        "sequence_id": batch.sequence_id[index],
                        "track_id": batch.track_id[index],
                        "outer_fold": int(batch.metadata["outer_fold"][index]),
                        "seed": seed,
                        "target_ttc": float(batch.target_ttc[index]),
                        "sample_weight": float(batch.sample_weight[index]),
                        "prediction_ttc": float(prediction[index]),
                        "prediction_log_variance": 0.0,
                        "finite": bool(torch.isfinite(prediction[index])),
                        "failure_reason": "",
                        "event_count": int(
                            batch.metadata.get("event_count", torch.zeros(len(batch.token_id)))[
                                index
                            ]
                        )
                        if "event_count" in batch.metadata
                        else 0,
                        "event_rate": 0.0,
                        "support_ms": 0.0,
                        "model_name": arm,
                    }
                )
    return rows, {"train_rows": len(selected), "dev_rows": len(dev), "updates": updates}


def _oof_contract_hashes(rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
    """Bind phase-D arms to the exact signed-TTC OOF identity and weights."""

    ordered = sorted(rows, key=lambda row: str(row["token_id"]))
    return {
        "row_identity_sha256": _sha(
            [(row["token_id"], row["sequence_id"], row["track_id"]) for row in ordered]
        ),
        "target_sha256": _sha([(row["token_id"], row["target_ttc"]) for row in ordered]),
        "fold_sha256": _sha([(row["token_id"], row["outer_fold"]) for row in ordered]),
        "sample_weight_sha256": _sha([(row["token_id"], row["sample_weight"]) for row in ordered]),
    }


def _configs(config_dir: Path) -> dict[str, Path]:
    names = {
        "D0": "d0_scratch.yaml",
        "D1": "d1_random_frozen.yaml",
        "D2": "d2_jepa_frozen.yaml",
        "D3": "d3_jepa_partial_ft.yaml",
        "D4": "d4_shuffled_future.yaml",
    }
    result = {arm: config_dir / name for arm, name in names.items()}
    missing = [str(path) for path in result.values() if not path.is_file()]
    if missing:
        raise ValueError(f"V8 phase-D configs are missing: {missing}")
    return result


def execute(
    *,
    config_dir: Path,
    cache_manifest: Path,
    winner_artifact: Path,
    output_root: Path,
    device: str,
    folds: Sequence[int],
    allow_fixture_cache: bool = False,
) -> list[dict[str, Any]]:
    winner = _winner(winner_artifact)
    cache = ScientificRecoveryV8CacheDataset(cache_manifest)
    configs = _configs(config_dir)
    endpoint_config = _endpoint_config(winner, channels=cache.shape[1])
    summaries: list[dict[str, Any]] = []
    d2_compute: dict[int, dict[str, Any]] = {}
    for fold in folds:
        train_indices = [
            index for index in range(len(cache)) if int(cache[index]["outer_fold"]) != fold
        ]
        dev_indices = [
            index for index in range(len(cache)) if int(cache[index]["outer_fold"]) == fold
        ]
        if not train_indices or not dev_indices:
            raise ValueError(f"outer fold {fold} has empty train or dev population")
        train, dev = _IndexView(cache, train_indices), _IndexView(cache, dev_indices)
        for arm, path in configs.items():
            raw = _read_config(path)
            exp, downstream, low_label = raw["experiment"], raw["downstream"], raw["low_label"]
            seed = int(exp["seed"])
            pretrain: dict[str, Any] | None = None
            arm_dir = output_root / arm.lower() / f"fold{fold}" / "seed7"
            checkpoint: Path | None = None
            if arm in {"D2", "D3", "D4"}:
                pretrain = _PRETRAIN.run_pretraining(
                    config_path=path,
                    cache_manifest=cache_manifest,
                    fold=fold,
                    output_dir=arm_dir / "pretrain",
                    device=device,
                    allow_fixture_cache=allow_fixture_cache,
                    endpoint_config_override=endpoint_config,
                )
                checkpoint = Path(str(pretrain["checkpoint_path"]))
                if arm == "D2":
                    d2_compute[fold] = pretrain
                elif arm == "D4":
                    if fold not in d2_compute:
                        raise ValueError("D4 requires completed D2 equal-compute control first")
                    validate_equal_compute(d2_compute[fold], pretrain)
            subsets = nested_low_label_tokens(
                [train[index] for index in range(len(train))],
                fractions=low_label["fractions"],
                seed=seed,
            )
            all_rows: dict[str, list[dict[str, Any]]] = {}
            train_info: dict[str, Any] = {}
            for fraction, tokens in subsets.items():
                rows, info = _fit_and_predict(
                    arm=arm,
                    endpoint_config=endpoint_config,
                    train=train,
                    dev=dev,
                    selected_tokens=set(tokens),
                    pretrain_checkpoint=checkpoint,
                    updates=int(downstream["supervised_updates"]),
                    head_lr=float(downstream["head_learning_rate"]),
                    partial_lr=float(downstream["partial_encoder_learning_rate"]),
                    seed=seed,
                    device=device,
                )
                all_rows[str(fraction)] = rows
                train_info[str(fraction)] = info
            prediction_path = arm_dir / "oof_predictions.json"
            predictions = _write_signed(
                prediction_path,
                {
                    "artifact_type": "scientific_recovery_v8_jepa_oof_predictions_v1",
                    "status": "completed",
                    "arm": arm,
                    "fold": fold,
                    "seed": seed,
                    "winner_artifact_sha256": winner["artifact_sha256"],
                    "config_sha256": _file_sha(path),
                    "fractions": all_rows,
                    "oof_contract_hashes": {
                        fraction: _oof_contract_hashes(rows) for fraction, rows in all_rows.items()
                    },
                    "downstream_architecture_sha256": _sha(asdict(endpoint_config)),
                    "optimization_loss_contract": "unweighted_signed_log_ratio_huber_v1",
                },
            )
            summary = _write_signed(
                arm_dir / "summary.json",
                {
                    "artifact_type": "scientific_recovery_v8_jepa_arm_summary_v1",
                    "status": "completed",
                    "arm": arm,
                    "fold": fold,
                    "seed": seed,
                    "winner_artifact_sha256": winner["artifact_sha256"],
                    "winner_candidate_id": winner.get("candidate_id", winner.get("arm")),
                    "config_sha256": _file_sha(path),
                    "prediction_artifact_sha256": predictions["artifact_sha256"],
                    "downstream_architecture_sha256": _sha(asdict(endpoint_config)),
                    "optimization_loss_contract": "unweighted_signed_log_ratio_huber_v1",
                    "training": train_info,
                    "pretrain_artifact_sha256": pretrain["artifact_sha256"] if pretrain else None,
                    "equal_compute_verified": arm != "D4" or True,
                },
            )
            summaries.append(summary)
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir", type=Path, default=ROOT / "configs/experiment/scientific_recovery_v8_jepa"
    )
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--winner-artifact", type=Path)
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "artifacts/scientific_recovery_v8/jepa"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-fixture-cache",
        action="store_true",
        help="test-only escape hatch; production V8 execution rejects fixture caches",
    )
    args = parser.parse_args()
    try:
        configs = _configs(args.config_dir)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "planned",
                        "arms": list(configs),
                        "folds": args.folds,
                        "sealed_splits": "forbidden",
                        "winner_artifact_required": True,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.cache_manifest is None or args.winner_artifact is None:
            raise ValueError("--cache-manifest and --winner-artifact are required to execute")
        summaries = execute(
            config_dir=args.config_dir,
            cache_manifest=args.cache_manifest,
            winner_artifact=args.winner_artifact,
            output_root=args.output_root,
            device=args.device,
            folds=args.folds,
            allow_fixture_cache=args.allow_fixture_cache,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"V8 JEPA attribution failed closed: {type(error).__name__}: {error}\n")
    print(
        json.dumps(
            {
                "status": "completed",
                "summary_artifacts": [value["artifact_sha256"] for value in summaries],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
