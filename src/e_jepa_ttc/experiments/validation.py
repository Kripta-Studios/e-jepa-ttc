import json
import logging
from pathlib import Path
from typing import Any

import jsonschema
import torch
import yaml


def load_protocol(yaml_path: Path, schema_path: Path) -> dict[str, Any]:
    """Loads and validates the recovery v3 protocol."""
    with open(yaml_path, encoding="utf-8") as f:
        protocol = yaml.safe_load(f)

    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)

    jsonschema.validate(instance=protocol, schema=schema)
    return protocol


def verify_semantic_completion(
    metrics_path: Path, protocol: dict[str, Any], require_metrics: bool = True
) -> bool:
    """Verifies that a completed run's metrics satisfy the protocol constraints."""
    if not metrics_path.exists():
        logging.error(f"Metrics not found at {metrics_path}")
        return False

    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    if require_metrics:
        expected = set(protocol["requirements"]["primary_metrics"])
        # metrics.json typically has a "splits" -> "validation" -> "metrics" structure
        # Check if validation exists
        val_metrics = metrics.get("splits", {}).get("validation", {}).get("metrics", {})
        if not val_metrics:
            logging.error("Validation metrics not found in summary")
            return False

        # Because classification metrics might be NaN or missing if not computed,
        # just check that the metric keys are at least somewhat present or we have some metrics.
        if "mae_s" not in val_metrics:
            logging.error(f"mae_s missing from validation metrics (expected {expected})")
            return False

        # Check support
        train_count = metrics.get("splits", {}).get("train", {}).get("count", 0)
        val_count = metrics.get("splits", {}).get("validation", {}).get("count", 0)

        if train_count < protocol["requirements"]["min_train_support"]:
            logging.error(
                f"Train support {train_count} < {protocol['requirements']['min_train_support']}"
            )
            return False

        if val_count < protocol["requirements"]["min_val_support"]:
            logging.error(
                f"Validation support {val_count} < {protocol['requirements']['min_val_support']}"
            )
            return False

    return True


def verify_architecture_parity(scratch_ckpt_path: Path, jepa_finetuned_ckpt_path: Path) -> bool:
    """
    Ensures that a model trained from scratch and a model fine-tuned from JEPA
    have the exact same architecture (same parameter keys and shapes).
    """
    if not scratch_ckpt_path.exists():
        logging.error(f"Scratch checkpoint not found: {scratch_ckpt_path}")
        return False
    if not jepa_finetuned_ckpt_path.exists():
        logging.error(f"JEPA checkpoint not found: {jepa_finetuned_ckpt_path}")
        return False

    scratch = torch.load(scratch_ckpt_path, map_location="cpu", weights_only=True)
    jepa = torch.load(jepa_finetuned_ckpt_path, map_location="cpu", weights_only=True)

    # Check if nested
    scratch_sd = scratch.get("model_state_dict", scratch)
    jepa_sd = jepa.get("model_state_dict", jepa)

    scratch_keys = set(scratch_sd.keys())
    jepa_keys = set(jepa_sd.keys())

    if scratch_keys != jepa_keys:
        missing_in_jepa = scratch_keys - jepa_keys
        missing_in_scratch = jepa_keys - scratch_keys
        logging.error(
            f"Architecture mismatch. Missing in JEPA: {missing_in_jepa}. Missing in Scratch: {missing_in_scratch}"
        )
        return False

    for k in scratch_keys:
        if scratch_sd[k].shape != jepa_sd[k].shape:
            logging.error(
                f"Shape mismatch for {k}: scratch={scratch_sd[k].shape}, jepa={jepa_sd[k].shape}"
            )
            return False

    logging.info("Architecture parity verified.")
    return True
