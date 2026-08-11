#!/usr/bin/env python
"""Freeze A6 transport-adapter configs from the already frozen A5-ANCHOR controls."""
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[1]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a YAML mapping")
    return payload

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config-dir", default="artifacts/configs/a5_postgate_recovery_v1")
    ap.add_argument("--protocol", default="configs/experiment/e_jepa_garl_event_causal_scale_a6_transport_adapter_v1.yaml")
    ap.add_argument("--output-dir", default="artifacts/configs/a6_transport_adapter_v1")
    args = ap.parse_args()

    protocol_path = ROOT / args.protocol
    protocol = read_yaml(protocol_path)
    source = protocol["source_evidence"]
    contract = protocol["adapter_contract"]

    checkpoint = ROOT / source["a4_checkpoint"]
    if sha256(checkpoint) != source["a4_checkpoint_sha256"]:
        raise ValueError("A4 checkpoint differs from A6 preregistration")

    diagnostic_path = ROOT / source["diagnostic_replication"]
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if diagnostic.get("artifact_type") != "a5_transport_signal_diagnostic_replication_v1":
        raise ValueError("A5 diagnostic replication artifact type is incompatible")
    if bool(diagnostic.get("passed")) is not bool(source["diagnostic_replication_required_passed"]):
        raise ValueError("A5 diagnostic replication decision differs from A6 preregistration")
    if int(diagnostic.get("passing_seeds", -1)) != int(source["diagnostic_replication_required_passing_seeds"]):
        raise ValueError("A5 diagnostic replication seed count differs from A6 preregistration")

    anchor_gate_path = ROOT / source["anchor_seed7_gate"]
    anchor_gate = json.loads(anchor_gate_path.read_text(encoding="utf-8"))
    if bool(anchor_gate.get("passed")) is not bool(source["anchor_seed7_required_passed"]):
        raise ValueError("A5-ANCHOR seed7 decision differs from A6 preregistration")
    anchor_summary = json.loads((ROOT / source["anchor_seed7_summary"]).read_text(encoding="utf-8"))
    if anchor_summary.get("artifact_sha256") != source["anchor_seed7_artifact_sha256"]:
        raise ValueError("A5-ANCHOR seed7 summary identity differs from A6 preregistration")

    base_dir = ROOT / args.base_config_dir
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}

    for seed in (7, 13, 23):
        anchored = read_yaml(base_dir / f"anchor_seed{seed}.yaml")
        model_source_path = ROOT / anchored["model_config"]
        model_payload = read_yaml(model_source_path)
        model_payload["transport_adapter_enabled"] = True
        model_payload["transport_adapter_depth"] = int(contract["transport_adapter_depth"])
        model_path = out / f"model_seed{seed}.yaml"
        model_path.write_text(yaml.safe_dump(model_payload, sort_keys=False), encoding="utf-8", newline="\n")
        files[model_path.name] = sha256(model_path)

        cfg = copy.deepcopy(anchored)
        cfg["experiment"]["name"] = f"e_jepa_garl_event_causal_scale_a6_transport_adapter_v1_seed{seed}"
        cfg["experiment"]["protocol_version"] = "causal_scale_a6_transport_adapter_v1"
        cfg["experiment"]["single_scientific_difference"] = (
            "keep_frozen_A4_geometry_but_add_identity_initialized_trainable_transport_only_feature_adapter"
        )
        cfg["model_config"] = model_path.relative_to(ROOT).as_posix()
        dc = cfg["decision_contract"]
        dc["expected_parameter_count"] = int(contract["expected_parameter_count"])
        change = dc["representation_change"]
        change["type"] = "a4_frozen_endpoint_plus_adaptive_transport_adapter"
        change["transport_adapter_enabled"] = True
        change["transport_adapter_depth"] = int(contract["transport_adapter_depth"])
        change["A4_endpoint_encoder_inherited_from_checkpoint"] = True
        change["A4_endpoint_encoder_frozen_for_entire_run"] = True
        dc.pop("anchor_contract", None)
        dc.pop("anchor_gate", None)
        dc["adapter_contract"] = {
            "initialization_mode": "shape_compatible",
            "initialization_checkpoint": source["a4_checkpoint"],
            "initialization_checkpoint_sha256": source["a4_checkpoint_sha256"],
            "parent_encoder_frozen_for_entire_run": True,
            "geometry_must_equal_parent_by_construction": True,
            "foreground_warmup_epochs": 0,
            "transport_adapter_depth": int(contract["transport_adapter_depth"]),
            "adapter_is_transport_only": True,
            "adapter_identity_initialized": True,
            "no_resolution_change": True,
            "no_radius_or_temperature_change": True,
            "no_DINO_lambda_change": True,
        }
        dc["adapter_gate"] = copy.deepcopy(protocol["adapter_gate"])
        dc["a6_protocol"] = args.protocol
        cfg_path = out / f"seed{seed}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8", newline="\n")
        files[cfg_path.name] = sha256(cfg_path)

    manifest = {
        "artifact_type": "a6_transport_adapter_frozen_configs_v1",
        "protocol": args.protocol,
        "protocol_sha256": sha256(protocol_path),
        "source_A4_checkpoint_sha256": source["a4_checkpoint_sha256"],
        "transport_radius": int(contract["transport_radius"]),
        "transport_temperature": float(contract["transport_temperature"]),
        "transport_adapter_depth": int(contract["transport_adapter_depth"]),
        "expected_parameter_count": int(contract["expected_parameter_count"]),
        "files": files,
        "private_test_opened": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
