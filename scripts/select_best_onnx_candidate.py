import argparse
import json
import sys
import hashlib
from pathlib import Path

def hash_file(path):
    if not path or not Path(path).exists():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4096), b""):
            h.update(b)
    return h.hexdigest()

def hash_dict(d):
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="artifacts/runs")
    parser.add_argument("--require-full-label", action="store_true")
    parser.add_argument("--require-commit", type=str, default=None)
    parser.add_argument("--require-protocol", type=str, default=None)
    parser.add_argument("--require-navigation", type=str, default="enabled")
    parser.add_argument("--require-model-name", type=str, default="event-tubelet-transformer")
    args = parser.parse_args()

    # Load protocol version from yaml
    protocol_path = Path("configs/recovery_v3_protocol.yaml")
    if not protocol_path.exists():
        print(json.dumps({"status": "rejected", "reason": "missing_protocol_file"}), file=sys.stderr)
        sys.exit(1)
        
    import yaml
    with protocol_path.open("r", encoding="utf-8") as f:
        protocol_data = yaml.safe_load(f)
    actual_protocol_version = str(protocol_data.get("protocol_version", "unknown"))
    actual_protocol_hash = hash_file(protocol_path)

    expected_protocol_version = args.require_protocol if args.require_protocol else actual_protocol_version

    runs_dir = Path(args.runs_dir)
    best_mae = float("inf")
    best_record = None

    if runs_dir.exists():
        for run_path in runs_dir.iterdir():
            if not run_path.is_dir() or "recovery" not in run_path.name:
                continue

            metrics_path = run_path / "metrics.json"
            if not metrics_path.exists():
                continue

            with metrics_path.open("r", encoding="utf-8") as f:
                try:
                    metrics = json.load(f)
                except Exception:
                    print(json.dumps({"run": run_path.name, "status": "rejected", "reason": "invalid_json"}), file=sys.stderr)
                    sys.exit(1)

            # Check explicit reasons
            if args.require_full_label and float(metrics.get("train_fraction", 0.0)) < 1.0:
                print(json.dumps({"run": run_path.name, "status": "rejected", "reason": "train_fraction_mismatch"}), file=sys.stderr)
                continue
            if args.require_commit and metrics.get("git_commit") != args.require_commit:
                print(json.dumps({"run": run_path.name, "status": "rejected", "reason": "commit_mismatch"}), file=sys.stderr)
                continue
            
            run_prot_version = str(metrics.get("protocol_version", "unknown"))
            if run_prot_version != expected_protocol_version:
                print(json.dumps({"run": run_path.name, "status": "rejected", "reason": "protocol_version_mismatch"}), file=sys.stderr)
                continue
                
            if metrics.get("navigation_mode") != args.require_navigation:
                print(json.dumps({"run": run_path.name, "status": "rejected", "reason": "navigation_mode_mismatch"}), file=sys.stderr)
                continue
            if metrics.get("model_name", "event-tubelet-transformer") != args.require_model_name:
                print(json.dumps({"run": run_path.name, "status": "rejected", "reason": "model_name_mismatch"}), file=sys.stderr)
                continue
            if metrics.get("final_test_opened") is True:
                print(json.dumps({"run": run_path.name, "status": "rejected", "reason": "final_test_exposure"}), file=sys.stderr)
                continue
            
            if not ("splits" in metrics and "validation" in metrics["splits"] and "metrics" in metrics["splits"]["validation"] and "mae_s" in metrics["splits"]["validation"]["metrics"]):
                print(json.dumps({"run": run_path.name, "status": "rejected", "reason": "missing_validation_metrics"}), file=sys.stderr)
                continue
            
            mae = float(metrics["splits"]["validation"]["metrics"]["mae_s"])
            ckpt = run_path / "tiny_cnn_best.pt"
            if not ckpt.exists():
                print(json.dumps({"run": run_path.name, "status": "rejected", "reason": "missing_checkpoint"}), file=sys.stderr)
                continue
                
            cache_path = metrics.get("cache", metrics.get("cache_path"))
            cache_sha256 = metrics.get("cache_sha256")
            
            model_config_dict = metrics.get("run_fingerprint_payload", {}).get("resolved_model_config", {})
            if not model_config_dict:
                model_config_dict = metrics.get("model_config", {})
                
            cfg_sha256 = hash_dict(model_config_dict)
            
            # The run's protocol hash must match the actual file's hash OR we update it? 
            # "Calcula el SHA-256 real del protocolo resuelto."
            run_protocol_hash = metrics.get("protocol_sha256", metrics.get("protocol_hash"))
            # If the run didn't save the correct hash, it fails, but for compatibility if it has no hash maybe we reject?
            # Wait, the instruction says: "El candidate sólo es válido si los tres coinciden con el run... Reemplaza: protocol_hash = best_metrics.get("protocol_version", ...) porque una versión no es un hash."
            
            if not run_protocol_hash or run_protocol_hash == "unknown":
                print(json.dumps({"run": run_path.name, "status": "rejected", "reason": "protocol_hash_missing"}), file=sys.stderr)
                continue
            
            git_commit = metrics.get("git_commit")
            
            if run_protocol_hash != actual_protocol_hash:
                print(json.dumps({"run": run_path.name, "status": "rejected", "reason": "protocol_hash_mismatch"}), file=sys.stderr)
                continue
                
            if mae < best_mae:
                best_mae = mae
                best_record = {
                    "artifact_type": "onnx_candidate_v3",
                    "schema_version": "3.0",
                    "checkpoint_path": ckpt.as_posix(),
                    "checkpoint_sha256": hash_file(ckpt),
                    "cache_path": cache_path,
                    "cache_sha256": cache_sha256,
                    "cache_sidecar_sha256": metrics.get("cache_sidecar_sha256", "unknown"),
                    "model_config": model_config_dict,
                    "model_config_sha256": cfg_sha256,
                    "protocol_version": actual_protocol_version,
                    "protocol_sha256": actual_protocol_hash,
                    "split_manifest_sha256": metrics.get("split_manifest_sha256", "unknown"),
                    "normalization_sha256": metrics.get("normalization_sha256", "unknown"),
                    "navigation_mode": metrics.get("navigation_mode"),
                    "selection_split": "validation",
                    "selection_metric": "mae_s",
                    "selection_metric_value": mae,
                    "code_commit": git_commit,
                }

    if best_record is None:
        print("Error: Could not find any valid trained checkpoint in validation.", file=sys.stderr)
        sys.exit(1)

    for key, val in best_record.items():
        if val == "unknown" or val == "missing" or val is None:
            print(json.dumps({"run": "best", "status": "rejected", "reason": f"missing_{key}"}), file=sys.stderr)
            sys.exit(1)

    print(json.dumps(best_record, indent=2))

if __name__ == "__main__":
    main()
