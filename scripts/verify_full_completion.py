import argparse
import json
import logging
import sys
from pathlib import Path

import jsonschema


def _load_schema(name: str) -> dict:
    schema_path = Path(__file__).resolve().parent.parent / "schemas" / name
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing schema: {schema_path}")
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def verify_dir_exists(path: Path) -> None:
    if not path.exists():
        logging.error(f"Missing required directory: {path}")
        sys.exit(1)


def verify_file_exists(path: Path, schema_name: str = None) -> None:
    if not path.exists():
        logging.error(f"Missing required file: {path}")
        sys.exit(1)
        
    if schema_name and path.suffix == ".json":
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            jsonschema.validate(instance=data, schema=_load_schema(schema_name))
        except Exception as e:
            logging.error(f"Schema validation failed for {path} against {schema_name}: {e}")
            sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--onnx-dir", type=Path, default=Path("artifacts/onnx/final"))
    args = parser.parse_args()

    runs_dir = args.runs_dir
    onnx_dir = args.onnx_dir

    verify_dir_exists(runs_dir)

    # 1. Verify EvTTC Matrix
    # We expect scratch, jepa, and downstream runs for seeds 7, 13, 21 and nav enabled/disabled.
    seeds = [7, 13, 21]
    nav_modes = ["enabled", "disabled"]
    fractions = ["frac10", "frac5"]

    for nav in nav_modes:
        # Check scratch full
        for seed in seeds:
            verify_file_exists(
                runs_dir
                / f"recovery_scratch_nav{nav}_seed{seed}_post_fix_v3_cache_verified"
                / "metrics.json", "training_run_v3.schema.json"
            )
            for frac in fractions:
                verify_file_exists(
                    runs_dir
                    / f"recovery_scratch_nav{nav}_seed{seed}_{frac}_post_fix_v3_cache_verified"
                    / "metrics.json", "training_run_v3.schema.json"
                )

        # Check JEPA Pretrain
        for seed in seeds:
            verify_file_exists(
                runs_dir
                / f"recovery_jepa_nav{nav}_seed{seed}_post_fix_v3_cache_verified"
                / "metrics.json", "training_run_v3.schema.json"
            )
            # Check JEPA downstream
            for downstream in seeds:
                verify_file_exists(
                    runs_dir
                    / (
                        f"recovery_downstream_ssl{seed}_nav{nav}_seed{downstream}"
                        "_post_fix_v3_cache_verified"
                    )
                    / "metrics.json", "training_run_v3.schema.json"
                )
                for frac in fractions:
                    verify_file_exists(
                        runs_dir
                        / (
                            f"recovery_downstream_ssl{seed}_nav{nav}_seed{downstream}_{frac}"
                            "_post_fix_v3_cache_verified"
                        )
                        / "metrics.json", "training_run_v3.schema.json"
                    )

    logging.info("EvTTC matrix verification passed.")

    # 2. Verify eAP Matrix
    eap_runs_dir = runs_dir / "eap_object_jepa_matrix"
    verify_dir_exists(eap_runs_dir)

    verify_file_exists(eap_runs_dir / "eap_split_statistics.json", "training_run_v3.schema.json")
    for seed in seeds:
        verify_file_exists(eap_runs_dir / "pretrain" / f"seed-{seed}" / "summary.json", "training_run_v3.schema.json")
        for frac in ["1", "0.1", "0.05"]:
            verify_file_exists(
                eap_runs_dir
                / "finetune"
                / "jepa"
                / f"fraction-{frac}"
                / f"seed-{seed}"
                / "summary.json", "training_run_v3.schema.json"
            )
            verify_file_exists(
                eap_runs_dir
                / "finetune"
                / "scratch"
                / f"fraction-{frac}"
                / f"seed-{seed}"
                / "summary.json", "training_run_v3.schema.json"
            )

    logging.info("eAP matrix verification passed.")

    # 3. Verify ONNX Output
    verify_dir_exists(onnx_dir)
    verify_file_exists(onnx_dir / "model.onnx")
    verify_file_exists(onnx_dir / "model_manifest.json", "onnx_manifest_v3.schema.json")
    verify_file_exists(onnx_dir / "equivalence.json", "onnx_equivalence_v3.schema.json")
    
    # We might not have benchmark checked here initially, but if it exists we should check it
    if (onnx_dir / "benchmark.json").exists():
        verify_file_exists(onnx_dir / "benchmark.json", "onnx_benchmark_v3.schema.json")

    # Load equivalence.json
    with open(onnx_dir / "equivalence.json") as f:
        equiv = json.load(f)
        mae = equiv.get("maximum_absolute_error", float("inf"))
        if mae > 1e-4:
            logging.error(f"ONNX export equivalence failed: Max absolute error {mae} > 1e-4")
            sys.exit(1)

    logging.info("ONNX validation passed.")

    # Ensure no INVALID_RUN.txt exists in runs dir
    invalid_files = list(runs_dir.rglob("INVALID_RUN.txt"))
    if invalid_files:
        logging.error(f"Found INVALID_RUN.txt in {invalid_files}. The run is tainted.")
        sys.exit(1)

    logging.info("ALL MATRICES COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
