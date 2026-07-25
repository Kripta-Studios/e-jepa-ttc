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
    with open(schema_path, encoding="utf-8") as f:
        return json.load(f)


def _load_protocol() -> dict:
    import yaml

    protocol_path = Path(__file__).resolve().parent.parent / "configs" / "recovery_v3_protocol.yaml"
    with open(protocol_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def verify_dir_exists(path: Path) -> None:
    if not path.exists():
        logging.error(f"Missing required directory: {path}")
        sys.exit(1)


def verify_file_exists(path: Path, schema_name: str = None, protocol: dict = None) -> None:
    if not path.exists():
        logging.error(f"Missing required file: {path}")
        sys.exit(1)

    if schema_name and path.suffix == ".json":
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            jsonschema.validate(instance=data, schema=_load_schema(schema_name))

            if schema_name == "training_run_v3.schema.json" and protocol:
                from e_jepa_ttc.experiments.validation import verify_semantic_completion

                if not verify_semantic_completion(path, protocol, require_metrics=True):
                    logging.error(f"Semantic validation against protocol failed for {path}")
                    sys.exit(1)
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

    protocol = _load_protocol()

    # 1. Verify EvTTC Matrix
    # We expect scratch, jepa, and downstream runs for configured seeds and nav modes.
    evttc_matrix = protocol.get("matrix", {})
    seeds = evttc_matrix.get("downstream_seeds", [7, 13, 21])
    ssl_seeds = evttc_matrix.get("ssl_seeds", [7, 13, 21])
    nav_modes = evttc_matrix.get("nav_modes", ["enabled", "disabled"])

    # Convert label fractions to frac strings (e.g., 0.10 -> frac10)
    # Skipping 1.0 because the full run is implicit in scratch full
    fractions_float = [f for f in evttc_matrix.get("label_fractions", []) if f < 1.0]
    fractions = [f"frac{int(round(f * 100))}" for f in fractions_float]

    from e_jepa_ttc.experiments.validation import verify_architecture_parity

    for nav in nav_modes:
        # Check scratch full
        for seed in seeds:
            scratch_ckpt = (
                runs_dir
                / f"recovery_scratch_nav{nav}_seed{seed}_post_fix_v3_cache_verified"
                / "tiny_cnn_best.pt"
            )
            verify_file_exists(
                runs_dir
                / f"recovery_scratch_nav{nav}_seed{seed}_post_fix_v3_cache_verified"
                / "metrics.json",
                "training_run_v3.schema.json",
                protocol=protocol,
            )
            for frac in fractions:
                verify_file_exists(
                    runs_dir
                    / f"recovery_scratch_nav{nav}_seed{seed}_{frac}_post_fix_v3_cache_verified"
                    / "metrics.json",
                    "training_run_v3.schema.json",
                    protocol=protocol,
                )

        # Check JEPA Pretrain
        for seed in ssl_seeds:
            verify_file_exists(
                runs_dir
                / f"recovery_jepa_nav{nav}_seed{seed}_post_fix_v3_cache_verified"
                / "metrics.json",
                "training_run_v3.schema.json",
                protocol=protocol,
            )
            # Check JEPA downstream
            for downstream in seeds:
                jepa_ckpt = (
                    runs_dir
                    / (
                        f"recovery_downstream_ssl{seed}_nav{nav}_seed{downstream}"
                        "_post_fix_v3_cache_verified"
                    )
                    / "tiny_cnn_best.pt"
                )
                verify_file_exists(
                    runs_dir
                    / (
                        f"recovery_downstream_ssl{seed}_nav{nav}_seed{downstream}"
                        "_post_fix_v3_cache_verified"
                    )
                    / "metrics.json",
                    "training_run_v3.schema.json",
                    protocol=protocol,
                )

                # Check architecture parity for the full downstream run
                scratch_ckpt = (
                    runs_dir
                    / f"recovery_scratch_nav{nav}_seed{downstream}_post_fix_v3_cache_verified"
                    / "tiny_cnn_best.pt"
                )
                if scratch_ckpt.exists() and jepa_ckpt.exists():
                    if not verify_architecture_parity(scratch_ckpt, jepa_ckpt):
                        logging.error(
                            f"Architecture parity failed for full downstream seed {downstream} nav {nav}"
                        )
                        sys.exit(1)

                for frac in fractions:
                    jepa_low_ckpt = (
                        runs_dir
                        / (
                            f"recovery_downstream_ssl{seed}_nav{nav}_seed{downstream}_{frac}"
                            "_post_fix_v3_cache_verified"
                        )
                        / "tiny_cnn_best.pt"
                    )

                    verify_file_exists(
                        runs_dir
                        / (
                            f"recovery_downstream_ssl{seed}_nav{nav}_seed{downstream}_{frac}"
                            "_post_fix_v3_cache_verified"
                        )
                        / "metrics.json",
                        "training_run_v3.schema.json",
                        protocol=protocol,
                    )

                    scratch_low_ckpt = (
                        runs_dir
                        / f"recovery_scratch_nav{nav}_seed{downstream}_{frac}_post_fix_v3_cache_verified"
                        / "tiny_cnn_best.pt"
                    )
                    if scratch_low_ckpt.exists() and jepa_low_ckpt.exists():
                        if not verify_architecture_parity(scratch_low_ckpt, jepa_low_ckpt):
                            logging.error(
                                f"Architecture parity failed for frac {frac} downstream seed {downstream} nav {nav}"
                            )
                            sys.exit(1)

    logging.info("EvTTC matrix verification passed.")

    # 2. Verify eAP Matrix
    eap_runs_dir = runs_dir / "eap_object_jepa_matrix"
    verify_dir_exists(eap_runs_dir)

    verify_file_exists(
        eap_runs_dir / "eap_split_statistics.json", "training_run_v3.schema.json", protocol=protocol
    )

    eap_matrix = protocol.get("eap_matrix", {})
    eap_ssl_seeds = eap_matrix.get("ssl_seeds", [17, 42, 73])
    eap_fractions = eap_matrix.get("label_fractions", [1.0, 0.1, 0.05])

    for seed in eap_ssl_seeds:
        verify_file_exists(
            eap_runs_dir / "pretrain" / f"seed-{seed}" / "summary.json",
            "training_run_v3.schema.json",
            protocol=protocol,
        )
        for frac in eap_fractions:
            verify_file_exists(
                eap_runs_dir
                / "finetune"
                / "jepa"
                / f"fraction-{frac}"
                / f"seed-{seed}"
                / "summary.json",
                "training_run_v3.schema.json",
                protocol=protocol,
            )
            verify_file_exists(
                eap_runs_dir
                / "finetune"
                / "scratch"
                / f"fraction-{frac}"
                / f"seed-{seed}"
                / "summary.json",
                "training_run_v3.schema.json",
                protocol=protocol,
            )

            # Note: Parity check for eAP matrix is already built into run_object_jepa_matrix.py,
            # but we can optionally re-run it here if needed.

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
