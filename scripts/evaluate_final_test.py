import argparse
import logging
import sys
from pathlib import Path

# Add src to path if needed, assuming run from repo root
sys.path.insert(0, str(Path("src").absolute()))

from e_jepa_ttc.experiments.test_lock import check_final_test_lock


def main():
    parser = argparse.ArgumentParser(description="Evaluate on the final test split.")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the model checkpoint to evaluate"
    )
    parser.add_argument(
        "--cache-manifest", type=str, required=True, help="Path to the cache manifest"
    )
    parser.add_argument(
        "--target-hash",
        type=str,
        default=None,
        help="Optional hash of the exact model/protocol to evaluate",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logging.info("Initiating final test evaluation protocol...")

    if not check_final_test_lock(claim_level="final", target_hash=args.target_hash):
        logging.error("Final test lock check failed. Evaluation aborted.")
        sys.exit(1)

    logging.info("Final test lock successfully validated.")
    logging.info(f"Evaluating checkpoint {args.checkpoint} on final test splits...")
    sys.exit("FINAL_TEST_EVALUATOR_NOT_IMPLEMENTED")


if __name__ == "__main__":
    main()
