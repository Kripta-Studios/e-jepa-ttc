import json
import logging
from datetime import UTC, datetime
from pathlib import Path


def check_final_test_lock(
    claim_level: str,
    target_hash: str | None = None,
    unlock_path: Path = Path("artifacts/audit/final_test_unlock.json"),
    ledger_path: Path = Path("artifacts/audit/final_test_burn_ledger.jsonl"),
) -> bool:
    """
    Enforces the scientific 'Final Test Lock'.
    If the claim_level is 'final', an explicit unlock file must exist and be valid.
    A record of the evaluation is appended to a burn ledger.
    The lock is single-use per authorization_hash.
    If target_hash is provided, the unlock file must specify it.
    """
    if claim_level != "final":
        return True

    logging.info("Final test claim requested. Checking final test lock...")

    if not unlock_path.exists():
        logging.error(f"Final test lock is active. Unlock file not found at {unlock_path}")
        return False

    try:
        with open(unlock_path, encoding="utf-8") as f:
            unlock_data = json.load(f)

        # Basic validation of unlock file
        required_keys = {"authorized_by", "reason", "authorization_hash", "timestamp"}
        if target_hash is not None:
            required_keys.add("target_hash")

        if not required_keys.issubset(unlock_data.keys()):
            logging.error(
                f"Unlock file missing required keys: {required_keys - set(unlock_data.keys())}"
            )
            return False

        if target_hash is not None and unlock_data["target_hash"] != target_hash:
            logging.error(
                f"Unlock target_hash {unlock_data['target_hash']} "
                f"does not match requested {target_hash}"
            )
            return False

    except Exception as e:
        logging.error(f"Failed to read unlock file: {e}")
        return False

    auth_hash = unlock_data["authorization_hash"]

    # Check burn ledger for single-use
    if ledger_path.exists():
        try:
            with open(ledger_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    if entry.get("authorization_hash") == auth_hash:
                        logging.error(
                            f"Final test lock token {auth_hash} has already been burned; "
                            "tokens are strictly single-use."
                        )
                        return False
        except Exception as e:
            logging.error(f"Failed to read burn ledger: {e}")
            return False

    # Append to burn ledger
    ledger_entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "action": "final_test_evaluation",
        "authorization_hash": auth_hash,
        "authorized_by": unlock_data["authorized_by"],
    }
    if target_hash is not None:
        ledger_entry["target_hash"] = target_hash

    try:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ledger_entry) + "\n")
        logging.warning("FINAL TEST EVALUATION RECORDED IN BURN LEDGER.")
    except Exception as e:
        logging.error(f"Failed to write to burn ledger: {e}")
        return False

    return True
