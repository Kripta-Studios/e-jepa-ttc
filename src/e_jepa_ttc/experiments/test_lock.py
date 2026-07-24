import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def check_final_test_lock(
    claim_level: str,
    unlock_path: Path = Path("artifacts/audit/final_test_unlock.json"),
    ledger_path: Path = Path("artifacts/audit/final_test_burn_ledger.jsonl")
) -> bool:
    """
    Enforces the scientific 'Final Test Lock'.
    If the claim_level is 'final', an explicit unlock file must exist and be valid.
    A record of the evaluation is appended to a burn ledger.
    """
    if claim_level != "final":
        return True
        
    logging.info("Final test claim requested. Checking final test lock...")
    
    if not unlock_path.exists():
        logging.error(f"Final test lock is active. Unlock file not found at {unlock_path}")
        return False
        
    try:
        with open(unlock_path, "r", encoding="utf-8") as f:
            unlock_data = json.load(f)
            
        # Basic validation of unlock file
        required_keys = {"authorized_by", "reason", "authorization_hash", "timestamp"}
        if not required_keys.issubset(unlock_data.keys()):
            logging.error(f"Unlock file missing required keys: {required_keys - set(unlock_data.keys())}")
            return False
            
    except Exception as e:
        logging.error(f"Failed to read unlock file: {e}")
        return False
        
    # Append to burn ledger
    ledger_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "final_test_evaluation",
        "authorization_hash": unlock_data["authorization_hash"],
        "authorized_by": unlock_data["authorized_by"]
    }
    
    try:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ledger_entry) + "\n")
        logging.warning("FINAL TEST EVALUATION RECORDED IN BURN LEDGER.")
    except Exception as e:
        logging.error(f"Failed to write to burn ledger: {e}")
        return False
        
    return True
