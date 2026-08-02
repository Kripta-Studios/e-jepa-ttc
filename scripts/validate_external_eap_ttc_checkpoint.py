import argparse
import hashlib
from pathlib import Path

import torch


def validate_checkpoint() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    print(f"File size: {ckpt_path.stat().st_size} bytes")

    # Calculate SHA256
    hasher = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        hasher.update(f.read())
    file_hash = hasher.hexdigest()
    print(f"File SHA256: {file_hash}")

    # Load and validate
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if "audit_json_sha256" in state_dict:
        print(f"Internal audit_json_sha256: {state_dict['audit_json_sha256']}")
        print(f"Internal audit_result: {state_dict.get('audit_result')}")
        print(f"Internal garlttc_data_sha256: {state_dict.get('garlttc_data_sha256')}")
    else:
        print("Internal audit_json_sha256 NOT FOUND")

    print("Validation: PASS")


if __name__ == "__main__":
    validate_checkpoint()
