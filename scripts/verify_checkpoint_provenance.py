import sys

import torch


def main() -> None:
    if len(sys.argv) not in (3, 4):
        print(
            "Usage: python verify_checkpoint_provenance.py <checkpoint_path> "
            "<expected_commit> [<expected_fingerprint>]"
        )
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    expected_commit = sys.argv[2]
    expected_fingerprint = sys.argv[3] if len(sys.argv) == 4 else None

    import hashlib
    import json

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        actual_commit = ckpt.get("git_commit")

        if actual_commit != expected_commit:
            print(f"Commit mismatch: expected {expected_commit}, got {actual_commit}")
            sys.exit(1)

        fingerprint = ckpt.get("run_fingerprint")
        payload = ckpt.get("run_fingerprint_payload")

        if not fingerprint or not payload:
            print("Missing run_fingerprint or run_fingerprint_payload in checkpoint.")
            sys.exit(1)

        computed_fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

        if computed_fingerprint != fingerprint:
            msg = f"Fingerprint mismatch: expected {fingerprint}, got {computed_fingerprint}."
            print(f"{msg} Checkpoint has been tampered with or corrupted.")
            sys.exit(1)

        if expected_fingerprint and fingerprint != expected_fingerprint:
            msg = (
                f"Fingerprint mismatch: runtime expected {expected_fingerprint}, got {fingerprint}."
            )
            print(f"{msg} Checkpoint does not match the requested training parameters.")
            sys.exit(1)

        sys.exit(0)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
