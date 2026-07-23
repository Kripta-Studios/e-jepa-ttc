import sys
import torch

def main():
    if len(sys.argv) != 3:
        print("Usage: python verify_checkpoint_provenance.py <checkpoint_path> <expected_commit>")
        sys.exit(1)
        
    checkpoint_path = sys.argv[1]
    expected_commit = sys.argv[2]
    
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        actual_commit = ckpt.get("git_commit")
        
        if actual_commit == expected_commit:
            sys.exit(0)
        else:
            print(f"Commit mismatch: expected {expected_commit}, got {actual_commit}")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
