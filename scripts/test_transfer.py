import hashlib

import torch

from e_jepa_ttc.models.dense_patch_ttc import BaseEventTubeletBackbone, DensePatchTTCHead


def hash_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def test_transfer(checkpoint_path: str) -> None:
    # 1. Instantiate Downstream
    print("Instantiating downstream model with random weights...")
    encoder = BaseEventTubeletBackbone(checkpoint_path=None, allow_random_initialization=True)
    head = DensePatchTTCHead(dim=192)
    
    # Hashes before
    encoder_hash_before = {name: hash_tensor(param) for name, param in encoder.named_parameters()}
    head_hash_before = {name: hash_tensor(param) for name, param in head.named_parameters()}
    
    # 2. Load Checkpoint
    print(f"Loading checkpoint {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["encoder_state_dict"]
    
    # To demonstrate missing/unexpected keys, let's load it into the encoder
    # The backbone expects `encoder_state_dict` structure which maps to `encoder.*` internally.
    # BaseEventTubeletBackbone holds `self.encoder` which is EventTubeletTransformerEncoder.
    # The checkpoint state dict is for the EventTubeletTransformerEncoder.
    
    res = encoder.encoder.load_state_dict(state, strict=False)
    
    print(f"\nMissing keys: {res.missing_keys}")
    print(f"Unexpected keys: {res.unexpected_keys}")
    
    # Hashes after
    encoder_hash_after = {name: hash_tensor(param) for name, param in encoder.named_parameters()}
    head_hash_after = {name: hash_tensor(param) for name, param in head.named_parameters()}
    
    # 3. Check what changed
    print("\nEncoder Hash Changes:")
    changed_encoder = 0
    for name in encoder_hash_before:
        if encoder_hash_before[name] != encoder_hash_after[name]:
            changed_encoder += 1
    print(f"{changed_encoder} / {len(encoder_hash_before)} encoder tensors CHANGED.")
    
    print("\nHead Hash Changes:")
    changed_head = 0
    for name in head_hash_before:
        if head_hash_before[name] != head_hash_after[name]:
            changed_head += 1
    print(f"{changed_head} / {len(head_hash_before)} head tensors CHANGED.")

if __name__ == "__main__":
    ckpt = "artifacts/runs/eap_ttc_smoke_seed42/eap_jepa_encoder_best.pt"
    test_transfer(ckpt)
