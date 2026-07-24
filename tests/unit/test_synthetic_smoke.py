import torch
from torch.utils.data import Dataset
import numpy as np

from e_jepa_ttc.models.tiny_cnn import TinyCNNEncoder, TinyCNNRegressor
from e_jepa_ttc.representations.voxel_grid import encode_voxel_grid
from e_jepa_ttc.evaluation.robustness import evaluate_robustness
from e_jepa_ttc.data.types import EventBatch

class MockEventDataset(Dataset):
    def __init__(self, size: int = 4):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> EventBatch:
        return EventBatch(
            x=np.array([10, 20, 30], dtype=np.int32),
            y=np.array([10, 20, 30], dtype=np.int32),
            t_us=np.array([1000, 2000, 3000], dtype=np.int64),
            polarity=np.array([1, -1, 1], dtype=np.int8),
            width=64,
            height=64,
            sequence_id="seq0",
            t_start_us=0,
            t_end_us=4000
        )

def test_synthetic_smoke_pipeline():
    device = torch.device("cpu")
    dataset = MockEventDataset(size=2)
    
    # Instantiate models
    encoder = TinyCNNEncoder(in_channels=10, width=16)
    regressor = TinyCNNRegressor(in_channels=10, width=16)
    
    # Mock pretraining step (encoder forward)
    # Convert EventBatch to VoxelGrid manually for the mock
    event_batch = dataset[0]
    voxel_grid = encode_voxel_grid(event_batch, bins=5)
    x = torch.from_numpy(voxel_grid).float().unsqueeze(0)
    
    encoded = encoder(x)
    assert encoded.shape == (1, 64)
    
    # Mock finetuning step
    pred = regressor(x)
    loss = pred.sum()
    loss.backward() # check gradients flow
    
    # Run robustness
    # The robustness evaluator should return a dictionary with the results
    robustness_results = evaluate_robustness(
        model=regressor,
        dataset=dataset,
        device=device,
        corruptions=None # use defaults
    )
    
    assert "corruptions_tested" in robustness_results
    assert robustness_results["corruptions_tested"] > 0
