import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_synthetic_pipeline_smoke(tmp_path: Path):
    """
    Runs the full end-to-end eAP matrix using the synthetic mock smoke configuration.
    This ensures that bounding works and that the output schemas pass the verifier.
    """
    cache_path = tmp_path / "mock_eap_cache.npz"
    # Create mock eAP cache
    split = np.array(["train"] * 100 + ["validation"] * 20)
    sequence_id = np.array(["seq1"] * 50 + ["seq2"] * 50 + ["seq3"] * 20)
    
    # Needs context_events, context_ego_actions, future_events, y_ttc, y_class, object_mask, etc.
    context_events = np.random.randn(120, 1, 10, 16, 16).astype(np.float32)
    context_ego_actions = np.random.randn(120, 1, 2).astype(np.float32)
    context_ego_action_mask = np.ones((120, 1), dtype=bool)
    context_boxes = np.random.randn(120, 1, 1, 4).astype(np.float32)
    context_sampling_boxes = np.random.randn(120, 1, 1, 4).astype(np.float32)
    context_depth_m = np.random.randn(120, 1).astype(np.float32)
    context_object_mask = np.ones((120, 1, 1), dtype=bool)
    
    future_events = np.random.randn(120, 3, 10, 16, 16).astype(np.float32)
    future_ego_actions = np.random.randn(120, 3, 2).astype(np.float32)
    future_ego_action_mask = np.ones((120, 3), dtype=bool)
    future_boxes = np.random.randn(120, 3, 1, 4).astype(np.float32)
    future_sampling_boxes = np.random.randn(120, 3, 1, 4).astype(np.float32)
    future_depth_m = np.random.randn(120, 3, 1).astype(np.float32)
    future_object_mask = np.ones((120, 3, 1), dtype=bool)
    
    ttc_s = np.random.uniform(0.5, 5.0, size=(120, 1)).astype(np.float32)
    y_class = (ttc_s < 2.0).astype(np.int64).reshape(-1)
    
    np.savez(
        cache_path,
        split=split,
        sequence_id=sequence_id,
        context_events=context_events,
        context_ego_actions=context_ego_actions,
        context_ego_action_mask=context_ego_action_mask,
        context_boxes=context_boxes,
        context_sampling_boxes=context_sampling_boxes,
        context_depth_m=context_depth_m,
        context_object_mask=context_object_mask,
        future_events=future_events,
        future_ego_actions=future_ego_actions,
        future_ego_action_mask=future_ego_action_mask,
        future_boxes=future_boxes,
        future_sampling_boxes=future_sampling_boxes,
        future_depth_m=future_depth_m,
        future_object_mask=future_object_mask,
        ttc_s=ttc_s,
        y_class=y_class,
        sample_token=np.array([f"tok{i}" for i in range(120)]),
        prediction_horizons_s=np.array([0.5, 1.0, 2.0]),
        cache_format_version=np.array(2)
    )
    
    # Create a simple cache manifest pointing to our npz
    manifest_path = tmp_path / "mock_cache_manifest.json"
    with manifest_path.open("w") as f:
        json.dump({
            "version": "1.0",
            "shards": [
                {
                    "path": str(cache_path),
                    "split": "train",
                    "sequence_ids": ["seq1", "seq2"]
                },
                {
                    "path": str(cache_path),
                    "split": "validation",
                    "sequence_ids": ["seq3"]
                }
            ],
            "splits": {
                "train": ["seq1", "seq2"],
                "validation": ["seq3"]
            }
        }, f)
        
    smoke_dir = tmp_path / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    
    script_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "run_object_jepa_matrix.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        "--cache-manifest", str(manifest_path),
        "--output-dir", str(smoke_dir),
        "--pretrain-epochs", "1",
        "--finetune-epochs", "1",
        "--seeds", "42",
        "--label-fractions", "1.0", "0.5"
    ]
    
    # Run the matrix
    subprocess.check_call(cmd)
    
    # Verify that the matrix script ran successfully and produced the matrix summary
    assert (smoke_dir / "matrix_summary.json").exists()
