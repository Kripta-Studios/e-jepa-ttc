"""Resume hashing includes optimizer tensors, RNG arrays and the next-sample cursor."""

import copy

import numpy as np
import torch

from operational.train40_system.checkpoint import payload_digest


def test_full_resume_hash_detects_tensor_rng_and_cursor_changes():
    payload = {
        "optimizer": {0: {"exp_avg": torch.ones(3)}},
        "numpy_rng": ("MT19937", np.asarray([1, 2], np.uint32)),
        "cursor": {"position": 32, "order": torch.arange(64)},
    }
    original = payload_digest(payload)
    assert payload_digest(copy.deepcopy(payload)) == original
    for field in ("optimizer", "numpy_rng", "cursor"):
        changed = copy.deepcopy(payload)
        if field == "optimizer":
            changed[field][0]["exp_avg"][1] = 2
        elif field == "numpy_rng":
            changed[field][1][0] = 3
        else:
            changed[field]["position"] = 33
        assert payload_digest(changed) != original
