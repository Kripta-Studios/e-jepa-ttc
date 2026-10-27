import threading
from types import SimpleNamespace

import pytest
import torch

from operational.oof_validated_prefetch import ValidatedFoldPrefetchRuntime


def setup_runtime(invalid=False):
    cursor = dict(epoch=1, position=0, order=[0, 1, 2, 3], losses=[])
    calls = []

    class Checkpoint:
        def restore(self):
            return cursor

    def validate(part, recipe):
        calls.append(threading.current_thread().name)
        if not bool(torch.isfinite(part.events).all()):
            raise ValueError("invalid data")

    def build(source, ids, recipe):
        parts = [
            SimpleNamespace(
                events=torch.tensor([float("nan") if invalid else i]),
                dinov3_relation_targets=None,
                dinov3_relation_valid=None,
            )
            for i in ids
        ]
        return SimpleNamespace(parts=parts)

    module = SimpleNamespace(
        ProducerCheckpoint=Checkpoint, _effective_batch=build, _validate_batch=validate
    )
    source = SimpleNamespace(identity={"modality": "rgb"}, population_size=4)
    recipe = SimpleNamespace(effective_batch_size=2)
    return module, source, recipe, calls, cursor


def test_validates_in_worker_and_consumes_once():
    module, source, recipe, calls, cursor = setup_runtime()
    original = module._validate_batch
    builder = module._effective_batch
    with ValidatedFoldPrefetchRuntime(module, parallel_rgb_decode=False) as runtime:
        module.ProducerCheckpoint().restore()
        batch = module._effective_batch(source, [0, 1], recipe)
        for part in batch.parts:
            module._validate_batch(part, recipe)
        assert all(name.startswith("oof-fold-prefetch") for name in calls)
        with pytest.raises(ValueError, match="single-use"):
            module._validate_batch(batch.parts[0], recipe)
        assert runtime.snapshot()["cpu_validation"]["consumed_parts"] == 2
    assert module._validate_batch is original and module._effective_batch is builder
    assert cursor["position"] == 0


@pytest.mark.parametrize("mutation", ["inplace", "replace", "recipe", "metadata"])
def test_changed_inputs_rejected(mutation):
    module, source, recipe, _, _ = setup_runtime()
    with ValidatedFoldPrefetchRuntime(module, parallel_rgb_decode=False):
        module.ProducerCheckpoint().restore()
        part = module._effective_batch(source, [0, 1], recipe).parts[0]
        if mutation == "inplace":
            part.events.add_(1)
        elif mutation == "replace":
            part.events = part.events.clone()
        elif mutation == "metadata":
            part.dinov3_relation_valid = object()
        else:
            recipe = SimpleNamespace(effective_batch_size=2)
        with pytest.raises(ValueError):
            module._validate_batch(part, recipe)


def test_invalid_data_propagates_and_symbols_restore():
    module, source, recipe, _, _ = setup_runtime(invalid=True)
    original = module._validate_batch
    with pytest.raises(ValueError, match="invalid data"):
        with ValidatedFoldPrefetchRuntime(module, parallel_rgb_decode=False):
            module.ProducerCheckpoint().restore()
            module._effective_batch(source, [0, 1], recipe)
    assert module._validate_batch is original
