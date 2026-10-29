"""Contract tests for the generic AGENTS.md public module layout."""

import numpy as np
import torch

from e_jepa_ttc.config import canonical_config_hash, load_config
from e_jepa_ttc.data.base import EventReader
from e_jepa_ttc.data.types import EventBatch
from e_jepa_ttc.losses.uncertainty import gaussian_nll
from e_jepa_ttc.models.target_encoder import update_target_encoder
from e_jepa_ttc.representations.normalize import normalize_window


def test_config_hash_is_order_independent(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("b: 2\na: 1\n", encoding="utf-8")
    payload = load_config(path)
    assert payload == {"b": 2, "a": 1}
    assert canonical_config_hash(payload) == canonical_config_hash({"a": 1, "b": 2})


def test_normalization_is_finite_and_bounded() -> None:
    values = normalize_window(np.asarray([[-2.0, 0.0, 4.0]], dtype=np.float32))
    assert np.all(np.isfinite(values))
    assert float(np.max(np.abs(values))) <= 1.1


def test_target_encoder_updates_without_gradients() -> None:
    online = torch.nn.Linear(2, 2, bias=False)
    target = torch.nn.Linear(2, 2, bias=False)
    target.load_state_dict({"weight": torch.zeros_like(online.weight)})
    online.weight.data.fill_(2.0)
    update_target_encoder(target, online, momentum=0.5)
    assert torch.allclose(target.weight, torch.ones_like(target.weight))
    assert all(parameter.grad is None for parameter in target.parameters())


def test_uncertainty_loss_is_finite() -> None:
    value = gaussian_nll(torch.tensor([1.0, -1.0]), torch.zeros(2))
    assert torch.isfinite(value)


def test_event_reader_protocol_is_runtime_usable() -> None:
    class Reader:
        def read_window(self, start_us: int, end_us: int) -> EventBatch:
            return EventBatch.empty(
                width=4,
                height=4,
                sequence_id="fixture",
                t_start_us=start_us,
                t_end_us=end_us,
            )

    reader: EventReader = Reader()
    assert reader.read_window(0, 10).duration_us == 10
