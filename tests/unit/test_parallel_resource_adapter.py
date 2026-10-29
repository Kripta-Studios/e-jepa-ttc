"""Check resource metadata admission before any optimizer or cache allocation."""

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("fail_before_cache", [False, True])
def test_archived_protocol_check_precedes_active_resource_restore(
    tmp_path, monkeypatch, fail_before_cache
):
    from operational.efficient_context import garl_train
    from operational.efficient_context import garl_train_parallel_authorized as adapter
    from operational.efficient_context.common import atomic_json
    from operational.efficient_context.parallel_inputs import ParallelInputCache

    out = tmp_path / "out"
    atomic_json(out / "garl/PROTOCOL.json", {"active_tree_RSS_limit_bytes": 12_000_000_000})
    policy = {"max_tree_rss_gib": 16_000_000_000 / 1024**3, "other_caps": "unchanged"}
    c = SimpleNamespace(out=out, policy=policy, require_resources=lambda: None)
    original_cache = garl_train.InputCache
    calls = []

    def init_cache(self, campaign, rows):
        assert campaign.policy is policy
        calls.append("active_resources_before_cache")

    def native_execute(campaign):
        assert campaign.policy["other_caps"] == "unchanged"
        assert int(campaign.policy["max_tree_rss_gib"] * 1024**3) == 12_000_000_000
        calls.append("complete_original_protocol_validation")
        if fail_before_cache:
            raise ValueError("scientific source mismatch")
        garl_train.InputCache(campaign, {})
        assert campaign.policy is policy
        calls.append("active_resources_before_fit")

    monkeypatch.setattr(ParallelInputCache, "__init__", init_cache)
    monkeypatch.setattr(adapter, "_original_execute", native_execute)
    if fail_before_cache:
        with pytest.raises(ValueError, match="scientific source mismatch"):
            adapter.execute(c)
        assert calls == ["complete_original_protocol_validation"]
    else:
        adapter.execute(c)
        assert calls == [
            "complete_original_protocol_validation",
            "active_resources_before_cache",
            "active_resources_before_fit",
        ]
    assert c.policy is policy and garl_train.InputCache is original_cache
