"""Do not clear benchmark stops before every optimizer state is durable."""

from scripts.runpod_concurrency_sequence_v1 import durable


def test_durable_requires_receipt_journal_and_no_pending_update():
    receipt = {"completed_updates": 200, "status": "PAUSED_RESOURCE"}
    journal = {"completed_updates": 200, "durable_updates": 200, "pending_update_upper": 0}
    assert durable(receipt, journal)
    assert not durable(receipt, {**journal, "pending_update_upper": 1})
    assert not durable(receipt, {**journal, "durable_updates": 100})
    assert not durable({**receipt, "status": "IN_PROGRESS"}, journal)
