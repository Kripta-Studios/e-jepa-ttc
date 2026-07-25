import pytest
import os
import e_jepa_ttc.artifacts.protocol as protocol_module

@pytest.fixture(autouse=True)
def mock_frozen_protocol(monkeypatch):
    """
    Mock the frozen protocol globally for unit tests so they don't depend on pipeline outputs.
    Tests that actually test the protocol logic should override this.
    """
    def mock_load():
        return {
            "protocol_version": "3.0",
            "protocol_sha256": "mock_protocol_hash",
            "code_commit": "mock_commit",
            "artifact_sha256": "mock_hash"
        }
    
    monkeypatch.setattr(protocol_module, "load_frozen_protocol", mock_load)
