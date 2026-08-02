from __future__ import annotations

from scripts.repair_eap_geo2_provenance import _upgrade_mapping


def test_geo2_provenance_is_distinct() -> None:
    payload = {
        "artifact_type": "eap_geo_on_demand_pretraining_v1",
        "pretraining_regime": "eap_geo",
        "geometry_target_version": "v2",
        "geometry_sampling_strategy": "balanced_tracks",
        "trainer_config": {},
        "provenance": {},
    }
    _upgrade_mapping(payload, artifact=True)
    assert payload["artifact_type"] == "eap_geo_v2_on_demand_pretraining_v1"
    assert payload["pretraining_regime"] == "eap_geo_v2"
    assert payload["uses_labels_for_window_sampling"] is True
    assert payload["provenance"]["uses_object_roi"] is False
    assert payload["provenance"]["uses_boxes_for_sampling"] is True
    assert payload["provenance"]["uses_category_for_sampling"] is True
    assert payload["provenance"]["uses_3d_for_sampling"] is True
    assert payload["provenance"]["uses_future_labels_for_sampling"] is False
