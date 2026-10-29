from __future__ import annotations

from e_jepa_ttc.training.eap_jepa import EAPJEPATrainerConfig, _sampling_provenance
from scripts.build_recovery_readiness_v4 import (
    SSL_PURE_SAMPLING_FIELDS,
    _ssl_pure_sampling_provenance,
)


def test_trainer_emits_complete_ssl_pure_sampling_provenance() -> None:
    provenance = _sampling_provenance(EAPJEPATrainerConfig(), geometry_enabled=False)
    assert set(provenance) == set(SSL_PURE_SAMPLING_FIELDS)
    assert all(value is False for value in provenance.values())


def test_trainer_preserves_geo_and_geo_v2_sampling_disclosures() -> None:
    geo = _sampling_provenance(EAPJEPATrainerConfig(), geometry_enabled=True)
    geo_v2 = _sampling_provenance(
        EAPJEPATrainerConfig(
            geometry_target_version="v2",
            geometry_sampling_strategy="balanced_tracks",
        ),
        geometry_enabled=True,
    )
    assert geo["uses_boxes_for_sampling"] is True
    assert geo["uses_depth_for_sampling"] is True
    assert geo["uses_3d_for_sampling"] is True
    assert geo_v2["uses_category_for_sampling"] is True
    assert geo_v2["uses_depth_for_sampling"] is False


def test_ssl_pure_auditor_requires_every_explicit_sampling_flag() -> None:
    provenance = dict.fromkeys(SSL_PURE_SAMPLING_FIELDS, False)
    assert _ssl_pure_sampling_provenance(provenance)

    for field in SSL_PURE_SAMPLING_FIELDS:
        missing = dict(provenance)
        del missing[field]
        assert not _ssl_pure_sampling_provenance(missing)


def test_ssl_pure_auditor_rejects_each_sampling_label_family() -> None:
    provenance = dict.fromkeys(SSL_PURE_SAMPLING_FIELDS, False)
    for field in SSL_PURE_SAMPLING_FIELDS:
        leaking = dict(provenance)
        leaking[field] = True
        assert not _ssl_pure_sampling_provenance(leaking)
