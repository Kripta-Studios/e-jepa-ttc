"""Resource-bounded CLI preparation under the currently acknowledged time scope."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256

from .configuration_preflight import open_acknowledged_source_configuration
from .coordination import shared_write_admission, verified_ack
from .lifecycle import admitted
from .source_gather_qa import audit_cached_source
from .source_preparation import prepare_source_identities


def prepare_configured_sources(
    local_paths: Path,
    source_config: Path,
    source_config_sha256: str,
    output: Path,
    *,
    other_reserved_bytes: int,
    resume: bool,
) -> dict:
    """Prepare source identities, not fits; refuse unacknowledged expanded scope.

    An expanded configuration is never silently reduced to D0. The original ACK
    currently covers only OLD8192. Expanded metadata can be inspected with the
    separate configuration preflight, but new query sources need the supplementary
    temporal recognition requested from the owner before this loader opens them.
    """
    if type(other_reserved_bytes) is not int or other_reserved_bytes < 0:
        raise ValueError("explicit nonnegative outstanding output reservation required")
    paths_hash = sha256(local_paths)
    if source_config.stat().st_size > 1_048_576 or sha256(source_config) != source_config_sha256:
        raise ValueError("source configuration changed")
    config = json.loads(source_config.read_text(encoding="utf-8"))
    # Check before resolving missing cache paths or reading any target payload.
    if any(key in config for key in ("expansion", "dense", "matched")):
        raise ValueError(
            "WAITING_EXPANDED_TIME_RECOGNITION: original ACK covers OLD8192 only; "
            "preserve D1/DENSE configuration and obtain the supplementary recognition "
            "specified in docs/SIMPLEX_T_STAGE70_EXPANSION_ACK_REQUEST.md"
        )
    sources, inspection = open_acknowledged_source_configuration(
        local_paths, source_config, source_config_sha256
    )
    paths = json.loads(local_paths.read_text(encoding="utf-8"))
    work = Path(paths["worktree"]).resolve(strict=True)
    try:
        if not output.resolve().is_relative_to(work):
            raise ValueError("source preparation output must remain inside the companion worktree")
        ack_hash = inspection["ack_sha256"]
        ack_path = Path(paths["shared_coordination"]) / "SIMPLEX_T_STAGE70_ACK.json"
        index_path = sources.index_root / "INDEX_MANIFEST.json"
        # Exact input-only ROI-context amendment already audited for OLD8192.
        index_hash = "93a4f62e5025c5046fc82fcb1428a428f8a8df869b34486b92d5c753b0f68a3b"

        def validate() -> None:
            if sha256(local_paths) != paths_hash or sha256(source_config) != source_config_sha256:
                raise ValueError("local/source configuration changed during preparation")
            ack = verified_ack(ack_path, ack_hash)
            if sha256(index_path) != index_hash:
                raise ValueError("source index differs from audited OLD8192 context amendment")
            index = json.loads(index_path.read_text(encoding="utf-8"))
            if index["ancestry"] != ack["producers"]["authoritative_historical_manifest"]:
                raise ValueError("context index changes acknowledged producer ancestry")

        def resource_ok() -> bool:
            snapshot = admitted([work])
            # 64 MiB reserved for preparation JSON and atomic rewrite overhead.
            return snapshot["has_headroom"] and shared_write_admission(
                snapshot["written_volume_free_bytes"][0], other_reserved_bytes + 67_108_864
            )

        validate()

        def audit_boundary() -> None:
            if not resource_ok():
                raise InterruptedError("PAUSED_RESOURCE: full source gather QA")

        torch.set_num_threads(4)
        if torch.get_num_interop_threads() != 2:
            torch.set_num_interop_threads(2)
        availability = {
            "d1": False,
            "density": False,
            "t3": True,
            "latent": True,
            "replicate_scalar": True,
            "replicate_latent": True,
        }
        return prepare_source_identities(
            output,
            sources=sources,
            availability=availability,
            configuration_sha256=source_config_sha256,
            authority_sha256=ack_hash,
            validate_prerequisites=validate,
            resource_ok=resource_ok,
            resume=resume,
            validate_loaded_source=lambda source: audit_cached_source(
                source, boundary=audit_boundary
            ),
        )
    finally:
        sources.release()
