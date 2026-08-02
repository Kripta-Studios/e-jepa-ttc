"""Dataset adapters and event data contracts."""

from e_jepa_ttc.data.carla_looming import (
    CarlaLoomingMetadata,
    CarlaLoomingSequence,
    CarlaLoomingWindowDataset,
)
from e_jepa_ttc.data.eap_highres_jepa import (
    BlockAwareBatchSampler,
    EAPHighResLabelFreeDataset,
    LabelFreeDataset,
    LabelFreeEAPDataset,
    MatchedEAPDataset,
    collate_label_free,
    make_label_free_loader,
)
from e_jepa_ttc.data.matched_eap_subset import (
    ALLOWED_PARQUET_COLUMNS,
    MatchedEAPSubsetBuilder,
    MatchedSubsetConfig,
    build_matched_eap_subset,
    load_matched_manifest,
    validate_code_commit,
    validate_matched_manifest,
)
from e_jepa_ttc.data.types import DatasetSequence, EventBatch, TTCWindowSample

__all__ = [
    "CarlaLoomingMetadata",
    "CarlaLoomingSequence",
    "CarlaLoomingWindowDataset",
    "DatasetSequence",
    "EventBatch",
    "TTCWindowSample",
    "ALLOWED_PARQUET_COLUMNS",
    "BlockAwareBatchSampler",
    "EAPHighResLabelFreeDataset",
    "LabelFreeEAPDataset",
    "LabelFreeDataset",
    "MatchedEAPDataset",
    "MatchedSubsetConfig",
    "MatchedEAPSubsetBuilder",
    "build_matched_eap_subset",
    "load_matched_manifest",
    "validate_code_commit",
    "validate_matched_manifest",
    "collate_label_free",
    "make_label_free_loader",
]
