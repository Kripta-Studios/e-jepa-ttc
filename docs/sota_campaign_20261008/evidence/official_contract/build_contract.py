"""Build a metadata-only official-comparison contract; never open benchmark payloads."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "data/protocols/evttc_official_comparison_v2.json"


def main() -> None:
    """Bind primary-document findings and exact remaining dependencies to hashes."""
    sources = json.loads((OUT / "SOURCES.json").read_text(encoding="utf-8"))
    hf_train = set((OUT / "dataset_splits_train.txt").read_text().split())
    github_train = set((OUT / "github_train.txt").read_text().split())
    release = json.loads((OUT / "model_model_release_metadata.json").read_text())
    local_paths = [
        "data/manifests/evttc_all32_local.yaml",
        "data/splits/evttc32_grouped_cv.yaml",
        "data/splits/evttc_all32_article_family_holdout.yaml",
        "artifacts/evttc_transfer_20261008/QUERY_MANIFEST.json",
        "artifacts/efficient_context_20261004/resumption_20261008/evaluation/AUTHORS_DRAFT.txt",
    ]
    local_hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in local_paths
    }
    manifest = yaml.safe_load((ROOT / local_paths[0]).read_text(encoding="utf-8"))
    cv = yaml.safe_load((ROOT / local_paths[1]).read_text(encoding="utf-8"))
    dev_ids = sorted(entry["sequence_id"] for entry in manifest["sequences"])
    labels = [
        "CCRs1-low",
        "CCRs1-medium",
        "CCRs1-high",
        "CCRs2-low",
        "CCRs2-medium",
        "CCRs2-high",
        "CCRm-low",
        "CCRm-medium",
        "Slider-750",
        "Slider-1000",
    ]
    candidates = {}
    for label in labels:
        prefix = label.replace("CCRs1-", "CCRs-1-").replace("CCRs2-", "CCRs-2-")
        candidates[label] = [sequence for sequence in dev_ids if sequence.startswith(prefix + "-")]
    dependencies = [
        {
            "id": "OFFICIAL_RECORDING_AND_QUERY_IDENTITY",
            "status": "UNRESOLVED",
            "required": (
                "Official file hashes/revisions, source recording IDs, crop start/end"
                " times and evaluation query timestamps for each benchmark label, "
                "including overlap/run variant."
            ),
            "reason": (
                "A short scenario label does not identify the recording or queried "
                "intervals; dev32 naming candidates are not proven mappings."
            ),
        },
        {
            "id": "CAMERA_ROI_AND_AVAILABILITY",
            "status": "UNRESOLVED",
            "required": (
                "Exact camera/lens for Garl TableVI; calibration, ROI "
                "sensor/projection, rectification/resize and annotation/detector "
                "availability rules; whether GT boxes/depth are used."
            ),
            "reason": (
                "Website distinguishes 8-mm/16-mm settings. Existing dev32 left-"
                "camera rotation-only ROI protocol is a local approximation, not a "
                "certified official adapter."
            ),
        },
        {
            "id": "TEMPORAL_TARGET_AND_SCORER_CONTRACT",
            "status": "UNRESOLVED",
            "required": (
                "Reference timestamp, history/event duration, RGB pairing and actual "
                "delta_t, GT alignment/interpolation, evaluated TTC range, "
                "zero/missing/failed prediction rules, sequence aggregation and "
                "runtime boundary."
            ),
            "reason": (
                "The pointwise RTE equation is documented; a complete executable "
                "official scoring/coverage contract has not been bound. Do not import"
                " eAP bins or choose exclusions from predictions."
            ),
        },
        {
            "id": "CHECKPOINT_TO_TABLEVI_BINDING",
            "status": "UNRESOLVED",
            "required": (
                "Author-linked SHA-256 checkpoint and exact code/config revision for "
                "TableVI, including whether released paper_ours_full.pth is that "
                "model."
            ),
            "reason": (
                "HF identifies a primary full checkpoint and its config/hash, but no "
                "inspected document binds this digest to the TableVI evaluation."
            ),
        },
        {
            "id": "CHECKPOINT_TRAINING_AND_SELECTION_LINEAGE",
            "status": "UNRESOLVED",
            "required": (
                "Per-checkpoint sequence manifests for all training, branch "
                "pretraining, checkpoint selection, validation and calibration; "
                "explain public TRAIN40 versus GitHub TRAIN46 and any EvTTC use."
            ),
            "reason": (
                "Paper states eAP training and no EvTTC fine-tuning; this is not a "
                "checkpoint-specific exclusion manifest. Six extra GitHub IDs are "
                "training-listed, not new holdouts."
            ),
        },
        {
            "id": "INDEPENDENT_GROUPS_AND_FREEZE",
            "status": "UNRESOLVED",
            "required": (
                "Map official clips to physical recording/session groups and prior "
                "dev32 usage; freeze models, adapters, queries, target eligibility "
                "and scorer before protected evaluation access."
            ),
            "reason": (
                "Local dev32 has a development-model-selection protocol and observed "
                "evaluations. Crops of those recordings cannot become independent by "
                "renaming them."
            ),
        },
    ]
    paper = "https://arxiv.org/html/2603.16303v1"
    evttc = "https://arxiv.org/html/2412.05053v2"
    react = "https://arxiv.org/html/2609.19204v1"
    competition = "https://nail-hnu.github.io/EvTTC/competition/"
    contract = {
        "schema": "evttc_official_comparison_v2",
        "observed_utc": datetime.now(UTC).isoformat(),
        "status": "METADATA_READY_OFFICIAL_REPRODUCTION_BLOCKED_CONTRACT_UNRESOLVED",
        "user_campaign_authorized": True,
        "authorization_is_not_the_blocker": True,
        "official_reproduction_ready": False,
        "independent_confirmation_ready": False,
        "protected_payload_access_performed": False,
        "protected_payload_access_gate": (
            "Only after an explicit frozen evaluation contract; no Stage76 or sealed "
            "test payload was inspected here."
        ),
        "protocols": {
            "garl_table_vi_three": {
                "source": paper,
                "locator": "Table VI and section VII-C1; metric section VII-A",
                "documented_labels": ["CCRs2-medium", "CCRs2-high", "CCRm-medium"],
                "published_garl_rte_percent": [8.31, 10.56, 12.93],
                "published_average_rte_percent": 10.60,
                "modality": "RGB plus events",
                "reported_transfer": "eAP-trained, EvTTC without fine-tuning",
                "average_is_consistent_with_equal_three_row_mean": True,
                "exact_query_weighting_verified": False,
                "checkpoint_digest_bound_by_paper": None,
                "exact_official_sequence_file_map": None,
            },
            "evttc_benchmark_ten": {
                "source": competition,
                "paper_source": evttc,
                "labels": labels,
                "documented_scope": (
                    "Eight road sequences plus two Slider sequences; distinct from "
                    "the Garl three-row table and local dev32."
                ),
                "metadata_only": True,
                "download_links_followed": False,
                "leaderboard_submission_performed": False,
                "official_complete_scorer_bound": False,
            },
            "react_in_domain": {
                "source": react,
                "locator": "Sections III-E/F, Table II caption, section IV-E1",
                "documented": (
                    "Trained on EvTTC; TableII uses in-domain validation and full-"
                    "field event input; five seeds. RTE averaged per sequence. ROI "
                    "crop ablation is separate."
                ),
                "table_ii_garl_reference_percent": 9.44,
                "table_ii_react_fp32_percent": 9.59,
                "table_ii_react_reported_spread": 0.74,
                "not_same_reported_garl_number_as_table_vi": True,
                "exact_train_validation_ids_or_baseline_query_alignment_bound": False,
                "comparability": (
                    "Literature context only until split, checkpoint and query "
                    "identity are resolved; do not combine its numbers with TableVI "
                    "or our development scores."
                ),
            },
        },
        "metric_contract": {
            "documented_garl_pointwise_rte_percent": (
                "abs(predicted_ttc - ground_truth_ttc) / abs(ground_truth_ttc) * 100"
            ),
            "react_equation_denominator": (
                "ground_truth_ttc, with evaluated cases described as positive/collision cases"
            ),
            "local_explicit_scorer": "operational/sota_eval/scoring.py",
            "local_scorer_ready": True,
            "local_policy_not_official_certification": {
                "eligibility": "GT finite and !=0; no prediction-dependent selection",
                "coverage": (
                    "Keep rows without GT, count prediction failures; full-cohort "
                    "score unavailable on failure"
                ),
                "micro_and_macro_sequence": "Report both explicitly; no eAP bin weighting",
                "bootstrap": "Paired whole sequences or verified recording groups, fixed seed",
            },
            "official_aggregation_and_invalid_rules": None,
            "invented_official_thresholds": False,
            "historical_inverse_ttc_relative_alias_is_not_rte": True,
        },
        "lineage": {
            "revisions": sources["revisions"],
            "hf_public_train_count": len(hf_train),
            "github_train_count": len(github_train),
            "github_extra_training_ids": sorted(github_train - hf_train),
            "hf_train_not_in_github": sorted(hf_train - github_train),
            "checkpoint_candidates": [
                record
                for record in release["checkpoints"]
                if record["name"] in ["paper_ours_full.pth", "paper_event_only_lhr.pth"]
            ],
            "publisher_release_metadata_fields": sorted(release),
            "train40_checkpoint_genealogy_verified": False,
            "sha_identifies_bytes_not_training_membership": True,
        },
        "development_overlap": {
            "dev32_sequence_count": len(dev_ids),
            "dev32_sequences": dev_ids,
            "declared_local_cv_role": cv["role"],
            "existing_cv_benchmark10_opened": cv["benchmark10_opened"],
            "candidate_mapping_status": "NAME_FAMILY_SPEED_ONLY_NOT_FILE_OR_TEMPORAL_IDENTITY",
            "candidate_matches": candidates,
            "confirmed_physical_recording_overlap": None,
            "independent_groups": None,
            "no_new_holdout_claim": True,
            "slider_absent_from_dev32_manifest": not any("Slider" in name for name in dev_ids),
        },
        "dependencies": dependencies,
        "next_actions": [
            (
                "Request the four concise author clarifications in AUTHORS_DRAFT.txt;"
                " draft remains unsent."
            ),
            (
                "Resolve recording/clip identity using metadata or publisher hash "
                "maps before declaring independence."
            ),
            (
                "Freeze the completed query, adapter, checkpoint and scorer contract;"
                " then proceed within authorized evaluation scope."
            ),
            (
                "Keep local development comparisons running independently with "
                "explicit exploratory status."
            ),
        ],
        "local_source_sha256": local_hashes,
        "public_source_receipts": "artifacts/sota_campaign_20261008/official_contract/SOURCES.json",
        "updates": 0,
    }
    assert len(dev_ids) == 32 and len(labels) == 10
    assert len(hf_train) == 40 and len(github_train) == 46
    assert set(contract["protocols"]["garl_table_vi_three"]["documented_labels"]).issubset(labels)
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    status = {
        key: contract[key]
        for key in [
            "schema",
            "status",
            "user_campaign_authorized",
            "authorization_is_not_the_blocker",
            "official_reproduction_ready",
            "independent_confirmation_ready",
            "dependencies",
            "next_actions",
        ]
    }
    status["protocol_sha256"] = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    (OUT / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"protocol": str(PROTOCOL.relative_to(ROOT)), "sha256": status["protocol_sha256"]}
        )
    )


if __name__ == "__main__":
    main()
