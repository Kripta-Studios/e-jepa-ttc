"""Bind current public Test12 submission metadata without inferring hidden accuracy."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

TRUSTED_INPUTS = "b78730abe68de7fbe06c6c98b25d5fa025c04d9303720b680ce0c0f575d2e526"
REVISION = "53fbaade17f3b76aa56dc7df2b527ce279062f59"


def digest(path: Path) -> str:
    """Hash the exact saved source bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Verify that the pinned official template and sealed native aliases match exactly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or digest(args.inputs) != TRUSTED_INPUTS:
        raise ValueError("Require a fresh receipt and the trusted input inventory")
    info_path = args.snapshot / "hf_dataset_info.json"
    card_path = args.snapshot / "pinned_dataset/README.md"
    template_path = args.snapshot / "pinned_dataset/sample_submission.json"
    info = json.loads(info_path.read_text(encoding="utf-8-sig"))
    card = card_path.read_text(encoding="utf-8")
    template = json.loads(template_path.read_text(encoding="utf-8"))
    public = json.loads(args.inputs.read_text(encoding="utf-8"))
    if (
        info["id"] != "NAIL-HNU/GarlTTC-dataset"
        or info["sha"] != REVISION
        or template["meta"]["format"] != "garlttc_prediction_v1"
        or set(template["results"]) != set(public["physical_aliases"])
        or len(template["results"]) != 6762
        or len(set(public["physical_aliases"].values())) != 5805
        or "Offline evaluation using the private test labels is not supported." not in card
    ):
        raise ValueError("Pinned public Test12 contract differs; reassess its scope")
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "repository": info["id"],
        "revision": REVISION,
        "repository_last_modified": info["last_modified"],
        "readme_url": f"https://huggingface.co/datasets/{info['id']}/blob/{REVISION}/README.md",
        "official_evaluation_url": "https://www.codabench.org/competitions/17289/",
        "public_file_inventory": [f["rfilename"] for f in info["siblings"]],
        "official_template_logical_queries": 6762,
        "trusted_physical_queries": 5805,
        "exact_logical_token_set_match": True,
        "private_test_targets_downloaded": False,
        "test12_accuracy_evaluated": False,
        "official_submission_sent": False,
        "required_for_test12_accuracy": "Authorized official CodaBench evaluation",
        "scope": "Submission identity and public availability only; no accuracy claim",
        "source_sha256": {
            str(p): digest(p) for p in (info_path, card_path, template_path, args.inputs)
        },
        "script_sha256": digest(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("Official template: 6762 aliases / 5805 physical queries; exact identity verified")


if __name__ == "__main__":
    main()
