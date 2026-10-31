"""Create a separate sampler-only producer from the admitted R2 runner."""

# ruff: noqa: E501 -- exact source anchors preserve the admitted producer text.

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENT_SHA256 = "e8c7291de32f39e7326857c0722bb04068f81f605f0ab22ebdb5240f317519db"


def digest(path: Path) -> str:
    """Return a file's SHA-256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rewrite(text: str, plan_sha: str) -> str:
    """Change only sampler admission, construction and its provenance snapshot."""
    replacements = {
        "    ResumablePermutationSampler,\n": "",
        "from e_jepa_ttc.detection.resident_input_cache import ResidentInputCache": "from e_jepa_ttc.detection.rare_sampling import load_schedule\n"
        "from e_jepa_ttc.detection.resident_input_cache import ResidentInputCache",
        "    manifest = validate_campaign_manifest(config)": "    manifest = validate_campaign_manifest(config)\n"
        "    prepared_sampler, sampling_receipt = load_schedule(\n"
        "        ROOT / 'sampling_plan.json',\n"
        f"        expected_sha256='{plan_sha}',\n"
        "        tokens=manifest['roles'][config.data.train_role]['sample_tokens'],\n"
        "        seed=config.seed, updates=config.optimization.updates,\n"
        "        batch_size=config.optimization.batch_size,\n"
        "        manifest_sha256=config.data.expected_manifest_sha256,\n"
        "    )",
        "    sampler = ResumablePermutationSampler(len(train_tokens), config.seed)": "    sampler = prepared_sampler\n    report['sampling'] = sampling_receipt",
        "    owned_paths = [": "    owned_paths = [\n        ROOT / 'sampling_plan.json',\n"
        "        ROOT / 'src/e_jepa_ttc/detection/rare_sampling.py',",
    }
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise ValueError(f"Frozen producer anchor differs: {old}")
        text = text.replace(old, new, 1)
    compile(text, "rare_runner.py", "exec")
    return text


def main() -> None:
    """Freeze a distinct producer; never edit historical source or checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "destination", "module", "plan"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    parent = args.source / "scripts/train_detection_campaign.py"
    if digest(parent) != PARENT_SHA256:
        raise ValueError("Expected the admitted evaluation-prefetch R2 producer")
    if args.destination.exists():
        raise FileExistsError(args.destination)
    text = rewrite(parent.read_text(), digest(args.plan))
    shutil.copytree(args.source, args.destination)
    runner = args.destination / "scripts/train_detection_campaign.py"
    runner.write_text(text, encoding="utf-8", newline="\n")
    module = args.destination / "src/e_jepa_ttc/detection/rare_sampling.py"
    shutil.copy2(args.module, module)
    shutil.copy2(args.plan, args.destination / "sampling_plan.json")
    receipt = {
        "parent_runner_sha256": PARENT_SHA256,
        "runner_sha256": digest(runner),
        "module_sha256": digest(module),
        "plan_sha256": digest(args.plan),
        "intervention": "R2-rare-v1; sampler only; new cold recipe, same 8192-update budget",
    }
    (args.destination / "RARE_SAMPLING_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
