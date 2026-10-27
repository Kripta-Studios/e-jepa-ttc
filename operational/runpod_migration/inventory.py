"""Build a read-only input inventory; never rent, upload, or start training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .paths import PathMap

GIB = 1024**3


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build(run: Path, output: Path) -> dict[str, Any]:
    """Inventory frozen inputs; bulk SHA values retain their manifest provenance."""
    repo = Path(__file__).resolve().parents[2]
    run = run.resolve(strict=True)
    config = read(run / "SCIENTIFIC_CONFIG.json")
    roots = {str(repo.parent): "workspace", "E:\\": "drives/E"}
    mapping = PathMap(roots, "/workspace/ejepa_bundle")
    records: dict[str, dict[str, Any]] = {}

    def add(
        path: Path,
        profiles: tuple[str, ...],
        expected: str | None = None,
        provenance: str = "",
        *,
        bulk: bool = False,
    ) -> None:
        path = path.resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        key = str(path)
        if key in records:
            records[key]["profiles"] = sorted(set(records[key]["profiles"]) | set(profiles))
            return
        size = path.stat().st_size
        actual = None if bulk else digest(path)
        if expected and actual and actual != expected:
            raise ValueError(f"Input binding changed: {path}")
        records[key] = {
            "source": key,
            "relative": mapping.relative(path),
            "bytes": size,
            "sha256": actual or expected,
            "hash_status": "VERIFIED_NOW"
            if actual
            else ("BOUND_MANIFEST_NOT_REHASHED" if expected else "HASH_AT_PACK_REQUIRED"),
            "hash_provenance": provenance,
            "profiles": sorted(profiles),
        }

    all_profiles = ("rgb_pilot", "all_producers", "rotations")
    # Byte-identical source closure, plus transitive imports outside the registered list.
    for folder in (repo / "src", repo / "operational"):
        for path in folder.rglob("*.py"):
            add(path, all_profiles)
    for name in ("pyproject.toml", "uv.lock", "README.md", "AGENTS.md"):
        add(repo / name, all_profiles)
    add(repo / "scripts/runpod_bootstrap.sh", all_profiles)
    for path in (repo / "docs/runpod_migration_20261026").iterdir():
        if path.is_file():
            add(path, all_profiles)
    for reference in (repo / "artifacts/runpod_migration_20261026").glob(
        "*_INPUT_REFERENCE_R*.json"
    ):
        add(reference, all_profiles)
    for field in ("source_sha256", "input_sha256"):
        for path, sha in config[field].items():
            add(Path(path), all_profiles, sha, f"SCIENTIFIC_CONFIG:{field}")
    add(run / "SCIENTIFIC_CONFIG.json", all_profiles)
    # Cached-head fitting verifies these original recipe-bound files, even when
    # a file is historical provenance rather than an input to the current fit.
    for name, sha in read(Path(config["recipe"]))["source_sha256"].items():
        if name.startswith("file:"):
            add(Path(name.removeprefix("file:")), ("rotations",), sha, "OOF_RECIPE:source_sha256")
    add(Path(config["active_rgb_runtime"]) / "PRODUCER_PLAN.json", all_profiles)
    contract = Path(config["fold_contract"])
    for path in contract.parent.iterdir():
        if path.suffix in {".json", ".parquet"}:
            add(path, all_profiles)
    source = Path(config["rgb_source"])
    descriptor = read(source)
    for role in ("P", "H", "V"):
        manifest_path = (source.parent / descriptor["role_manifests"][role]).resolve()
        add(manifest_path, all_profiles)
        manifest = read(manifest_path)
        add(Path(manifest["rows_path"]), all_profiles, manifest["rows_sha256"])
        binding = (source.parent / descriptor["cache_bindings"][role]).resolve()
        add(binding, all_profiles)
        cache_path = Path(read(binding)["cache_manifest_path"])
        add(cache_path, all_profiles, read(binding)["cache_manifest_sha256"])
        for shard in read(cache_path)["shards"]:
            add(Path(shard["path"]), all_profiles, shard["sha256"], str(cache_path), bulk=True)
    event_root = Path(config["event_root"])
    for path in event_root.glob("*.json"):
        add(path, all_profiles)
    for name in ("TRAIN40_INDEX.npz", "TRAIN40_ROWS.parquet"):
        add(event_root / name, all_profiles)
    add(
        event_root / "H8_FEATURES.npz",
        ("rotations",),
        read(event_root / "H8_FEATURE_MANIFEST.json")["files"][0]["sha256"],
    )
    teacher_path = Path(config["teacher"])
    teacher = read(teacher_path)
    for shard in teacher["ordered_shards"]:
        add(
            Path(teacher["directory"]) / shard["name"],
            all_profiles,
            shard["sha256"],
            str(teacher_path),
            bulk=True,
        )
    event_cache = Path(read(event_root / "PREPARE_FREEZE.json")["cache_root"])
    for path in event_cache.glob("shard_*.npz"):
        receipt = path.with_suffix(".json")
        add(receipt, ("all_producers", "rotations"))
        add(path, ("all_producers", "rotations"), read(receipt)["sha256"], str(receipt), bulk=True)
    raw_root = Path(config["raw_root"])
    add(raw_root / "data/train.parquet", all_profiles)
    for item in read(event_root / "RAW_PLAN.json")["files"]:
        add(
            raw_root / item["filename"],
            ("rotations",),
            item["sha256"],
            str(event_root / "RAW_PLAN.json"),
            bulk=True,
        )
    # History RGB decoding uses published TRAIN tar shards, not test media.
    for path in (raw_root / "data/train").rglob("*.tar"):
        add(path, ("rotations",), bulk=True)
    totals = {}
    for profile in all_profiles:
        selected = [x for x in records.values() if profile in x["profiles"]]
        total = sum(x["bytes"] for x in selected)
        # Separate existing input bytes from a prospective output allowance.
        output_allowance = 10 * GIB if profile == "rgb_pilot" else 50 * GIB
        reserve = max(20 * GIB, math.ceil(total * 0.15))
        capacity_gb = math.ceil((total + output_allowance + reserve) / 1e9 / 25) * 25
        totals[profile] = dict(
            files=len(selected),
            input_bytes=total,
            input_gib=round(total / GIB, 3),
            output_allowance_bytes=output_allowance,
            free_reserve_bytes=reserve,
            recommended_persistent_decimal_gb=capacity_gb,
            unhashed_files=sum(x["sha256"] is None for x in selected),
        )
    packages = [
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "pyarrow",
        "h5py",
        "hdf5plugin",
        "pillow",
        "psutil",
        "omegaconf",
        "pydantic",
        "pyyaml",
        "scipy",
        "scikit-learn",
        "einops",
        "safetensors",
        "transformers",
        "gdown",
        "jsonschema",
        "huggingface-hub",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    origin_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    if len(origin_commit) != 40:
        raise ValueError("Missing source repository base commit")
    result = dict(
        schema="ejepa_runpod_inventory_v1",
        created_utc=datetime.now(UTC).isoformat(),
        scientific_config_sha256=digest(run / "SCIENTIFIC_CONFIG.json"),
        source_run=str(run),
        source_repo=str(repo),
        origin_git_commit=origin_commit,
        origin_git_commit_is_base_only=True,
        exact_execution_code_authority="original SHA bindings plus bundle manifest",
        roots=roots,
        bundle_root="/workspace/ejepa_bundle",
        profiles=totals,
        packages=versions,
        files=sorted(records.values(), key=lambda x: x["relative"]),
        execution_ready=False,
        test12_inference_authorized=False,
        notes=[
            "RGB pilot covers prepared RGB producers only, not a full rotation.",
            "Existing raw digests are verified during packing/remote preflight.",
            "No trained checkpoints imported: fresh unstarted folds only.",
            "Final TRAIN40 fitting/export is not part of this rotation bundle.",
        ],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.run, args.output)
    print(json.dumps(result["profiles"], indent=2))


if __name__ == "__main__":
    main()
