"""Fixed QA-IQ versus native-context IQn comparison; development only, no automatic promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_campaign.data import read
from operational.phase_observation_cache.integrity import load_bound_observations
from operational.phase_robust.model import fit_normalization
from operational.phase_startup.model import ExpertRouter
from scripts.audit_phase_native_source_closure import DEPENDENCIES, verify_snapshot
from scripts.train_phase_robust import load_arrays, scored
from scripts.train_phase_startup import fit_router

SEEDS = (7, 13, 21)
EPOCHS = 300
PLAN = Path("docs/superpowers/plans/2026-10-31-phase-native-context-augmentation.md")


def validate_variant_identity(aug: dict[str, np.ndarray]) -> None:
    """Require precisely one last-1/2/3 variant for each unique selected query."""
    required = {"token", "group", "keep", "native", "status", "reason"}
    if set(aug) != required:
        raise ValueError("Augmentation schema differs")
    n = len(aug["token"])
    if any(aug[k].shape != (n,) for k in required - {"native"}):
        raise ValueError("Augmentation columns are not aligned")
    if aug["native"].shape != (n, 3) or aug["native"].dtype != np.float32:
        raise ValueError("Expected FP32 F0/RGB/event vectors")
    if aug["keep"].dtype.kind not in "iu" or not np.isin(aug["keep"], [1, 2, 3]).all():
        raise ValueError("Unknown context restart length")
    if any(aug[k].dtype.kind not in "US" for k in ("token", "group", "status", "reason")):
        raise ValueError("Expected textual augmentation identities")
    if not np.isin(aug["status"], ["fused", "event_only", "unavailable"]).all():
        raise ValueError("Unknown native route status")
    if np.isinf(aug["native"]).any() or np.any(aug["reason"] == ""):
        raise ValueError("Invalid native estimate or missing reason")
    unavailable = aug["status"] == "unavailable"
    event_only = aug["status"] == "event_only"
    if not np.isnan(aug["native"][unavailable]).all():
        raise ValueError("Unavailable route must not fabricate native estimates")
    if event_only.any() and (
        not np.isnan(aug["native"][event_only, 1]).all()
        or not np.isfinite(aug["native"][event_only][:, [0, 2]]).all()
        or not np.array_equal(aug["native"][event_only, 0], aug["native"][event_only, 2])
    ):
        raise ValueError("Event-only route differs from frozen fallback")
    for token in np.unique(aug["token"]):
        mask = aug["token"] == token
        if sorted(aug["keep"][mask].tolist()) != [1, 2, 3] or len(set(aug["group"][mask])) != 1:
            raise ValueError("Missing/duplicate variant or inconsistent group")


def join_augmentations(
    clean: dict[str, np.ndarray], aug: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Join labels and unchanged geometric channels only after external seals pass."""
    validate_variant_identity(aug)
    if clean["token"].ndim != 1 or len(set(clean["token"])) != len(clean["token"]):
        raise ValueError("Clean tokens must be unique")
    lookup = {str(token): i for i, token in enumerate(clean["token"])}
    if any(str(token) not in lookup for token in aug["token"]):
        raise ValueError("Augmentation query missing from eligible population")
    indices = np.array([lookup[str(token)] for token in aug["token"]], np.int64)
    if not np.array_equal(clean["group"][indices], aug["group"]):
        raise ValueError("Augmentation acquisition group differs from clean identity")
    joined = {
        k: clean[k][indices].copy()
        for k in ("geometry", "quality", "target", "family", "group", "token")
    }
    joined.update({k: aug[k].copy() for k in ("native", "keep", "status", "reason")})
    return joined, indices


def augmentation_weights(tokens: np.ndarray, families: np.ndarray) -> np.ndarray:
    """Equal family mass, equal query mass within family, divided among admitted variants."""
    if tokens.shape != families.shape or tokens.ndim != 1 or not len(tokens):
        raise ValueError("Need nonempty aligned variant tokens/families")
    for token in np.unique(tokens):
        if len(set(families[tokens == token])) != 1:
            raise ValueError("One query cannot belong to different families")
    weights = np.empty(len(tokens), np.float32)
    unique_families = np.unique(families)
    for family in unique_families:
        family_tokens = np.unique(tokens[families == family])
        for token in family_tokens:
            mask = tokens == token
            weights[mask] = 1 / (len(unique_families) * len(family_tokens) * int(mask.sum()))
    return weights


def fold_masks(
    clean: dict[str, np.ndarray], augmented: dict[str, np.ndarray], indices: np.ndarray, family: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exclude every variant of a held-out family and check physical-group separation."""
    test = clean["family"] == family
    train = ~test
    if not test.any() or not train.any() or set(clean["group"][train]) & set(clean["group"][test]):
        raise ValueError("Empty fold or acquisition-group leakage")
    if not np.array_equal(clean["token"][indices], augmented["token"]) or not np.array_equal(
        clean["group"][indices], augmented["group"]
    ):
        raise ValueError("Augmented fold mapping changed")
    aug_train, aug_test = train[indices], test[indices]
    if set(augmented["group"][aug_train]) & set(clean["group"][test]):
        raise ValueError("Augmented acquisition-group leakage")
    return train, test, aug_train, aug_test


def verify_sealed_artifacts(root: Path, receipt: dict) -> None:
    """Verify a complete expansion seal before reading any target array."""
    if (
        receipt.get("schema") != "phase_native_context_augmentations_v1"
        or receipt.get("status") != "COMPLETE"
        or receipt.get("queries") != 149
        or receipt.get("augmentations") != 447
        or receipt.get("clean_exact") is not True
        or receipt.get("disk_roundtrip_exact") is not True
        or receipt.get("targets_read") is not False
        or receipt.get("optimizer_updates") != 0
        or receipt.get("native_order") != ["F0", "RGB", "event"]
    ):
        raise ValueError("Expansion is incomplete, unverified or outside the fixed contract")
    artifacts = receipt.get("artifacts", {})
    if not {"FREEZE.json", "CLEAN.npz", "AUGMENTATIONS.npz"}.issubset(artifacts):
        raise ValueError("Missing sealed expansion artifacts")
    for relative, digest in artifacts.items():
        path = (root / relative).resolve()
        if (
            not path.is_relative_to(root.resolve())
            or not path.is_file()
            or file_sha(path) != digest
        ):
            raise ValueError(f"Changed, missing or escaping expansion artifact: {relative}")
    if file_sha(root / "FREEZE.json") != receipt.get("freeze_sha256"):
        raise ValueError("Expansion freeze differs")


def validate_expansion(cache: Path, population: Path, original_config: Path) -> tuple[dict, dict]:
    """Validate aggregate/per-query provenance and population identities, without labels."""
    complete = read(cache / "COMPLETE.json")
    verify_sealed_artifacts(cache, complete)
    freeze, original = read(cache / "FREEZE.json"), read(original_config)
    if complete["inputs"] != freeze["inputs"]:
        raise ValueError("Expansion input binding differs")
    for relative, digest in freeze["source_hashes"].items():
        source = Path("executed_source") / relative
        path = (cache / source).resolve()
        if (
            not path.is_relative_to(cache.resolve())
            or str(source) not in complete["artifacts"]
            or not path.is_file()
            or file_sha(path) != digest
        ):
            raise ValueError("Missing or changed archived extraction source")
    if not freeze["source_hashes"]:
        raise ValueError("Missing extraction source binding")
    for path in (population, original_config):
        if freeze["inputs"].get(str(path)) != file_sha(path):
            raise ValueError("Expansion uses a different population or exclusion contract")
    with np.load(population, allow_pickle=False) as data:
        tokens, groups, families = data["token"], data["group"], data["family"]
    excluded = set(original["excluded_road"] + original["excluded_slider"])
    if (
        len(tokens) != 2608
        or len(set(tokens)) != 2608
        or len(set(groups)) != 25
        or len(set(families)) != 8
        or set(groups) & excluded
        or sorted(set(groups)) != original["fit_groups"]
    ):
        raise ValueError("Original eligible population or exclusions changed")
    clean, augmented = load_arrays(cache / "CLEAN.npz"), load_arrays(cache / "AUGMENTATIONS.npz")
    if set(clean) != {"token", "group", "native", "status", "reason"}:
        raise ValueError("Clean cache schema differs")
    if (
        clean["native"].shape != (149, 3)
        or clean["native"].dtype != np.float32
        or any(clean[k].shape != (149,) for k in ("token", "group", "status", "reason"))
        or len(set(clean["token"])) != 149
    ):
        raise ValueError("Clean cache dimensions or unique identities differ")
    validate_variant_identity(augmented)
    if len(augmented["token"]) != 447 or not np.array_equal(
        augmented["token"], np.repeat(clean["token"], 3)
    ):
        raise ValueError("Augmentation ordering differs from clean identity")
    if not np.array_equal(augmented["keep"], np.tile([1, 2, 3], 149)):
        raise ValueError("Augmentation keep ordering differs")
    if (
        clean["token"].tolist() != freeze["selected_tokens"]
        or clean["group"].tolist() != freeze["selected_groups"]
    ):
        raise ValueError("Frozen selection differs from cache")
    lookup = {str(token): i for i, token in enumerate(tokens)}
    for token, group in zip(clean["token"], clean["group"], strict=True):
        if str(token) not in lookup or groups[lookup[str(token)]] != group:
            raise ValueError("Selected query missing or acquisition group swapped")
    for group in set(groups):
        expected = 77 if group == "Slider-500" else 3
        if int(np.sum(clean["group"] == group)) != expected:
            raise ValueError("Selection must contain three per road group and all77Slider")
    if set(clean["token"][clean["group"] == "Slider-500"]) != {
        f"Slider-500:{i}" for i in range(77)
    }:
        raise ValueError("Slider selection is incomplete")
    for key in ("group", "status", "reason"):
        if not np.array_equal(augmented[key], np.repeat(clean[key], 3)):
            raise ValueError("Variant route/group identity differs from clean cache")
    for i, token in enumerate(clean["token"]):
        relative = Path("queries") / hashlib.sha256(str(token).encode()).hexdigest()[:24]
        destination = cache / relative
        required = [
            relative / name
            for name in (
                "COMPLETE.json",
                "PARITY.json",
                "NATIVE.npz",
                "OBSERVATION_SEAL.json",
                "QUERY.json",
            )
        ]
        if any(str(path) not in complete["artifacts"] for path in required):
            raise ValueError("Per-query evidence is not sealed")
        query = read(destination / "COMPLETE.json")
        binding = query["binding"]

        def canonical(value: object) -> str:
            return hashlib.sha256(
                json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()

        if (
            binding["token"] != str(token)
            or binding["group"] != str(clean["group"][i])
            or query.get("clean_exact") is not True
            or query.get("disk_roundtrip_exact") is not True
            or binding["model_sha256"] != canonical(freeze["upstream"])
            or binding["source_sha256"] != canonical(freeze["source_hashes"])
            or binding["query_sha256"] != canonical(read(destination / "QUERY.json"))
            or binding["query_us"] != read(destination / "QUERY.json")["anchor_us"]
            or query["status"] != clean["status"][i]
            or query["reason"] != clean["reason"][i]
        ):
            raise ValueError("Per-query identity/parity differs")
        for name, digest in query["artifacts"].items():
            path = (destination / name).resolve()
            if not path.is_relative_to(destination.resolve()) or file_sha(path) != digest:
                raise ValueError("Per-query artifact seal differs")
        parity = read(destination / "PARITY.json")
        if parity.get("clean_exact") is not True or parity.get("disk_roundtrip_exact") is not True:
            raise ValueError("Per-query clean disk parity did not pass")
        load_bound_observations(destination, binding)
        with np.load(destination / "NATIVE.npz", allow_pickle=False) as data:
            expected = np.vstack((clean["native"][i], augmented["native"][3 * i : 3 * i + 3]))
            if set(data.files) != {"native"} or not np.array_equal(
                data["native"], expected, equal_nan=True
            ):
                raise ValueError("Aggregate native values differ from sealed query replay")
    return augmented, {
        "complete_sha256": file_sha(cache / "COMPLETE.json"),
        "freeze_sha256": file_sha(cache / "FREEZE.json"),
        "queries": 149,
        "variants": 447,
    }


def validate_source_closure(path: Path, cache: Path) -> None:
    """Require the supplemental numerical-source proof, bound to this exact expansion."""
    if not path.is_file():
        raise ValueError("Missing completed source closure")
    closure = read(path)
    if (
        closure.get("schema") != "phase_native_source_closure_v1"
        or closure.get("status") != "PASS"
        or closure.get("snapshot_to_completion_unchanged") is not True
        or closure.get("frozen_numerical_sources_match") is not True
        or closure.get("labels_opened") is not False
        or closure.get("producer_calls") != 0
        or closure.get("extraction_root") != str(cache.resolve())
        or closure.get("expanded_complete_sha256") != file_sha(cache / "COMPLETE.json")
        or closure.get("expanded_freeze_sha256") != file_sha(cache / "FREEZE.json")
        or closure.get("snapshot_sha256") != file_sha(path.parent / "SNAPSHOT.json")
    ):
        raise ValueError("Source closure failed or expansion/snapshot identity differs")
    complete = read(cache / "COMPLETE.json")
    snapshot = read(path.parent / "SNAPSHOT.json")
    dependencies = closure.get("dependency_sha256", {})
    if (
        set(dependencies) != set(DEPENDENCIES)
        or snapshot.get("dependency_sha256") != dependencies
        or closure.get("original_artifacts_verified") != len(complete["artifacts"])
        or set(closure.get("dependency_equivalence", {})) != set(dependencies)
    ):
        raise ValueError("Source closure dependency/artifact inventory differs")
    verify_snapshot(path.parent, dependencies)
    for name, item in closure["dependency_equivalence"].items():
        if (
            item["current_sha256"] != dependencies[name]
            or item["frozen_sha256"] != snapshot["archived_dependency_sha256"][name]
            or item["equivalence"]
            not in {"byte_identical", "ast_and_lexical_tokens_identical_line_wrapping_only"}
            or (
                item["equivalence"] == "byte_identical"
                and item["current_sha256"] != item["frozen_sha256"]
            )
        ):
            raise ValueError("Invalid numerical-source equivalence binding")
    audit_path = Path("scripts/audit_phase_native_source_closure.py")
    if closure.get("audit_source_sha256") != file_sha(audit_path):
        raise ValueError("Source closure audit implementation differs")


def load_training_data(
    population: Path, original_config: Path, cache: Path, source_closure: Path | None = None
) -> tuple[dict, dict, np.ndarray, dict]:
    """Open targets only after the complete source/query/parity chain has passed."""
    augmented, admission = validate_expansion(cache, population, original_config)
    if source_closure is None:
        raise ValueError("Explicit supplemental source closure is required before targets")
    validate_source_closure(source_closure, cache)
    admission["source_closure_sha256"] = file_sha(source_closure)
    clean = load_arrays(population)
    joined, indices = join_augmentations(clean, augmented)
    return clean, joined, indices, admission


def load_historical_iq(
    baseline: Path, population: Path, original_config: Path, tokens: np.ndarray
) -> np.ndarray:
    """Bind the historical control vector for descriptive reproduction checks only."""
    config = read(baseline / "CONFIG.json")
    for path in (population, original_config):
        if config["inputs"].get(str(path)) != file_sha(path):
            raise ValueError("Historical IQ uses a different population or exclusion contract")
    if config["seeds"] != list(SEEDS) or config["epochs"] != EPOCHS:
        raise ValueError("Historical IQ seeds or update budget differs")
    with np.load(baseline / "OOF_PREDICTIONS.npz", allow_pickle=False) as data:
        values = data["IQ"]
        if values.shape != tokens.shape or not np.isfinite(values).all():
            raise ValueError("Historical IQ vector has wrong shape or unavailable estimates")
        if "token" in data and not np.array_equal(data["token"], tokens):
            raise ValueError("Historical IQ ordered tokens differ")
    return values


def admitted(a: dict[str, np.ndarray]) -> np.ndarray:
    """Original IQ admission, including geometry-only observations."""
    return np.isfinite(a["target"]) & (np.isfinite(a["native"]).any(1) | np.isfinite(a["geometry"]))


def fit_native(
    clean: dict,
    augmented: dict,
    train: np.ndarray,
    aug_train: np.ndarray,
    seed: int,
    destination: Path,
) -> ExpertRouter:
    """Use clean-only normalization and replace the original augmentation loss half."""
    destination.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    model = ExpertRouter("quality", "inverse")
    if sum(p.numel() for p in model.parameters()) != 688:
        raise ValueError("Preregistered router capacity differs")
    cmask, amask = train & admitted(clean), aug_train & admitted(augmented)
    cf, af = clean["family"][cmask], augmented["family"][amask]
    if set(cf) != set(af) or not len(cf):
        raise ValueError("Every training family must have admitted clean and restart rows")
    tensors = [
        torch.from_numpy(clean[k][cmask]) for k in ("native", "geometry", "quality", "target")
    ]
    atensors = [
        torch.from_numpy(augmented[k][amask]) for k in ("native", "geometry", "quality", "target")
    ]
    native, geometry, quality, target = tensors
    center, scale = fit_normalization(model.inputs(native, geometry, quality).detach().numpy())
    model.center.copy_(torch.from_numpy(center))
    model.scale.copy_(torch.from_numpy(scale))
    seconds_scale = target.abs().median().clamp(min=0.2)
    cw = np.array([1 / np.sum(cf == family) for family in cf], np.float32)
    clean_mass = torch.from_numpy(cw / cw.sum())
    aug_mass = torch.from_numpy(augmentation_weights(augmented["token"][amask], af))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    binding = {
        "mode": "IQn",
        "seed": seed,
        "coordinate": "inverse",
        "epochs": EPOCHS,
        "fit_groups": sorted(set(clean["group"][train])),
        "clean_rows": int(cmask.sum()),
        "augmented_rows": int(amask.sum()),
        "augmented_queries": len(set(augmented["token"][amask])),
        "seconds_scale_clean_train_only": float(seconds_scale),
        "parameters": 688,
    }
    atomic_receipt(destination / "FIT_CONFIG.json", binding)
    history, started = [], time.perf_counter()

    def objective(values: list[torch.Tensor], mass: torch.Tensor) -> torch.Tensor:
        prediction = model(*values[:3])
        truth = values[3]
        error = (prediction - truth).abs()
        return (mass * (error / truth.abs().clamp(min=0.2) + error / seconds_scale)).sum()

    try:
        for epoch in range(EPOCHS):
            clean_loss, aug_loss = objective(tensors, clean_mass), objective(atensors, aug_mass)
            loss = 0.5 * clean_loss + 0.5 * aug_loss
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite native context loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": float(loss.detach()),
                    "clean": float(clean_loss.detach()),
                    "native": float(aug_loss.detach()),
                }
            )
    except Exception as error:
        torch.save(
            {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()},
            destination / "failed_checkpoint.pt",
        )
        atomic_receipt(
            destination / "FAILED.json", {**binding, "error": repr(error), "loss": history}
        )
        raise
    torch.save(
        {"state_dict": model.state_dict(), "seed": seed, "mode": "IQn", "coordinate": "inverse"},
        destination / "checkpoint.pt",
    )
    atomic_receipt(
        destination / "COMPLETE.json",
        {
            **binding,
            "loss": history,
            "seconds": time.perf_counter() - started,
            "checkpoint_sha256": file_sha(destination / "checkpoint.pt"),
        },
    )
    return model.eval()


def predict(model: torch.nn.Module, arrays: dict, mask: np.ndarray) -> np.ndarray:
    """Evaluate explicit clean or restart rows without updating a model."""
    with torch.inference_mode():
        return model(
            *(torch.from_numpy(arrays[k][mask]) for k in ("native", "geometry", "quality"))
        ).numpy()


def run(args: argparse.Namespace) -> None:
    """Freeze one two-arm experiment; later refit/eAP checks remain separate gated stages."""
    if args.output.exists():
        raise ValueError("Use a new immutable campaign directory")
    clean, augmented, indices, admission = load_training_data(
        args.population, args.original_config, args.cache, args.source_closure
    )
    historical_iq = load_historical_iq(
        args.baseline, args.population, args.original_config, clean["token"]
    )
    if args.preflight_only:
        print("PREFLIGHT PASS: sealed149/447 identities and parity; no fitting")
        return
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    folds = sorted(set(clean["family"]))
    sources = [
        Path(__file__),
        Path("tests/unit/test_phase_native_context_fusion.py"),
        PLAN,
        Path("scripts/train_phase_startup.py"),
        Path("scripts/train_phase_robust.py"),
        Path("operational/phase_startup/model.py"),
        Path("operational/phase_robust/model.py"),
        Path("operational/phase_campaign/evaluation.py"),
        Path("operational/phase_adapt_cache.py"),
        Path("operational/phase_observation_cache/integrity.py"),
        Path("scripts/audit_phase_native_source_closure.py"),
    ]
    source_hashes = {str(p): file_sha(p) for p in sources}
    input_paths = [
        args.population,
        args.original_config,
        args.cache / "COMPLETE.json",
        args.cache / "FREEZE.json",
        args.cache / "AUGMENTATIONS.npz",
        args.cache / "CLEAN.npz",
        args.baseline / "CONFIG.json",
        args.baseline / "OOF_PREDICTIONS.npz",
        args.source_closure,
        args.source_closure.parent / "SNAPSHOT.json",
    ]
    config = {
        "schema": "phase_native_context_fusion_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seeds": list(SEEDS),
        "epochs": EPOCHS,
        "folds": folds,
        "fits": 48,
        "parameters": 688,
        "control": "Original fit_router QA inverse, unchanged single-expert corruption",
        "candidate": "IQn replaces augmented0.5 with real context restarts; geometry unchanged",
        "normalization": "clean fit rows only",
        "augmentation_weights": "equal family/query; inverse admitted variant count",
        "selection": "clean OOF only; road andSlider macro MAE/RAE no worse than IQ; fullcoverage",
        "secondary": "restart OOF diagnostic only, never selection",
        "admission": admission,
        "source_hashes": source_hashes,
        "inputs": {str(p): file_sha(p) for p in input_paths},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "threads": 2,
        "fit_groups": sorted(set(clean["group"])),
        "scope": "eligible exposed development; no benchmark or eAP evaluation",
    }
    atomic_receipt(args.output / "CONFIG.json", config)
    for path in sources:
        destination = args.output / "executed_source" / path.resolve().relative_to(Path.cwd())
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    clean_oof = {
        f"{mode}_seed{seed}": np.full(len(clean["token"]), np.nan, np.float32)
        for mode in ("IQ", "IQn")
        for seed in SEEDS
    }
    aug_oof = {key: np.full(len(augmented["token"]), np.nan, np.float32) for key in clean_oof}
    started, completed = time.perf_counter(), 0
    for family in folds:
        train, test, aug_train, aug_test = fold_masks(clean, augmented, indices, family)
        for mode in ("IQ", "IQn"):
            for seed in SEEDS:
                destination = args.output / "folds" / family / mode / str(seed)
                atomic_receipt(
                    destination / "ADMISSION.json",
                    {
                        "mode": mode,
                        "seed": seed,
                        "fit_groups": sorted(set(clean["group"][train])),
                        "heldout_groups": sorted(set(clean["group"][test])),
                        "fit_token_sha256": hashlib.sha256(
                            clean["token"][train].tobytes()
                        ).hexdigest(),
                        "augmented_fit_pairs": [
                            [str(t), int(k)]
                            for t, k in zip(
                                augmented["token"][aug_train],
                                augmented["keep"][aug_train],
                                strict=True,
                            )
                        ],
                    },
                )
                try:
                    # Each fit owns a fresh subdirectory; the control API remains unchanged.
                    fit_dir = destination / "fit"
                    model = (
                        fit_router(clean, train, "QA", seed, fit_dir, "inverse")
                        if mode == "IQ"
                        else fit_native(clean, augmented, train, aug_train, seed, fit_dir)
                    )
                    key = f"{mode}_seed{seed}"
                    clean_oof[key][test] = predict(model, clean, test)
                    aug_oof[key][aug_test] = predict(model, augmented, aug_test)
                    save_npz(
                        destination / "OOF.npz",
                        {"prediction": clean_oof[key][test], "token": clean["token"][test]},
                    )
                    save_npz(
                        destination / "RESTART_OOF.npz",
                        {
                            "prediction": aug_oof[key][aug_test],
                            "token": augmented["token"][aug_test],
                            "keep": augmented["keep"][aug_test],
                        },
                    )
                except Exception as error:
                    atomic_receipt(
                        destination / "FAILED.json",
                        {"mode": mode, "seed": seed, "family": family, "error": repr(error)},
                    )
                    raise
                completed += 1
                print(
                    f"FIT {completed}/48 {family} {mode} seed={seed} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    for mode in ("IQ", "IQn"):
        clean_oof[mode] = np.mean([clean_oof[f"{mode}_seed{s}"] for s in SEEDS], axis=0)
        aug_oof[mode] = np.mean([aug_oof[f"{mode}_seed{s}"] for s in SEEDS], axis=0)
    atomic_receipt(
        args.output / "IQ_REPRODUCTION.json",
        {
            "informational_only": True,
            "used_for_selection": False,
            "exact": bool(np.array_equal(clean_oof["IQ"], historical_iq)),
            "allclose_rtol1e-6_atol1e-6": bool(
                np.allclose(clean_oof["IQ"], historical_iq, rtol=1e-6, atol=1e-6)
            ),
            "maximum_absolute_difference_seconds": float(
                np.max(np.abs(clean_oof["IQ"] - historical_iq))
            ),
            "historical_oof_sha256": file_sha(args.baseline / "OOF_PREDICTIONS.npz"),
            "identity": "Exact population/config hashes and shape; tokens checked when present",
        },
    )
    save_npz(
        args.output / "OOF_PREDICTIONS.npz",
        {**clean_oof, "token": clean["token"], "group": clean["group"]},
    )
    save_npz(
        args.output / "RESTART_OOF_PREDICTIONS.npz",
        {
            **aug_oof,
            "token": augmented["token"],
            "group": augmented["group"],
            "keep": augmented["keep"],
            "status": augmented["status"],
            "reason": augmented["reason"],
        },
    )
    masks = {
        "all": np.ones(len(clean["token"]), bool),
        "road": clean["family"] != "Slider",
        "slider": clean["family"] == "Slider",
    }
    reports = {
        name: {domain: scored(clean, values, mask) for domain, mask in masks.items()}
        for name, values in clean_oof.items()
    }
    secondary = {
        name: {
            str(keep): scored(augmented, values, augmented["keep"] == keep) for keep in (1, 2, 3)
        }
        for name, values in aug_oof.items()
    }
    gates = {
        f"{domain}_{key}": reports["IQn"][domain][key] <= reports["IQ"][domain][key]
        for domain in ("road", "slider")
        for key in ("macro_record_mae_seconds", "macro_record_relative_error_pct")
    }
    full = all(reports[mode][domain]["coverage"] == 1 for mode in ("IQ", "IQn") for domain in masks)
    for path, digest in {**config["inputs"], **source_hashes}.items():
        if file_sha(Path(path)) != digest:
            raise ValueError("Frozen source/input changed during experiment")
    atomic_receipt(
        args.output / "SELECTION.json",
        {
            "selected": "IQn" if full and all(gates.values()) else None,
            "gates": gates,
            "full_coverage": full,
            "report": reports,
            "secondary_restart_report": secondary,
            "secondary_used_for_selection": False,
            "fits_completed": completed,
            "promoted": False,
            "config_sha256": file_sha(args.output / "CONFIG.json"),
            "oof_sha256": file_sha(args.output / "OOF_PREDICTIONS.npz"),
            "restart_oof_sha256": file_sha(args.output / "RESTART_OOF_PREDICTIONS.npz"),
            "elapsed_seconds": time.perf_counter() - started,
            "next_step": "separate preregistered refit/preservation only if every gate passes",
        },
    )
    print(
        f"COMPLETE selected={'IQn' if full and all(gates.values()) else None}; "
        "no automatic refit or promotion",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("population", "original-config", "cache", "source-closure", "baseline", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
