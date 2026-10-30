"""Fixed leave-family-out F1 experiments; current-fit isolation, historical exposure declared."""

from __future__ import annotations

import argparse
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.evttc_transfer_audit.common import canonical_sha, file_sha
from operational.final_refit.training import PhaseOnlyFinal
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_adapt_training import configure_cuda, fit_head, load_head
from operational.phase_campaign.data import load_data, read
from operational.phase_campaign.evaluation import (
    metrics,
    paired_group_bootstrap,
    phase_interval_metrics,
)
from operational.phase_campaign.protocol import family_folds, fold_indices
from scripts.final_train40_transfer import HEAD, verify_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "candidate", "protocol", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    configure_cuda()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    values, rows, receipt = load_data(args.data)
    records = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))["records"]
    folds = family_folds(records)
    paths, parent = verify_candidate(args.candidate)
    initial = torch.load(paths[HEAD], map_location="cpu", weights_only=False)["model_state_dict"]
    sources = {
        str(p).replace("\\", "/"): file_sha(p)
        for p in [
            Path(__file__).relative_to(Path.cwd()),
            Path("operational/phase_adapt_training.py"),
            Path("operational/phase_adapt.py"),
            Path("operational/final_refit/training.py"),
            Path("src/e_jepa_ttc/rgb_port/fusion.py"),
            Path("src/e_jepa_ttc/simplex_t/phase.py"),
            *sorted(Path("operational/phase_campaign").glob("*.py")),
        ]
    }
    recipe = {
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "max_epochs": 30,
        "max_updates": 2000,
        "grad_clip": 1.0,
        "selection": "last_endpoint",
        "domain_weight": "equal",
        "group_weight": "equal_within_domain",
    }
    config = {
        "recipe": recipe,
        "folds": folds,
        "seeds": [7, 13, 21],
        "parent": parent,
        "data_receipt_sha256": file_sha(args.data / "COMPLETE.json"),
        "protocol_sha256": file_sha(args.protocol),
        "sources": sources,
        "scope": "held_out_for_current_head_fit_historical_upstream_exposure_unresolved",
        "group_unit": "scenario_family_proxy_not_certified_acquisition",
        "promotion": "exploratory_no_practical_margin_or_external_independence",
    }
    atomic_receipt(args.output / "CONFIG.json", config)
    atomic_receipt(
        args.output / "ENVIRONMENT.json",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        },
    )
    baseline = PhaseOnlyFinal().eval()
    baseline.load_state_dict(initial)
    all_seeds = {}
    for seed in config["seeds"]:
        collected = []
        for fold in folds:
            train, test = fold_indices(values, fold)
            destination = args.output / f"seed_{seed}" / fold["family"]
            binding = {
                "campaign": canonical_sha(config),
                "parent": parent,
                "fold": fold,
                "train_tokens_sha256": canonical_sha(values["sample_token"][train].tolist()),
                "test_tokens_sha256": canonical_sha(values["sample_token"][test].tolist()),
            }
            training = destination / "fit"
            current_recipe = {**recipe, "seed": seed}
            if not (training / "COMPLETE.json").exists():
                print(
                    f"START seed={seed} heldout={fold['family']} "
                    f"train={len(train)} test={len(test)}",
                    flush=True,
                )
                fit_head(
                    initial,
                    {k: v[train] for k, v in values.items()},
                    current_recipe,
                    training,
                    device="cuda",
                    binding=binding,
                )
            result = read(training / "COMPLETE.json")
            checkpoint = training / "checkpoint.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if payload["binding"] != binding or payload["recipe"] != current_recipe:
                raise ValueError("Fold checkpoint binding differs")
            model = load_head(checkpoint, result["checkpoint_sha256"])
            with torch.inference_mode():
                x = torch.from_numpy(values["features"][test])
                w = torch.from_numpy(values["rgb_width_logits"][test])
                before, after = baseline(x, w), model(x, w)
                target_phase = values["target_phase"][test]
                truth = phase_to_ttc(torch.from_numpy(target_phase)).numpy()
            predictions = {
                "indices": test,
                "target": truth,
                "target_phase": target_phase,
                "group": values["group_id"][test],
                "family": np.full(len(test), fold["family"]),
                "sample_token": values["sample_token"][test],
                "F0": before["ttc"].numpy(),
                "F1": after["ttc"].numpy(),
                "F0_q10": before["q10"].numpy(),
                "F0_q90": before["q90"].numpy(),
                "F1_q10": after["q10"].numpy(),
                "F1_q90": after["q90"].numpy(),
            }
            save_npz(destination / "heldout_predictions.npz", predictions)
            collected.append(predictions)
            atomic_receipt(
                destination / "HELDOUT.json",
                {
                    "binding": binding,
                    "F0": metrics(truth, predictions["F0"]),
                    "F1": metrics(truth, predictions["F1"]),
                    "predictions_sha256": file_sha(destination / "heldout_predictions.npz"),
                },
            )
            print(f"DONE seed={seed} heldout={fold['family']}", flush=True)
        merged = {key: np.concatenate([part[key] for part in collected]) for key in collected[0]}
        if len(set(merged["sample_token"])) != len(merged["sample_token"]):
            raise ValueError("Repeated heldout query")
        save_npz(args.output / f"seed_{seed}" / "heldout_predictions.npz", merged)
        report = {"models": {}, "by_recording": {}, "intervals": {}, "regimes": {}}
        names = sorted(set(merged["group"]))
        families = np.array(
            [merged["family"][np.flatnonzero(merged["group"] == name)[0]] for name in names]
        )
        for model_name in ("F0", "F1"):
            report["models"][model_name] = metrics(merged["target"], merged[model_name])
            report["by_recording"][model_name] = {
                str(name): metrics(
                    merged["target"][merged["group"] == name],
                    merged[model_name][merged["group"] == name],
                )
                for name in names
            }
            report["models"][model_name]["macro_recording_relative_error"] = float(
                np.mean(
                    [
                        v["relative_absolute_error_pct"]
                        for v in report["by_recording"][model_name].values()
                    ]
                )
            )
            report["intervals"][model_name] = phase_interval_metrics(
                merged["target_phase"], merged[model_name + "_q10"], merged[model_name + "_q90"]
            )
            report["regimes"][model_name] = {}
            for name, mask in {
                "negative": merged["target"] < 0,
                "positive_under_1s": (merged["target"] > 0) & (merged["target"] < 1),
                "1_to_4s": (merged["target"] >= 1) & (merged["target"] < 4),
                "over_4s": merged["target"] >= 4,
            }.items():
                report["regimes"][model_name][name] = metrics(
                    merged["target"][mask], merged[model_name][mask]
                )
        report["paired_family_bootstrap"] = paired_group_bootstrap(
            np.array(
                [report["by_recording"]["F0"][str(n)]["relative_absolute_error_pct"] for n in names]
            ),
            np.array(
                [report["by_recording"]["F1"][str(n)]["relative_absolute_error_pct"] for n in names]
            ),
            families,
        )
        all_seeds[str(seed)] = report
        atomic_receipt(args.output / f"seed_{seed}" / "REPORT.json", report)
    if any(file_sha(Path(name)) != digest for name, digest in sources.items()):
        raise ValueError("Campaign sources changed during execution")
    atomic_receipt(
        args.output / "REPORT.json",
        {"config": config, "seeds": all_seeds, "data_population": receipt, "status": "COMPLETE"},
    )
    print("COMPLETE: all family holdouts and three seeds", flush=True)


if __name__ == "__main__":
    main()
