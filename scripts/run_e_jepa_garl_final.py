"""Run the cache-free event-only Garl candidate from screen to frozen evaluation.

The runner deliberately separates training, freezing, label-free EvTTC
prediction, scoring, and offline submission validation. It never uploads a
submission and it never opens EvTTC labels during prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts" / "train_e_jepa_tubelet_lhr.py"
PREDICTOR = ROOT / "scripts" / "predict_garl_evttc_table_vi.py"
SCORER = ROOT / "scripts" / "evaluate_garl_evttc_table_vi.py"
SUBMISSION_VALIDATOR = ROOT / "scripts" / "validate_garlttc_submission.py"
PROFILE_CONFIGS = {
    "screen": ROOT / "configs" / "experiment" / "e_jepa_garl_event_screen_v1.yaml",
    "stable-screen": (
        ROOT
        / "configs"
        / "experiment"
        / "e_jepa_garl_event_dense_level_dynamics_stable_scratch_screen_v3.yaml"
    ),
    "full": ROOT / "configs" / "experiment" / "e_jepa_garl_event_full_v1.yaml",
}
STABLE_TRANSFER_CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "e_jepa_garl_event_dense_level_dynamics_stable_transfer_screen_v3.yaml"
)
PROFILE_SPLITS = {
    "screen": ROOT / "data" / "splits" / "eap_pilot12_v1.json",
    "stable-screen": ROOT / "data" / "splits" / "eap_pilot12_v1.json",
    "full": ROOT / "data" / "splits" / "eap_train40_v1.json",
}
PROFILE_SEEDS = {
    "screen": (7,),
    "stable-screen": (7,),
    "full": (7, 13, 23),
}
PROFILE_OUTPUT_NAMES = {
    "screen": "e_jepa_garl_event_screen_v1",
    "stable-screen": "e_jepa_garl_event_stable_screen_v3",
    "full": "e_jepa_garl_event_full_v1",
}
STAGES = ("train", "freeze", "evttc-predict", "evttc-score", "submission-validate")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_stages(raw: Sequence[str]) -> tuple[str, ...]:
    """Validate and expand the requested ordered stages."""

    values = tuple(raw)
    if values == ("all",):
        return STAGES
    invalid = sorted(set(values).difference(STAGES))
    if invalid:
        raise ValueError(f"Unknown stages: {invalid}")
    if len(set(values)) != len(values):
        raise ValueError("Stages cannot be repeated")
    positions = [STAGES.index(value) for value in values]
    if positions != sorted(positions):
        raise ValueError(f"Stages must follow this order: {list(STAGES)}")
    return values


def training_command(
    *,
    profile: str,
    seed: int,
    eap_root: Path,
    garlttc_root: Path,
    split: Path,
    output_root: Path,
    device: str,
    resume: bool,
    pretrained: Path | None = None,
) -> list[str]:
    """Build one canonical trainer command without a dense-cache argument."""

    command = [
        sys.executable,
        str(TRAINER),
        "--eap-root",
        str(eap_root),
        "--garlttc-root",
        str(garlttc_root),
        "--split",
        str(split),
        "--config",
        str(
            STABLE_TRANSFER_CONFIG
            if profile == "stable-screen" and pretrained is not None
            else PROFILE_CONFIGS[profile]
        ),
        "--output-dir",
        str(output_root / f"seed-{seed}"),
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    if pretrained is not None:
        command.extend(("--pretrained", str(pretrained)))
    if resume:
        command.append("--resume")
    return command


def freeze_candidate(output_root: Path, seeds: Sequence[int]) -> dict[str, Any]:
    """Freeze the best full-profile seed using only Garl validation metrics."""

    candidates: list[dict[str, Any]] = []
    commits: set[str] = set()
    configs: set[str] = set()
    datasets: set[str] = set()
    for seed in seeds:
        summary_path = output_root / f"seed-{seed}" / "summary.json"
        summary = _read_json(summary_path)
        trainer = summary.get("train_config")
        if not isinstance(trainer, Mapping) or trainer.get("run_scope") != "full_candidate":
            raise ValueError(f"Run is not a full_candidate: {summary_path}")
        if summary.get("status") != "completed" or summary.get("git_dirty") is not False:
            raise ValueError(f"Run is incomplete or dirty: {summary_path}")
        if summary.get("downstream_evaluation_eligible") is not True:
            raise ValueError(f"Run is not eligible for downstream evaluation: {summary_path}")
        if int(summary.get("seed", -1)) != seed:
            raise ValueError(f"Seed mismatch in {summary_path}")
        checkpoint = Path(str(summary["best_checkpoint"]))
        if not checkpoint.is_file() or _sha256(checkpoint) != summary.get(
            "best_checkpoint_sha256"
        ):
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
        commits.add(str(summary["git_commit"]))
        configs.add(str(summary["config_hash"]))
        datasets.add(str(summary["dataset_manifest_hash"]))
        candidates.append(
            {
                "seed": seed,
                "score": float(summary["best_score"]),
                "checkpoint": checkpoint.resolve().as_posix(),
                "checkpoint_sha256": _sha256(checkpoint),
                "summary": summary_path.resolve().as_posix(),
                "summary_sha256": _sha256(summary_path),
            }
        )
    if len(candidates) != 3 or len(commits) != 1 or len(configs) != 1 or len(datasets) != 1:
        raise ValueError("Freeze requires three comparable seeds from one commit/config/dataset")
    selected = min(candidates, key=lambda item: (float(item["score"]), int(item["seed"])))
    result = {
        "artifact_type": "e_jepa_garl_event_multiseed_freeze_v1",
        "status": "frozen_for_external_evaluation",
        "created_at": datetime.now(UTC).isoformat(),
        "selection_dataset": "GarlTTC public validation",
        "selection_uses_evttc": False,
        "seeds": list(seeds),
        "git_commit": next(iter(commits)),
        "config_hash": next(iter(configs)),
        "dataset_manifest_hash": next(iter(datasets)),
        "candidates": candidates,
        "selected_seed": selected["seed"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "claim_eligible": False,
        "claim_blocker": "EvTTC Table VI and official eAP/CodaBench evaluation are pending",
    }
    _write_json(output_root / "freeze.json", result)
    return result


def _run(command: Sequence[str], *, dry_run: bool) -> None:
    print(json.dumps(list(command), ensure_ascii=False))
    if not dry_run:
        subprocess.run(list(command), cwd=ROOT, check=True)


def _checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint.resolve()
    freeze = _read_json(args.output_root.resolve() / "freeze.json")
    return Path(str(freeze["selected_checkpoint"]))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILE_CONFIGS), default="screen")
    parser.add_argument("--stages", nargs="+", default=["train"])
    parser.add_argument("--eap-root", type=Path)
    parser.add_argument("--garlttc-root", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--pretrained",
        type=Path,
        help="Optional exact Dense Level-Dynamics checkpoint for a training-only profile.",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--evttc-config", type=Path)
    parser.add_argument(
        "--evttc-protocol",
        type=Path,
        default=ROOT / "data" / "protocols" / "garl_evttc_table_vi_v1.yaml",
    )
    parser.add_argument("--evttc-predictions", type=Path)
    parser.add_argument("--evttc-targets", type=Path)
    parser.add_argument("--evttc-metrics", type=Path)
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--sample-submission", type=Path)
    parser.add_argument("--submission-validation", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        stages = resolve_stages(args.stages)
        seeds = tuple(args.seeds or PROFILE_SEEDS[args.profile])
        expected_seeds = PROFILE_SEEDS[args.profile]
        if args.profile == "full" and tuple(seeds) != expected_seeds:
            raise ValueError(f"Full profile requires predeclared seeds {list(expected_seeds)}")
        if args.profile == "full" and args.pretrained is not None:
            raise ValueError(
                "Full multiseed transfer requires one seed-matched checkpoint per run; "
                "do not pass one --pretrained checkpoint to all full seeds."
            )
        output_root = (
            args.output_root
            or ROOT / "artifacts" / "runs" / PROFILE_OUTPUT_NAMES[args.profile]
        ).resolve()
        args.output_root = output_root
        split = (args.split or PROFILE_SPLITS[args.profile]).resolve()
        if "train" in stages:
            if args.eap_root is None or args.garlttc_root is None:
                raise ValueError("train requires --eap-root and --garlttc-root")
            for seed in seeds:
                _run(
                    training_command(
                        profile=args.profile,
                        seed=seed,
                        eap_root=args.eap_root.resolve(),
                        garlttc_root=args.garlttc_root.resolve(),
                        split=split,
                        output_root=output_root,
                        device=args.device,
                        resume=args.resume,
                        pretrained=(
                            args.pretrained.resolve() if args.pretrained is not None else None
                        ),
                    ),
                    dry_run=args.dry_run,
                )
        if "freeze" in stages:
            if args.profile != "full":
                raise ValueError("Only the three-seed full profile can be frozen")
            if args.dry_run:
                print(json.dumps(["freeze", str(output_root / "freeze.json")]))
            else:
                freeze_candidate(output_root, seeds)
        if "evttc-predict" in stages:
            if args.evttc_config is None or args.evttc_predictions is None:
                raise ValueError("evttc-predict requires --evttc-config and --evttc-predictions")
            _run(
                [
                    sys.executable,
                    str(PREDICTOR),
                    "--checkpoint",
                    str(_checkpoint(args)),
                    "--config",
                    str(args.evttc_config.resolve()),
                    "--protocol",
                    str(args.evttc_protocol.resolve()),
                    "--output",
                    str(args.evttc_predictions.resolve()),
                ],
                dry_run=args.dry_run,
            )
        if "evttc-score" in stages:
            if (
                args.evttc_predictions is None
                or args.evttc_targets is None
                or args.evttc_metrics is None
            ):
                raise ValueError(
                    "evttc-score requires --evttc-predictions, --evttc-targets and "
                    "--evttc-metrics"
                )
            _run(
                [
                    sys.executable,
                    str(SCORER),
                    "score",
                    "--predictions",
                    str(args.evttc_predictions.resolve()),
                    "--targets",
                    str(args.evttc_targets.resolve()),
                    "--protocol",
                    str(args.evttc_protocol.resolve()),
                    "--output",
                    str(args.evttc_metrics.resolve()),
                ],
                dry_run=args.dry_run,
            )
        if "submission-validate" in stages:
            if (
                args.submission is None
                or args.sample_submission is None
                or args.submission_validation is None
            ):
                raise ValueError(
                    "submission-validate requires --submission, --sample-submission and "
                    "--submission-validation"
                )
            _run(
                [
                    sys.executable,
                    str(SUBMISSION_VALIDATOR),
                    "--submission",
                    str(args.submission.resolve()),
                    "--sample",
                    str(args.sample_submission.resolve()),
                    "--output",
                    str(args.submission_validation.resolve()),
                ],
                dry_run=args.dry_run,
            )
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
