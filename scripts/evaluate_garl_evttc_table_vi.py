"""Score a frozen EvTTC prediction payload against a separate TTC file.

The prediction payload is deliberately checked before the target file is read.
This keeps the zero-shot predict/score boundary explicit and prevents a local
evaluation artifact from silently becoming a label-aware prediction run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from e_jepa_ttc.data.evttc_garl_adapter import reject_labels_from_predict_payload
from e_jepa_ttc.evaluation.garl_evttc_zero_shot import score_zero_shot_predictions
from e_jepa_ttc.utils.io import write_structured

if __package__:
    from scripts.table_vi_label_free import canonical_token_schema, read_mapping
else:
    from table_vi_label_free import (  # pyright: ignore[reportMissingImports]
        canonical_token_schema,
        read_mapping,
    )

_FORBIDDEN_ROW_KEYS = frozenset(
    {"ttc", "ttc_s", "frame_ttc", "target_ttc", "target_ttc_s", "gt_ttc", "future_labels"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return value


def _prediction_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    reject_labels_from_predict_payload(payload)
    rows = payload.get("predictions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Prediction payload must contain a non-empty predictions list.")
    output: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise ValueError(f"Prediction row {index} is not an object.")
        leaked = sorted(_FORBIDDEN_ROW_KEYS.intersection(raw))
        if leaked:
            raise ValueError(f"Prediction row {index} contains forbidden labels: {leaked}")
        if "predicted_ttc_s" not in raw:
            raise ValueError(f"Prediction row {index} has no predicted_ttc_s.")
        output.append(dict(raw))
    return output


def _target_rows(path: Path) -> dict[tuple[str, str, str, int], float]:
    payload = _read(path)
    rows = payload.get("targets")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Target payload must contain a non-empty targets list.")
    output: dict[tuple[str, str, str, int], float] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise ValueError(f"Target row {index} is not an object.")
        identity = (
            str(raw.get("sequence_id", "")),
            str(raw.get("sample_token", "")),
            str(raw.get("track_id", "")),
            int(raw.get("timestamp_us", 0)),
        )
        if identity in output:
            raise ValueError(f"Duplicate target identity: {identity}")
        output[identity] = float(raw["target_ttc_s"])
    return output


def score(predictions_path: Path, targets_path: Path, output_path: Path) -> dict[str, Any]:
    """Score predictions after independently validating the predict payload."""

    prediction_payload = _read(predictions_path)
    rows = _prediction_rows(prediction_payload)
    targets = _target_rows(targets_path)
    identities = [
        (
            str(row.get("sequence_id", "")),
            str(row.get("sample_token", "")),
            str(row.get("track_id", "")),
            int(row.get("timestamp_us", 0)),
        )
        for row in rows
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("Prediction payload contains duplicate sample identities.")
    missing = [identity for identity in identities if identity not in targets]
    if missing:
        raise ValueError(f"Target payload is missing {len(missing)} prediction identities.")
    truth = [targets[identity] for identity in identities]
    predicted = [float(row["predicted_ttc_s"]) for row in rows]
    metrics = score_zero_shot_predictions(truth, predicted)
    result: dict[str, Any] = {
        "artifact_type": "garl_evttc_table_vi_score_v1",
        "schema_version": "v1",
        "evidence_type": "separate_predict_and_score_payloads",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": "garl_signed_v1",
        "predictions_path": predictions_path.as_posix(),
        "predictions_sha256": _sha256(predictions_path),
        "targets_path": targets_path.as_posix(),
        "targets_sha256": _sha256(targets_path),
        "sample_count": len(rows),
        "training_updates_on_target_dataset": 0,
        "benchmark10_opened": False,
        "metrics": metrics,
    }
    write_structured(output_path, result)
    return result


def _validate_protocol_preflight(protocol_path: Path) -> dict[str, Any]:
    """Validate declarations that keep Table VI prediction label-free."""

    protocol = read_mapping(protocol_path)
    canonical_token_schema(protocol, location="protocol")
    if protocol.get("predict_score_separation") is not True:
        raise ValueError("Protocol must declare predict_score_separation=true.")
    if protocol.get("labels_used_by_predict") is not False:
        raise ValueError("Protocol must declare labels_used_by_predict=false.")
    if protocol.get("selection_uses_evttc") is not False:
        raise ValueError("Protocol must declare selection_uses_evttc=false.")
    return protocol


def predict_preflight(
    checkpoint_path: Path,
    protocol_path: Path,
    manifest_paths: Sequence[Path],
    output_path: Path,
) -> None:
    """Reject PLAN's incomplete predict alias before opening inputs or a checkpoint."""

    del checkpoint_path, manifest_paths, output_path
    protocol = _validate_protocol_preflight(protocol_path)
    if protocol.get("zero_shot_completed") is not False:
        raise ValueError("Protocol must explicitly declare zero_shot_completed=false.")
    raise ValueError(
        "The PLAN.md 'predict' alias is preflight-only: raw --manifests do not define "
        "the frozen model input schema or normalization. Provide a real inference config "
        "and a verified label-free input manifest via "
        "scripts/predict_garl_evttc_table_vi.py --checkpoint ... --config ... "
        "--protocol ... --output .... No zero-shot prediction was run."
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--targets", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="command")

    predict_parser = commands.add_parser(
        "predict",
        help="Preflight the PLAN.md prediction alias without opening labels.",
    )
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--protocol", type=Path, required=True)
    predict_parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    predict_parser.add_argument("--output", type=Path, required=True)

    score_parser = commands.add_parser("score", help="Score a separate prediction payload.")
    score_parser.add_argument("--predictions", type=Path, required=True)
    score_parser.add_argument("--targets", type=Path)
    score_parser.add_argument("--protocol", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "predict":
        try:
            predict_preflight(args.checkpoint, args.protocol, args.manifests, args.output)
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
        raise AssertionError("predict_preflight must not return")
    if args.command == "score":
        try:
            _validate_protocol_preflight(args.protocol)
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
        if args.targets is None:
            parser.error(
                "The PLAN.md 'score' alias requires explicit --targets; the portable "
                "protocol intentionally contains no target path. No score was computed."
            )
        result = score(args.predictions, args.targets, args.output)
    else:
        missing = [
            option
            for option, value in (
                ("--predictions", args.predictions),
                ("--targets", args.targets),
                ("--output", args.output),
            )
            if value is None
        ]
        if missing:
            parser.error(
                "legacy score mode requires " + ", ".join(missing) + "; or use predict/score"
            )
        result = score(args.predictions, args.targets, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
