"""Localize original native replay drift before changing any platform or tolerance."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np

from e_jepa_ttc.rgb_port.features import EVENT_PHASE17_NAMES, RGB_PHASE17_NAMES
from operational.final_refit.test12 import TRUSTED_INPUTS_SHA256
from operational.phase_robust.bundle import verified_upstream_paths
from operational.release_research.protocol import ResearchRun, sha256, write_json
from scripts.release_research import read, sources, verify_complete


def audit(capture: Path, baseline: Path, upstream: Path, output: Path) -> None:
    """Compare archived producer observations column by column without reading targets."""
    verify_complete(capture)
    freeze = read(capture / "FREEZE.json")
    if freeze["config"]["targets_read"] or freeze["config"]["candidate_evaluated"]:
        raise ValueError("Only original label-free captures are admitted")
    manifest = read(capture / "OBSERVATIONS.json")
    queries = {q["sample_token"]: q for q in read(capture / "QUERIES.json")["queries"]}
    parent = read(upstream / "BUNDLE.json")["parent"]
    receipt = read(baseline / "COMPLETE.json")
    if (
        receipt["status"] != "COMPLETE"
        or receipt["parent"] != parent
        or receipt["inputs_sha256"] != TRUSTED_INPUTS_SHA256
    ):
        raise ValueError("Native reference ancestry differs")
    paths = verified_upstream_paths(upstream)
    archives = [
        a
        for a in receipt["archives"].values()
        if any(q["sequence_id"] in Path(a["path"]).name for q in queries.values())
    ]
    inputs = {
        **sources(),
        "source:scripts/release_native_drift_audit.py": Path(__file__),
        "source:src/e_jepa_ttc/rgb_port/features.py": Path("src/e_jepa_ttc/rgb_port/features.py"),
        "capture_complete": capture / "COMPLETE.json",
        "capture_manifest": capture / "OBSERVATIONS.json",
        "capture_queries": capture / "QUERIES.json",
        "baseline_complete": baseline / "COMPLETE.json",
        **{"normalizer:" + m: paths[m + "_normalizer"] for m in ("event", "rgb")},
        **{"archive:" + str(i): Path(a["path"]) for i, a in enumerate(archives)},
    }
    for modality, items in manifest.items():
        for identity, item in items.items():
            inputs[modality + ":" + identity] = capture / item["folder"] / "PAIR_INPUTS.npz"
    run = ResearchRun(
        output,
        inputs,
        {
            "experiment_id": "native-original-column-drift-v1",
            "targets_read": False,
            "optimizer_updates": 0,
            "candidate_evaluated": False,
            "scope": "platform diagnosis, not accuracy evaluation or tolerance selection",
        },
    )
    try:
        reference: dict[str, dict] = {"event": {}, "rgb": {}}
        seen = set()
        for entry in archives:
            path = Path(entry["path"])
            if sha256(path) != entry["sha256"]:
                raise ValueError("Historical archive changed")
            with tarfile.open(path) as archive:

                def member(name: str) -> bytes:
                    stream = archive.extractfile(name)
                    if stream is None:
                        raise ValueError("Missing archived member")
                    return stream.read()

                names = archive.getnames()
                admission = json.loads(
                    member(next(n for n in names if n.endswith(".admission.json")))
                )
                if (
                    admission["model_seal_sha256"] != parent["model_seal_sha256"]
                    or admission["inputs_sha256"] != TRUSTED_INPUTS_SHA256
                ):
                    raise ValueError("Sequence admission differs")
                for name in names:
                    if "/diagnostics/" not in name or not name.endswith(".json"):
                        continue
                    metadata = json.loads(member(name))
                    token = metadata["sample_token"]
                    if token not in queries:
                        continue
                    if (
                        token in seen
                        or metadata["private_targets_read"]
                        or metadata["run_binding_sha256"] != admission["run_binding_sha256"]
                    ):
                        raise ValueError("Invalid archived diagnostic")
                    seen.add(token)
                    if any(
                        metadata[k] != queries[token][k]
                        for k in (
                            "event_history",
                            "rgb_history",
                            "query_time_us",
                            "sequence_id",
                            "track_id",
                        )
                    ):
                        raise ValueError("History identity differs")
                    payload = member(name.removesuffix(".json") + ".npz")
                    if hashlib.sha256(payload).hexdigest() != metadata["npz_sha256"]:
                        raise ValueError("Diagnostic hash differs")
                    with np.load(io.BytesIO(payload), allow_pickle=False) as stored:
                        for modality in ("event", "rgb"):
                            for i, identity in enumerate(stored[modality + "_new_ids"]):
                                identity = str(identity)
                                if identity not in manifest[modality]:
                                    continue
                                value = {
                                    "raw17": stored[modality + "_new_raw"][i],
                                    "expert3": stored[modality + "_new_experts"][i],
                                }
                                if identity in reference[modality]:
                                    for key in value:
                                        np.testing.assert_array_equal(
                                            value[key], reference[modality][identity][key]
                                        )
                                reference[modality][identity] = value
        if seen != set(queries):
            raise ValueError("Missing query diagnostics")
        rows = []
        summaries = {}
        for modality, names in (("event", EVENT_PHASE17_NAMES), ("rgb", RGB_PHASE17_NAMES)):
            if set(reference[modality]) != set(manifest[modality]):
                raise ValueError(
                    "Original observations are not fully covered by archived diagnostics"
                )
            norm = read(paths[modality + "_normalizer"])
            scale = np.array(norm["scale"], dtype=np.float32)
            mean = np.array(norm["mean"], dtype=np.float32)
            differences, normalized = [], []
            for identity, item in manifest[modality].items():
                filename = capture / item["folder"] / "PAIR_INPUTS.npz"
                if sha256(filename) != item["pair_sha256"]:
                    raise ValueError("Captured pair changed")
                with np.load(filename, allow_pickle=False) as captured:
                    actual = captured["call0.raw17"].reshape(17)
                    expert = captured["call0.expert3"].reshape(3)
                expected = reference[modality][identity]["raw17"].reshape(17)
                difference = actual.astype(float) - expected.astype(float)
                normalized_difference = ((actual - mean) / scale).astype(float) - (
                    (expected - mean) / scale
                ).astype(float)
                differences.append(np.abs(difference))
                normalized.append(np.abs(normalized_difference))
                rows.append(
                    {
                        "modality": modality,
                        "identity": identity,
                        "raw_difference": dict(zip(names, difference.tolist(), strict=True)),
                        "normalized_difference": dict(
                            zip(names, normalized_difference.tolist(), strict=True)
                        ),
                        "expert_max_abs": float(
                            np.max(np.abs(expert - reference[modality][identity]["expert3"]))
                        ),
                    }
                )
            raw_max = np.max(differences, axis=0)
            norm_max = np.max(normalized, axis=0)
            summaries[modality] = {
                "observations": len(differences),
                "columns": {
                    name: {
                        "raw_max_abs": float(raw_max[i]),
                        "normalized_max_abs": float(norm_max[i]),
                        "scale": float(scale[i]),
                    }
                    for i, name in enumerate(names)
                },
                "raw_statistics_exact": bool(np.all(raw_max[:2] == 0)),
                "producer_diagnostics_changed": bool(np.any(raw_max[2:8] != 0)),
            }
        write_json(output / "OBSERVATION_DIFFERENCES.json", rows)
        write_json(output / "METRICS.json", summaries)
        lines = [
            "# Localización de deriva en el original nativo",
            "",
            "Comparación sin targets y sin ejecutar el candidato; tolerancias intactas.",
            "",
            "| Modalidad | Campo | Máximo Δ crudo | Máximo Δ normalizado |",
            "|---|---|---:|---:|",
        ]
        for modality, summary in summaries.items():
            for name, field in summary["columns"].items():
                lines.append(
                    f"| {modality} | {name} | {field['raw_max_abs']:.9g} | "
                    f"{field['normalized_max_abs']:.9g} |"
                )
        lines += [
            "",
            "La normalización amplifica diferencias según su escala congelada.",
            "Este diagnóstico localiza el primer bloque observado que difiere,",
            "pero no identifica por sí solo el kernel, dispositivo o política numérica causante.",
            "No prueba un fallo del candidato ni autoriza relajar la paridad histórica.",
        ]
        (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def main() -> None:
    """Inspect a sealed native original capture against the historical archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument(
        "--baseline", type=Path, default=Path("output/phase_safe_fusion/test12_audit")
    )
    parser.add_argument(
        "--upstream", type=Path, default=Path("output/phase_submission/FINAL_MODEL/upstream")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.capture, args.baseline, args.upstream, args.output)


if __name__ == "__main__":
    main()
