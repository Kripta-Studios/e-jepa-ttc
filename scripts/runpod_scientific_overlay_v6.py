"""Add bit-exact local RGB I/O to an unchanged R3/R_A5 v5 checkpoint identity."""

# ruff: noqa: ANN401

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from operational.rgb_port.accounting import atomic_write_json
from operational.runpod_migration.inventory import digest, read
from operational.runpod_migration.local_cache import route_rgb_source
from operational.runpod_migration.runtime_prefetch import ByteBoundedFoldPrefetchRuntime

if __package__:
    from . import runpod_scientific_overlay_v5 as v5
else:
    import runpod_scientific_overlay_v5 as v5


def _remove_argument(name: str) -> Path:
    if sys.argv.count(name) != 1:
        raise ValueError(f"Exactly one {name} is required")
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv):
        raise ValueError(f"Missing value for {name}")
    value = Path(sys.argv[index + 1])
    del sys.argv[index : index + 2]
    return value


def _validate_proofs(
    input_proof_path: Path,
    resume_proof_path: Path,
    readiness_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    input_proof, resume_proof = read(input_proof_path), read(resume_proof_path)
    input_route = input_proof.get("routing", {})
    resume_input = resume_proof.get("input_proof", {})
    resume_route = resume_input.get("routing", {})
    readiness_sha = digest(readiness_path)
    if (
        input_proof.get("schema") != "ejepa_runpod_local_cache_preflight_v1"
        or input_proof.get("status") != "PASS"
        or input_proof.get("rotation") != 3
        or input_proof.get("fit") != "R_A5"
        or input_proof.get("optimizer_updates") != 0
        or input_proof.get("training_called") is not False
        or input_proof.get("scientific_execution_allowed") is not False
        or input_proof.get("logical_vs_local_bit_exact") is not True
        or input_proof.get("python_numpy_torch_rng_unchanged") is not True
        or input_route.get("readiness_sha256") != readiness_sha
        or resume_proof.get("schema") != "ejepa_local_cache_resume_probe_v1"
        or resume_proof.get("status") != "PASS"
        or resume_proof.get("rotation") != 3
        or resume_proof.get("fit") != "R_A5"
        or resume_proof.get("technical_optimizer_updates") != 8
        or resume_proof.get("scientific_optimizer_updates") != 0
        or resume_proof.get("scientific_execution_allowed") is not False
        or resume_proof.get("readiness_sha256") != readiness_sha
        or resume_input.get("logical_vs_local_bit_exact") is not True
        or resume_input.get("python_numpy_torch_rng_unchanged") is not True
        or resume_route.get("readiness_sha256") != readiness_sha
        or input_route.get("content_identity_sha256")
        != resume_route.get("content_identity_sha256")
    ):
        raise ValueError("Local-cache input/resume proofs are incomplete or disagree")
    return input_proof, resume_proof


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] != "producer":
        raise ValueError("Scientific v6 supports only producer R3/R_A5")
    stage = _remove_argument("--local-stage").resolve(strict=True)
    readiness_path = _remove_argument("--local-readiness").resolve(strict=True)
    input_proof_path = _remove_argument("--local-input-proof").resolve(strict=True)
    resume_proof_path = _remove_argument("--local-resume-proof").resolve(strict=True)
    input_proof, resume_proof = _validate_proofs(
        input_proof_path,
        resume_proof_path,
        readiness_path,
    )
    repository = Path(__file__).resolve().parents[1]
    expected_local_cache = digest(repository / "operational/runpod_migration/local_cache.py")
    expected_runtime = digest(repository / "operational/runpod_migration/runtime_prefetch.py")
    expected_bounded = digest(repository / "operational/runpod_migration/bounded_prefetch.py")
    if (
        input_proof.get("runtime_sources", {}).get("local_cache") != expected_local_cache
        or resume_proof.get("local_cache_source_sha256") != expected_local_cache
        or resume_proof.get("runtime_prefetch_source_sha256") != expected_runtime
        or resume_proof.get("bounded_prefetch_source_sha256") != expected_bounded
    ):
        raise ValueError("Local-cache proof runtime bytes differ from v6")
    expected_content = input_proof["routing"]["content_identity_sha256"]
    expected_route_map = input_proof["routing"]["route_map_sha256"]
    original_prepare = v5.technical.prepare

    def prepare(bundle: Path, run: Path, budget_bytes: int) -> dict[str, Any]:
        setup = original_prepare(bundle, run, budget_bytes)
        from operational.final_refit import scientific_fold_producers as science
        from operational.runpod_migration import runner

        original_train = science.train

        def train(*pos: Any, **kw: Any) -> dict[str, Any]:
            source = kw.get("source")
            scientific_run = Path(kw["run"])
            if source is None:
                raise ValueError("v6 requires the explicit frozen RGB source")
            routing = route_rgb_source(
                source,
                logical_root=bundle,
                stage_root=stage,
                readiness_path=readiness_path,
            )
            if (
                routing["content_identity_sha256"] != expected_content
                or routing["route_map_sha256"] != expected_route_map
            ):
                raise ValueError("Scientific local route differs from its numeric proofs")
            v5_admission = scientific_run / "SCIENTIFIC_CLOUD_ADMISSION.json"
            if not v5_admission.is_file():
                raise ValueError("v5 scientific admission must precede v6 runtime routing")
            admission = {
                "schema": "ejepa_local_cache_runtime_admission_v1",
                "status": "PASS",
                "task": {"rotation": 3, "fit": "R_A5"},
                "v5_scientific_admission_sha256": digest(v5_admission),
                "migration_binding_sha256": digest(scientific_run / "MIGRATION_BINDING.json"),
                "input_proof_sha256": digest(input_proof_path),
                "resume_proof_sha256": digest(resume_proof_path),
                "readiness_sha256": digest(readiness_path),
                "content_identity_sha256": routing["content_identity_sha256"],
                "route_map_sha256": routing["route_map_sha256"],
                "v6_source_sha256": digest(Path(__file__)),
                "v5_source_sha256": digest(Path(str(v5.__file__))),
                "local_cache_source_sha256": expected_local_cache,
                "bounded_prefetch_source_sha256": expected_bounded,
                "runtime_prefetch_source_sha256": expected_runtime,
                "checkpoint_identity_extended": False,
                "checkpoint_identity_unchanged_for_v5_resume": True,
                "scientific_source_identity_unchanged": True,
                "input_bit_exact": True,
                "resume_state_bit_exact": True,
                "sampler_rng_loss_precision_seed_unchanged": True,
                "checkpoint_journal_progress_remain_on_durable_volume": True,
                "prefetch_depth": 2,
                "prefetch_reservation_bytes_per_batch": 192 * 1024**2,
                "decoder_max_pending_bytes": 128 * 1024**2,
                "shared_transient_limit_bytes": 512 * 1024**2,
                "zero_io_claimed": False,
            }
            runner._write_once(
                scientific_run / "LOCAL_CACHE_RUNTIME_ADMISSION.json", admission
            )

            def stats(value: dict[str, object]) -> None:
                atomic_write_json(
                    scientific_run / "LOCAL_CACHE_RUNTIME_STATS.json",
                    {
                        "schema": "ejepa_local_cache_runtime_stats_v1",
                        "admission_sha256": digest(
                            scientific_run / "LOCAL_CACHE_RUNTIME_ADMISSION.json"
                        ),
                        "runtime": value,
                    },
                )

            with ByteBoundedFoldPrefetchRuntime(science, stats_callback=stats):
                return original_train(*pos, **kw)

        science.train = train
        return setup

    v5.technical.prepare = prepare
    try:
        return v5.main()
    finally:
        v5.technical.prepare = original_prepare


if __name__ == "__main__":
    raise SystemExit(main())
