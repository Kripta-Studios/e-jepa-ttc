"""Direct executable entrypoints must not bypass scientific branch gates."""

import json
import sys
from pathlib import Path

import pytest
import torch

from e_jepa_ttc.artifacts import training_authorization as authorization
from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact


@pytest.mark.parametrize(
    "mutation", [None, "input", "qa_log", "smoke", "runtime", "invocation", "dirty"]
)
def test_complete_authorization_binds_runtime_inputs_qa_and_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str | None
) -> None:
    def write(path: Path, value: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sign_artifact(value)), encoding="utf-8")

    source, log = tmp_path / "input", tmp_path / "qa.log"
    source.write_bytes(b"input")
    log.write_bytes(b"passed")
    bindings = {
        "source": {"path": str(source), "bytes": 5, "sha256": compute_file_hash(str(source))}
    }
    qa_path = tmp_path / "qa/QA_MANIFEST.json"
    write(
        qa_path,
        {
            "training_commit": "abc",
            "status": "passed",
            "input_bindings": bindings,
            "evidence": {"log": {"path": str(log), "sha256": compute_file_hash(str(log))}},
        },
    )
    smoke_path = tmp_path / "qa/STAGE64_REAL_TRAIN_ONLY_SMOKE.json"
    write(
        smoke_path,
        {
            "status": "passed",
            "identity": {"training_commit": "abc", "microbatch": 8, "device": "cuda"},
        },
    )
    write(
        tmp_path / "stage63/X3_FEASIBILITY_V2.json",
        {"integrity_passed": True, "training_ready": True},
    )
    write(
        tmp_path / "TRAINING_LOCK.json",
        {
            "training_commit": "abc",
            "authorization_version": "complete_identity_v2",
            "python": "wrong" if mutation == "runtime" else sys.version,
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "invocation": {"microbatch": 8, "device": "cuda"},
            "input_bindings": bindings,
            "qa_sha256": compute_file_hash(str(qa_path)),
            "smoke_sha256": compute_file_hash(str(smoke_path)),
        },
    )
    if mutation == "input":
        source.write_bytes(b"other")
    elif mutation == "qa_log":
        log.write_bytes(b"edited")
    elif mutation == "smoke":
        smoke_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        authorization.subprocess,
        "check_output",
        lambda command, **kwargs: (
            "abc\n" if "rev-parse" in command else (" M tracked.py" if mutation == "dirty" else "")
        ),
    )
    invocation = {"microbatch": 4 if mutation == "invocation" else 8, "device": "cuda"}
    if mutation is None:
        authorization.validate_training_authorization(
            tmp_path, tmp_path, stage=64, invocation=invocation
        )
    else:
        with pytest.raises(ValueError):
            authorization.validate_training_authorization(
                tmp_path, tmp_path, stage=64, invocation=invocation
            )


def test_completed_result_resume_rejects_changed_output(tmp_path: Path) -> None:
    path = tmp_path / "oof.csv"
    path.write_text("token,prediction\na,2\n", encoding="utf-8")
    result = {"output_bindings": authorization.bind_output_files(tmp_path, [path])}
    authorization.verify_output_bindings(tmp_path, result)
    path.write_text("token,prediction\na,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="output identity"):
        authorization.verify_output_bindings(tmp_path, result)


@pytest.mark.parametrize("seed", [13, 23])
@pytest.mark.parametrize("passed", [False, True])
def test_replica_requires_seed7_all_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seed: int, passed: bool
) -> None:
    def artifact(path: Path) -> dict[str, object]:
        if path.name == "X3_FEASIBILITY_V2.json":
            return {"integrity_passed": True, "training_ready": True}
        return {
            "status": "RAW_ALL_GATES_PASSED",
            "training_lock_sha256": "lock",
            "all_endpoints_frozen_before_evaluation": True,
            "gates": {"all_passed": passed},
        }

    monkeypatch.setattr(authorization, "read_signed", artifact)
    if passed:
        authorization.validate_branch(tmp_path, stage=64, seed=seed, lock_sha256="lock")
    else:
        with pytest.raises(ValueError, match="all seed7"):
            authorization.validate_branch(tmp_path, stage=64, seed=seed, lock_sha256="lock")


@pytest.mark.parametrize("status", ["INTEGRITY_BLOCKED", "RESOURCE_BLOCKED", "NUMERICAL_FAILURE"])
def test_technical_failure_never_authorizes_stage65(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    monkeypatch.setattr(
        authorization,
        "read_signed",
        lambda path: (
            {"integrity_passed": True}
            if path.name == "X3_FEASIBILITY_V2.json"
            else {"status": status, "training_lock_sha256": "lock"}
        ),
    )
    with pytest.raises(ValueError, match="scientific failure"):
        authorization.validate_branch(tmp_path, stage=65, seed=7, lock_sha256="lock")


def test_replica_attempt_blocks_stage65_even_if_raw_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "stage64/seed13").mkdir(parents=True)
    monkeypatch.setattr(
        authorization,
        "read_signed",
        lambda path: {
            "integrity_passed": True,
            "decision": "RAW_DATA_UNAVAILABLE",
        },
    )
    with pytest.raises(ValueError, match="replica attempt"):
        authorization.validate_branch(tmp_path, stage=65, seed=7, lock_sha256="lock")
