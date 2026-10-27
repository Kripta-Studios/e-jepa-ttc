"""Recover a missing package receipt without rewriting any ZIP or prediction.

This is an additive, local recovery tool. It preserves the frozen packager's
exact ZIP comparison and existing-receipt validation, adding only the missing
atomic sidecar write after that ZIP has passed verification.
"""

from __future__ import annotations

import ast
import copy
import inspect
import json
import os
import uuid
from pathlib import Path
from typing import Any


def durable_write_once(path: Path, value: dict[str, Any]) -> None:
    """Publish the same canonical bytes as the original writer, with fsync."""
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError("Refusing to overwrite a changed scientific receipt")
        return
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def recovery_packager(original: Any) -> Any:
    """Inject the original receipt construction into its missing-sidecar branch."""
    tree = ast.parse(inspect.getsource(original))
    function = tree.body[0]
    if not isinstance(function, ast.FunctionDef):
        raise ValueError("Frozen packager is not a function")
    branches = [
        node
        for node in function.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "output.exists()"
    ]
    receipt_starts = [
        index
        for index, node in enumerate(function.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "receipt" for target in node.targets)
    ]
    if len(branches) != 1 or len(receipt_starts) != 1:
        raise ValueError("Frozen packager control flow changed")
    branch = branches[0]
    if (
        len(branch.body) != 4
        or ast.unparse(branch.body[0]) != "verify_zip(output)"
        or not isinstance(branch.body[1], ast.Assign)
        or ast.unparse(branch.body[1].targets[0]) != "existing"
    ):
        raise ValueError("Frozen existing-ZIP verification changed")
    tail = function.body[receipt_starts[0] :]
    if (
        len(tail) != 4
        or ast.unparse(tail[-1]) != "return receipt"
        or ast.unparse(tail[-2]) != "_write_once(output.with_suffix('.json'), receipt)"
    ):
        raise ValueError("Frozen receipt construction changed")
    missing = ast.parse("if not output.with_suffix('.json').exists():\n    pass").body[0]
    missing.body = copy.deepcopy(tail[:-1])
    # Original verify_zip precedes this insertion, original receipt checks follow.
    branch.body.insert(1, missing)
    function.name = "_recover_frozen_package"
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    namespace["_write_once"] = durable_write_once
    exec(compile(tree, "<missing-package-receipt-recovery>", "exec"), namespace)
    return namespace[function.name]


def recover_package_receipt(final: Any, run: Any) -> dict[str, Any]:
    """Require trusted public inputs, sealed predictions and an exact existing ZIP."""
    run = run.resolve(strict=True)
    root = run / "final/test12"
    archive = root / "PHASE_ONLY_S7_TRAIN40_SCIENTIFIC_OOF.zip"
    if not archive.is_file():
        raise FileNotFoundError("Recovery requires an already published ZIP")
    cfg = final._config(run)
    if (
        final.sha256(cfg["test12_inputs"]) != final.TRUSTED_INPUTS_SHA256
        or final.sha256(cfg["official_test_inputs"]) != final.TRUSTED_OFFICIAL_TEST_INPUTS_SHA256
    ):
        raise ValueError("Official input identities differ")
    _, seal_sha = final._scientific_seal(run / "final/MODEL_SEAL.json")
    receipt_path = root / "PREDICTION_RECEIPT.json"
    prediction = final._read(receipt_path)
    inputs = final._read(cfg["test12_inputs"])
    physical_count = len(set(inputs["physical_aliases"].values()))
    binding = final.canonical_sha256(
        {"model_seal_sha256": seal_sha, "inputs_sha256": final.TRUSTED_INPUTS_SHA256}
    )
    if (
        prediction.get("schema") != final.PREDICTION_SCHEMA
        or prediction.get("candidate") != final.self_candidate()
        or prediction.get("status") != "COMPLETE"
        or prediction.get("model_seal_sha256") != seal_sha
        or prediction.get("inputs_sha256") != final.TRUSTED_INPUTS_SHA256
        or prediction.get("predictions_sha256") != final.sha256(root / "predictions.csv")
        or prediction.get("population") != final.OFFICIAL_TEST_QUERIES
        or prediction.get("physical_queries") != physical_count
        or prediction.get("run_binding_sha256") != binding
        or prediction.get("computed_under_seal") is not True
        or prediction.get("private_targets_read") is not False
        or prediction.get("external_submission") is not False
    ):
        raise ValueError("Prediction receipt is not the exact sealed public candidate")
    before = final.sha256(archive)
    result = recovery_packager(final.package_scientific_predictions)(run)
    if final.sha256(archive) != before:
        raise ValueError("Recovery unexpectedly modified ZIP bytes")
    # The unchanged original also accepts the repaired receipt on an ordinary resume.
    if final.package_scientific_predictions(run) != result:
        raise ValueError("Original packager rejected the recovery")
    return result


def main() -> None:
    import runpod_final_bootstrap as bootstrap
    import runpod_final_cache_compat

    mapping = bootstrap.setup()
    runpod_final_cache_compat.install(mapping)
    from operational.final_refit import scientific_oof_final as final

    result = recover_package_receipt(final, final.Path(str(bootstrap.RUN)))
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
