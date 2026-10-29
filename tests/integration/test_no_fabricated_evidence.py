import ast
import os
import re
from pathlib import Path


def test_no_fabricated_evidence_in_scripts():
    """
    Regression test to ensure no orchestration scripts (powershell or python)
    contain mechanisms that generate dummy success artifacts to bypass verifiers.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    scripts_dir = repo_root / "scripts"

    suspicious_patterns = [
        re.compile(r"fake[_ -]?(?:hash|sha|metric|data|evidence)", re.IGNORECASE),
        re.compile(r"pseudo[_ -]?hash", re.IGNORECASE),
        re.compile(r"dummy\s*=\s*\{", re.IGNORECASE),
    ]

    # Explicitly allowed files that legimately parse or use these names,
    # but do NOT fabricate them to skip stages.
    allowed_files = [
        "audit_cache.py",
        "aggregate_results.py",
        "run_evttc_architecture_matrix.py",
        "run_evttc_final_pipeline.py",
        "run_object_jepa_matrix.py",
        "select_best_onnx_candidate.py",
    ]

    failures = []

    for root, _, files in os.walk(scripts_dir):
        for file in files:
            if file in allowed_files:
                continue

            if file.endswith(".ps1") or file.endswith(".py"):
                filepath = Path(root) / file
                try:
                    with open(filepath, encoding="utf-8") as f:
                        content = f.read()

                    for i, line in enumerate(content.split("\n")):
                        # Skip comment lines
                        if line.strip().startswith("#"):
                            continue

                        for pattern in suspicious_patterns:
                            if pattern.search(line):
                                failures.append(
                                    f"{file}:{i + 1} -> Suspicious pattern found: {line.strip()}"
                                )
                    if filepath.suffix == ".py":
                        tree = ast.parse(content, filename=str(filepath))
                        for node in ast.walk(tree):
                            if not isinstance(node, ast.Dict):
                                continue
                            keys = {
                                key.value
                                for key in node.keys
                                if isinstance(key, ast.Constant) and isinstance(key.value, str)
                            }
                            values = {
                                value.value
                                for value in node.values
                                if isinstance(value, ast.Constant) and isinstance(value.value, str)
                            }
                            if "status" in keys and "success" in values:
                                failures.append(
                                    f"{file}:{getattr(node, 'lineno', 0)} -> literal success status"
                                )
                except (UnicodeDecodeError, SyntaxError) as exc:
                    failures.append(f"{file}: parse failure: {exc}")
                    pass

    assert len(failures) == 0, "Fabricated evidence mechanisms detected in scripts:\n" + "\n".join(
        failures
    )


def test_no_debug_artifacts_in_root():
    repo_root = Path(__file__).resolve().parent.parent.parent
    forbidden_items = [
        "debug_test_dir",
        "test_empty",
        "audit.json",
        "master_smoke.log",
        "completion_manifest.json",
    ]

    failures = []

    # Check exact forbidden names in root
    for item in forbidden_items:
        path = repo_root / item
        if path.exists():
            failures.append(f"Forbidden artifact found in root: {item}")

    # Check for unauthorized generated files in root
    for file in os.listdir(repo_root):
        if (
            file.endswith(".onnx")
            or file.endswith(".npz")
            or file.endswith(".h5")
            or file.endswith(".hdf5")
            or file.endswith(".bag")
            or file.endswith(".mp4")
        ):
            failures.append(f"Unauthorized generated file in root: {file}")

    assert not failures, "Found debug or untracked artifacts: " + "\n".join(failures)
