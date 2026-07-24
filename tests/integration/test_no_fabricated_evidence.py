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
        re.compile(r"\{\s*['\"]status['\"]\s*:\s*['\"]success['\"]\s*\}", re.IGNORECASE),
        re.compile(r"dummy\s*=\s*\{", re.IGNORECASE),
        re.compile(r"['\"]summary\.json['\"]", re.IGNORECASE),
        re.compile(r"['\"]matrix_summary\.json['\"]", re.IGNORECASE),
        re.compile(r"['\"]manifest\.json['\"]", re.IGNORECASE),
        re.compile(r"['\"]eap_split_statistics\.json['\"]", re.IGNORECASE),
    ]

    # Explicitly allowed files that legimately parse or use these names,
    # but do NOT fabricate them to skip stages.
    allowed_files = [
        "verify_smoke_completion.py",
        "verify_full_completion.py",
        "audit_cache.py",
        "aggregate_results.py",
        "run_object_jepa_matrix.py"
    ]

    failures = []
    
    for root, _, files in os.walk(scripts_dir):
        for file in files:
            if file in allowed_files:
                continue
                
            if file.endswith(".ps1") or file.endswith(".py"):
                filepath = Path(root) / file
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    for i, line in enumerate(content.split('\n')):
                        # Skip comment lines
                        if line.strip().startswith('#'):
                            continue
                            
                        # If a .ps1 file is creating a dummy dictionary or writing status success
                        for pattern in suspicious_patterns:
                            if pattern.search(line):
                                failures.append(f"{file}:{i+1} -> Suspicious pattern found: {line.strip()}")
                except UnicodeDecodeError:
                    pass
                    
    assert len(failures) == 0, "Fabricated evidence mechanisms detected in scripts:\n" + "\n".join(failures)
