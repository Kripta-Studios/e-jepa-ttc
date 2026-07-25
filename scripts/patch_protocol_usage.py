import os
from pathlib import Path
import re

def patch_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Add import if missing
    if "from e_jepa_ttc.artifacts.protocol import get_current_protocol_identity" not in content:
        # insert after the last standard import (e.g. after import json or pathlib)
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                last_import = i
        lines.insert(last_import + 1, "from e_jepa_ttc.artifacts.protocol import get_current_protocol_identity")
        content = '\n'.join(lines)
        
    # Find where it injects protocol_version.
    # Typically: "protocol_version": "2.0",
    # We want to replace it with:
    # "protocol_version": get_current_protocol_identity()[0],
    # "protocol_sha256": get_current_protocol_identity()[1],
    
    # We can use regex to replace it
    # We will just replace `"protocol_version": "2.0",` with:
    # `"protocol_version": get_current_protocol_identity()[0], "protocol_sha256": get_current_protocol_identity()[1],`
    
    # Ensure it's padded correctly
    
    content = re.sub(
        r'"protocol_version":\s*"2\.0",',
        r'"protocol_version": get_current_protocol_identity()[0],\n            "protocol_sha256": get_current_protocol_identity()[1],',
        content
    )

    # Some of them might not have the comma
    content = re.sub(
        r'"protocol_version":\s*"2\.0"',
        r'"protocol_version": get_current_protocol_identity()[0],\n            "protocol_sha256": get_current_protocol_identity()[1]',
        content
    )
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"Patched {filepath}")

def main():
    repo_root = Path(__file__).resolve().parent.parent
    training_dir = repo_root / "src" / "e_jepa_ttc" / "training"
    
    for py_file in ["jepa.py", "object_jepa.py", "supervised.py"]:
        path = training_dir / py_file
        if path.exists():
            patch_file(path)

if __name__ == "__main__":
    main()
