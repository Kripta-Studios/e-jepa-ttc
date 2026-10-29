#!/usr/bin/env bash
# Run only inside the selected Pod, after transferring and verifying the bundle.
set -euo pipefail
bundle="${1:?Usage: runpod_bootstrap.sh /workspace/ejepa_bundle}"
repo="$bundle/workspace/e-jepa-ttc-v18-joint-measurement"
test -f "$bundle/BUNDLE_MANIFEST.json"
test -d "$repo"
python3 -m pip install 'uv>=0.9,<1'
uv python install 3.11.15
uv venv --python 3.11.15 /workspace/ejepa_env
python=/workspace/ejepa_env/bin/python
uv pip install --python "$python" --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0+cu128 torchvision==0.26.0+cu128
"$python" - "$bundle/BUNDLE_MANIFEST.json" /workspace/ejepa_requirements.txt <<'PY'
import json, pathlib, sys
packages = json.loads(pathlib.Path(sys.argv[1]).read_text())["packages"]
pathlib.Path(sys.argv[2]).write_text("\n".join(
    f"{name}=={version}" for name, version in sorted(packages.items())
    if name not in {"torch", "torchvision"}) + "\n")
PY
uv pip install --python "$python" -r /workspace/ejepa_requirements.txt
uv pip install --python "$python" --no-deps -e "$repo"
export PYTHONPATH="$repo/src:$repo"
"$python" -m operational.runpod_migration.bundle verify --root "$bundle"
"$python" -c 'import sys,torch; print(sys.version); print(torch.__version__,torch.version.cuda); print(torch.cuda.get_device_name())'
# No training is started here. Save this environment evidence before preflight.
uv pip freeze --python "$python" > /workspace/ejepa_environment.lock.txt
