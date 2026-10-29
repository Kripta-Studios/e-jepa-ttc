#!/usr/bin/env bash
# Operational overlay: immutable scientific bundle is never edited.
set -euo pipefail
bundle="${1:-/opt/ejepa_bundle}"
repo="$bundle/workspace/e-jepa-ttc-v18-joint-measurement"
test -f "$bundle/BUNDLE_MANIFEST.json"
test -d "$repo"
export UV_CACHE_DIR=/tmp/ejepa_uv_cache
export UV_PYTHON_INSTALL_DIR=/opt/ejepa_python
export UV_LINK_MODE=copy
python3 -m pip install 'uv>=0.9,<1'
uv python install 3.11.15
uv venv --python 3.11.15 /opt/ejepa_env
python=/opt/ejepa_env/bin/python
uv pip install --python "$python" --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0+cu128 torchvision==0.26.0+cu128
"$python" - "$bundle/BUNDLE_MANIFEST.json" /opt/ejepa_requirements.txt <<'PY'
import json, pathlib, sys
packages = json.loads(pathlib.Path(sys.argv[1]).read_text())["packages"]
pathlib.Path(sys.argv[2]).write_text("\n".join(
    f"{name}=={version}" for name, version in sorted(packages.items())
    if name not in {"torch", "torchvision"}) + "\n")
PY
uv pip install --python "$python" -r /opt/ejepa_requirements.txt
# PYTHONPATH avoids editable-install metadata inside the immutable bundle.
export PYTHONPATH="$repo/src:$repo"
"$python" -m operational.runpod_migration.bundle verify --root "$bundle"
"$python" -c 'import sys,torch; print(sys.version); print(torch.__version__,torch.version.cuda); print(torch.cuda.get_device_name())'
uv pip freeze --python "$python" > /opt/ejepa_environment.lock.txt
# No training, source rewriting, global-volume writes, or automatic cache deletion.
