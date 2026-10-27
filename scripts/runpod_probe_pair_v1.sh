#!/usr/bin/env bash
# Explicit eight-update technical probes; no scientific producer command.
set -euo pipefail
repo=/workspace/ejepa_bundle/workspace/e-jepa-ttc-v18-joint-measurement
export PYTHONPATH="$repo/src:$repo"
export PYTHONDONTWRITEBYTECODE=1
for rotation in 2 3; do
  run="/workspace/ejepa_technical/r${rotation}_rgb_v3"
  output="/workspace/ejepa_operations/r${rotation}"
  test ! -e "$run/technical_resume_probe"
  nohup /workspace/ejepa_env/bin/python -u /workspace/ejepa_operations/runpod_technical_overlay_v3.py \
    resume-probe --bundle /workspace/ejepa_bundle --run "$run" \
    --rotation "$rotation" --fit R_A5 --budget-bytes 46000000000 \
    --assignment "$output/TECHNICAL_R${rotation}.json" \
    > "$output/RESUME_PROBE.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$output/PROBE_PID"
done
