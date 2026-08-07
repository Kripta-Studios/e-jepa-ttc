param([switch]$Force)
$ErrorActionPreference = "Stop"
$Root = "artifacts\debug\object_event_v4_12_directional_sign\screen-seed-7"
$Out = "artifacts\debug\object_event_v4_13_selective_dual_head"
uv run --no-sync python scripts\preflight_object_event_v4_13.py `
  --v412-summary "$Root\summary.json" `
  --predictions "$Root\validation_predictions.csv"
if ($LASTEXITCODE -ne 0) { throw "v4.13 preflight failed with exit code $LASTEXITCODE" }
$argsList = @(
  "scripts\analyze_object_event_v4_13_selective_dual_head.py",
  "--config", "configs\experiment\e_jepa_garl_object_event_selective_dual_head_v4_13.yaml",
  "--v412-summary", "$Root\summary.json",
  "--predictions", "$Root\validation_predictions.csv",
  "--output-dir", $Out
)
if ($Force) { $argsList += "--force" }
uv run --no-sync python @argsList
exit $LASTEXITCODE
