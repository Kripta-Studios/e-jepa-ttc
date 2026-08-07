[CmdletBinding()]
param([switch]$Force)

$ErrorActionPreference = "Stop"
$Out = "artifacts\debug\object_event_v4_21_box_pseudoflow_target_audit"
$Cache = "artifacts\cache\garl_object_event_common_roi_screen_v4\manifest.json"
$V419 = "artifacts\debug\object_event_v4_19_dense_correspondence\summary.json"
$V420 = "artifacts\debug\object_event_v4_20_box_pseudoflow\summary.json"
$Train = "artifacts\debug\object_event_v4_10_multiseed\ensemble_train_predictions.csv"
$Validation = "artifacts\debug\object_event_v4_10_multiseed\ensemble_validation_predictions.csv"
$Config = "configs\experiment\e_jepa_garl_object_event_box_pseudoflow_target_audit_v4_21.yaml"
$V48Config = "configs\experiment\e_jepa_garl_object_event_dense_motion_v4_8.yaml"

& uv run --no-sync python scripts\preflight_object_event_v4_21.py `
    --cache-manifest $Cache `
    --v419-summary $V419 `
    --v420-summary $V420 `
    --ensemble-train $Train `
    --ensemble-validation $Validation
if ($LASTEXITCODE -ne 0) { throw "v4.21 preflight failed with exit code $LASTEXITCODE" }

$Args = @(
    "scripts\analyze_object_event_v4_21_box_pseudoflow_target_audit.py",
    "--cache-manifest", $Cache,
    "--v48-config", $V48Config,
    "--config", $Config,
    "--ensemble-train", $Train,
    "--ensemble-validation", $Validation,
    "--v419-summary", $V419,
    "--v420-summary", $V420,
    "--output-dir", $Out
)
if ($Force) { $Args += "--force" }
& uv run --no-sync python @Args
if ($LASTEXITCODE -ne 0) { throw "v4.21 target audit failed with exit code $LASTEXITCODE" }
Write-Host "v4.21 completed. Inspect summary.decision; no model was trained."
