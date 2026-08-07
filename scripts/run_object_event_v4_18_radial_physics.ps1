[CmdletBinding()]
param(
    [string]$Device = "cuda",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Out = "artifacts\debug\object_event_v4_18_radial_physics"
$Cache = "artifacts\cache\garl_object_event_common_roi_screen_v4\manifest.json"
$V417 = "artifacts\debug\object_event_v4_17_signed_anchor_temporal\screen\summary.json"
$Train = "artifacts\debug\object_event_v4_10_multiseed\ensemble_train_predictions.csv"
$Validation = "artifacts\debug\object_event_v4_10_multiseed\ensemble_validation_predictions.csv"
$Config = "configs\experiment\e_jepa_garl_object_event_radial_physics_v4_18.yaml"
$V48Config = "configs\experiment\e_jepa_garl_object_event_dense_motion_v4_8.yaml"
$V412Config = "configs\experiment\e_jepa_garl_object_event_directional_sign_probe_v4_12.yaml"
$Checkpoints = @(
    "7=artifacts\debug\object_event_v4_8_dense_motion\screen-seed-7\best_gate_passing.pt",
    "13=artifacts\debug\object_event_v4_8_dense_motion\screen-seed-13\best_gate_passing.pt",
    "23=artifacts\debug\object_event_v4_8_dense_motion\screen-seed-23\best_gate_passing.pt"
)

$PreflightArgs = @(
    "scripts\preflight_object_event_v4_18.py",
    "--cache-manifest", $Cache,
    "--v417-summary", $V417,
    "--ensemble-train", $Train,
    "--ensemble-validation", $Validation
)
foreach ($Checkpoint in $Checkpoints) {
    $PreflightArgs += @("--v48-checkpoint", $Checkpoint)
}
& uv run --no-sync python @PreflightArgs
if ($LASTEXITCODE -ne 0) {
    throw "v4.18 preflight failed with exit code $LASTEXITCODE"
}

$Args = @(
    "scripts\analyze_object_event_v4_18_radial_physics.py",
    "--cache-manifest", $Cache,
    "--v48-config", $V48Config,
    "--v412-config", $V412Config,
    "--config", $Config,
    "--ensemble-train", $Train,
    "--ensemble-validation", $Validation,
    "--output-dir", $Out,
    "--device", $Device
)
foreach ($Checkpoint in $Checkpoints) {
    $Args += @("--v48-checkpoint", $Checkpoint)
}
if ($Force) {
    $Args += "--force"
}

& uv run --no-sync python @Args
if ($LASTEXITCODE -ne 0) {
    throw "v4.18 radial physics experiment failed with exit code $LASTEXITCODE"
}

Write-Host "v4.18 completed. This experiment is decision-oriented and has no scientific exit-code gate."
