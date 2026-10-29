[CmdletBinding()]
param(
    [string]$Device = "cuda",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Out = "artifacts\debug\object_event_v4_23_joint_geometry_ttc"
$Cache = "artifacts\cache\garl_object_event_common_roi_screen_v4\manifest.json"
$V422 = "artifacts\debug\object_event_v4_22_encoder_geometry\summary.json"
$Train = "artifacts\debug\object_event_v4_10_multiseed\ensemble_train_predictions.csv"
$Validation = "artifacts\debug\object_event_v4_10_multiseed\ensemble_validation_predictions.csv"
$Config = "configs\experiment\e_jepa_garl_object_event_joint_geometry_ttc_v4_23.yaml"
$V48Config = "configs\experiment\e_jepa_garl_object_event_dense_motion_v4_8.yaml"
$Checkpoints = @(
    "7=artifacts\debug\object_event_v4_22_encoder_geometry\adapted_seed_7.pt",
    "13=artifacts\debug\object_event_v4_22_encoder_geometry\adapted_seed_13.pt",
    "23=artifacts\debug\object_event_v4_22_encoder_geometry\adapted_seed_23.pt"
)

$PreflightArgs = @(
    "scripts\preflight_object_event_v4_23.py",
    "--cache-manifest", $Cache,
    "--v422-summary", $V422
)
foreach ($Checkpoint in $Checkpoints) { $PreflightArgs += @("--adapted-checkpoint", $Checkpoint) }
& uv run --no-sync python @PreflightArgs
if ($LASTEXITCODE -ne 0) { throw "v4.23 preflight failed with exit code $LASTEXITCODE" }

$Args = @(
    "scripts\analyze_object_event_v4_23_joint_geometry_ttc.py",
    "--cache-manifest", $Cache,
    "--v48-config", $V48Config,
    "--config", $Config,
    "--ensemble-train", $Train,
    "--ensemble-validation", $Validation,
    "--v422-summary", $V422,
    "--output-dir", $Out,
    "--device", $Device
)
foreach ($Checkpoint in $Checkpoints) { $Args += @("--adapted-checkpoint", $Checkpoint) }
if ($Force) { $Args += "--force" }
& uv run --no-sync python @Args
if ($LASTEXITCODE -ne 0) { throw "v4.23 joint geometry+TTC experiment failed with exit code $LASTEXITCODE" }
Write-Host "v4.23 completed. Inspect summary.decision; no scientific pass/fail exit gate."
