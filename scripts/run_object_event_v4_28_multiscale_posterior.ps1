[CmdletBinding()]
param(
    [string]$Device = "cuda",
    [switch]$Force
)
$ErrorActionPreference = "Stop"
$Out = "artifacts\debug\object_event_v4_28_multiscale_posterior"
$Cache = "artifacts\cache\garl_object_event_common_roi_screen_v4\manifest.json"
$V48Config = "configs\experiment\e_jepa_garl_object_event_dense_motion_v4_8.yaml"
$Config = "configs\experiment\e_jepa_garl_object_event_multiscale_posterior_v4_28.yaml"
$V427Summary = "artifacts\debug\object_event_v4_27_scale_correlation_lhr\summary.json"
$Validation = "artifacts\debug\object_event_v4_10_multiseed\ensemble_validation_predictions.csv"
$Adapted = @(
    "7=artifacts\debug\object_event_v4_22_encoder_geometry\adapted_seed_7.pt",
    "13=artifacts\debug\object_event_v4_22_encoder_geometry\adapted_seed_13.pt",
    "23=artifacts\debug\object_event_v4_22_encoder_geometry\adapted_seed_23.pt"
)
$Preflight = @(
    "scripts\preflight_object_event_v4_28.py",
    "--cache-manifest", $Cache,
    "--v48-config", $V48Config,
    "--v427-summary", $V427Summary
)
foreach ($Checkpoint in $Adapted) { $Preflight += @("--adapted-checkpoint", $Checkpoint) }
& uv run --no-sync python @Preflight
if ($LASTEXITCODE -ne 0) { throw "v4.28 preflight failed with exit code $LASTEXITCODE" }

$Args = @(
    "scripts\analyze_object_event_v4_28_multiscale_posterior.py",
    "--cache-manifest", $Cache,
    "--v48-config", $V48Config,
    "--config", $Config,
    "--v427-summary", $V427Summary,
    "--ensemble-validation", $Validation,
    "--output-dir", $Out,
    "--device", $Device
)
foreach ($Checkpoint in $Adapted) { $Args += @("--adapted-checkpoint", $Checkpoint) }
if ($Force) { $Args += "--force" }
& uv run --no-sync python @Args
if ($LASTEXITCODE -ne 0) { throw "v4.28 failed with exit code $LASTEXITCODE" }
Write-Host "v4.28 completed. Inspect summary.json and oof_arm_ranking.csv."
