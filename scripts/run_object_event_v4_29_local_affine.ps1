[CmdletBinding()]
param(
    [string]$Device = "cuda",
    [switch]$Force
)
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
$Out = "artifacts\debug\object_event_v4_29_local_affine"
$Cache = "artifacts\cache\garl_object_event_common_roi_screen_v4\manifest.json"
$V48Config = "configs\experiment\e_jepa_garl_object_event_dense_motion_v4_8.yaml"
$Config = "configs\experiment\e_jepa_garl_object_event_local_affine_v4_29.yaml"
$V427 = "artifacts\debug\object_event_v4_27_scale_correlation_lhr\summary.json"
$V428 = "artifacts\debug\object_event_v4_28_multiscale_posterior\summary.json"
$V410 = "artifacts\debug\object_event_v4_10_multiseed\summary.json"
$Validation = "artifacts\debug\object_event_v4_10_multiseed\ensemble_validation_predictions.csv"
$Checkpoints = @(
    "7=artifacts\debug\object_event_v4_22_encoder_geometry\adapted_seed_7.pt",
    "13=artifacts\debug\object_event_v4_22_encoder_geometry\adapted_seed_13.pt",
    "23=artifacts\debug\object_event_v4_22_encoder_geometry\adapted_seed_23.pt"
)
$Preflight = @("scripts\preflight_object_event_v4_29.py", "--cache-manifest", $Cache, "--v48-config", $V48Config, "--config", $Config, "--v427-summary", $V427, "--v428-summary", $V428, "--v410-summary", $V410, "--ensemble-validation", $Validation)
foreach ($Checkpoint in $Checkpoints) { $Preflight += @("--adapted-checkpoint", $Checkpoint) }
& uv run --no-sync python @Preflight
if ($LASTEXITCODE -ne 0) { throw "v4.29 preflight failed with exit code $LASTEXITCODE" }
$Run = @("scripts\analyze_object_event_v4_29_local_affine.py", "--cache-manifest", $Cache, "--v48-config", $V48Config, "--config", $Config, "--v427-summary", $V427, "--v428-summary", $V428, "--v410-summary", $V410, "--ensemble-validation", $Validation, "--output-dir", $Out, "--device", $Device)
foreach ($Checkpoint in $Checkpoints) { $Run += @("--adapted-checkpoint", $Checkpoint) }
if ($Force) { $Run += "--force" }
& uv run --no-sync python @Run
if ($LASTEXITCODE -ne 0) { throw "v4.29 analyzer failed with exit code $LASTEXITCODE" }
Write-Host "v4.29 completed; inspect summary.json."
}
finally {
    Pop-Location
}
