[CmdletBinding()]
param(
    [string]$Device = "cuda",
    [switch]$Force
)
$ErrorActionPreference = "Stop"
$Out = "artifacts\debug\object_event_v4_27_scale_correlation_lhr"
$Cache = "artifacts\cache\garl_object_event_common_roi_screen_v4\manifest.json"
$V48Config = "configs\experiment\e_jepa_garl_object_event_dense_motion_v4_8.yaml"
$Config = "configs\experiment\e_jepa_garl_object_event_scale_correlation_lhr_v4_27.yaml"
$Validation = "artifacts\debug\object_event_v4_10_multiseed\ensemble_validation_predictions.csv"
$Adapted = @(
    "7=artifacts\debug\object_event_v4_22_encoder_geometry\adapted_seed_7.pt",
    "13=artifacts\debug\object_event_v4_22_encoder_geometry\adapted_seed_13.pt",
    "23=artifacts\debug\object_event_v4_22_encoder_geometry\adapted_seed_23.pt"
)
$Preflight = @(
    "scripts\preflight_object_event_v4_27.py",
    "--cache-manifest", $Cache,
    "--v48-config", $V48Config
)
foreach ($Checkpoint in $Adapted) { $Preflight += @("--adapted-checkpoint", $Checkpoint) }
& uv run --no-sync python @Preflight
if ($LASTEXITCODE -ne 0) { throw "v4.27 preflight failed with exit code $LASTEXITCODE" }

$Args = @(
    "scripts\analyze_object_event_v4_27_scale_correlation_lhr.py",
    "--cache-manifest", $Cache,
    "--v48-config", $V48Config,
    "--config", $Config,
    "--ensemble-validation", $Validation,
    "--output-dir", $Out,
    "--device", $Device
)
foreach ($Checkpoint in $Adapted) { $Args += @("--adapted-checkpoint", $Checkpoint) }
if ($Force) { $Args += "--force" }
& uv run --no-sync python @Args
if ($LASTEXITCODE -ne 0) { throw "v4.27 failed with exit code $LASTEXITCODE" }
Write-Host "v4.27 completed. Inspect summary.json first; validation files exist only if the OOF gate passed."
exit 0
