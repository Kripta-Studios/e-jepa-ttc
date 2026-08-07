[CmdletBinding()]
param(
    [string]$Device = "cuda",
    [switch]$Force
)
$ErrorActionPreference = "Stop"
$Out = "artifacts\debug\object_event_v4_25_geometry_readout"
$Cache = "artifacts\cache\garl_object_event_common_roi_screen_v4\manifest.json"
$V48Config = "configs\experiment\e_jepa_garl_object_event_dense_motion_v4_8.yaml"
$V424Config = "configs\experiment\e_jepa_garl_object_event_train_only_orchestrator_v4_24.yaml"
$V424Summary = "artifacts\debug\object_event_v4_24_train_only_orchestrator\summary.json"
$Train = "artifacts\debug\object_event_v4_10_multiseed\ensemble_train_predictions.csv"
$Validation = "artifacts\debug\object_event_v4_10_multiseed\ensemble_validation_predictions.csv"
$Adapted = @(
    "7=artifacts\debug\object_event_v4_22_encoder_geometry\adapted_seed_7.pt",
    "13=artifacts\debug\object_event_v4_22_encoder_geometry\adapted_seed_13.pt",
    "23=artifacts\debug\object_event_v4_22_encoder_geometry\adapted_seed_23.pt"
)
$Champions = @(
    "7=artifacts\debug\object_event_v4_24_train_only_orchestrator\champion_geometry_only_regularized_seed_7.pt",
    "13=artifacts\debug\object_event_v4_24_train_only_orchestrator\champion_geometry_only_regularized_seed_13.pt",
    "23=artifacts\debug\object_event_v4_24_train_only_orchestrator\champion_geometry_only_regularized_seed_23.pt"
)
$Preflight = @("scripts\preflight_object_event_v4_25.py","--v424-summary",$V424Summary)
foreach ($Checkpoint in $Champions) { $Preflight += @("--champion-checkpoint",$Checkpoint) }
& uv run --no-sync python @Preflight
if ($LASTEXITCODE -ne 0) { throw "v4.25 preflight failed with exit code $LASTEXITCODE" }
$Args = @(
    "scripts\analyze_object_event_v4_25_geometry_readout.py",
    "--cache-manifest",$Cache,
    "--v48-config",$V48Config,
    "--v424-config",$V424Config,
    "--v424-summary",$V424Summary,
    "--ensemble-train",$Train,
    "--ensemble-validation",$Validation,
    "--output-dir",$Out,
    "--device",$Device
)
foreach ($Checkpoint in $Champions) { $Args += @("--champion-checkpoint",$Checkpoint) }
foreach ($Checkpoint in $Adapted) { $Args += @("--adapted-checkpoint",$Checkpoint) }
if ($Force) { $Args += "--force" }
& uv run --no-sync python @Args
if ($LASTEXITCODE -ne 0) { throw "v4.25 analysis failed with exit code $LASTEXITCODE" }
Write-Host "v4.25 completed. Inspect summary.json and train_only_readout_ranking.csv."
exit 0
