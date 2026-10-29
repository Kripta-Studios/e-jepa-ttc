[CmdletBinding()]
param(
    [string]$Device = "cuda",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Out = "artifacts\debug\object_event_v4_24_train_only_orchestrator"
$Cache = "artifacts\cache\garl_object_event_common_roi_screen_v4\manifest.json"
$V422 = "artifacts\debug\object_event_v4_22_encoder_geometry\summary.json"
$V423 = "artifacts\debug\object_event_v4_23_joint_geometry_ttc\summary.json"
$Train = "artifacts\debug\object_event_v4_10_multiseed\ensemble_train_predictions.csv"
$Validation = "artifacts\debug\object_event_v4_10_multiseed\ensemble_validation_predictions.csv"
$Config = "configs\experiment\e_jepa_garl_object_event_train_only_orchestrator_v4_24.yaml"
$V48Config = "configs\experiment\e_jepa_garl_object_event_dense_motion_v4_8.yaml"
$Checkpoints = @(
    "7=artifacts\debug\object_event_v4_22_encoder_geometry\adapted_seed_7.pt",
    "13=artifacts\debug\object_event_v4_22_encoder_geometry\adapted_seed_13.pt",
    "23=artifacts\debug\object_event_v4_22_encoder_geometry\adapted_seed_23.pt"
)

$Preflight = @(
    "scripts\preflight_object_event_v4_24.py",
    "--cache-manifest", $Cache,
    "--v422-summary", $V422,
    "--v423-summary", $V423
)
foreach ($Checkpoint in $Checkpoints) { $Preflight += @("--adapted-checkpoint", $Checkpoint) }
& uv run --no-sync python @Preflight
if ($LASTEXITCODE -ne 0) { throw "v4.24 preflight failed with exit code $LASTEXITCODE" }

$Args = @(
    "scripts\analyze_object_event_v4_24_orchestrator.py",
    "--cache-manifest", $Cache,
    "--v48-config", $V48Config,
    "--config", $Config,
    "--ensemble-train", $Train,
    "--ensemble-validation", $Validation,
    "--v422-summary", $V422,
    "--v423-summary", $V423,
    "--output-dir", $Out,
    "--device", $Device
)
foreach ($Checkpoint in $Checkpoints) { $Args += @("--adapted-checkpoint", $Checkpoint) }
if ($Force) { $Args += "--force" }
& uv run --no-sync python @Args
if ($LASTEXITCODE -ne 0) { throw "v4.24 orchestrator failed with exit code $LASTEXITCODE" }

Write-Host "v4.24 completed. Inspect summary.json, stage1_arm_ranking.csv and stage2_arm_ranking.csv."
exit 0
