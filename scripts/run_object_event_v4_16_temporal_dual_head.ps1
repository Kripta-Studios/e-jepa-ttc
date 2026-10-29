[CmdletBinding()]
param(
    [string]$Device = "cuda",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Root = "artifacts\debug\object_event_v4_16_temporal_dual_head"
$Overfit = Join-Path $Root "overfit96"
$Screen = Join-Path $Root "screen"
$Cache = "artifacts\cache\garl_object_event_common_roi_screen_v4\manifest.json"
$V415 = "artifacts\debug\object_event_v4_15_shared_odd_projection\screen\summary.json"
$Train = "artifacts\debug\object_event_v4_10_multiseed\ensemble_train_predictions.csv"
$Validation = "artifacts\debug\object_event_v4_10_multiseed\ensemble_validation_predictions.csv"
$Config = "configs\experiment\e_jepa_garl_object_event_temporal_dual_head_v4_16.yaml"
$V48Config = "configs\experiment\e_jepa_garl_object_event_dense_motion_v4_8.yaml"
$V412Config = "configs\experiment\e_jepa_garl_object_event_directional_sign_probe_v4_12.yaml"
$Checkpoints = @(
    "7=artifacts\debug\object_event_v4_8_dense_motion\screen-seed-7\best_gate_passing.pt",
    "13=artifacts\debug\object_event_v4_8_dense_motion\screen-seed-13\best_gate_passing.pt",
    "23=artifacts\debug\object_event_v4_8_dense_motion\screen-seed-23\best_gate_passing.pt"
)

function Invoke-PythonChecked {
    param([string]$Label, [string[]]$Arguments, [int[]]$Allowed = @(0))
    & uv run --no-sync python @Arguments
    $Code = $LASTEXITCODE
    $script:LastPythonCode = $Code
    if ($Allowed -notcontains $Code) {
        throw "$Label failed with exit code $Code"
    }
}

$PreflightArgs = @(
    "scripts\preflight_object_event_v4_16.py",
    "--cache-manifest", $Cache,
    "--v415-summary", $V415,
    "--ensemble-train", $Train,
    "--ensemble-validation", $Validation
)
foreach ($Checkpoint in $Checkpoints) {
    $PreflightArgs += @("--v48-checkpoint", $Checkpoint)
}
Invoke-PythonChecked -Label "v4.16 preflight" -Arguments $PreflightArgs

if ($Force -and (Test-Path $Root)) {
    Remove-Item -Recurse -Force $Root
}
New-Item -ItemType Directory -Force -Path $Root | Out-Null

$Common = @(
    "scripts\train_e_jepa_object_event_v4_16.py",
    "--cache-manifest", $Cache,
    "--v48-config", $V48Config,
    "--v412-config", $V412Config,
    "--config", $Config,
    "--ensemble-train", $Train,
    "--ensemble-validation", $Validation,
    "--device", $Device
)
foreach ($Checkpoint in $Checkpoints) {
    $Common += @("--v48-checkpoint", $Checkpoint)
}

$ReuseOverfit = $false
if (Test-Path (Join-Path $Overfit "summary.json")) {
    $Summary = Get-Content (Join-Path $Overfit "summary.json") -Raw | ConvertFrom-Json
    $ReuseOverfit = ($Summary.status -eq "overfit_passed")
}
if ($ReuseOverfit) {
    Write-Host "Reusing passed v4.16 overfit: $Overfit"
}
else {
    if (Test-Path $Overfit) { Remove-Item -Recurse -Force $Overfit }
    Invoke-PythonChecked -Label "v4.16 overfit" -Arguments ($Common + @(
        "--output-dir", $Overfit,
        "--mode", "overfit"
    )) -Allowed @(0, 2)
    $Code = $script:LastPythonCode
    if ($Code -eq 2) {
        Write-Host "v4.16 overfit completed but failed scientific gates."
        exit 2
    }
}

if (Test-Path $Screen) {
    Write-Host "Replacing previous or incomplete v4.16 screen directory: $Screen"
    Remove-Item -Recurse -Force $Screen
}
Invoke-PythonChecked -Label "v4.16 screen" -Arguments ($Common + @(
    "--output-dir", $Screen,
    "--mode", "screen"
)) -Allowed @(0, 2)
$ScreenCode = $script:LastPythonCode
exit $ScreenCode
