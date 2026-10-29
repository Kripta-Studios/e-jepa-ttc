param(
    [string]$Device = "cuda",
    [int]$Seed = 7,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param(
        [string]$Label,
        [scriptblock]$Command,
        [int[]]$AllowedExitCodes = @(0)
    )
    & $Command
    $Code = $LASTEXITCODE
    if ($AllowedExitCodes -notcontains $Code) {
        throw "$Label failed with exit code $Code"
    }
    return $Code
}

$Cache = "artifacts/cache/garl_object_event_common_roi_screen_v4/manifest.json"
$V48Config = "configs/experiment/e_jepa_garl_object_event_dense_motion_v4_8.yaml"
$ProbeConfig = "configs/experiment/e_jepa_garl_object_event_directional_sign_probe_v4_12.yaml"
$V48Checkpoint = "artifacts/debug/object_event_v4_8_dense_motion/screen-seed-$Seed/best_gate_passing.pt"
$V410Root = "artifacts/debug/object_event_v4_10_multiseed"
$TrainPredictions = "$V410Root/ensemble_train_predictions.csv"
$ValidationPredictions = "$V410Root/ensemble_validation_predictions.csv"
$V410Summary = "$V410Root/summary.json"
$OutRoot = "artifacts/debug/object_event_v4_12_directional_sign"
$Overfit = "$OutRoot/overfit64"
$Screen = "$OutRoot/screen-seed-$Seed"

Invoke-Checked "v4.12 preflight" {
    uv run --no-sync python scripts/preflight_object_event_v4_12.py `
        --cache-manifest $Cache `
        --v48-checkpoint $V48Checkpoint `
        --ensemble-train $TrainPredictions `
        --ensemble-validation $ValidationPredictions `
        --v410-summary $V410Summary
}

$ForceArgs = @()
if ($Force) {
    $ForceArgs = @("--force")
}

$ReuseOverfit = $false
if (-not $Force -and (Test-Path "$Overfit/summary.json")) {
    $ExistingOverfit = Get-Content "$Overfit/summary.json" -Raw | ConvertFrom-Json
    if ($ExistingOverfit.passed) {
        Write-Host "Reusing passed v4.12 overfit: $Overfit/summary.json"
        $ReuseOverfit = $true
    }
}

if (-not $ReuseOverfit) {
    $OverfitCode = Invoke-Checked "v4.12 overfit" {
        uv run --no-sync python scripts/train_e_jepa_object_event_v4_12.py `
            --cache-manifest $Cache `
            --v48-config $V48Config `
            --probe-config $ProbeConfig `
            --v48-checkpoint $V48Checkpoint `
            --ensemble-train $TrainPredictions `
            --ensemble-validation $ValidationPredictions `
            --output-dir $Overfit `
            --device $Device `
            --mode overfit `
            @ForceArgs
    } @(0, 2)
}

$OverfitSummary = Get-Content "$Overfit/summary.json" -Raw | ConvertFrom-Json
if (-not $OverfitSummary.passed) {
    Write-Host "v4.12 overfit completed but failed scientific gates."
    exit 2
}

$ScreenForceArgs = $ForceArgs
if (-not $Force -and (Test-Path $Screen) -and -not (Test-Path "$Screen/summary.json")) {
    Write-Host "Replacing incomplete v4.12 screen directory: $Screen"
    $ScreenForceArgs = @("--force")
}

$ScreenCode = Invoke-Checked "v4.12 screen" {
    uv run --no-sync python scripts/train_e_jepa_object_event_v4_12.py `
        --cache-manifest $Cache `
        --v48-config $V48Config `
        --probe-config $ProbeConfig `
        --v48-checkpoint $V48Checkpoint `
        --ensemble-train $TrainPredictions `
        --ensemble-validation $ValidationPredictions `
        --output-dir $Screen `
        --device $Device `
        --mode screen `
        @ScreenForceArgs
} @(0, 2)

exit $ScreenCode
