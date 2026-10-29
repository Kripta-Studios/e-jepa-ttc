param(
    [string]$Device = "cuda",
    [switch]$Force,
    [switch]$SkipTraining
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

$Seeds = @(7, 13, 23)
$Cache = "artifacts/cache/garl_object_event_common_roi_screen_v4/manifest.json"
$V48Config = "configs/experiment/e_jepa_garl_object_event_dense_motion_v4_8.yaml"
$ProbeSource = "configs/experiment/e_jepa_garl_object_event_directional_sign_probe_v4_12.yaml"
$FusionConfig = "configs/experiment/e_jepa_garl_object_event_selective_dual_head_v4_13.yaml"
$AggregateConfig = "configs/experiment/e_jepa_garl_object_event_locked_dual_head_multiseed_v4_14.yaml"
$V48Root = "artifacts/debug/object_event_v4_8_dense_motion"
$V410Root = "artifacts/debug/object_event_v4_10_multiseed"
$V412Seed7 = "artifacts/debug/object_event_v4_12_directional_sign/screen-seed-7"
$V413Seed7 = "artifacts/debug/object_event_v4_13_selective_dual_head"
$Runtime = "artifacts/debug/object_event_v4_14_runtime"
$DirectionRoot = "$Runtime/v412"
$FusionRoot = "$Runtime/v413"
$ConfigRoot = "$Runtime/configs"
$Out = "artifacts/debug/object_event_v4_14_locked_multiseed"

Invoke-Checked "v4.14 preflight" {
    uv run --no-sync python scripts/preflight_object_event_v4_14.py `
        --config $AggregateConfig `
        --v410-summary "$V410Root/summary.json" `
        --v413-summary "$V413Seed7/summary.json" `
        --v48-root $V48Root
}

# Materialize seed-7 results into the common true-seed runtime layout.
$Seed7Direction = "$DirectionRoot/seed-7/screen"
$Seed7Fusion = "$FusionRoot/seed-7"
if ($Force -and (Test-Path "$DirectionRoot/seed-7")) { Remove-Item "$DirectionRoot/seed-7" -Recurse -Force }
if ($Force -and (Test-Path $Seed7Fusion)) { Remove-Item $Seed7Fusion -Recurse -Force }
if (-not (Test-Path "$Seed7Direction/summary.json")) {
    New-Item -ItemType Directory -Path $Seed7Direction -Force | Out-Null
    Copy-Item "$V412Seed7/summary.json" "$Seed7Direction/summary.json"
    Copy-Item "$V412Seed7/validation_predictions.csv" "$Seed7Direction/validation_predictions.csv"
    if (Test-Path "$V412Seed7/validation_per_sequence.csv") {
        Copy-Item "$V412Seed7/validation_per_sequence.csv" "$Seed7Direction/validation_per_sequence.csv"
    }
}
if (-not (Test-Path "$Seed7Fusion/summary.json")) {
    New-Item -ItemType Directory -Path $Seed7Fusion -Force | Out-Null
    Copy-Item "$V413Seed7/summary.json" "$Seed7Fusion/summary.json"
    Copy-Item "$V413Seed7/validation_predictions.csv" "$Seed7Fusion/validation_predictions.csv"
    if (Test-Path "$V413Seed7/validation_per_sequence.csv") {
        Copy-Item "$V413Seed7/validation_per_sequence.csv" "$Seed7Fusion/validation_per_sequence.csv"
    }
}

foreach ($Seed in @(13, 23)) {
    $SeedConfig = "$ConfigRoot/seed-$Seed/v412_seed_$Seed.yaml"
    Invoke-Checked "v4.14 materialize config seed $Seed" {
        uv run --no-sync python scripts/prepare_object_event_v4_14_seed_config.py `
            --source $ProbeSource `
            --seed $Seed `
            --output $SeedConfig
    }

    $Checkpoint = "$V48Root/screen-seed-$Seed/best_gate_passing.pt"
    $Overfit = "$DirectionRoot/seed-$Seed/overfit64"
    $Screen = "$DirectionRoot/seed-$Seed/screen"
    $Fusion = "$FusionRoot/seed-$Seed"

    if (-not $SkipTraining) {
        $ReuseOverfit = $false
        if (-not $Force -and (Test-Path "$Overfit/summary.json")) {
            $Summary = Get-Content "$Overfit/summary.json" -Raw | ConvertFrom-Json
            if ($Summary.passed -and [int]$Summary.train_config.seed -eq $Seed) {
                Write-Host "Reusing passed v4.12 overfit seed ${Seed}: $Overfit/summary.json"
                $ReuseOverfit = $true
            }
        }
        if (-not $ReuseOverfit) {
            $CommandArgs = @(
                "scripts/train_e_jepa_object_event_v4_12.py",
                "--cache-manifest", $Cache,
                "--v48-config", $V48Config,
                "--probe-config", $SeedConfig,
                "--v48-checkpoint", $Checkpoint,
                "--ensemble-train", "$V410Root/ensemble_train_predictions.csv",
                "--ensemble-validation", "$V410Root/ensemble_validation_predictions.csv",
                "--output-dir", $Overfit,
                "--device", $Device,
                "--mode", "overfit"
            )
            if ($Force -or (Test-Path $Overfit)) { $CommandArgs += "--force" }
            Invoke-Checked "v4.12 overfit seed $Seed" { uv run --no-sync python @CommandArgs } @(0, 2)
        }
        $OverfitSummary = Get-Content "$Overfit/summary.json" -Raw | ConvertFrom-Json
        if (-not $OverfitSummary.passed) {
            throw "v4.12 overfit seed $Seed failed scientific gates"
        }

        $ReuseScreen = $false
        if (-not $Force -and (Test-Path "$Screen/summary.json")) {
            $Summary = Get-Content "$Screen/summary.json" -Raw | ConvertFrom-Json
            if ($Summary.artifact_type -eq "object_event_v4_12_reversal_balanced_directional_sign" -and [int]$Summary.train_config.seed -eq $Seed) {
                Write-Host "Reusing complete v4.12 screen seed $Seed ($($Summary.status))"
                $ReuseScreen = $true
            }
        }
        if (-not $ReuseScreen) {
            $CommandArgs = @(
                "scripts/train_e_jepa_object_event_v4_12.py",
                "--cache-manifest", $Cache,
                "--v48-config", $V48Config,
                "--probe-config", $SeedConfig,
                "--v48-checkpoint", $Checkpoint,
                "--ensemble-train", "$V410Root/ensemble_train_predictions.csv",
                "--ensemble-validation", "$V410Root/ensemble_validation_predictions.csv",
                "--output-dir", $Screen,
                "--device", $Device,
                "--mode", "screen"
            )
            if ($Force -or (Test-Path $Screen)) { $CommandArgs += "--force" }
            Invoke-Checked "v4.12 screen seed $Seed" { uv run --no-sync python @CommandArgs } @(0, 2)
        }

        $ReuseFusion = $false
        if (-not $Force -and (Test-Path "$Fusion/summary.json")) {
            $Summary = Get-Content "$Fusion/summary.json" -Raw | ConvertFrom-Json
            if ($Summary.status -in @("selective_fusion_passed", "selective_fusion_failed")) {
                Write-Host "Reusing complete v4.13 fusion seed $Seed ($($Summary.status))"
                $ReuseFusion = $true
            }
        }
        if (-not $ReuseFusion) {
            $CommandArgs = @(
                "scripts/analyze_object_event_v4_13_selective_dual_head.py",
                "--config", $FusionConfig,
                "--v412-summary", "$Screen/summary.json",
                "--predictions", "$Screen/validation_predictions.csv",
                "--output-dir", $Fusion
            )
            if ($Force -or (Test-Path $Fusion)) { $CommandArgs += "--force" }
            Invoke-Checked "v4.13 fusion seed $Seed" { uv run --no-sync python @CommandArgs } @(0, 2)
        }
    }
}

$AggregateArgs = @(
    "scripts/aggregate_object_event_v4_14_multiseed.py",
    "--direction-root", $DirectionRoot,
    "--fusion-root", $FusionRoot,
    "--config", $AggregateConfig,
    "--output-dir", $Out
)
if ($Force -or (Test-Path $Out)) { $AggregateArgs += "--force" }
uv run --no-sync python @AggregateArgs
exit $LASTEXITCODE
