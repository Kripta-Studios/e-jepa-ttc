param(
    [string]$Device = "cuda",
    [int[]]$Seeds = @(7, 13, 23),
    [switch]$Force,
    [switch]$SkipTraining
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Cache = "artifacts\cache\garl_object_event_common_roi_screen_v4\manifest.json"
$V42Root = "artifacts\runs\e_jepa_garl_object_event_screen_v4_2\scratch"
$V45Summary = "artifacts\debug\object_event_v4_5_paired_mid_multiseed\summary.json"
$V46Root = "artifacts\debug\object_event_v4_6_height_ratio"
$V47Root = "artifacts\debug\object_event_v4_7_highres_extent"
$V48Root = "artifacts\debug\object_event_v4_8_dense_motion"
$V49Root = "artifacts\debug\object_event_v4_9_fixed_fusion"
$RuntimeRoot = "artifacts\debug\object_event_v4_10_runtime"
$Output = "artifacts\debug\object_event_v4_10_multiseed"

$V46BaseConfig = "configs\experiment\e_jepa_garl_object_event_height_ratio_v4_6.yaml"
$V47BaseConfig = "configs\experiment\e_jepa_garl_object_event_highres_extent_v4_7.yaml"
$V48BaseConfig = "configs\experiment\e_jepa_garl_object_event_dense_motion_v4_8.yaml"
$V49Config = "configs\experiment\e_jepa_garl_object_event_fixed_fusion_v4_9.yaml"
$V410Config = "configs\experiment\e_jepa_garl_object_event_fixed_fusion_multiseed_v4_10.yaml"

function Invoke-PythonChecked {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [int[]]$AllowedExitCodes = @(0),
        [string]$Label = "Python command"
    )
    & uv run --no-sync python @Arguments
    $Code = $LASTEXITCODE
    if ($AllowedExitCodes -notcontains $Code) {
        throw "$Label failed with exit code $Code"
    }
}


function Test-SummarySeed {
    param(
        [Parameter(Mandatory = $true)][string]$SummaryPath,
        [Parameter(Mandatory = $true)][int]$ExpectedSeed
    )
    if (-not (Test-Path -LiteralPath $SummaryPath)) { return $false }
    try {
        $Payload = Get-Content $SummaryPath -Raw | ConvertFrom-Json
        return ([int]$Payload.train_config.seed -eq $ExpectedSeed)
    }
    catch {
        return $false
    }
}

function Test-V49Seed {
    param(
        [Parameter(Mandatory = $true)][string]$SummaryPath,
        [Parameter(Mandatory = $true)][int]$ExpectedSeed
    )
    if (-not (Test-Path -LiteralPath $SummaryPath)) { return $false }
    try {
        $Payload = Get-Content $SummaryPath -Raw | ConvertFrom-Json
        if ($Payload.passed -ne $true -or [double]$Payload.fusion_config.alpha -ne 0.5) {
            return $false
        }
        $V48SummaryPath = [string]$Payload.source_artifacts.v4_8_summary
        return (Test-SummarySeed -SummaryPath $V48SummaryPath -ExpectedSeed $ExpectedSeed)
    }
    catch {
        return $false
    }
}
function Get-PreferredCheckpoint {
    param([Parameter(Mandatory = $true)][string]$Directory)
    $Candidates = @(
        (Join-Path $Directory "eligible.pt"),
        (Join-Path $Directory "best_gate_passing.pt"),
        (Join-Path $Directory "best_observed.pt")
    )
    foreach ($Candidate in $Candidates) {
        if (Test-Path -LiteralPath $Candidate) { return $Candidate }
    }
    throw "No checkpoint found in $Directory"
}

$SeedStrings = $Seeds | ForEach-Object { "$_" }
$PreflightArguments = @(
    "scripts\preflight_object_event_v4_10.py",
    "--v42-run-root", $V42Root,
    "--v49-run-root", $V49Root,
    "--v49-config", $V49Config,
    "--seeds"
) + $SeedStrings
Invoke-PythonChecked -Arguments $PreflightArguments -Label "v4.10 preflight"

foreach ($Seed in $Seeds) {
    $V49SeedRoot = Join-Path $V49Root "seed-$Seed"
    $V49Summary = Join-Path $V49SeedRoot "summary.json"
    if ((Test-V49Seed -SummaryPath $V49Summary -ExpectedSeed $Seed) -and -not $Force) {
        Write-Host "Reusing eligible true-seed v4.9 seed ${Seed}: $V49Summary" -ForegroundColor DarkCyan
        continue
    }
    if ($SkipTraining) {
        throw "Missing eligible v4.9 seed $Seed while -SkipTraining was requested"
    }

    $SeedConfigRoot = Join-Path $RuntimeRoot "configs\seed-$Seed"
    Invoke-PythonChecked -Arguments @(
        "scripts\prepare_object_event_v4_10_seed_configs.py",
        "--seed", "$Seed",
        "--v46-config", $V46BaseConfig,
        "--v47-config", $V47BaseConfig,
        "--v48-config", $V48BaseConfig,
        "--output-dir", $SeedConfigRoot
    ) -Label "materialize seed $Seed configs"
    $V46Config = Join-Path $SeedConfigRoot "v46_seed_${Seed}.yaml"
    $V47Config = Join-Path $SeedConfigRoot "v47_seed_${Seed}.yaml"
    $V48Config = Join-Path $SeedConfigRoot "v48_seed_${Seed}.yaml"

    $V42SeedRoot = Join-Path $V42Root "seed-$Seed"
    $V42Summary = Join-Path $V42SeedRoot "summary.json"
    $V42Checkpoint = Get-PreferredCheckpoint -Directory $V42SeedRoot

    $V46Screen = Join-Path $V46Root "screen-seed-$Seed"
    $V46ScreenSummary = Join-Path $V46Screen "summary.json"
    if ($Force -or -not (Test-SummarySeed -SummaryPath $V46ScreenSummary -ExpectedSeed $Seed)) {
        Invoke-PythonChecked -Arguments @(
            "scripts\preflight_object_event_v4_6.py",
            "--cache-manifest", $Cache,
            "--v42-checkpoint", $V42Checkpoint,
            "--v45-summary", $V45Summary,
            "--config", $V46Config,
            "--seed", "$Seed"
        ) -Label "v4.6 preflight seed $Seed"
        $V46Overfit = Join-Path $V46Root "overfit64-seed-$Seed"
        $V46OverfitArgs = @(
            "scripts\train_e_jepa_object_event_v4_6.py",
            "--cache-manifest", $Cache,
            "--config", $V46Config,
            "--initial-checkpoint", $V42Checkpoint,
            "--output-dir", $V46Overfit,
            "--device", $Device,
            "--mode", "overfit"
        )
        if ($Force -or (Test-Path -LiteralPath $V46Overfit)) { $V46OverfitArgs += "--force" }
        Invoke-PythonChecked -Arguments $V46OverfitArgs -Label "v4.6 overfit seed $Seed"

        $V46ScreenArgs = @(
            "scripts\train_e_jepa_object_event_v4_6.py",
            "--cache-manifest", $Cache,
            "--config", $V46Config,
            "--initial-checkpoint", $V42Checkpoint,
            "--output-dir", $V46Screen,
            "--device", $Device,
            "--mode", "screen"
        )
        if ($Force -or (Test-Path -LiteralPath $V46Screen)) { $V46ScreenArgs += "--force" }
        Invoke-PythonChecked -Arguments $V46ScreenArgs -AllowedExitCodes @(0, 2) -Label "v4.6 screen seed $Seed"
    }

    $V46Checkpoint = Get-PreferredCheckpoint -Directory $V46Screen
    $V46Predictions = Join-Path $V46Screen "validation_predictions.csv"
    $V47Screen = Join-Path $V47Root "screen-seed-$Seed"
    $V47ScreenSummary = Join-Path $V47Screen "summary.json"
    if ($Force -or -not (Test-SummarySeed -SummaryPath $V47ScreenSummary -ExpectedSeed $Seed)) {
        Invoke-PythonChecked -Arguments @(
            "scripts\preflight_object_event_v4_7.py",
            "--cache-manifest", $Cache,
            "--v46-summary", $V46ScreenSummary,
            "--v46-checkpoint", $V46Checkpoint,
            "--v46-validation-predictions", $V46Predictions,
            "--config", $V47Config
        ) -Label "v4.7 preflight seed $Seed"
        $V47Overfit = Join-Path $V47Root "overfit64-seed-$Seed"
        $V47OverfitArgs = @(
            "scripts\train_e_jepa_object_event_v4_7.py",
            "--cache-manifest", $Cache,
            "--config", $V47Config,
            "--initial-v46-checkpoint", $V46Checkpoint,
            "--output-dir", $V47Overfit,
            "--device", $Device,
            "--mode", "overfit"
        )
        if ($Force -or (Test-Path -LiteralPath $V47Overfit)) { $V47OverfitArgs += "--force" }
        Invoke-PythonChecked -Arguments $V47OverfitArgs -Label "v4.7 overfit seed $Seed"

        $V47ScreenArgs = @(
            "scripts\train_e_jepa_object_event_v4_7.py",
            "--cache-manifest", $Cache,
            "--config", $V47Config,
            "--initial-v46-checkpoint", $V46Checkpoint,
            "--output-dir", $V47Screen,
            "--device", $Device,
            "--mode", "screen"
        )
        if ($Force -or (Test-Path -LiteralPath $V47Screen)) { $V47ScreenArgs += "--force" }
        Invoke-PythonChecked -Arguments $V47ScreenArgs -AllowedExitCodes @(0, 2) -Label "v4.7 screen seed $Seed"
    }

    $V47Checkpoint = Get-PreferredCheckpoint -Directory $V47Screen
    $V48Screen = Join-Path $V48Root "screen-seed-$Seed"
    $V48ScreenSummary = Join-Path $V48Screen "summary.json"
    if ($Force -or -not (Test-SummarySeed -SummaryPath $V48ScreenSummary -ExpectedSeed $Seed)) {
        Invoke-PythonChecked -Arguments @(
            "scripts\preflight_object_event_v4_8.py",
            "--cache-manifest", $Cache,
            "--v47-summary", $V47ScreenSummary,
            "--v47-checkpoint", $V47Checkpoint
        ) -Label "v4.8 preflight seed $Seed"
        $V48Overfit = Join-Path $V48Root "overfit64-seed-$Seed"
        $V48OverfitArgs = @(
            "scripts\train_e_jepa_object_event_v4_8.py",
            "--cache-manifest", $Cache,
            "--config", $V48Config,
            "--initial-v47-checkpoint", $V47Checkpoint,
            "--output-dir", $V48Overfit,
            "--device", $Device,
            "--mode", "overfit"
        )
        if ($Force -or (Test-Path -LiteralPath $V48Overfit)) { $V48OverfitArgs += "--force" }
        Invoke-PythonChecked -Arguments $V48OverfitArgs -Label "v4.8 overfit seed $Seed"

        $V48ScreenArgs = @(
            "scripts\train_e_jepa_object_event_v4_8.py",
            "--cache-manifest", $Cache,
            "--config", $V48Config,
            "--initial-v47-checkpoint", $V47Checkpoint,
            "--output-dir", $V48Screen,
            "--device", $Device,
            "--mode", "screen"
        )
        if ($Force -or (Test-Path -LiteralPath $V48Screen)) { $V48ScreenArgs += "--force" }
        Invoke-PythonChecked -Arguments $V48ScreenArgs -Label "v4.8 screen seed $Seed"
    }

    $V42Train = Join-Path $V42SeedRoot "train_predictions.csv"
    $V42Validation = Join-Path $V42SeedRoot "validation_predictions.csv"
    $V48Train = Join-Path $V48Screen "train_predictions.csv"
    $V48Validation = Join-Path $V48Screen "validation_predictions.csv"
    Invoke-PythonChecked -Arguments @(
        "scripts\preflight_object_event_v4_9.py",
        "--v42-summary", $V42Summary,
        "--v42-train-predictions", $V42Train,
        "--v42-validation-predictions", $V42Validation,
        "--v48-summary", $V48ScreenSummary,
        "--v48-train-predictions", $V48Train,
        "--v48-validation-predictions", $V48Validation
    ) -Label "v4.9 preflight seed $Seed"
    $V49Args = @(
        "scripts\analyze_object_event_v4_9_fixed_fusion.py",
        "--config", $V49Config,
        "--v42-summary", $V42Summary,
        "--v42-train-predictions", $V42Train,
        "--v42-validation-predictions", $V42Validation,
        "--v48-summary", $V48ScreenSummary,
        "--v48-train-predictions", $V48Train,
        "--v48-validation-predictions", $V48Validation,
        "--output-dir", $V49SeedRoot
    )
    if ($Force -or (Test-Path -LiteralPath $V49SeedRoot)) { $V49Args += "--force" }
    Invoke-PythonChecked -Arguments $V49Args -Label "v4.9 fixed fusion seed $Seed"
}

$AggregateArgs = @(
    "scripts\aggregate_object_event_v4_10_multiseed.py",
    "--run-root", $V49Root,
    "--config", $V410Config,
    "--output-dir", $Output
)
if ($Force -or (Test-Path -LiteralPath $Output)) { $AggregateArgs += "--force" }
& uv run --no-sync python @AggregateArgs
exit $LASTEXITCODE
