param(
    [int]$Seed = 7,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Config = "configs\experiment\e_jepa_garl_object_event_fixed_fusion_v4_9.yaml"
$V42Root = "artifacts\runs\e_jepa_garl_object_event_screen_v4_2\scratch\seed-$Seed"
$V48Root = "artifacts\debug\object_event_v4_8_dense_motion\screen-seed-$Seed"
$Output = "artifacts\debug\object_event_v4_9_fixed_fusion\seed-$Seed"

$V42Summary = Join-Path $V42Root "summary.json"
$V42Train = Join-Path $V42Root "train_predictions.csv"
$V42Validation = Join-Path $V42Root "validation_predictions.csv"
$V48Summary = Join-Path $V48Root "summary.json"
$V48Train = Join-Path $V48Root "train_predictions.csv"
$V48Validation = Join-Path $V48Root "validation_predictions.csv"

uv run --no-sync python scripts\preflight_object_event_v4_9.py `
    --v42-summary $V42Summary `
    --v42-train-predictions $V42Train `
    --v42-validation-predictions $V42Validation `
    --v48-summary $V48Summary `
    --v48-train-predictions $V48Train `
    --v48-validation-predictions $V48Validation
if ($LASTEXITCODE -ne 0) { throw "Falló el preflight v4.9" }

$ForceArg = @()
if ($Force) { $ForceArg = @("--force") }

uv run --no-sync python scripts\analyze_object_event_v4_9_fixed_fusion.py `
    --config $Config `
    --v42-summary $V42Summary `
    --v42-train-predictions $V42Train `
    --v42-validation-predictions $V42Validation `
    --v48-summary $V48Summary `
    --v48-train-predictions $V48Train `
    --v48-validation-predictions $V48Validation `
    --output-dir $Output `
    @ForceArg
exit $LASTEXITCODE
