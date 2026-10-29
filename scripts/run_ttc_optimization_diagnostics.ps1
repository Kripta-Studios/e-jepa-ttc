param(
    [Parameter(Mandatory = $true)]
    [string]$EapRoot,

    [Parameter(Mandatory = $true)]
    [string]$GarlRoot,

    [string]$Split = "data\splits\eap_pilot12_v1.json",

    [string]$Config =
        "configs\experiment\e_jepa_garl_event_dense_level_dynamics_stable_scratch_screen_v3.yaml",

    [string]$FailedCheckpoint =
        "artifacts\runs\stable_screen_v3\scratch\seed-7\last.pt",

    [int[]]$Sizes = @(16, 32, 64, 128, 256),

    [int]$OptimizerSteps = 800,

    [int]$EvalEvery = 100,

    [switch]$SkipGradientAudits,

    [switch]$Force
)

$ErrorActionPreference = "Stop"

$ScaleScript = "artifacts\debug\diagnose_ttc_scale_sweep.py"
$GradientScript = "artifacts\debug\diagnose_ttc_gradient_conflict.py"
$OutputRoot = "artifacts\debug\ttc_scale_sweep_scratch"

foreach ($Path in @(
    $EapRoot,
    $GarlRoot,
    $Split,
    $Config,
    $ScaleScript,
    $GradientScript
)) {
    if (-not (Test-Path $Path)) {
        throw "No existe: $Path"
    }
}

if (-not $SkipGradientAudits) {
    Write-Host "`n===== GRADIENTES: SCRATCH INICIAL =====" -ForegroundColor Cyan

    & uv run --no-sync python $GradientScript `
        --eap-root $EapRoot `
        --garlttc-root $GarlRoot `
        --split $Split `
        --config $Config `
        --selection-samples 256 `
        --samples-per-bucket 8 `
        --batch-size 2 `
        --num-workers 2 `
        --seed 7 `
        --device cuda `
        --output "artifacts\debug\gradient_conflict_scratch_init.json"

    if ($LASTEXITCODE -ne 0) {
        throw "Falló la auditoría de gradientes scratch"
    }

    if (Test-Path $FailedCheckpoint) {
        Write-Host "`n===== GRADIENTES: CHECKPOINT COLAPSADO =====" -ForegroundColor Cyan

        & uv run --no-sync python $GradientScript `
            --eap-root $EapRoot `
            --garlttc-root $GarlRoot `
            --split $Split `
            --config $Config `
            --checkpoint $FailedCheckpoint `
            --selection-samples 256 `
            --samples-per-bucket 8 `
            --batch-size 2 `
            --num-workers 2 `
            --seed 7 `
            --device cuda `
            --output "artifacts\debug\gradient_conflict_failed_checkpoint.json"

        if ($LASTEXITCODE -ne 0) {
            throw "Falló la auditoría de gradientes del checkpoint"
        }
    }
    else {
        Write-Warning "No existe el checkpoint fallido: $FailedCheckpoint"
    }
}

Write-Host "`n===== BARRIDO DE ESCALA =====" -ForegroundColor Cyan

$ScaleArgs = @(
    "run",
    "--no-sync",
    "python",
    $ScaleScript,
    "--eap-root",
    $EapRoot,
    "--garlttc-root",
    $GarlRoot,
    "--split",
    $Split,
    "--config",
    $Config,
    "--output-root",
    $OutputRoot,
    "--validation-samples",
    "256",
    "--optimizer-steps",
    "$OptimizerSteps",
    "--eval-every",
    "$EvalEvery",
    "--batch-size",
    "2",
    "--accumulation-steps",
    "8",
    "--learning-rate",
    "0.0001",
    "--weight-decay",
    "0",
    "--max-grad-norm",
    "1",
    "--num-workers",
    "2",
    "--seed",
    "7",
    "--device",
    "cuda",
    "--sizes"
)

foreach ($Size in $Sizes) {
    $ScaleArgs += "$Size"
}

if ($Force) {
    $ScaleArgs += "--force"
}

& uv @ScaleArgs

if ($LASTEXITCODE -ne 0) {
    throw "Falló el barrido de escala"
}

Write-Host "`n===== RESULTADO =====" -ForegroundColor Green
Get-Content "$OutputRoot\summary.json" -Raw
