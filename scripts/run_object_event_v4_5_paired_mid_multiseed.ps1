param(
    [string]$CacheManifest = "artifacts\cache\garl_object_event_common_roi_screen_v4\manifest.json",
    [string]$TrainConfig = "configs\experiment\e_jepa_garl_object_event_paired_mid_v4_5.yaml",
    [string]$AggregateConfig = "configs\experiment\e_jepa_garl_object_event_paired_mid_multiseed_v4_5.yaml",
    [string]$V42RunRoot = "artifacts\runs\e_jepa_garl_object_event_screen_v4_2\scratch",
    [string]$RunRoot = "artifacts\runs\e_jepa_garl_object_event_paired_mid_v4_5",
    [string]$V44Summary = "artifacts\debug\object_event_v4_4_geometry_cv\summary.json",
    [string]$AggregateOutput = "artifacts\debug\object_event_v4_5_paired_mid_multiseed",
    [string]$Device = "cuda",
    [int[]]$Seeds = @(7, 13, 23),
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$PreflightArguments = @(
    "run", "--no-sync", "python",
    "scripts\preflight_object_event_v4_5.py",
    "--cache-manifest", $CacheManifest,
    "--v42-run-root", $V42RunRoot,
    "--v44-summary", $V44Summary,
    "--train-config", $TrainConfig,
    "--aggregate-config", $AggregateConfig,
    "--seeds"
) + ($Seeds | ForEach-Object { "$_" })
& uv @PreflightArguments
if ($LASTEXITCODE -ne 0) { throw "Falló el preflight v4.5" }

foreach ($Seed in $Seeds) {
    $V42SeedDir = Join-Path $V42RunRoot "seed-$Seed"
    $Eligible = Join-Path $V42SeedDir "eligible.pt"
    $BestObserved = Join-Path $V42SeedDir "best_observed.pt"
    $InitialCheckpoint = if (Test-Path -LiteralPath $Eligible) { $Eligible } else { $BestObserved }
    if (-not (Test-Path -LiteralPath $InitialCheckpoint)) {
        throw "No existe checkpoint v4.2 para seed ${Seed}: $InitialCheckpoint"
    }

    $OutputDir = Join-Path $RunRoot "seed-$Seed"
    $Summary = Join-Path $OutputDir "summary.json"
    if ((Test-Path -LiteralPath $Summary) -and -not $Force) {
        Write-Host "Reutilizando v4.5 seed ${Seed}: $Summary" -ForegroundColor DarkCyan
        continue
    }

    $Arguments = @(
        "scripts\train_e_jepa_object_event_v4_5.py",
        "--cache-manifest", $CacheManifest,
        "--config", $TrainConfig,
        "--initial-checkpoint", $InitialCheckpoint,
        "--output-dir", $OutputDir,
        "--device", $Device,
        "--seed", "$Seed"
    )
    if ($Force -or (Test-Path -LiteralPath $OutputDir)) { $Arguments += "--force" }

    Write-Host "Afinando v4.5 MiD/reciprocity seed $Seed" -ForegroundColor Green
    uv run --no-sync python @Arguments
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -eq 1) { throw "Fallo operacional en v4.5 seed $Seed" }
    if ($ExitCode -eq 2) {
        Write-Warning "Seed $Seed terminó, pero no superó todos sus gates; se agregará igualmente."
    }
}

$AggregateArguments = @(
    "run", "--no-sync", "python",
    "scripts\aggregate_object_event_v4_5_multiseed.py",
    "--run-root", $RunRoot,
    "--baseline-summary", $V44Summary,
    "--config", $AggregateConfig,
    "--output-dir", $AggregateOutput
)
if ($Force -or (Test-Path -LiteralPath $AggregateOutput)) { $AggregateArguments += "--force" }
& uv @AggregateArguments
$AggregateExit = $LASTEXITCODE
if ($AggregateExit -eq 1) { throw "Falló operacionalmente la agregación v4.5" }
if ($AggregateExit -eq 2) {
    Write-Warning "V4.5 terminó correctamente, pero no superó los gates científicos."
    exit 2
}
Write-Host "V4.5 paired reciprocal MiD multiseed superado." -ForegroundColor Green
exit 0
