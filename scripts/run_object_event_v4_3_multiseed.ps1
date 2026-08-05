param(
    [string]$CacheManifest = "artifacts\cache\garl_object_event_common_roi_screen_v4\manifest.json",
    [string]$TrainConfig = "configs\experiment\e_jepa_garl_object_event_screen_v4_2.yaml",
    [string]$AggregateConfig = "configs\experiment\e_jepa_garl_object_event_multiseed_v4_3.yaml",
    [string]$RunRoot = "artifacts\runs\e_jepa_garl_object_event_screen_v4_2\scratch",
    [string]$AggregateOutput = "artifacts\debug\object_event_v4_3_multiseed",
    [string]$Device = "cuda",
    [int[]]$Seeds = @(7, 13, 23),
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$Seed7Summary = Join-Path $RunRoot "seed-7\summary.json"
uv run --no-sync python scripts\preflight_object_event_v4_3.py `
    --cache-manifest $CacheManifest `
    --seed7-summary $Seed7Summary
if ($LASTEXITCODE -ne 0) { throw "Falló el preflight v4.3" }

foreach ($Seed in $Seeds) {
    $OutputDir = Join-Path $RunRoot "seed-$Seed"
    $Summary = Join-Path $OutputDir "summary.json"
    if ((Test-Path -LiteralPath $Summary) -and -not $Force) {
        Write-Host "Reutilizando seed ${Seed}: $Summary" -ForegroundColor DarkCyan
        continue
    }

    $Arguments = @(
        "scripts\train_e_jepa_object_event_v4_2.py",
        "--cache-manifest", $CacheManifest,
        "--config", $TrainConfig,
        "--output-dir", $OutputDir,
        "--device", $Device,
        "--seed", "$Seed"
    )
    if ($Force -or (Test-Path -LiteralPath $OutputDir)) {
        $Arguments += "--force"
    }

    Write-Host "Entrenando v4.2 event-only seed $Seed" -ForegroundColor Green
    uv run --no-sync python @Arguments
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -eq 1) {
        throw "Fallo operacional en seed $Seed"
    }
    if ($ExitCode -eq 2) {
        Write-Warning "Seed $Seed terminó, pero no superó sus gates individuales; se agregará igualmente."
    }
}

uv run --no-sync python scripts\aggregate_object_event_v4_3_multiseed.py `
    --run-root $RunRoot `
    --config $AggregateConfig `
    --output-dir $AggregateOutput
$AggregateExit = $LASTEXITCODE
if ($AggregateExit -eq 1) { throw "Falló operacionalmente la agregación v4.3" }
if ($AggregateExit -eq 2) {
    Write-Warning "V4.3 multiseed terminó correctamente, pero no superó los gates de robustez."
    exit 2
}
Write-Host "V4.3 multiseed robusto superado." -ForegroundColor Green
