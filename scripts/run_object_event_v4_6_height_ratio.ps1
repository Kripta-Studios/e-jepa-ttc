param(
    [string]$CacheManifest = "artifacts\cache\garl_object_event_common_roi_screen_v4\manifest.json",
    [string]$Config = "configs\experiment\e_jepa_garl_object_event_height_ratio_v4_6.yaml",
    [string]$V42RunRoot = "artifacts\runs\e_jepa_garl_object_event_screen_v4_2\scratch",
    [string]$V45Summary = "artifacts\debug\object_event_v4_5_paired_mid_multiseed\summary.json",
    [string]$OutputRoot = "artifacts\debug\object_event_v4_6_height_ratio",
    [string]$Device = "cuda",
    [int]$Seed = 7,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$SeedDir = Join-Path $V42RunRoot "seed-$Seed"
$Eligible = Join-Path $SeedDir "eligible.pt"
$BestObserved = Join-Path $SeedDir "best_observed.pt"
$Checkpoint = if (Test-Path -LiteralPath $Eligible) { $Eligible } else { $BestObserved }
if (-not (Test-Path -LiteralPath $Checkpoint)) {
    throw "No existe checkpoint v4.2 para seed ${Seed}: $Checkpoint"
}

& uv run --no-sync python scripts\preflight_object_event_v4_6.py `
    --cache-manifest $CacheManifest `
    --v42-checkpoint $Checkpoint `
    --v45-summary $V45Summary `
    --config $Config `
    --seed $Seed
if ($LASTEXITCODE -ne 0) { throw "Falló el preflight v4.6" }

$OverfitOutput = Join-Path $OutputRoot "overfit64"
$OverfitArguments = @(
    "scripts\train_e_jepa_object_event_v4_6.py",
    "--cache-manifest", $CacheManifest,
    "--config", $Config,
    "--initial-checkpoint", $Checkpoint,
    "--output-dir", $OverfitOutput,
    "--device", $Device,
    "--mode", "overfit"
)
if ($Force -or (Test-Path -LiteralPath $OverfitOutput)) { $OverfitArguments += "--force" }

Write-Host "Ejecutando v4.6 overfit64 seed $Seed" -ForegroundColor Green
& uv run --no-sync python @OverfitArguments
$OverfitExit = $LASTEXITCODE
if ($OverfitExit -eq 1) { throw "Fallo operacional en v4.6 overfit64" }
if ($OverfitExit -eq 2) {
    Write-Warning "V4.6 no sobreajusta foreground/height ratio. No se lanza el screen completo."
    exit 2
}

$ScreenOutput = Join-Path $OutputRoot "screen-seed-$Seed"
$ScreenArguments = @(
    "scripts\train_e_jepa_object_event_v4_6.py",
    "--cache-manifest", $CacheManifest,
    "--config", $Config,
    "--initial-checkpoint", $Checkpoint,
    "--output-dir", $ScreenOutput,
    "--device", $Device,
    "--mode", "screen"
)
if ($Force -or (Test-Path -LiteralPath $ScreenOutput)) { $ScreenArguments += "--force" }

Write-Host "Ejecutando v4.6 screen completo seed $Seed" -ForegroundColor Green
& uv run --no-sync python @ScreenArguments
$ScreenExit = $LASTEXITCODE
if ($ScreenExit -eq 1) { throw "Fallo operacional en v4.6 screen" }
if ($ScreenExit -eq 2) {
    Write-Warning "V4.6 terminó correctamente, pero no superó los gates científicos."
    exit 2
}
Write-Host "V4.6 learned foreground height-ratio superado." -ForegroundColor Green
exit 0
