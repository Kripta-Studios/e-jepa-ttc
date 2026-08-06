param(
    [string]$CacheManifest = "artifacts\cache\garl_object_event_common_roi_screen_v4\manifest.json",
    [string]$V46Root = "artifacts\debug\object_event_v4_6_height_ratio",
    [string]$OutputRoot = "artifacts\debug\object_event_v4_7_highres_extent",
    [string]$Device = "cuda",
    [int]$Seed = 7,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Config = "configs\experiment\e_jepa_garl_object_event_highres_extent_v4_7.yaml"
$V46Summary = Join-Path $V46Root "screen-seed-$Seed\summary.json"
$V46Predictions = Join-Path $V46Root "screen-seed-$Seed\validation_predictions.csv"
$V46Checkpoint = Join-Path $V46Root "screen-seed-$Seed\best_observed.pt"
if (-not (Test-Path $V46Checkpoint)) {
    $V46Checkpoint = Join-Path $V46Root "screen-seed-$Seed\eligible.pt"
}
if (-not (Test-Path $V46Checkpoint)) {
    throw "No se encontró checkpoint v4.6 en $V46Root\screen-seed-$Seed"
}

uv run --no-sync python scripts\preflight_object_event_v4_7.py `
    --cache-manifest $CacheManifest `
    --v46-summary $V46Summary `
    --v46-checkpoint $V46Checkpoint `
    --v46-validation-predictions $V46Predictions `
    --config $Config
if ($LASTEXITCODE -ne 0) { throw "Falló el preflight v4.7" }

$Overfit = Join-Path $OutputRoot "overfit64"
$Screen = Join-Path $OutputRoot "screen-seed-$Seed"
$ForceArg = @()
if ($Force) { $ForceArg = @("--force") }

uv run --no-sync python scripts\train_e_jepa_object_event_v4_7.py `
    --cache-manifest $CacheManifest `
    --config $Config `
    --initial-v46-checkpoint $V46Checkpoint `
    --output-dir $Overfit `
    --device $Device `
    --mode overfit `
    @ForceArg
$OverfitExit = $LASTEXITCODE
if ($OverfitExit -eq 1) { throw "Falló operacionalmente el overfit v4.7" }
if ($OverfitExit -ne 0) {
    Write-Warning "El overfit v4.7 terminó, pero falló un gate científico. No se ejecutará el screen."
    exit 2
}

uv run --no-sync python scripts\train_e_jepa_object_event_v4_7.py `
    --cache-manifest $CacheManifest `
    --config $Config `
    --initial-v46-checkpoint $V46Checkpoint `
    --output-dir $Screen `
    --device $Device `
    --mode screen `
    @ForceArg
$ScreenExit = $LASTEXITCODE
if ($ScreenExit -eq 1) { throw "Falló operacionalmente el screen v4.7" }
exit $ScreenExit
