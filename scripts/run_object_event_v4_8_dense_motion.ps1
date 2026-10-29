param(
    [string]$Device = "cuda",
    [int]$Seed = 7,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Cache = "artifacts\cache\garl_object_event_common_roi_screen_v4\manifest.json"
$Config = "configs\experiment\e_jepa_garl_object_event_dense_motion_v4_8.yaml"
$V47Root = "artifacts\debug\object_event_v4_7_highres_extent\screen-seed-$Seed"
$V47Summary = Join-Path $V47Root "summary.json"
$V47Checkpoint = Join-Path $V47Root "best_observed.pt"
if (Test-Path (Join-Path $V47Root "eligible.pt")) {
    $V47Checkpoint = Join-Path $V47Root "eligible.pt"
}
$OutputRoot = "artifacts\debug\object_event_v4_8_dense_motion"
$Overfit = Join-Path $OutputRoot "overfit64"
$Screen = Join-Path $OutputRoot "screen-seed-$Seed"

uv run --no-sync python scripts\preflight_object_event_v4_8.py `
    --cache-manifest $Cache `
    --v47-summary $V47Summary `
    --v47-checkpoint $V47Checkpoint
if ($LASTEXITCODE -ne 0) { throw "Falló el preflight v4.8" }

$ForceArg = @()
if ($Force) { $ForceArg = @("--force") }

uv run --no-sync python scripts\train_e_jepa_object_event_v4_8.py `
    --cache-manifest $Cache `
    --config $Config `
    --initial-v47-checkpoint $V47Checkpoint `
    --output-dir $Overfit `
    --device $Device `
    --mode overfit `
    @ForceArg
$OverfitExit = $LASTEXITCODE
if ($OverfitExit -eq 1) { throw "Falló operacionalmente el overfit v4.8" }
if ($OverfitExit -eq 2) {
    Write-Warning "El overfit v4.8 terminó, pero falló un gate científico. No se ejecutará el screen."
    exit 2
}

$OverfitCheckpoint = Join-Path $Overfit "eligible.pt"
if (-not (Test-Path $OverfitCheckpoint)) {
    throw "El overfit v4.8 pasó pero no produjo eligible.pt"
}

uv run --no-sync python scripts\train_e_jepa_object_event_v4_8.py `
    --cache-manifest $Cache `
    --config $Config `
    --initial-v47-checkpoint $V47Checkpoint `
    --output-dir $Screen `
    --device $Device `
    --mode screen `
    @ForceArg
exit $LASTEXITCODE
