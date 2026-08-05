param(
    [string]$CacheManifest = "artifacts\cache\garl_object_event_common_roi_screen_v4\manifest.json",
    [string]$Config = "configs\experiment\e_jepa_garl_object_event_screen_v4_2.yaml",
    [string]$OutputDir = "artifacts\runs\e_jepa_garl_object_event_screen_v4_2\scratch\seed-7",
    [string]$Device = "cuda",
    [int]$Seed = 7,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

uv run --no-sync python scripts\preflight_object_event_v4_2.py `
    --cache-manifest $CacheManifest
if ($LASTEXITCODE -ne 0) { throw "Falló el preflight v4.2" }

$Arguments = @(
    "scripts\train_e_jepa_object_event_v4_2.py",
    "--cache-manifest", $CacheManifest,
    "--config", $Config,
    "--output-dir", $OutputDir,
    "--device", $Device,
    "--seed", "$Seed"
)
if ($Force) { $Arguments += "--force" }

uv run --no-sync python @Arguments
if ($LASTEXITCODE -eq 2) {
    throw "El screen v4.2 terminó correctamente, pero no superó los gates científicos. Revisa summary.json."
}
if ($LASTEXITCODE -ne 0) { throw "Falló operacionalmente el screen v4.2" }

Write-Host "V4.2 event-only screen superado." -ForegroundColor Green
