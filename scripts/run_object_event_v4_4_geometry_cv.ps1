param(
    [string]$CacheManifest = "artifacts\cache\garl_object_event_common_roi_screen_v4\manifest.json",
    [string]$RunRoot = "artifacts\runs\e_jepa_garl_object_event_screen_v4_2\scratch",
    [string]$Config = "configs\experiment\e_jepa_garl_object_event_geometry_cv_v4_4.yaml",
    [string]$OutputDir = "artifacts\debug\object_event_v4_4_geometry_cv",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

uv run --no-sync python scripts\preflight_object_event_v4_4.py `
    --cache-manifest $CacheManifest `
    --run-root $RunRoot `
    --config $Config
if ($LASTEXITCODE -ne 0) {
    throw "Falló el preflight v4.4"
}

$Arguments = @(
    "run", "--no-sync", "python",
    "scripts\diagnose_object_event_v4_4_geometry_cv.py",
    "--cache-manifest", $CacheManifest,
    "--run-root", $RunRoot,
    "--config", $Config,
    "--output-dir", $OutputDir
)
if ($Force) {
    $Arguments += "--force"
}

& uv @Arguments
$ExitCode = $LASTEXITCODE
if ($ExitCode -notin @(0, 2)) {
    throw "Fallo operacional v4.4 con código $ExitCode"
}
exit $ExitCode
