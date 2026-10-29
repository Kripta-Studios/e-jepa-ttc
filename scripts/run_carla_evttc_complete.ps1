[CmdletBinding()]
param(
    [ValidateSet("Smoke", "Full")]
    [string]$Profile = "Full",
    [ValidateSet("all", "carla", "evttc-control", "transfer", "compare")]
    [string[]]$Stages = @("all"),
    [string]$CarlaRoot = "datasets/CARLA_DVS_Looming_Dataset/random_spawn",
    [string]$CarlaRunDir = "",
    [int]$CarlaWorkers = 8,
    [int]$EvttcWorkers = 12,
    [switch]$Resume,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "No existe $Python. Ejecuta primero: uv sync --locked --all-groups --no-editable"
}

Push-Location $RepoRoot
try {
    $Arguments = @(
        "scripts/run_carla_evttc_complete.py",
        "--profile", $Profile.ToLowerInvariant(),
        "--stages"
    )
    $Arguments += $Stages
    $Arguments += @(
        "--carla-root", $CarlaRoot,
        "--carla-workers", "$CarlaWorkers",
        "--evttc-workers", "$EvttcWorkers"
    )
    if (-not [string]::IsNullOrWhiteSpace($CarlaRunDir)) {
        $Arguments += @("--carla-run-dir", $CarlaRunDir)
    }
    if ($Resume) { $Arguments += "--resume" }
    if ($DryRun) { $Arguments += "--dry-run" }
    $env:PYTHONPATH = Join-Path $RepoRoot "src"
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "El pipeline CARLA→EvTTC terminó con código $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
