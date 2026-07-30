[CmdletBinding()]
param(
    [ValidateSet("Analysis", "Full")]
    [string]$Profile = "Analysis",
    [ValidateSet("both", "ssl", "geo")]
    [string[]]$Objectives = @("both"),
    [ValidateSet("all", "pretrain", "evttc-control", "transfer", "compare")]
    [string[]]$Stages = @("all"),
    [string]$EapRoot = "E:\eAP_dataset",
    [string]$EapSplit = "",
    [int[]]$Folds = @(),
    [int[]]$Seeds = @(),
    [int]$EapWorkers = 8,
    [int]$EvttcWorkers = 12,
    [int]$EapBatchSize = 24,
    [int]$EapGradientAccumulation = 2,
    [switch]$Resume,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "No existe $Python. Ejecuta: uv sync --locked --all-groups --no-editable"
}

Push-Location $RepoRoot
try {
    $Arguments = @(
        "scripts/run_eap_evttc_complete.py",
        "--profile", $Profile.ToLowerInvariant(),
        "--objectives"
    )
    $Arguments += $Objectives
    $Arguments += "--stages"
    $Arguments += $Stages
    $Arguments += @(
        "--eap-root", $EapRoot,
        "--eap-workers", "$EapWorkers",
        "--evttc-workers", "$EvttcWorkers",
        "--eap-batch-size", "$EapBatchSize",
        "--eap-gradient-accumulation", "$EapGradientAccumulation"
    )
    if ($EapSplit) { $Arguments += @("--eap-split", $EapSplit) }
    if ($Folds.Count -gt 0) { $Arguments += @("--folds") + $Folds }
    if ($Seeds.Count -gt 0) { $Arguments += @("--seeds") + $Seeds }
    if ($Resume) { $Arguments += "--resume" }
    if ($DryRun) { $Arguments += "--dry-run" }
    $env:PYTHONPATH = Join-Path $RepoRoot "src"
    $env:PYTHONUTF8 = "1"
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "El pipeline eAP→EvTTC terminó con código $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
