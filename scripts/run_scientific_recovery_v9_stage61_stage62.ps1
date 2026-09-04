param(
    [Parameter(Mandatory=$true)][string]$Package,
    [Parameter(Mandatory=$true)][string]$HandoffRoot,
    [Parameter(Mandatory=$true)][string]$ReferenceRoot,
    [Parameter(Mandatory=$true)][string]$CacheRoot,
    [Parameter(Mandatory=$true)][string]$RouterRoot,
    [string]$CampaignRoot = "artifacts/scientific_recovery_v9_stage61_stage62",
    [string]$PythonExe = "",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:OMP_NUM_THREADS = "16"
$env:MKL_NUM_THREADS = "16"
$env:OPENBLAS_NUM_THREADS = "16"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = (Join-Path $repo "src") + ";" + $repo
$campaign = [System.IO.Path]::GetFullPath((Join-Path $repo $CampaignRoot))
$features = Join-Path $campaign "feature_cache"
$router = Join-Path $campaign "stage61"
$x2 = Join-Path $campaign "stage62"

if ((Test-Path -LiteralPath $campaign) -and -not $Resume) {
    throw "Campaign root already exists; use -Resume only for an identity-preserving continuation."
}
New-Item -ItemType Directory -Force -Path $campaign | Out-Null

function Invoke-StagePython {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
    if ($PythonExe) {
        & $PythonExe @Arguments
    } else {
        uv run python @Arguments
    }
    if ($LASTEXITCODE -ne 0) { throw "Python stage failed: $($Arguments -join ' ')" }
}

Invoke-StagePython scripts/preflight_scientific_recovery_v9_stage61.py `
    --package $Package --handoff-root $HandoffRoot --reference-root $ReferenceRoot `
    --cache-root $CacheRoot --router-root $RouterRoot `
    --output (Join-Path $campaign "PREFLIGHT.json") --require-clean

if (-not (Test-Path -LiteralPath $features)) {
    Invoke-StagePython scripts/build_scientific_recovery_v9_stage61_reference.py `
        --reference-root $ReferenceRoot --cache-root $CacheRoot --router-root $RouterRoot `
        --output-root $features --device cuda --batch-size 32
}

if (-not (Test-Path -LiteralPath (Join-Path $router "aggregate_seed7/STAGE61_GATE.json"))) {
    Invoke-StagePython scripts/run_scientific_recovery_v9_stage61.py `
        --feature-root $features --router-root $RouterRoot --reference-root $ReferenceRoot `
        --output-root $router --seed 7 --device cuda
}
$stage61 = Get-Content -Raw -LiteralPath (Join-Path $router "aggregate_seed7/STAGE61_GATE.json") | ConvertFrom-Json
if (-not $stage61.gate_passed) {
    if (-not (Test-Path -LiteralPath (Join-Path $x2 "aggregate_seed7/STAGE62_GATE.json"))) {
        Invoke-StagePython scripts/run_scientific_recovery_v9_stage62.py `
            --feature-root $features --router-root $RouterRoot --reference-root $ReferenceRoot `
            --output-root $x2 --seed 7 --device cuda
    }
}

Invoke-StagePython scripts/audit_scientific_recovery_v9_x3_feasibility.py `
    --cache-root $CacheRoot --output (Join-Path $campaign "X3_FEASIBILITY.json")

Invoke-StagePython scripts/aggregate_scientific_recovery_v9_stage61.py `
    --campaign-root $campaign --output (Join-Path $campaign "ESSENTIAL_INVENTORY.json")
Invoke-StagePython scripts/analyze_scientific_recovery_v9_stage61.py `
    --campaign-root $campaign --repo $repo
Invoke-StagePython scripts/package_scientific_recovery_v9_stage61.py `
    --campaign-root $campaign --handoff-root $HandoffRoot --repo $repo
