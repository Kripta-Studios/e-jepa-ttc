<#
Plan:    ./scripts/runpod_test12_parity_probe.ps1
Execute: ./scripts/runpod_test12_parity_probe.ps1 -RawRoot D:/test12_media -Execute
Download the single pinned H5 using the command printed by planning first.
This wrapper never downloads data or launches training.
#>
param(
    [string]$Python = "",
    [string]$CodeRoot = "",
    [string]$RawRoot = "",
    [string]$Output = "",
    [switch]$Execute
)
$ErrorActionPreference = 'Stop'
$probeRepo = Split-Path -Parent $PSScriptRoot
$probeSibling = Split-Path -Parent $probeRepo
if (-not $Python) { $Python = Join-Path $probeSibling 'e-jepa-ttc/.venv/Scripts/python.exe' }
if (-not $CodeRoot) { $CodeRoot = Join-Path $probeSibling 'e-jepa-ttc-v16-multimodal-validation' }
if (-not $RawRoot) { $RawRoot = Join-Path $probeRepo 'artifacts/test12_staged_media' }
if (-not $Output) { $Output = Join-Path $probeRepo 'artifacts/test12_single_query_parity' }
$probeFinal = Join-Path $probeRepo 'artifacts/runpod_event_migration_20261026/final_20261027'
$probeArgs = @(
    (Join-Path $PSScriptRoot 'runpod_test12_parity_probe.py'),
    '--frozen-inputs', (Join-Path $CodeRoot 'artifacts/multimodal_research_20261012/test12_phase_only/INPUTS.json'),
    '--event-freeze', (Join-Path $CodeRoot 'artifacts/research_independent_20261012T165833Z/DEPENDENCIES/OFFICIAL_TEST12/H8_CURRENT_SENSOR_INPUT_FREEZE.json'),
    '--public-inputs', (Join-Path $probeFinal 'public_test12/data/test_inputs.parquet'),
    '--inventory', (Join-Path $probeFinal 'TEST12_EXACT_INVENTORY.json'),
    '--code-root', $CodeRoot, '--raw-root', $RawRoot, '--output', $Output
)
if ($Execute) { $probeArgs += '--execute' }
& $Python @probeArgs
exit $LASTEXITCODE
