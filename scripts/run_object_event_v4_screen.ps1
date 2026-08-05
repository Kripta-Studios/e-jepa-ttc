param(
    [Parameter(Mandatory = $true)]
    [string]$EapRoot,

    [Parameter(Mandatory = $true)]
    [string]$GarlTtcRoot,

    [Parameter(Mandatory = $true)]
    [string]$LevelCheckpoint,

    [string]$Split = "artifacts\manifests\eap_level_dynamics_v1.json",
    [string]$Device = "cuda",
    [int]$Workers = 4,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"

function Test-AssignmentSplit {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }
    try {
        $value = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        return (
            $null -ne $value.assignments -and
            $null -ne $value.assignments.train -and
            $null -ne $value.assignments.validation -and
            @($value.assignments.train).Count -gt 0 -and
            @($value.assignments.validation).Count -gt 0
        )
    }
    catch {
        return $false
    }
}

function Resolve-AssignmentSplit {
    param([Parameter(Mandatory = $true)][string]$Candidate)

    if (Test-AssignmentSplit -Path $Candidate) {
        return (Resolve-Path -LiteralPath $Candidate).Path
    }

    $LegacyManifest =
        "artifacts\cache\garl_object_lhr_screen_v2\manifest.json"
    if (Test-Path -LiteralPath $LegacyManifest -PathType Leaf) {
        $Legacy = Get-Content -LiteralPath $LegacyManifest -Raw |
            ConvertFrom-Json
        $Fallback = [string]$Legacy.split_path
        if (
            -not [string]::IsNullOrWhiteSpace($Fallback) -and
            (Test-AssignmentSplit -Path $Fallback)
        ) {
            $Resolved = (Resolve-Path -LiteralPath $Fallback).Path
            Write-Warning (
                "El split solicitado no contiene assignments. " +
                "Se reutiliza el split sellado por la caché v3: $Resolved"
            )
            return $Resolved
        }
    }

    $Keys = @()
    if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
        try {
            $Value = Get-Content -LiteralPath $Candidate -Raw |
                ConvertFrom-Json
            $Keys = @(
                $Value.PSObject.Properties.Name | Sort-Object
            )
        }
        catch { }
    }
    throw (
        "Object Event v4 necesita un split con assignments.train y " +
        "assignments.validation. Candidate=$Candidate; " +
        "top_level_keys=$($Keys -join ','). " +
        "eap_level_dynamics_v1.json es un manifest de subset, no un split."
    )
}

$Split = Resolve-AssignmentSplit -Candidate $Split
Write-Host "Split v4 resuelto: $Split" -ForegroundColor DarkCyan

$CacheDir = "artifacts\cache\garl_object_event_common_roi_screen_v4"
$Manifest = "$CacheDir\manifest.json"
$RunRoot = "artifacts\runs\e_jepa_garl_object_event_screen_v4"

uv run --no-sync python scripts\preflight_object_event_v4.py
if ($LASTEXITCODE -ne 0) { throw "Falló el preflight de código v4" }

$CacheArgs = @(
    "scripts\build_eap_object_event_v4_cache.py",
    "--eap-root", $EapRoot,
    "--garlttc-root", $GarlTtcRoot,
    "--split", $Split,
    "--output-dir", $CacheDir,
    "--profile", "screen",
    "--workers", "$Workers"
)
if ($Resume) { $CacheArgs += "--resume" }
uv run --no-sync python @CacheArgs
if ($LASTEXITCODE -ne 0) { throw "Falló la caché común v4" }

uv run --no-sync python scripts\preflight_object_event_v4.py --manifest $Manifest
if ($LASTEXITCODE -ne 0) { throw "Falló el contrato de caché v4" }

uv run --no-sync python scripts\audit_object_event_v4_cache.py `
  --manifest $Manifest `
  --samples 256 `
  --output "artifacts\debug\object_event_v4_cache_audit.json"
if ($LASTEXITCODE -ne 0) { throw "Falló la auditoría de señal de caché v4" }

$TrainCommon = @(
    "scripts\train_e_jepa_object_event_v4.py",
    "--cache-manifest", $Manifest,
    "--config", "configs\experiment\e_jepa_garl_object_event_screen_v4.yaml",
    "--seed", "7",
    "--device", $Device
)
if ($Resume) { $TrainCommon += "--resume" }

uv run --no-sync python @TrainCommon `
  --output-dir "$RunRoot\scratch\seed-7"
if ($LASTEXITCODE -ne 0) { throw "Scratch v4 no completó o no pasó gates" }

uv run --no-sync python @TrainCommon `
  --output-dir "$RunRoot\level-transfer\seed-7" `
  --pretrained $LevelCheckpoint
if ($LASTEXITCODE -ne 0) { throw "Level-transfer v4 no completó o no pasó gates" }

uv run --no-sync python scripts\compare_object_event_v4_runs.py `
  --scratch "$RunRoot\scratch\seed-7\summary.json" `
  --level "$RunRoot\level-transfer\seed-7\summary.json" `
  --output "artifacts\debug\object_event_v4_screen_comparison.csv"
if ($LASTEXITCODE -ne 0) { throw "Falló la comparación v4" }

Write-Host "Object Event TTC v4 screen finalizado." -ForegroundColor Green
Write-Host "Sube:" -ForegroundColor Green
Write-Host "artifacts\debug\object_event_v4_cache_audit.json" -ForegroundColor Cyan
Write-Host "artifacts\debug\object_event_v4_screen_comparison.csv" -ForegroundColor Cyan
Write-Host "$RunRoot\scratch\seed-7\summary.json" -ForegroundColor Cyan
Write-Host "$RunRoot\level-transfer\seed-7\summary.json" -ForegroundColor Cyan
