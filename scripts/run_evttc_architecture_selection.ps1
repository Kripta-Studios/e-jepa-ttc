[CmdletBinding()]
param(
    [ValidateSet("Validate", "Smoke", "Screen", "Confirm")]
    [string]$Mode = "Screen",
    [ValidateSet("Core", "Garl", "All")]
    [string]$Stage = "Core",
    [ValidateSet("HistoricalBase", "GroupedCV")]
    [string]$Protocol = "HistoricalBase",
    [int]$Fold = 0,
    [int]$Seed = 7,
    [string[]]$Variants = @(),
    [int]$Workers = 4,
    [int]$BatchSize = 0,
    [switch]$Resume,
    [switch]$AllFolds,
    [switch]$AllSeeds,
    [switch]$SkipUnitTests,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "No existe $Python. Ejecuta primero: uv sync --dev"
}

Push-Location $RepoRoot
try {
    if (-not $SkipUnitTests) {
        & $Python -m pytest `
            tests/unit/test_oge_architecture.py `
            tests/unit/test_causal_geometry.py `
            tests/unit/test_evttc_object_cache.py `
            tests/unit/test_object_evaluation.py `
            tests/unit/test_grouped_cv.py `
            tests/unit/test_benchmark10_guard.py `
            tests/unit/test_training_controls.py `
            tests/unit/test_storage_guard.py `
            tests/unit/test_submission_format.py `
            -q
        if ($LASTEXITCODE -ne 0) {
            throw "Fallaron los tests unitarios de arquitectura."
        }
    }

    if ($Mode -eq "Validate") {
        & $Python -m ruff check src scripts tests
        if ($LASTEXITCODE -ne 0) {
            throw "Ruff encontró errores."
        }
        exit 0
    }

    if ($Protocol -eq "HistoricalBase" -and $AllFolds) {
        throw "HistoricalBase tiene un único split. Usa GroupedCV con -AllFolds."
    }
    $ProtocolArg = if ($Protocol -eq "HistoricalBase") {
        "historical_base"
    }
    else {
        "grouped_cv"
    }
    $Folds = if ($AllFolds) { @(0, 1, 2, 3, 4) } else { @($Fold) }
    $Seeds = if ($AllSeeds) { @(7, 13, 21) } else { @($Seed) }
    $CoreVariants = @(
        "A0_MATCHED_GLOBAL",
        "A1_MATCHED_DENSE_BLOCK",
        "A2_MATCHED_DENSE_ATTNRES",
        "K1_OBJECT_KDA",
        "A4_GT_GEOMETRY"
    )
    $GarlVariants = @(
        "G0_RGB_DIRECT",
        "G1_EVENT_DIRECT",
        "G2_RGBE_DIRECT_EARLY",
        "G3_RGB_LHR",
        "G4_EVENT_LHR",
        "G5_RGBE_LHR_EARLY",
        "G6_RGBE_LHR_LATE",
        "G7_RGBE_LHR_LATE_FOREGROUND"
    )
    $SelectedVariants = if ($Variants.Count -gt 0) {
        $Variants
    }
    elseif ($Stage -eq "Core") {
        $CoreVariants
    }
    elseif ($Stage -eq "Garl") {
        $GarlVariants
    }
    else {
        $GarlVariants + $CoreVariants
    }
    foreach ($CurrentFold in $Folds) {
        foreach ($CurrentSeed in $Seeds) {
            $RunnerArgs = @(
                "scripts/run_evttc_architecture_matrix.py",
                "--mode", $Mode.ToLowerInvariant(),
                "--stage-role", $Stage.ToLowerInvariant(),
                "--split-protocol", $ProtocolArg,
                "--fold", "$CurrentFold",
                "--seed", "$CurrentSeed",
                "--workers", "$Workers"
            )
            if ($Resume) {
                $RunnerArgs += "--resume"
            }
            if ($DryRun) {
                $RunnerArgs += "--dry-run"
            }
            if ($BatchSize -gt 0) {
                $RunnerArgs += @("--batch-size", "$BatchSize")
            }
            $RunnerArgs += "--variants"
            $RunnerArgs += $SelectedVariants

            & $Python @RunnerArgs
            if ($LASTEXITCODE -ne 0) {
                throw "La matriz EvTTC terminó con error en fold=$CurrentFold seed=$CurrentSeed."
            }
        }
    }

    if (-not $DryRun) {
        $ResultRoot = (
            "artifacts/runs/evttc32_architecture_v4_" +
            "$ProtocolArg" +
            "_$($Mode.ToLowerInvariant())"
        )
        $StageRoot = Join-Path $ResultRoot $Stage.ToLowerInvariant()
        & $Python scripts/aggregate_evttc_architecture_selection.py `
            --root $StageRoot `
            --output "$StageRoot/aggregate.json"
        if ($LASTEXITCODE -ne 0) {
            throw "No se pudo agregar la matriz EvTTC."
        }
    }
}
finally {
    Pop-Location
}
