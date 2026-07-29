[CmdletBinding()]
param(
    [string]$RepoRoot = "",
    [int]$ExpectedSequenceCount = 32,
    [int]$MinimumFreeGB = 45,
    [switch]$PreflightOnly,
    [switch]$WithRobustness,
    [switch]$SkipSmoke,
    [switch]$SkipHoldoutEvaluation
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# ---------------------------------------------------------------------------
# E-JEPA-TTC EvTTC-32 article ablation pipeline v6 (Windows PowerShell 5.1)
# Compatible with Windows PowerShell 5.1.
#
# It performs:
#   1. Fail-fast preflight.
#   2. Validation of 32 EvTTC sequences.
#   3. Reversible promotion of evttc_complete_staging -> evttc.
#   4. Manifest, preregistered 19/5/8 family split, temporal index.
#   5. Separate train/validation and family-holdout voxel caches.
#   6. Protocol/config regeneration and protocol freeze.
#   7. One-epoch smoke tests for all 11 ablation arms.
#   8. Local Git commit of the preregistered protocol.
#   9. 33 JEPA pretrainings and 33 downstream fine-tunings.
#  10. Validation-only ranking and selection.
#  11. Optional robustness for BASE and the selected arm.
#  12. One-time family-holdout evaluation for BASE and the selected arm.
#
# It does NOT:
#   - push to GitHub;
#   - evaluate Slider-750 / Slider-1000;
#   - claim an official/final result.
# ---------------------------------------------------------------------------

$ScriptStarted = Get-Date
$Timestamp = $ScriptStarted.ToString("yyyyMMdd_HHmmss")
$HadFatalError = $false
$RobustnessFailed = $false
$HoldoutFailures = New-Object System.Collections.ArrayList

function Resolve-RepositoryRoot {
    param([string]$RequestedRoot)

    if ($RequestedRoot) {
        return (Resolve-Path -LiteralPath $RequestedRoot).Path
    }

    if ($PSScriptRoot) {
        $candidate = Split-Path -Parent $PSScriptRoot
        if (Test-Path -LiteralPath (Join-Path $candidate ".git")) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    $current = (Get-Location).Path
    if (Test-Path -LiteralPath (Join-Path $current ".git")) {
        return $current
    }

    throw "No se pudo localizar la raíz del repositorio. Usa -RepoRoot."
}

$RepoRoot = Resolve-RepositoryRoot -RequestedRoot $RepoRoot
Set-Location -LiteralPath $RepoRoot

$AuditRoot = Join-Path $RepoRoot "artifacts\audit\evttc32_overnight"
$RunAuditDir = Join-Path $AuditRoot $Timestamp
New-Item -ItemType Directory -Force -Path $RunAuditDir | Out-Null

$MasterLog = Join-Path $RunAuditDir "overnight.log"
$StatusJson = Join-Path $RunAuditDir "status.json"
$FailureMarker = Join-Path $RunAuditDir "FAILED.txt"
$SuccessMarker = Join-Path $RunAuditDir "SUCCESS.txt"

function Write-Log {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )

    $line = "[{0}] [{1}] {2}" -f (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"), $Level, $Message
    Write-Host $line
    $line | Out-File -LiteralPath $MasterLog -Append -Encoding utf8
}

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Log ("=" * 72)
    Write-Log $Message "STEP"
    Write-Log ("=" * 72)
}

function Assert-Path {
    param(
        [string]$Path,
        [string]$Description
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Falta $Description`: $Path"
    }
}

function Invoke-NativeStreaming {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$Label,
        [switch]$ContinueOnError
    )

    Write-Log ("START {0}" -f $Label)
    Write-Log ('COMMAND: "{0}" {1}' -f $FilePath, ($Arguments -join " "))

    $previousPreference = $ErrorActionPreference
    $exitCode = -1
    $lines = New-Object System.Collections.ArrayList

    try {
        # En Windows PowerShell 5.1 las líneas stderr de procesos nativos
        # pueden convertirse en ErrorRecord. No deben abortar antes de poder
        # consultar el código de salida real del proceso.
        $ErrorActionPreference = "Continue"

        & $FilePath @Arguments 2>&1 | ForEach-Object {
            $lineText = $_.ToString()
            [void]$lines.Add($lineText)
            Write-Host $lineText
            $lineText | Out-File -LiteralPath $MasterLog -Append -Encoding utf8
        }

        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }

    if ($exitCode -ne 0) {
        $tail = @($lines | Select-Object -Last 30) -join "`n"
        $message = "$Label falló con código $exitCode."
        if ($tail) {
            $message += "`nÚltima salida del proceso:`n$tail"
        }

        Write-Log $message "ERROR"

        if ($ContinueOnError) {
            return $false
        }

        throw $message
    }

    Write-Log ("DONE {0}" -f $Label)
    return $true
}

function Invoke-NativeCapture {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$Label
    )

    Write-Log ("START {0}" -f $Label)
    Write-Log ('COMMAND: "{0}" {1}' -f $FilePath, ($Arguments -join " "))

    $lines = New-Object System.Collections.ArrayList
    $previousPreference = $ErrorActionPreference
    $exitCode = -1

    try {
        $ErrorActionPreference = "Continue"

        & $FilePath @Arguments 2>&1 | ForEach-Object {
            $lineText = $_.ToString()
            [void]$lines.Add($lineText)
            Write-Host $lineText
            $lineText | Out-File -LiteralPath $MasterLog -Append -Encoding utf8
        }

        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }

    if ($exitCode -ne 0) {
        $tail = @($lines | Select-Object -Last 30) -join "`n"
        $message = "$Label falló con código $exitCode."
        if ($tail) {
            $message += "`nÚltima salida del proceso:`n$tail"
        }
        throw $message
    }

    Write-Log ("DONE {0}" -f $Label)
    return ($lines -join "`n")
}

function Get-CanonicalSequenceId {
    param([string]$SequenceId)

    return ($SequenceId `
        -replace "-overlap-100$", "" `
        -replace "-overlap-50$", "" `
        -replace "-overlap-0$", "")
}

function Test-FileReadable {
    param([string]$Path)

    $stream = $null
    try {
        $stream = [System.IO.File]::Open(
            $Path,
            [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read,
            [System.IO.FileShare]::ReadWrite
        )
        return $true
    }
    catch {
        return $false
    }
    finally {
        if ($stream) {
            $stream.Dispose()
        }
    }
}

function Get-DatasetAudit {
    param([string]$DatasetRoot)

    $root = (Resolve-Path -LiteralPath $DatasetRoot).Path.TrimEnd("\")
    $partialExtensions = @(".crdownload", ".part", ".partial", ".download", ".tmp")

    $partialFiles = @(
        Get-ChildItem -LiteralPath $root -Recurse -File |
        Where-Object { $partialExtensions -contains $_.Extension.ToLowerInvariant() }
    )

    if ($partialFiles.Count -gt 0) {
        Write-Host "Archivos parciales encontrados:" -ForegroundColor Red
        $partialFiles | Select-Object FullName, Length | Format-Table -AutoSize
        throw "Hay descargas parciales en $root."
    }

    $ttcFiles = @(
        Get-ChildItem -LiteralPath $root -Recurse -File -Filter "ttc.csv"
    )

    $rows = @(
        foreach ($ttc in $ttcFiles) {
            $sequenceDir = $ttc.Directory.FullName
            $relativePath = $sequenceDir.Substring($root.Length).TrimStart("\")
            $sequenceId = (($relativePath -split "\\") | Where-Object { $_ }) -join "-"
            $canonical = Get-CanonicalSequenceId -SequenceId $sequenceId

            $eventHdf5 = @(
                Get-ChildItem -LiteralPath $sequenceDir -File -Filter "*.hdf5" |
                Where-Object { $_.Name.ToLowerInvariant() -ne "gt.hdf5" }
            )

            $gtHdf5 = @(
                Get-ChildItem -LiteralPath $sequenceDir -File -Filter "gt.hdf5"
            )

            $labelDirectories = @(
                (Join-Path $sequenceDir "bbox_segmentation"),
                (Join-Path $sequenceDir "leftlabel")
            ) | Where-Object { Test-Path -LiteralPath $_ }

            $labelCount = 0
            foreach ($labelDirectory in $labelDirectories) {
                $labelCount += @(
                    Get-ChildItem -LiteralPath $labelDirectory -Recurse -File -Filter "*.json"
                ).Count
            }

            $eventReadable = $false
            $eventSize = 0
            $eventName = ""

            if ($eventHdf5.Count -eq 1) {
                $eventReadable = Test-FileReadable -Path $eventHdf5[0].FullName
                $eventSize = $eventHdf5[0].Length
                $eventName = $eventHdf5[0].Name
            }

            [PSCustomObject]@{
                SequenceId       = $sequenceId
                CanonicalId      = $canonical
                RelativePath     = $relativePath
                TTCBytes         = $ttc.Length
                TTCReadable      = (Test-FileReadable -Path $ttc.FullName)
                EventHDF5Count   = $eventHdf5.Count
                EventHDF5Name    = $eventName
                EventBytes       = $eventSize
                EventReadable    = $eventReadable
                GTHDF5Count      = $gtHdf5.Count
                LabelCount       = $labelCount
                Valid            = (
                    $ttc.Length -gt 10 -and
                    (Test-FileReadable -Path $ttc.FullName) -and
                    $eventHdf5.Count -eq 1 -and
                    $eventSize -gt 10MB -and
                    $eventReadable
                )
            }
        }
    )

    $expectedCanonical = @(
        "CCRs-1-low-100",
        "CCRs-1-medium-100",
        "CCRs-1-high-100",
        "CCRs-1-low-50",
        "CCRs-1-medium-50",
        "CCRs-1-high-50",
        "CCRs-1-low-0",
        "CCRs-1-medium-0",
        "CCRs-1-high-0",
        "CCRs-2-low-100",
        "CCRs-2-medium-100",
        "CCRs-2-high-100",
        "CCRs-3-low-100",
        "CCRs-3-medium-100",
        "CCRs-side-low",
        "CCRs-side-medium",
        "CCRs-side-high",
        "CCRm-low-100",
        "CCRm-medium-100",
        "CCRm-low-50",
        "CCRm-medium-50",
        "CCRm-low-0",
        "CCRm-medium-0",
        "CPLA-low",
        "CPLA-medium",
        "CPLA-high",
        "CPNA-low",
        "CPNA-medium",
        "CPNA-high",
        "CPNAO-low",
        "CPNAO-medium",
        "CPNAO-high"
    ) | Sort-Object -Unique

    $detectedCanonical = @($rows.CanonicalId | Sort-Object -Unique)
    $missing = @(
        Compare-Object $expectedCanonical $detectedCanonical |
        Where-Object { $_.SideIndicator -eq "<=" } |
        Select-Object -ExpandProperty InputObject
    )
    $unexpected = @(
        Compare-Object $expectedCanonical $detectedCanonical |
        Where-Object { $_.SideIndicator -eq "=>" } |
        Select-Object -ExpandProperty InputObject
    )
    $duplicates = @(
        $rows |
        Group-Object CanonicalId |
        Where-Object { $_.Count -gt 1 }
    )
    $invalid = @($rows | Where-Object { -not $_.Valid })

    $rows |
        Sort-Object SequenceId |
        Format-Table SequenceId, EventHDF5Count, @{Name="EventGB";Expression={[math]::Round($_.EventBytes / 1GB, 3)}}, GTHDF5Count, LabelCount, Valid -AutoSize

    Write-Log "Secuencias con ttc.csv: $($rows.Count)"
    Write-Log "IDs canónicos únicos: $($detectedCanonical.Count)"
    Write-Log "HDF5 de eventos: $(($rows | Measure-Object EventHDF5Count -Sum).Sum)"
    Write-Log "gt.hdf5 adicionales: $(($rows | Measure-Object GTHDF5Count -Sum).Sum)"
    Write-Log "Secuencias sin JSON de bbox/segmentación: $(@($rows | Where-Object { $_.LabelCount -eq 0 }).Count)"

    if ($missing.Count -gt 0) {
        Write-Host "FALTAN:" -ForegroundColor Red
        $missing | ForEach-Object { Write-Host $_ }
    }
    if ($unexpected.Count -gt 0) {
        Write-Host "INESPERADAS:" -ForegroundColor Red
        $unexpected | ForEach-Object { Write-Host $_ }
    }
    if ($duplicates.Count -gt 0) {
        Write-Host "DUPLICADAS:" -ForegroundColor Red
        $duplicates | Select-Object Name, Count | Format-Table -AutoSize
    }
    if ($invalid.Count -gt 0) {
        Write-Host "INVÁLIDAS:" -ForegroundColor Red
        $invalid | Format-Table SequenceId, TTCBytes, EventHDF5Count, EventBytes, EventReadable -AutoSize
    }

    if ($rows.Count -ne $ExpectedSequenceCount) {
        throw "Se detectaron $($rows.Count) secuencias; se esperaban $ExpectedSequenceCount."
    }
    if ($detectedCanonical.Count -ne $ExpectedSequenceCount) {
        throw "No hay $ExpectedSequenceCount IDs canónicos únicos."
    }
    if ($missing.Count -gt 0 -or $unexpected.Count -gt 0 -or $duplicates.Count -gt 0) {
        throw "La lista de secuencias no coincide con el catálogo EvTTC-32 esperado."
    }
    if ($invalid.Count -gt 0) {
        throw "Hay secuencias con ttc.csv/HDF5 incompletos, pequeños o bloqueados."
    }

    return $rows
}

function Get-UnexpectedGitChanges {
    $allowed = @(
        "scripts/run_evttc32_article_ablation_v4.ps1",
        "scripts/run_evttc32_article_ablation_v5.ps1",
        "scripts/run_evttc32_article_ablation_v6.ps1",
        "configs/recovery_v3_protocol.yaml",
        "configs/experiment/evttc32_article_ablation_matrix.yaml",
        "data/manifests/evttc_all32_local.yaml",
        "data/splits/evttc_all32_article_family_holdout.yaml"
    )

    $status = @(& git status --porcelain --untracked-files=normal)
    if ($LASTEXITCODE -ne 0) {
        throw "No se pudo consultar git status."
    }

    $unexpected = @()
    foreach ($line in $status) {
        if (-not $line) {
            continue
        }

        $path = ""
        if ($line.Length -ge 4) {
            $path = $line.Substring(3).Trim().Replace("\", "/")
        }

        # Rename records can contain "old -> new".
        if ($path -match " -> ") {
            $path = ($path -split " -> ")[-1]
        }

        if ($allowed -notcontains $path) {
            $unexpected += $line
        }
    }

    return $unexpected
}

function Set-AwakeMode {
    if (-not ("ExecutionState.NativeMethods" -as [type])) {
        Add-Type -Namespace ExecutionState -Name NativeMethods -MemberDefinition @"
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
"@
    }

    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
    [void][ExecutionState.NativeMethods]::SetThreadExecutionState(2147483713)
}

function Reset-AwakeMode {
    try {
        if ("ExecutionState.NativeMethods" -as [type]) {
            # ES_CONTINUOUS
            [void][ExecutionState.NativeMethods]::SetThreadExecutionState(2147483648)
        }
    }
    catch {
        # Never hide the primary result because reset failed.
    }
}

function Test-PendingReboot {
    $paths = @(
        "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending",
        "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired"
    )

    foreach ($path in $paths) {
        if (Test-Path $path) {
            return $true
        }
    }

    try {
        $sessionManager = Get-ItemProperty `
            "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager" `
            -Name PendingFileRenameOperations `
            -ErrorAction SilentlyContinue
        if ($sessionManager.PendingFileRenameOperations) {
            return $true
        }
    }
    catch {
    }

    return $false
}

function Write-Status {
    param(
        [string]$Status,
        [string]$Stage,
        [string]$Message
    )

    $payload = [ordered]@{
        status = $Status
        stage = $Stage
        message = $Message
        started_at = $ScriptStarted.ToString("o")
        updated_at = (Get-Date).ToString("o")
        repo_root = $RepoRoot
        audit_directory = $RunAuditDir
        log = $MasterLog
        with_robustness = [bool]$WithRobustness
        skip_smoke = [bool]$SkipSmoke
        skip_holdout_evaluation = [bool]$SkipHoldoutEvaluation
    }

    $payload | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $StatusJson -Encoding UTF8
}

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$StagingRoot = Join-Path $RepoRoot "datasets\evttc_complete_staging"
$DatasetRoot = Join-Path $RepoRoot "datasets\evttc"

$ManifestPath = Join-Path $RepoRoot "data\manifests\evttc_all32_local.yaml"
$SplitPath = Join-Path $RepoRoot "data\splits\evttc_all32_article_family_holdout.yaml"
$IndexPath = Join-Path $RepoRoot "data\cache\evttc_all32_index.json"

$TrainValDir = Join-Path $RepoRoot "artifacts\features\evttc32_trainval"
$TrainValCache = Join-Path $TrainValDir "cache.npz"

$HoldoutDir = Join-Path $RepoRoot "artifacts\features\evttc32_family_holdout"
$HoldoutCache = Join-Path $HoldoutDir "cache.npz"

$ProtocolPath = Join-Path $RepoRoot "configs\recovery_v3_protocol.yaml"
$SourceConfigPath = Join-Path $RepoRoot "configs\experiment\flowmimic_e0_e1_multiseed.yaml"
$ConfigPath = Join-Path $RepoRoot "configs\experiment\evttc32_article_ablation_matrix.yaml"
$FrozenProtocolPath = Join-Path $RepoRoot "artifacts\audit\recovery_v3\frozen_protocol.json"

$RunRootRelative = "artifacts/runs/evttc32_article_ablation"
$RunRoot = Join-Path $RepoRoot ($RunRootRelative -replace "/", "\")
$SummaryRelative = "artifacts/metrics/evttc32_article_ablation_summary.json"
$SummaryPath = Join-Path $RepoRoot ($SummaryRelative -replace "/", "\")
$RobustnessRelative = "artifacts/metrics/evttc32_article_selected_robustness.json"
$RobustnessPath = Join-Path $RepoRoot ($RobustnessRelative -replace "/", "\")

try {
    Start-Transcript -Path (Join-Path $RunAuditDir "transcript.txt") -Force | Out-Null
}
catch {
    Write-Log "No se pudo iniciar Start-Transcript; se seguirá usando overnight.log." "WARN"
}

try {
    Set-AwakeMode
    Write-Status -Status "running" -Stage "initializing" -Message "Inicializando."

    # -----------------------------------------------------------------------
    # PREFLIGHT
    # -----------------------------------------------------------------------
    Write-Step "PREFLIGHT: sistema, repositorio, GPU, disco y dataset"

    if ($PSVersionTable.PSVersion.Major -lt 5) {
        throw "Se requiere Windows PowerShell 5.1 o superior."
    }

    Assert-Path -Path (Join-Path $RepoRoot ".git") -Description "repositorio Git"
    Assert-Path -Path $Python -Description "Python de .venv"
    Assert-Path -Path $ProtocolPath -Description "protocolo recovery_v3"
    Assert-Path -Path $SourceConfigPath -Description "config E0/E1 original"
    Assert-Path -Path (Join-Path $RepoRoot "scripts\run_flowmimic_multiseed.py") -Description "runner E0/E1"
    Assert-Path -Path (Join-Path $RepoRoot "scripts\freeze_protocol.py") -Description "congelador de protocolo"

    & git rev-parse --is-inside-work-tree | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "La ruta no es un worktree Git."
    }

    $branch = (& git branch --show-current).Trim()
    $commit = (& git rev-parse HEAD).Trim()
    Write-Log "Rama: $branch"
    Write-Log "Commit inicial: $commit"

    $gitName = (& git config user.name).Trim()
    $gitEmail = (& git config user.email).Trim()
    if (-not $gitName -or -not $gitEmail) {
        throw "Git user.name/user.email no están configurados; el commit automático fallaría."
    }
    Write-Log "Identidad Git: $gitName <$gitEmail>"

    $activeProcesses = @(
        Get-CimInstance Win32_Process |
        Where-Object {
            $_.CommandLine -and
            $_.CommandLine -match "run_flowmimic_multiseed|evaluate_flowmimic_robustness|e_jepa_ttc pretrain|e_jepa_ttc train"
        }
    )

    if ($activeProcesses.Count -gt 0) {
        $activeProcesses | Select-Object ProcessId, Name, CommandLine | Format-Table -Wrap
        throw "Hay otro entrenamiento/evaluación E-JEPA-TTC activo."
    }

    $battery = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
    if ($battery) {
        Write-Log "Batería: $($battery.EstimatedChargeRemaining)% | BatteryStatus=$($battery.BatteryStatus)"
        if ($battery.BatteryStatus -eq 1) {
            throw "El portátil parece estar descargándose. Conéctalo a corriente antes del run nocturno."
        }
    }

    $repoDrive = (Get-Item -LiteralPath $RepoRoot).PSDrive
    $freeGB = [math]::Round($repoDrive.Free / 1GB, 2)
    Write-Log "Espacio libre en $($repoDrive.Name): $freeGB GB"
    if ($freeGB -lt $MinimumFreeGB) {
        throw "Solo hay $freeGB GB libres; se requieren al menos $MinimumFreeGB GB."
    }

    # Do not use `python -c` here. Windows PowerShell 5.1 can strip
    # embedded double quotes from native command arguments, turning valid
    # Python such as print("ok") into invalid code such as print(ok).
    $ImportCheckPath = Join-Path $RunAuditDir "python_cuda_import_check.py"

    @'
import h5py
import jsonschema
import numpy
import torch
import yaml

print("imports_ok")
print("h5py=" + h5py.__version__)
print("jsonschema=" + getattr(jsonschema, "__version__", "installed"))
print("numpy=" + numpy.__version__)
print("torch=" + torch.__version__)
print("cuda_available=" + str(torch.cuda.is_available()))
print(
    "gpu="
    + (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "NONE"
    )
)
print("yaml=" + getattr(yaml, "__version__", "installed"))
'@ | Set-Content -LiteralPath $ImportCheckPath -Encoding UTF8

    $importOutput = Invoke-NativeCapture `
        -FilePath $Python `
        -Arguments @($ImportCheckPath) `
        -Label "Python/CUDA import check"

    if ($importOutput -notmatch "imports_ok") {
        throw "No se pudieron importar las dependencias requeridas."
    }
    if ($importOutput -notmatch "cuda_available=True") {
        throw "PyTorch no detecta CUDA."
    }

    $unexpectedChanges = @(Get-UnexpectedGitChanges)
    if ($unexpectedChanges.Count -gt 0) {
        Write-Host "Cambios Git no gestionados por el script:" -ForegroundColor Red
        $unexpectedChanges | ForEach-Object { Write-Host $_ }
        throw "Limpia, confirma o guarda esos cambios antes de continuar."
    }

    $ValidationRoot = $null
    if (Test-Path -LiteralPath $StagingRoot) {
        $ValidationRoot = $StagingRoot
        Write-Log "Se validará staging: $StagingRoot"
    }
    elseif (Test-Path -LiteralPath $DatasetRoot) {
        $ValidationRoot = $DatasetRoot
        Write-Log "No existe staging; se validará el dataset definitivo: $DatasetRoot"
    }
    else {
        throw "No existe ni datasets\evttc_complete_staging ni datasets\evttc."
    }

    $datasetAudit = Get-DatasetAudit -DatasetRoot $ValidationRoot
    Write-Log "PASS: auditoría estructural EvTTC-32."

    $TempManifest = Join-Path $env:TEMP ("evttc32_preflight_{0}.yaml" -f $Timestamp)

    $scanText = Invoke-NativeCapture `
        -FilePath $Python `
        -Arguments @(
            "-m", "e_jepa_ttc",
            "data", "scan",
            "--root", $ValidationRoot,
            "--output", $TempManifest
        ) `
        -Label "Scanner real del repo"

    $scanJson = $scanText | ConvertFrom-Json
    if ([int]$scanJson.sequence_count -ne $ExpectedSequenceCount) {
        throw "El scanner del repo detectó $($scanJson.sequence_count), no $ExpectedSequenceCount."
    }

    $validateText = Invoke-NativeCapture `
        -FilePath $Python `
        -Arguments @(
            "-m", "e_jepa_ttc",
            "data", "validate",
            "--manifest", $TempManifest
        ) `
        -Label "Validator real del repo"

    $validateJson = $validateText | ConvertFrom-Json
    if ([int]$validateJson.sequence_count -ne $ExpectedSequenceCount) {
        throw "El validator no devolvió $ExpectedSequenceCount secuencias."
    }

    $badValidation = @(
        $validateJson.sequences |
        Where-Object {
            $null -eq $_.event_layout -or
            [int]$_.ttc_rows -le 0 -or
            [int]$_.hdf5_dataset_count -le 0
        }
    )
    if ($badValidation.Count -gt 0) {
        $badValidation | Select-Object sequence_id, ttc_rows, hdf5_dataset_count, event_layout | Format-List
        throw "Hay HDF5/CSV que el pipeline no puede interpretar."
    }

    Remove-Item -LiteralPath $TempManifest -Force -ErrorAction SilentlyContinue

    Write-Log "PASS PREFLIGHT: 32 secuencias legibles, CUDA activa, disco suficiente y Git controlado." "PASS"

    if ($PreflightOnly) {
        Write-Status -Status "preflight_passed" -Stage "preflight" -Message "Todas las comprobaciones previas han pasado."
        "PREFLIGHT PASSED" | Set-Content -LiteralPath $SuccessMarker -Encoding UTF8
        Write-Host ""
        Write-Host "PREFLIGHT PASSED" -ForegroundColor Green
        Write-Host "Log: $MasterLog"
        return
    }

    # -----------------------------------------------------------------------
    # PROMOTE STAGING
    # -----------------------------------------------------------------------
    Write-Step "PROMOCIÓN REVERSIBLE DEL DATASET"

    if (Test-Path -LiteralPath $StagingRoot) {
        $backupRelative = "datasets\evttc_backup_$Timestamp"
        $backupRoot = Join-Path $RepoRoot ($backupRelative -replace "/", "\")

        Write-Log "Backup previsto: $backupRoot"

        try {
            if (Test-Path -LiteralPath $DatasetRoot) {
                Rename-Item -LiteralPath $DatasetRoot -NewName ("evttc_backup_$Timestamp")
                Write-Log "Dataset anterior renombrado a $backupRelative"
            }

            Rename-Item -LiteralPath $StagingRoot -NewName "evttc"
            Write-Log "Staging promovido a datasets\evttc"
        }
        catch {
            Write-Log "Falló la promoción; intentando rollback." "ERROR"

            if (
                -not (Test-Path -LiteralPath $DatasetRoot) -and
                (Test-Path -LiteralPath $backupRoot)
            ) {
                Rename-Item -LiteralPath $backupRoot -NewName "evttc"
            }

            throw
        }
    }
    else {
        Write-Log "datasets\evttc ya es la ubicación definitiva; no se renombra."
    }

    $datasetAudit = Get-DatasetAudit -DatasetRoot $DatasetRoot
    Write-Log "PASS: datasets\evttc contiene las 32 secuencias."

    # -----------------------------------------------------------------------
    # MANIFEST + SPLIT + INDEX
    # -----------------------------------------------------------------------
    Write-Step "MANIFEST, SPLIT DIAGNÓSTICO 19/5/8 E ÍNDICE"

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ManifestPath) | Out-Null
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $SplitPath) | Out-Null
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $IndexPath) | Out-Null

    Invoke-NativeStreaming `
        -FilePath $Python `
        -Arguments @(
            "-m", "e_jepa_ttc",
            "data", "scan",
            "--root", "datasets\evttc",
            "--output", "data\manifests\evttc_all32_local.yaml"
        ) `
        -Label "Crear manifest definitivo" | Out-Null

    Invoke-NativeStreaming `
        -FilePath $Python `
        -Arguments @(
            "-m", "e_jepa_ttc",
            "data", "validate",
            "--manifest", "data\manifests\evttc_all32_local.yaml"
        ) `
        -Label "Validar manifest definitivo" | Out-Null

    $SplitHelper = Join-Path $RunAuditDir "create_split.py"
    @'
from pathlib import Path
import yaml

from e_jepa_ttc.data.evttc import read_manifest
from e_jepa_ttc.data.split import read_splits, validate_split_groups

manifest_path = Path("data/manifests/evttc_all32_local.yaml")
split_path = Path("data/splits/evttc_all32_article_family_holdout.yaml")

document = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
ids = [str(item["sequence_id"]) for item in document["sequences"]]

if len(ids) != 32 or len(set(ids)) != 32:
    raise RuntimeError(f"Manifest inválido: total={len(ids)}, únicos={len(set(ids))}")

def canonical(value: str) -> str:
    for suffix in ("-overlap-100", "-overlap-50", "-overlap-0"):
        if value.endswith(suffix):
            return value[:-len(suffix)]
    return value

expected = {
    "CCRs-1-low-100", "CCRs-1-medium-100", "CCRs-1-high-100",
    "CCRs-1-low-50", "CCRs-1-medium-50", "CCRs-1-high-50",
    "CCRs-1-low-0", "CCRs-1-medium-0", "CCRs-1-high-0",
    "CCRs-2-low-100", "CCRs-2-medium-100", "CCRs-2-high-100",
    "CCRs-3-low-100", "CCRs-3-medium-100",
    "CCRs-side-low", "CCRs-side-medium", "CCRs-side-high",
    "CCRm-low-100", "CCRm-medium-100",
    "CCRm-low-50", "CCRm-medium-50",
    "CCRm-low-0", "CCRm-medium-0",
    "CPLA-low", "CPLA-medium", "CPLA-high",
    "CPNA-low", "CPNA-medium", "CPNA-high",
    "CPNAO-low", "CPNAO-medium", "CPNAO-high",
}

observed = {canonical(value) for value in ids}
if observed != expected:
    raise RuntimeError(
        "IDs canónicos incorrectos.\n"
        f"Faltan: {sorted(expected - observed)}\n"
        f"Inesperados: {sorted(observed - expected)}"
    )

# Predeclared family holdout:
# complete CCRs-2, CCRs-3 and CPNAO families are kept out of fitting and
# hyperparameter selection. They are not discarded; they are opened only
# after the validation-selected article model has been fixed.
test_canonical = {
    "CCRs-2-low-100",
    "CCRs-2-medium-100",
    "CCRs-2-high-100",
    "CCRs-3-low-100",
    "CCRs-3-medium-100",
    "CPNAO-low",
    "CPNAO-medium",
    "CPNAO-high",
}

validation_canonical = {
    "CCRs-1-high-0",
    "CCRs-side-high",
    "CCRm-medium-0",
    "CPLA-high",
    "CPNA-high",
}

splits = {"train": [], "validation": [], "test": []}
for sequence_id in ids:
    normalized = canonical(sequence_id)
    if normalized in test_canonical:
        splits["test"].append(sequence_id)
    elif normalized in validation_canonical:
        splits["validation"].append(sequence_id)
    else:
        splits["train"].append(sequence_id)

for values in splits.values():
    values.sort()

counts = {name: len(values) for name, values in splits.items()}
if counts != {"train": 19, "validation": 5, "test": 8}:
    raise RuntimeError(f"Split inesperado: {counts}")

payload = {
    "version": 2,
    "protocol": "evttc-32-article-family-holdout-2026-07-27",
    "status": "predeclared_family_holdout_diagnostic",
    "evaluation_role": "article_ablation_and_family_generalization",
    "allowed_claim_levels": ["development", "diagnostic"],
    "test_was_previously_used_for_model_selection": False,
    "notes": (
        "Predeclared 19/5/8 sequence split. All 32 sequences participate in the "
        "protocol. Train fits weights, validation selects among the ablations, "
        "and the complete CCRs-2, CCRs-3 and CPNAO families remain physically "
        "outside all fitting and selection caches. The family holdout is opened "
        "once for BASE and the validation-selected best arm. It is not the "
        "official ten-sequence EvTTC benchmark because Slider-750/1000 are absent."
    ),
    "manifest": manifest_path.as_posix(),
    "splits": splits,
}

split_path.write_text(
    yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)

sequences = read_manifest(manifest_path)
loaded_splits = read_splits(split_path)
validate_split_groups(sequences, loaded_splits)

assigned = {value for values in loaded_splits.values() for value in values}
if assigned != set(ids):
    raise RuntimeError("El split no asigna exactamente las 32 secuencias.")

print(yaml.safe_dump({"counts": counts, "splits": splits}, sort_keys=False))
'@ | Set-Content -LiteralPath $SplitHelper -Encoding UTF8

    Invoke-NativeStreaming `
        -FilePath $Python `
        -Arguments @($SplitHelper) `
        -Label "Crear y validar split 19/5/8" | Out-Null

    Invoke-NativeStreaming `
        -FilePath $Python `
        -Arguments @(
            "-m", "e_jepa_ttc",
            "data", "index",
            "--manifest", "data\manifests\evttc_all32_local.yaml",
            "--output", "data\cache\evttc_all32_index.json",
            "--context-ms", "100",
            "--stride-ms", "20",
            "--horizons-ms", "20", "60", "100", "240", "500",
            "--clip-ttc-min", "0.1",
            "--clip-ttc-max", "12.0"
        ) `
        -Label "Crear índice temporal de las 32 secuencias" | Out-Null

    $IndexCheck = Join-Path $RunAuditDir "check_index.py"
    @'
from collections import Counter
from pathlib import Path

from e_jepa_ttc.data.evttc import read_manifest
from e_jepa_ttc.utils.io import read_structured

manifest = read_manifest("data/manifests/evttc_all32_local.yaml")
index = read_structured("data/cache/evttc_all32_index.json")
windows = index.get("windows", [])

counts = Counter(str(item["sequence_id"]) for item in windows)
missing = [item.sequence_id for item in manifest if counts[item.sequence_id] == 0]

if missing:
    raise RuntimeError(f"Secuencias sin ventanas temporales: {missing}")

print(f"window_count={len(windows)}")
print(f"sequence_count={len(counts)}")
for sequence_id in sorted(counts):
    print(f"{sequence_id}: {counts[sequence_id]}")
'@ | Set-Content -LiteralPath $IndexCheck -Encoding UTF8

    Invoke-NativeStreaming `
        -FilePath $Python `
        -Arguments @($IndexCheck) `
        -Label "Comprobar cobertura del índice" | Out-Null

    # -----------------------------------------------------------------------
    # CACHES
    # -----------------------------------------------------------------------
    Write-Step "CONSTRUCCIÓN DE CACHÉS TRAIN/VALIDATION Y HOLDOUT"

    New-Item -ItemType Directory -Force -Path $TrainValDir | Out-Null
    New-Item -ItemType Directory -Force -Path $HoldoutDir | Out-Null

    if (Test-Path -LiteralPath $TrainValCache) {
        Write-Log (
            "REUSE: ya existe artifacts\features\evttc32_trainval\cache.npz; " +
            "se validará antes de continuar."
        )
    }
    else {
        Invoke-NativeStreaming `
            -FilePath $Python `
            -Arguments @(
                "-m", "e_jepa_ttc",
                "cache", "voxel",
                "--manifest", "data\manifests\evttc_all32_local.yaml",
                "--split", "data\splits\evttc_all32_article_family_holdout.yaml",
                "--index", "data\cache\evttc_all32_index.json",
                "--output", "artifacts\features\evttc32_trainval\cache.npz",
                "--width", "160",
                "--height", "90",
                "--bins", "5",
                "--no-normalize",
                "--metadata-channels",
                "--navigation-channels",
                "--include-split", "train",
                "--include-split", "validation"
            ) `
            -Label "Construir caché train+validation" | Out-Null
    }

    if (Test-Path -LiteralPath $HoldoutCache) {
        Write-Log (
            "REUSE: ya existe artifacts\features\evttc32_family_holdout\cache.npz; " +
            "se validará antes de continuar."
        )
    }
    else {
        Invoke-NativeStreaming `
            -FilePath $Python `
            -Arguments @(
                "-m", "e_jepa_ttc",
                "cache", "voxel",
                "--manifest", "data\manifests\evttc_all32_local.yaml",
                "--split", "data\splits\evttc_all32_article_family_holdout.yaml",
                "--index", "data\cache\evttc_all32_index.json",
                "--output", "artifacts\features\evttc32_family_holdout\cache.npz",
                "--width", "160",
                "--height", "90",
                "--bins", "5",
                "--no-normalize",
                "--metadata-channels",
                "--navigation-channels",
                "--include-split", "test"
            ) `
            -Label "Construir caché holdout diagnóstico" | Out-Null
    }

    $CacheCheck = Join-Path $RunAuditDir "check_caches.py"
    @'
import numpy as np

def inspect(path: str, expected_splits: set[str], expected_counts: dict[str, int]):
    with np.load(path, allow_pickle=False) as data:
        required = {
            "x", "y_ttc", "sequence_id", "split",
            "cache_format_version", "source_manifest_sha256",
            "split_manifest_sha256", "preprocessing_config_sha256",
        }
        missing = required - set(data.files)
        if missing:
            raise RuntimeError(f"{path}: faltan claves {sorted(missing)}")

        if int(data["cache_format_version"]) < 2:
            raise RuntimeError(f"{path}: cache_format_version < 2")

        splits = data["split"].astype(str)
        sequences = data["sequence_id"].astype(str)
        physical_splits = set(splits.tolist())

        if physical_splits != expected_splits:
            raise RuntimeError(
                f"{path}: splits {sorted(physical_splits)} != {sorted(expected_splits)}"
            )

        for split_name, expected_sequence_count in expected_counts.items():
            sequence_count = len(set(sequences[splits == split_name].tolist()))
            if sequence_count != expected_sequence_count:
                raise RuntimeError(
                    f"{path}: {split_name} tiene {sequence_count} secuencias; "
                    f"se esperaban {expected_sequence_count}"
                )

        if data["x"].shape[0] == 0:
            raise RuntimeError(f"{path}: caché vacío")

        if np.count_nonzero(data["x"][: min(100, data["x"].shape[0])]) == 0:
            raise RuntimeError(f"{path}: muestra inicial completamente vacía")

        print(
            path,
            "shape=", data["x"].shape,
            "splits=", sorted(physical_splits),
            "sequences=", len(set(sequences.tolist())),
        )

inspect(
    "artifacts/features/evttc32_trainval/cache.npz",
    {"train", "validation"},
    {"train": 19, "validation": 5},
)

inspect(
    "artifacts/features/evttc32_family_holdout/cache.npz",
    {"test"},
    {"test": 8},
)

print("CACHE_CHECK_PASS")
'@ | Set-Content -LiteralPath $CacheCheck -Encoding UTF8

    $cacheCheckOutput = Invoke-NativeCapture `
        -FilePath $Python `
        -Arguments @($CacheCheck) `
        -Label "Auditar cachés"

    if ($cacheCheckOutput -notmatch "CACHE_CHECK_PASS") {
        throw "La auditoría de cachés no terminó en PASS."
    }

    # Recheck disk after cache materialization.
    $repoDrive = (Get-Item -LiteralPath $RepoRoot).PSDrive
    $freeAfterCacheGB = [math]::Round($repoDrive.Free / 1GB, 2)
    Write-Log "Espacio libre después de cachés: $freeAfterCacheGB GB"
    if ($freeAfterCacheGB -lt 12) {
        throw "Quedan menos de 12 GB libres después de construir los cachés."
    }

    # -----------------------------------------------------------------------
    # PROTOCOL + ARTICLE MATRIX CONFIG
    # -----------------------------------------------------------------------
    Write-Step "ACTUALIZAR PROTOCOLO, DEFINIR MATRIZ DE ABLACIONES Y CONGELAR"

    Copy-Item `
        -LiteralPath $ProtocolPath `
        -Destination (Join-Path $RunAuditDir "recovery_v3_protocol.before.yaml") `
        -Force

    $PrepareHelper = Join-Path $RunAuditDir "prepare_article_matrix.py"
    @'
from hashlib import sha256
from pathlib import Path
import yaml

protocol_path = Path("configs/recovery_v3_protocol.yaml")
source_config_path = Path("configs/experiment/flowmimic_e0_e1_multiseed.yaml")
config_path = Path("configs/experiment/evttc32_article_ablation_matrix.yaml")
manifest_path = Path("data/manifests/evttc_all32_local.yaml")
split_path = Path("data/splits/evttc_all32_article_family_holdout.yaml")
index_path = Path("data/cache/evttc_all32_index.json")
cache_path = Path("artifacts/features/evttc32_trainval/cache.npz")
holdout_cache_path = Path("artifacts/features/evttc32_family_holdout/cache.npz")

def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

source = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
split_document = yaml.safe_load(split_path.read_text(encoding="utf-8"))
splits = split_document["splits"]

protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
protocol["cache_format_version"] = 2
protocol["resources"]["evttc_dataset_manifest"] = manifest_path.as_posix()
protocol["resources"]["evttc_split_manifest"] = split_path.as_posix()
protocol["resources"]["evttc_cache"] = cache_path.as_posix()
protocol["requirements"]["forbidden_ordinary_splits"] = ["test"]
protocol["requirements"]["dataset"]["name"] = "evttc"
protocol["requirements"]["dataset"]["version"] = "EvTTC-32-local-article-v5"
protocol["requirements"]["dataset"]["manifest_hash"] = file_hash(manifest_path)

protocol_path.write_text(
    yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)

arms = {
    "BASE": {
        "description": "JEPA control without FlowMimic",
        "flowmimic_alignment_weight": 0.0,
        "flowmimic_inverse_ttc_weight": 0.0,
        "navigation_mode": "enabled",
        "motion_conditioning": True,
        "variance_weight": 1.0,
        "temporal_horizons_ms": [20, 60, 100, 240, 500],
        "mask_ratio": 0.45,
        "group": "main_objective",
    },
    "ALIGN": {
        "description": "FlowMimic latent alignment only",
        "flowmimic_alignment_weight": 0.25,
        "flowmimic_inverse_ttc_weight": 0.0,
        "navigation_mode": "enabled",
        "motion_conditioning": True,
        "variance_weight": 1.0,
        "temporal_horizons_ms": [20, 60, 100, 240, 500],
        "mask_ratio": 0.45,
        "group": "main_objective",
    },
    "INVERSE": {
        "description": "Synthetic inverse-TTC auxiliary head only",
        "flowmimic_alignment_weight": 0.0,
        "flowmimic_inverse_ttc_weight": 0.10,
        "navigation_mode": "enabled",
        "motion_conditioning": True,
        "variance_weight": 1.0,
        "temporal_horizons_ms": [20, 60, 100, 240, 500],
        "mask_ratio": 0.45,
        "group": "main_objective",
    },
    "BOTH": {
        "description": "Latent alignment plus synthetic inverse-TTC",
        "flowmimic_alignment_weight": 0.25,
        "flowmimic_inverse_ttc_weight": 0.10,
        "navigation_mode": "enabled",
        "motion_conditioning": True,
        "variance_weight": 1.0,
        "temporal_horizons_ms": [20, 60, 100, 240, 500],
        "mask_ratio": 0.45,
        "group": "main_objective",
    },
    "NO_NAV": {
        "description": "BOTH without navigation channels at pretrain/downstream",
        "flowmimic_alignment_weight": 0.25,
        "flowmimic_inverse_ttc_weight": 0.10,
        "navigation_mode": "disabled",
        "motion_conditioning": True,
        "variance_weight": 1.0,
        "temporal_horizons_ms": [20, 60, 100, 240, 500],
        "mask_ratio": 0.45,
        "group": "causal_ablation",
    },
    "NO_MOTION": {
        "description": "BOTH without causal context-motion conditioning",
        "flowmimic_alignment_weight": 0.25,
        "flowmimic_inverse_ttc_weight": 0.10,
        "navigation_mode": "enabled",
        "motion_conditioning": False,
        "variance_weight": 1.0,
        "temporal_horizons_ms": [20, 60, 100, 240, 500],
        "mask_ratio": 0.45,
        "group": "causal_ablation",
    },
    "NO_VARIANCE": {
        "description": "BOTH with the variance regularizer weight set to zero",
        "flowmimic_alignment_weight": 0.25,
        "flowmimic_inverse_ttc_weight": 0.10,
        "navigation_mode": "enabled",
        "motion_conditioning": True,
        "variance_weight": 0.0,
        "temporal_horizons_ms": [20, 60, 100, 240, 500],
        "mask_ratio": 0.45,
        "group": "regularization_ablation",
    },
    "H_SHORT": {
        "description": "BOTH with short future horizons",
        "flowmimic_alignment_weight": 0.25,
        "flowmimic_inverse_ttc_weight": 0.10,
        "navigation_mode": "enabled",
        "motion_conditioning": True,
        "variance_weight": 1.0,
        "temporal_horizons_ms": [20, 60, 100],
        "mask_ratio": 0.45,
        "group": "horizon_sensitivity",
    },
    "H_LONG": {
        "description": "BOTH with long future horizons",
        "flowmimic_alignment_weight": 0.25,
        "flowmimic_inverse_ttc_weight": 0.10,
        "navigation_mode": "enabled",
        "motion_conditioning": True,
        "variance_weight": 1.0,
        "temporal_horizons_ms": [100, 240, 500],
        "mask_ratio": 0.45,
        "group": "horizon_sensitivity",
    },
    "MASK_025": {
        "description": "BOTH with mask ratio 0.25",
        "flowmimic_alignment_weight": 0.25,
        "flowmimic_inverse_ttc_weight": 0.10,
        "navigation_mode": "enabled",
        "motion_conditioning": True,
        "variance_weight": 1.0,
        "temporal_horizons_ms": [20, 60, 100, 240, 500],
        "mask_ratio": 0.25,
        "group": "mask_sensitivity",
    },
    "MASK_065": {
        "description": "BOTH with mask ratio 0.65",
        "flowmimic_alignment_weight": 0.25,
        "flowmimic_inverse_ttc_weight": 0.10,
        "navigation_mode": "enabled",
        "motion_conditioning": True,
        "variance_weight": 1.0,
        "temporal_horizons_ms": [20, 60, 100, 240, 500],
        "mask_ratio": 0.65,
        "group": "mask_sensitivity",
    },
}

matrix = {
    "schema_version": "1.0",
    "experiment": {
        "id": "evttc32-article-ablation-v5-2026-07-27",
        "claim_level": "diagnostic",
        "seeds": [7, 13, 21],
        "selection_metric": "validation_mae_s_mean",
        "selection_direction": "minimize",
        "baseline_arm": "BASE",
        "arms": arms,
    },
    "protocol": {
        "version": "3.0",
        "sha256": file_hash(protocol_path),
        "final_test_opened": False,
        "selection_split": "validation",
        "train_sequences": splits["train"],
        "validation_sequences": splits["validation"],
        "family_holdout_sequences": splits["test"],
        "holdout_policy": (
            "The holdout is absent from all fitting/selection caches and is opened "
            "only after all arm/seed validation results have selected one best arm."
        ),
    },
    "data": {
        "cache": cache_path.as_posix(),
        "cache_sha256": file_hash(cache_path),
        "holdout_cache": holdout_cache_path.as_posix(),
        "holdout_cache_sha256": file_hash(holdout_cache_path),
        "manifest": manifest_path.as_posix(),
        "split": split_path.as_posix(),
        "index": index_path.as_posix(),
        "train_splits": ["train"],
        "validation_splits": ["validation"],
        "evaluation_splits": ["train", "validation"],
    },
    "pretrain": source["pretrain"],
    "downstream": source["downstream"],
    "robustness": source.get("robustness", {}),
    "outputs": {
        "run_root": "artifacts/runs/evttc32_article_ablation",
        "summary": "artifacts/metrics/evttc32_article_ablation_summary.json",
        "selection": "artifacts/metrics/evttc32_article_selection.json",
        "robustness": "artifacts/metrics/evttc32_article_selected_robustness.json",
        "holdout_root": "artifacts/metrics/evttc32_article_family_holdout",
    },
    "limitations": [
        "The complete official EvTTC ten-sequence benchmark is not evaluated.",
        "Slider-750 and Slider-1000 require a separate adapter.",
        "The family holdout is predeclared for this experiment but is not the official benchmark.",
        "Inverse-TTC weight 0.10 is a preregistered engineering choice, not an exhaustive sweep.",
    ],
}

config_path.write_text(
    yaml.safe_dump(matrix, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)

print("protocol_sha256=" + matrix["protocol"]["sha256"])
print("cache_sha256=" + matrix["data"]["cache_sha256"])
print("arms=" + ",".join(arms))
print("train_sequences=" + str(len(splits["train"])))
print("validation_sequences=" + str(len(splits["validation"])))
print("family_holdout_sequences=" + str(len(splits["test"])))
print("CONFIG_PREPARED")
'@ | Set-Content -LiteralPath $PrepareHelper -Encoding UTF8

    $prepareOutput = Invoke-NativeCapture `
        -FilePath $Python `
        -Arguments @($PrepareHelper) `
        -Label "Preparar protocolo y matriz de ablaciones"

    if ($prepareOutput -notmatch "CONFIG_PREPARED") {
        throw "No se completó la preparación del protocolo/matriz."
    }

    Invoke-NativeStreaming `
        -FilePath $Python `
        -Arguments @("scripts\freeze_protocol.py") `
        -Label "Congelar protocolo EvTTC-32 article-v5" | Out-Null

    Assert-Path -Path $FrozenProtocolPath -Description "artefacto de protocolo congelado"

    # The matrix config must contain the identity produced by the frozen protocol.
    $RefreshConfigHelper = Join-Path $RunAuditDir "refresh_matrix_protocol_identity.py"
    @'
from pathlib import Path
import json
import yaml

config_path = Path("configs/experiment/evttc32_article_ablation_matrix.yaml")
frozen_path = Path("artifacts/audit/recovery_v3/frozen_protocol.json")

config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
config["protocol"]["version"] = str(frozen["protocol_version"])
config["protocol"]["sha256"] = str(frozen["protocol_sha256"])
config_path.write_text(
    yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
print("MATRIX_PROTOCOL_REFRESHED")
'@ | Set-Content -LiteralPath $RefreshConfigHelper -Encoding UTF8

    Invoke-NativeStreaming `
        -FilePath $Python `
        -Arguments @($RefreshConfigHelper) `
        -Label "Sincronizar identidad del protocolo en la matriz" | Out-Null

    # -----------------------------------------------------------------------
    # GENERATE ARTICLE MATRIX RUNNER
    # -----------------------------------------------------------------------
    $MatrixRunner = Join-Path $RunAuditDir "run_article_matrix.py"
    @'
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping at {path}")
    return payload


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected object at {path}")
    return payload


def run(command: list[str], label: str) -> None:
    print(f"\n===== {label} =====", flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}")


def checkpoint_from_metrics(metrics_path: Path) -> Path:
    payload = read_json(metrics_path)
    checkpoint = Path(str(payload["best_checkpoint"]))
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def ssl_complete(
    metrics_path: Path,
    arm: dict[str, Any],
    seed: int,
    epochs: int,
) -> bool:
    if not metrics_path.is_file():
        return False
    try:
        payload = read_json(metrics_path)
        checkpoint = Path(str(payload.get("best_checkpoint", "")))
        return bool(
            checkpoint.is_file()
            and int(payload.get("seed", -1)) == seed
            and int(payload.get("epochs", -1)) == epochs
            and abs(
                float(payload.get("flowmimic_alignment_weight", -999))
                - float(arm["flowmimic_alignment_weight"])
            )
            < 1e-12
            and abs(
                float(payload.get("flowmimic_inverse_ttc_weight", -999))
                - float(arm["flowmimic_inverse_ttc_weight"])
            )
            < 1e-12
            and bool(payload.get("navigation_channels")) == (
                str(arm["navigation_mode"]) == "enabled"
            )
            and bool(payload.get("motion_conditioning")) == bool(
                arm["motion_conditioning"]
            )
            and abs(
                float(payload.get("variance_weight", -999))
                - float(arm["variance_weight"])
            )
            < 1e-12
            and [int(value) for value in payload.get("temporal_horizons_ms", [])]
            == [int(value) for value in arm["temporal_horizons_ms"]]
            and abs(float(payload.get("mask_ratio", -999)) - float(arm["mask_ratio"]))
            < 1e-12
        )
    except Exception:
        return False


def downstream_complete(
    metrics_path: Path,
    seed: int,
    epochs: int,
    navigation_mode: str,
) -> bool:
    if not metrics_path.is_file():
        return False
    try:
        payload = read_json(metrics_path)
        checkpoint = Path(str(payload.get("best_checkpoint", "")))
        predictions = Path(str(payload.get("predictions_path", "")))
        return bool(
            checkpoint.is_file()
            and predictions.is_file()
            and int(payload.get("seed", -1)) == seed
            and int(payload.get("epochs", -1)) == epochs
            and str(payload.get("navigation_mode")) == navigation_mode
            and payload.get("final_test_opened") is False
            and "test" not in set(payload.get("evaluation_splits", []))
        )
    except Exception:
        return False


def pretrain_command(
    config: dict[str, Any],
    arm: dict[str, Any],
    seed: int,
    output_dir: Path,
    epochs: int,
) -> list[str]:
    common = config["pretrain"]
    command = [
        sys.executable,
        "-m",
        "e_jepa_ttc",
        "pretrain",
        "jepa",
        "--cache",
        str(config["data"]["cache"]),
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(common["batch_size"]),
        "--learning-rate",
        str(common["learning_rate"]),
        "--seed",
        str(seed),
        "--device",
        str(common["device"]),
        "--model",
        str(common["model"]),
        "--navigation-mode",
        str(arm["navigation_mode"]),
        "--pretrain-splits",
        *[str(value) for value in config["data"]["train_splits"]],
        "--validation-splits",
        *[str(value) for value in config["data"]["validation_splits"]],
        "--temporal-horizons-ms",
        *[str(value) for value in arm["temporal_horizons_ms"]],
        "--max-target-slop-ms",
        str(common["max_target_slop_ms"]),
        "--mask-ratio",
        str(arm["mask_ratio"]),
        "--block-count",
        str(common["block_count"]),
        "--mask-mode",
        str(common["mask_mode"]),
        "--ema-momentum",
        str(common["ema_momentum"]),
        "--regularizer",
        str(common["regularizer"]),
        "--variance-weight",
        str(arm["variance_weight"]),
        "--min-std",
        str(common["min_std"]),
        "--dense-predictor",
        str(common["dense_predictor"]),
        "--flowmimic-alignment-weight",
        str(arm["flowmimic_alignment_weight"]),
        "--flowmimic-inverse-ttc-weight",
        str(arm["flowmimic_inverse_ttc_weight"]),
        "--flowmimic-minimum-ttc-s",
        str(common["flowmimic_minimum_ttc_s"]),
        "--flowmimic-maximum-ttc-s",
        str(common["flowmimic_maximum_ttc_s"]),
    ]
    if not bool(arm["motion_conditioning"]):
        command.append("--no-motion-conditioning")
    return command


def downstream_command(
    config: dict[str, Any],
    arm: dict[str, Any],
    seed: int,
    checkpoint: Path,
    output_dir: Path,
    epochs: int,
) -> list[str]:
    common = config["downstream"]
    command = [
        sys.executable,
        "-m",
        "e_jepa_ttc",
        "train",
        "tiny-cnn",
        "--cache",
        str(config["data"]["cache"]),
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(common["batch_size"]),
        "--learning-rate",
        str(common["learning_rate"]),
        "--seed",
        str(seed),
        "--device",
        str(common["device"]),
        "--model",
        str(common["model"]),
        "--navigation-mode",
        str(arm["navigation_mode"]),
        "--pretrained-encoder",
        str(checkpoint),
        "--train-splits",
        *[str(value) for value in config["data"]["train_splits"]],
        "--validation-splits",
        *[str(value) for value in config["data"]["validation_splits"]],
        "--evaluation-splits",
        *[str(value) for value in config["data"]["evaluation_splits"]],
    ]
    if bool(common.get("freeze_encoder", False)):
        command.append("--freeze-encoder")
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    config = read_yaml(args.config)
    arms = config["experiment"]["arms"]
    seeds = [7] if args.mode == "smoke" else [
        int(seed) for seed in config["experiment"]["seeds"]
    ]
    ssl_epochs = 1 if args.mode == "smoke" else int(config["pretrain"]["epochs"])
    ft_epochs = 1 if args.mode == "smoke" else int(config["downstream"]["epochs"])
    root = args.output_root or Path(config["outputs"]["run_root"])

    print(
        json.dumps(
            {
                "mode": args.mode,
                "arm_count": len(arms),
                "seeds": seeds,
                "ssl_epochs": ssl_epochs,
                "downstream_epochs": ft_epochs,
                "run_root": root.as_posix(),
            },
            indent=2,
        ),
        flush=True,
    )

    for arm_name, arm in arms.items():
        for seed in seeds:
            ssl_dir = root / arm_name.lower() / f"seed{seed}" / f"ssl{ssl_epochs}"
            ft_dir = root / arm_name.lower() / f"seed{seed}" / f"ft{ft_epochs}"
            ssl_metrics = ssl_dir / "metrics.json"
            ft_metrics = ft_dir / "metrics.json"

            if ssl_complete(ssl_metrics, arm, seed, ssl_epochs):
                print(f"SKIP complete SSL {arm_name} seed {seed}", flush=True)
            else:
                run(
                    pretrain_command(config, arm, seed, ssl_dir, ssl_epochs),
                    f"{arm_name} seed {seed} SSL",
                )

            checkpoint = checkpoint_from_metrics(ssl_metrics)

            if downstream_complete(
                ft_metrics,
                seed,
                ft_epochs,
                str(arm["navigation_mode"]),
            ):
                print(f"SKIP complete downstream {arm_name} seed {seed}", flush=True)
            else:
                run(
                    downstream_command(
                        config,
                        arm,
                        seed,
                        checkpoint,
                        ft_dir,
                        ft_epochs,
                    ),
                    f"{arm_name} seed {seed} downstream",
                )

    print("ARTICLE_MATRIX_RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
'@ | Set-Content -LiteralPath $MatrixRunner -Encoding UTF8

    # -----------------------------------------------------------------------
    # SMOKE ALL ARMS
    # -----------------------------------------------------------------------
    if (-not $SkipSmoke) {
        Write-Step "SMOKE DE LAS ONCE ABLACIONES: UNA ÉPOCA SSL + DOWNSTREAM"

        $SmokeRoot = Join-Path $RunAuditDir "smoke_all_arms"

        Invoke-NativeStreaming `
            -FilePath $Python `
            -Arguments @(
                $MatrixRunner,
                "--config",
                "configs\experiment\evttc32_article_ablation_matrix.yaml",
                "--mode",
                "smoke",
                "--output-root",
                $SmokeRoot
            ) `
            -Label "Smoke de las once variantes" | Out-Null

        Write-Log "PASS: smoke de todas las ablaciones completo." "PASS"
    }
    else {
        Write-Log "Smoke omitido por -SkipSmoke." "WARN"
    }

    # -----------------------------------------------------------------------
    # LOCAL COMMIT
    # -----------------------------------------------------------------------
    Write-Step "COMMIT LOCAL DEL PROTOCOLO Y MATRIZ PRERREGISTRADA"

    $pathsToAdd = @(
        "configs/recovery_v3_protocol.yaml",
        "configs/experiment/evttc32_article_ablation_matrix.yaml",
        "data/manifests/evttc_all32_local.yaml",
        "data/splits/evttc_all32_article_family_holdout.yaml"
    )

    $scriptFullPath = $MyInvocation.MyCommand.Path
    if (
        $scriptFullPath -and
        $scriptFullPath.StartsWith(
            $RepoRoot,
            [System.StringComparison]::OrdinalIgnoreCase
        )
    ) {
        $scriptRelative = $scriptFullPath.Substring($RepoRoot.Length).TrimStart("\").Replace("\", "/")
        if ($pathsToAdd -notcontains $scriptRelative) {
            $pathsToAdd += $scriptRelative
        }
    }

    foreach ($path in $pathsToAdd) {
        if (Test-Path -LiteralPath (Join-Path $RepoRoot ($path -replace "/", "\"))) {
            & git add -- $path
            if ($LASTEXITCODE -ne 0) {
                throw "git add falló para $path"
            }
        }
    }

    # frozen_protocol.json suele estar ignorado y no necesita formar parte del
    # commit. `git ls-files --error-unmatch` escribe en stderr y Windows
    # PowerShell 5.1 puede convertirlo en una excepción con ErrorAction=Stop.
    # Se consulta sin error-unmatch y solo se añade si Git ya lo rastrea.
    $trackedFrozen = @(
        & git ls-files -- "artifacts/audit/recovery_v3/frozen_protocol.json"
    )
    if ($LASTEXITCODE -ne 0) {
        throw "No se pudo consultar si frozen_protocol.json está rastreado."
    }

    if ($trackedFrozen.Count -gt 0) {
        & git add -- "artifacts/audit/recovery_v3/frozen_protocol.json"
        if ($LASTEXITCODE -ne 0) {
            throw "git add falló para frozen_protocol.json"
        }
    }
    else {
        Write-Log (
            "frozen_protocol.json existe como artefacto local/ignorado; " +
            "no se añade al commit."
        )
    }

    $staged = (& git diff --cached --name-only)
    if ($staged) {
        Invoke-NativeStreaming `
            -FilePath "git" `
            -Arguments @(
                "commit",
                "-m",
                "Add EvTTC-32 article ablation matrix and family holdout"
            ) `
            -Label "Commit local article-v4" | Out-Null
    }
    else {
        Write-Log "No había cambios nuevos para confirmar."
    }

    $dirtyAfterCommit = @(& git status --porcelain --untracked-files=normal)
    if ($dirtyAfterCommit.Count -gt 0) {
        $dirtyAfterCommit | ForEach-Object { Write-Host $_ }
        throw "El worktree no quedó limpio antes de la matriz experimental."
    }

    $runCommit = (& git rev-parse HEAD).Trim()
    Write-Log "Commit de la matriz: $runCommit"

    # -----------------------------------------------------------------------
    # FULL ARTICLE ABLATION MATRIX
    # -----------------------------------------------------------------------
    Write-Step "MATRIZ COMPLETA: 11 BRAZOS × 3 SEMILLAS × SSL/FINE-TUNING"

    Write-Status `
        -Status "running" `
        -Stage "article_ablation_matrix" `
        -Message "Ejecutando 11 brazos con semillas 7, 13 y 21."

    Invoke-NativeStreaming `
        -FilePath $Python `
        -Arguments @(
            $MatrixRunner,
            "--config",
            "configs\experiment\evttc32_article_ablation_matrix.yaml",
            "--mode",
            "full"
        ) `
        -Label "Matriz EvTTC-32 article-v4" | Out-Null

    # -----------------------------------------------------------------------
    # VALIDATION SUMMARY AND MODEL SELECTION
    # -----------------------------------------------------------------------
    Write-Step "RESUMEN MULTISEMILLA Y SELECCIÓN SOLO CON VALIDATION"

    $SummaryHelper = Join-Path $RunAuditDir "summarize_article_matrix.py"
    @'
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(path)
    return payload


config_path = Path("configs/experiment/evttc32_article_ablation_matrix.yaml")
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
run_root = Path(config["outputs"]["run_root"])
summary_path = Path(config["outputs"]["summary"])
selection_path = Path(config["outputs"]["selection"])
arms = config["experiment"]["arms"]
seeds = [int(seed) for seed in config["experiment"]["seeds"]]
ssl_epochs = int(config["pretrain"]["epochs"])
ft_epochs = int(config["downstream"]["epochs"])

rows: list[dict[str, Any]] = []
for arm_name, arm in arms.items():
    for seed in seeds:
        ssl_path = (
            run_root
            / arm_name.lower()
            / f"seed{seed}"
            / f"ssl{ssl_epochs}"
            / "metrics.json"
        )
        ft_path = (
            run_root
            / arm_name.lower()
            / f"seed{seed}"
            / f"ft{ft_epochs}"
            / "metrics.json"
        )
        if not ssl_path.is_file() or not ft_path.is_file():
            raise FileNotFoundError(f"Incomplete {arm_name} seed {seed}")

        ssl = read_json(ssl_path)
        ft = read_json(ft_path)
        if ft.get("final_test_opened") is not False:
            raise RuntimeError(f"{arm_name} seed {seed} opened test during selection")
        if "test" in set(ft.get("evaluation_splits", [])):
            raise RuntimeError(f"{arm_name} seed {seed} evaluated test during selection")

        validation = ft["splits"]["validation"]
        metrics = validation["metrics"]
        row = {
            "arm": arm_name,
            "group": arm["group"],
            "seed": seed,
            "validation_sequence_count": int(validation["sequence_count"]),
            "mae_s": float(metrics["mae_s"]),
            "rmse_s": float(metrics["rmse_s"]),
            "median_abs_error_s": float(metrics["median_abs_error_s"]),
            "mean_abs_relative_error_pct": float(
                metrics["mean_abs_relative_error_pct"]
            ),
            "ssl_best_validation_loss": float(ssl["best_loss"]),
            "ssl_best_epoch": int(ssl["best_epoch"]),
            "ssl_elapsed_seconds": float(ssl["elapsed_seconds"]),
            "downstream_best_epoch": int(ft["best_epoch"]),
            "downstream_elapsed_seconds": float(ft["elapsed_seconds"]),
            "ssl_metrics_path": ssl_path.as_posix(),
            "downstream_metrics_path": ft_path.as_posix(),
            "checkpoint_path": str(ft["best_checkpoint"]),
            "per_sequence": validation.get("per_sequence", {}),
        }
        if row["validation_sequence_count"] != 5:
            raise RuntimeError(
                f"{arm_name} seed {seed}: validation sequence count "
                f"{row['validation_sequence_count']} != 5"
            )
        rows.append(row)

aggregates: list[dict[str, Any]] = []
for arm_name, arm in arms.items():
    selected = [row for row in rows if row["arm"] == arm_name]
    if len(selected) != 3:
        raise RuntimeError(f"{arm_name}: expected 3 seeds, got {len(selected)}")

    aggregate = {
        "arm": arm_name,
        "group": arm["group"],
        "description": arm["description"],
        "seed_count": len(selected),
        "mae_s_mean": statistics.mean(row["mae_s"] for row in selected),
        "mae_s_std": statistics.stdev(row["mae_s"] for row in selected),
        "rmse_s_mean": statistics.mean(row["rmse_s"] for row in selected),
        "rmse_s_std": statistics.stdev(row["rmse_s"] for row in selected),
        "median_abs_error_s_mean": statistics.mean(
            row["median_abs_error_s"] for row in selected
        ),
        "mean_abs_relative_error_pct_mean": statistics.mean(
            row["mean_abs_relative_error_pct"] for row in selected
        ),
        "ssl_best_validation_loss_mean": statistics.mean(
            row["ssl_best_validation_loss"] for row in selected
        ),
    }
    aggregates.append(aggregate)

aggregates.sort(key=lambda item: (item["mae_s_mean"], item["arm"]))
baseline = next(item for item in aggregates if item["arm"] == "BASE")
eligible = [item for item in aggregates if item["arm"] != "BASE"]
winner = min(eligible, key=lambda item: (item["mae_s_mean"], item["arm"]))

paired = []
for arm_name in arms:
    if arm_name == "BASE":
        continue
    differences = []
    for seed in seeds:
        base_row = next(
            row for row in rows if row["arm"] == "BASE" and row["seed"] == seed
        )
        arm_row = next(
            row for row in rows if row["arm"] == arm_name and row["seed"] == seed
        )
        differences.append(arm_row["mae_s"] - base_row["mae_s"])

    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(20260727)
    samples = rng.choice(values, size=(10000, values.size), replace=True).mean(axis=1)
    paired.append(
        {
            "arm": arm_name,
            "arm_minus_base_mae_s_mean": float(values.mean()),
            "lower_95": float(np.quantile(samples, 0.025)),
            "upper_95": float(np.quantile(samples, 0.975)),
            "wins_vs_base": int(np.sum(values < 0)),
            "seed_count": int(values.size),
        }
    )

payload = {
    "artifact_type": "evttc32_article_ablation_validation_summary",
    "claim_level": "diagnostic",
    "selection_used_holdout": False,
    "selection_split": "validation",
    "validation_sequence_count": 5,
    "arm_count": len(arms),
    "seed_count_per_arm": 3,
    "rows": rows,
    "aggregates_ranked_by_validation_mae": aggregates,
    "paired_against_base": paired,
    "selection": {
        "baseline_arm": "BASE",
        "selected_best_nonbaseline_arm": winner["arm"],
        "selected_validation_mae_s_mean": winner["mae_s_mean"],
        "baseline_validation_mae_s_mean": baseline["mae_s_mean"],
        "selected_improvement_vs_base_pct": (
            100.0
            * (baseline["mae_s_mean"] - winner["mae_s_mean"])
            / baseline["mae_s_mean"]
        ),
        "selection_rule": (
            "Minimum three-seed mean validation MAE among all non-BASE "
            "preregistered arms; holdout remained physically absent."
        ),
    },
    "limitations": config.get("limitations", []),
}

summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(
    json.dumps(payload, indent=2, sort_keys=True),
    encoding="utf-8",
)
selection_path.write_text(
    json.dumps(payload["selection"], indent=2, sort_keys=True),
    encoding="utf-8",
)

csv_path = summary_path.with_suffix(".csv")
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "arm",
            "group",
            "description",
            "seed_count",
            "mae_s_mean",
            "mae_s_std",
            "rmse_s_mean",
            "rmse_s_std",
            "median_abs_error_s_mean",
            "mean_abs_relative_error_pct_mean",
            "ssl_best_validation_loss_mean",
        ],
    )
    writer.writeheader()
    writer.writerows(aggregates)

markdown_path = summary_path.with_suffix(".md")
lines = [
    "# EvTTC-32 article ablation validation summary",
    "",
    "| Rank | Arm | Group | MAE mean (s) | MAE std | RMSE mean (s) |",
    "|---:|---|---|---:|---:|---:|",
]
for rank, item in enumerate(aggregates, start=1):
    lines.append(
        f"| {rank} | {item['arm']} | {item['group']} | "
        f"{item['mae_s_mean']:.6f} | {item['mae_s_std']:.6f} | "
        f"{item['rmse_s_mean']:.6f} |"
    )
lines.extend(
    [
        "",
        f"Selected non-baseline arm: **{winner['arm']}**",
        "",
        (
            "Selection was performed only on the five validation sequences. "
            "The family holdout was not opened."
        ),
    ]
)
markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(json.dumps(payload["selection"], indent=2, sort_keys=True))
print("ARTICLE_SUMMARY_COMPLETE")
'@ | Set-Content -LiteralPath $SummaryHelper -Encoding UTF8

    Invoke-NativeStreaming `
        -FilePath $Python `
        -Arguments @($SummaryHelper) `
        -Label "Resumir y seleccionar por validation" | Out-Null

    Assert-Path -Path $SummaryPath -Description "resumen de ablaciones"
    $SelectionPath = Join-Path $RepoRoot "artifacts\metrics\evttc32_article_selection.json"
    Assert-Path -Path $SelectionPath -Description "selección del mejor brazo"

    # -----------------------------------------------------------------------
    # ROBUSTNESS: BASE VS VALIDATION-SELECTED ARM
    # -----------------------------------------------------------------------
    if ($WithRobustness) {
        Write-Step "ROBUSTEZ EN VALIDATION: BASE VS BRAZO SELECCIONADO"

        $RobustnessHelper = Join-Path $RunAuditDir "evaluate_selected_robustness.py"
        @'
from __future__ import annotations

import gc
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from e_jepa_ttc.data.ml_cache import build_voxel_cache
from e_jepa_ttc.representations.corruptions import EventCorruptionSpec
from e_jepa_ttc.training.supervised import evaluate_supervised_checkpoint


def metric_subset(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "mae_s": float(metrics["mae_s"]),
        "rmse_s": float(metrics["rmse_s"]),
        "median_abs_error_s": float(metrics["median_abs_error_s"]),
        "mean_abs_relative_error_pct": float(
            metrics["mean_abs_relative_error_pct"]
        ),
    }


config = yaml.safe_load(
    Path("configs/experiment/evttc32_article_ablation_matrix.yaml").read_text(
        encoding="utf-8"
    )
)
selection = json.loads(
    Path("artifacts/metrics/evttc32_article_selection.json").read_text(
        encoding="utf-8"
    )
)
selected_arm = str(selection["selected_best_nonbaseline_arm"])
arms = ("BASE", selected_arm)
seeds = [int(seed) for seed in config["experiment"]["seeds"]]
run_root = Path(config["outputs"]["run_root"])
ssl_epochs = int(config["pretrain"]["epochs"])
ft_epochs = int(config["downstream"]["epochs"])

with np.load(config["data"]["cache"], allow_pickle=False) as cache:
    width = int(cache["width"])
    height = int(cache["height"])
    bins = int(cache["bins"])
    normalize = bool(cache["normalize"])
    metadata_channels = bool(cache["metadata_channels"])
    navigation_channels = bool(cache["navigation_channels"])

models = []
for arm in arms:
    for seed in seeds:
        metrics_path = (
            run_root
            / arm.lower()
            / f"seed{seed}"
            / f"ft{ft_epochs}"
            / "metrics.json"
        )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        checkpoint = Path(str(metrics["best_checkpoint"]))
        models.append(
            {
                "arm": arm,
                "seed": seed,
                "checkpoint": checkpoint,
                "model_name": str(metrics["model_name"]),
                "clean": metric_subset(
                    metrics["splits"]["validation"]["metrics"]
                ),
            }
        )

conditions = []
for kind, severities in config.get("robustness", {}).get("conditions", {}).items():
    for severity in severities:
        conditions.append((str(kind), float(severity)))

rows = []
for model in models:
    rows.append(
        {
            "arm": model["arm"],
            "seed": model["seed"],
            "condition": "clean",
            "kind": "none",
            "severity": 0.0,
            "metrics": model["clean"],
            "mae_degradation_relative_pct": 0.0,
        }
    )

cache_records = []
with tempfile.TemporaryDirectory(prefix="evttc32_article_robustness_") as temporary:
    root = Path(temporary)
    for index, (kind, severity) in enumerate(conditions):
        condition = f"{kind}={severity:g}"
        print(f"BUILD {condition} ({index + 1}/{len(conditions)})", flush=True)
        output = root / f"condition_{index:02d}.npz"
        summary = build_voxel_cache(
            manifest_path=Path(config["data"]["manifest"]),
            split_path=Path(config["data"]["split"]),
            index_path=Path(config["data"]["index"]),
            output_path=output,
            width=width,
            height=height,
            bins=bins,
            normalize=normalize,
            metadata_channels=metadata_channels,
            navigation_channels=navigation_channels,
            include_splits=["validation"],
            corruption=EventCorruptionSpec(
                kind=kind,
                severity=severity,
                seed=int(config.get("robustness", {}).get("corruption_seed", 20260727)),
            ),
        )
        with np.load(output, allow_pickle=False) as cache:
            sequences = sorted(set(cache["sequence_id"].astype(str).tolist()))
            physical_splits = set(cache["split"].astype(str).tolist())
        if physical_splits != {"validation"} or len(sequences) != 5:
            raise RuntimeError(
                f"{condition}: splits={physical_splits}, sequences={sequences}"
            )
        cache_records.append(
            {
                "condition": condition,
                "window_count": int(summary["window_count"]),
                "sequence_ids": sequences,
                "cache_sha256": str(summary["cache_sha256"]),
            }
        )

        for model in models:
            print(
                f"EVAL {condition} {model['arm']} seed {model['seed']}",
                flush=True,
            )
            result = evaluate_supervised_checkpoint(
                cache_path=output,
                checkpoint_path=model["checkpoint"],
                output_path=None,
                batch_size=int(config["downstream"]["batch_size"]),
                device_name=str(config["downstream"]["device"]),
                evaluation_splits=("validation",),
                model_name=model["model_name"],
                allow_final_test_evaluation=False,
            )
            metrics = metric_subset(
                result["splits"]["validation"]["metrics"]
            )
            clean = model["clean"]
            rows.append(
                {
                    "arm": model["arm"],
                    "seed": model["seed"],
                    "condition": condition,
                    "kind": kind,
                    "severity": severity,
                    "metrics": metrics,
                    "mae_degradation_relative_pct": (
                        100.0
                        * (metrics["mae_s"] - clean["mae_s"])
                        / clean["mae_s"]
                    ),
                }
            )
            gc.collect()

aggregates = []
for condition in ["clean", *[f"{kind}={severity:g}" for kind, severity in conditions]]:
    for arm in arms:
        selected = [
            row
            for row in rows
            if row["condition"] == condition and row["arm"] == arm
        ]
        mae = np.asarray(
            [row["metrics"]["mae_s"] for row in selected],
            dtype=np.float64,
        )
        degradation = np.asarray(
            [row["mae_degradation_relative_pct"] for row in selected],
            dtype=np.float64,
        )
        aggregates.append(
            {
                "condition": condition,
                "arm": arm,
                "seed_count": len(selected),
                "mae_s_mean": float(mae.mean()),
                "mae_s_std": float(mae.std(ddof=1)),
                "mae_degradation_relative_pct_mean": float(
                    degradation.mean()
                ),
            }
        )

payload = {
    "artifact_type": "evttc32_article_selected_validation_robustness",
    "final_test_opened": False,
    "selection_used_holdout": False,
    "baseline_arm": "BASE",
    "selected_arm": selected_arm,
    "conditions": conditions,
    "cache_records": cache_records,
    "rows": rows,
    "aggregates": aggregates,
}

destination = Path(config["outputs"]["robustness"])
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(
    json.dumps(payload, indent=2, sort_keys=True),
    encoding="utf-8",
)
print("ARTICLE_ROBUSTNESS_COMPLETE")
'@ | Set-Content -LiteralPath $RobustnessHelper -Encoding UTF8

        $robustOk = Invoke-NativeStreaming `
            -FilePath $Python `
            -Arguments @($RobustnessHelper) `
            -Label "Robustez BASE vs seleccionado" `
            -ContinueOnError

        if (-not $robustOk) {
            $RobustnessFailed = $true
            Write-Log `
                "La robustez falló, pero la matriz y selección están completas." `
                "WARN"
        }
    }

    # -----------------------------------------------------------------------
    # ONE-TIME FAMILY HOLDOUT: ONLY BASE AND SELECTED ARM
    # -----------------------------------------------------------------------
    if (-not $SkipHoldoutEvaluation) {
        Write-Step "APERTURA ÚNICA DEL FAMILY HOLDOUT: BASE VS SELECCIONADO"

        $HoldoutHelper = Join-Path $RunAuditDir "evaluate_family_holdout.py"
        @'
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import yaml

from e_jepa_ttc.training.supervised import evaluate_supervised_checkpoint


def metric_subset(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "mae_s": float(metrics["mae_s"]),
        "rmse_s": float(metrics["rmse_s"]),
        "median_abs_error_s": float(metrics["median_abs_error_s"]),
        "mean_abs_relative_error_pct": float(
            metrics["mean_abs_relative_error_pct"]
        ),
    }


config = yaml.safe_load(
    Path("configs/experiment/evttc32_article_ablation_matrix.yaml").read_text(
        encoding="utf-8"
    )
)
selection = json.loads(
    Path("artifacts/metrics/evttc32_article_selection.json").read_text(
        encoding="utf-8"
    )
)
selected_arm = str(selection["selected_best_nonbaseline_arm"])
arms = ("BASE", selected_arm)
seeds = [int(seed) for seed in config["experiment"]["seeds"]]
run_root = Path(config["outputs"]["run_root"])
ft_epochs = int(config["downstream"]["epochs"])
output_root = Path(config["outputs"]["holdout_root"])
output_root.mkdir(parents=True, exist_ok=True)

rows = []
for arm in arms:
    for seed in seeds:
        metrics_path = (
            run_root
            / arm.lower()
            / f"seed{seed}"
            / f"ft{ft_epochs}"
            / "metrics.json"
        )
        training_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        checkpoint = Path(str(training_metrics["best_checkpoint"]))
        output = output_root / f"{arm.lower()}_seed{seed}.json"

        result = evaluate_supervised_checkpoint(
            cache_path=Path(config["data"]["holdout_cache"]),
            checkpoint_path=checkpoint,
            output_path=output,
            batch_size=int(config["downstream"]["batch_size"]),
            device_name=str(config["downstream"]["device"]),
            evaluation_splits=("test",),
            model_name=str(training_metrics["model_name"]),
            allow_final_test_evaluation=True,
        )
        split = result["splits"]["test"]
        if int(split["sequence_count"]) != 8:
            raise RuntimeError(
                f"{arm} seed {seed}: holdout sequence count "
                f"{split['sequence_count']} != 8"
            )
        rows.append(
            {
                "arm": arm,
                "seed": seed,
                "sequence_count": int(split["sequence_count"]),
                **metric_subset(split["metrics"]),
                "per_sequence": split.get("per_sequence", {}),
                "output_path": output.as_posix(),
            }
        )

aggregates = {}
for arm in arms:
    selected = [row for row in rows if row["arm"] == arm]
    aggregates[arm] = {
        "seed_count": len(selected),
        "mae_s_mean": statistics.mean(row["mae_s"] for row in selected),
        "mae_s_std": statistics.stdev(row["mae_s"] for row in selected),
        "rmse_s_mean": statistics.mean(row["rmse_s"] for row in selected),
        "median_abs_error_s_mean": statistics.mean(
            row["median_abs_error_s"] for row in selected
        ),
        "mean_abs_relative_error_pct_mean": statistics.mean(
            row["mean_abs_relative_error_pct"] for row in selected
        ),
    }

paired = []
for seed in seeds:
    base = next(
        row for row in rows if row["arm"] == "BASE" and row["seed"] == seed
    )
    best = next(
        row
        for row in rows
        if row["arm"] == selected_arm and row["seed"] == seed
    )
    paired.append(
        {
            "seed": seed,
            "base_mae_s": base["mae_s"],
            "selected_mae_s": best["mae_s"],
            "selected_minus_base_mae_s": best["mae_s"] - base["mae_s"],
            "selected_improvement_pct": (
                100.0 * (base["mae_s"] - best["mae_s"]) / base["mae_s"]
            ),
        }
    )

payload = {
    "artifact_type": "evttc32_article_family_holdout_summary",
    "claim_level": "diagnostic_family_generalization",
    "final_test_opened": True,
    "official_evttc_benchmark": False,
    "slider_750_1000_evaluated": False,
    "baseline_arm": "BASE",
    "selected_arm": selected_arm,
    "selection_was_completed_before_holdout": True,
    "holdout_sequence_count": 8,
    "rows": rows,
    "aggregates": aggregates,
    "paired": paired,
    "limitations": config.get("limitations", []),
}

summary = output_root / "summary.json"
summary.write_text(
    json.dumps(payload, indent=2, sort_keys=True),
    encoding="utf-8",
)

lines = [
    "EvTTC-32 family holdout",
    "=======================",
    "",
    f"Selected arm: {selected_arm}",
    f"BASE MAE: {aggregates['BASE']['mae_s_mean']:.6f} s",
    (
        f"{selected_arm} MAE: "
        f"{aggregates[selected_arm]['mae_s_mean']:.6f} s"
    ),
    "",
]
for item in paired:
    lines.append(
        f"seed {item['seed']}: improvement "
        f"{item['selected_improvement_pct']:.3f}%"
    )
(output_root / "summary.txt").write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print("\n".join(lines))
print("FAMILY_HOLDOUT_COMPLETE")
'@ | Set-Content -LiteralPath $HoldoutHelper -Encoding UTF8

        Invoke-NativeStreaming `
            -FilePath $Python `
            -Arguments @($HoldoutHelper) `
            -Label "Evaluar family holdout una sola vez" | Out-Null
    }
    else {
        Write-Log `
            "Family holdout no abierto por -SkipHoldoutEvaluation." `
            "WARN"
    }

    # -----------------------------------------------------------------------
    # FINAL REPORT
    # -----------------------------------------------------------------------
    Write-Step "INFORME FINAL ARTICLE-V4"

    $FinalReport = Join-Path $RunAuditDir "FINAL_REPORT.txt"
    $selection = Get-Content -LiteralPath $SelectionPath -Raw | ConvertFrom-Json

    $reportLines = New-Object System.Collections.ArrayList
    [void]$reportLines.Add("E-JEPA-TTC EvTTC-32 article ablation pipeline v4")
    [void]$reportLines.Add("================================================")
    [void]$reportLines.Add("")
    [void]$reportLines.Add("Status: COMPLETED")
    [void]$reportLines.Add("Started: $($ScriptStarted.ToString('o'))")
    [void]$reportLines.Add("Completed: $((Get-Date).ToString('o'))")
    [void]$reportLines.Add("Commit: $runCommit")
    [void]$reportLines.Add("Config: configs/experiment/evttc32_article_ablation_matrix.yaml")
    [void]$reportLines.Add("Summary: $SummaryRelative")
    [void]$reportLines.Add("Selected arm: $($selection.selected_best_nonbaseline_arm)")
    [void]$reportLines.Add("Robustness requested: $([bool]$WithRobustness)")
    [void]$reportLines.Add("Robustness failed: $RobustnessFailed")
    [void]$reportLines.Add("Holdout skipped: $([bool]$SkipHoldoutEvaluation)")
    [void]$reportLines.Add("Log: $MasterLog")
    [void]$reportLines.Add("")
    [void]$reportLines.Add("Scientific scope:")
    [void]$reportLines.Add("- 32 total sequences")
    [void]$reportLines.Add("- 19 train sequences")
    [void]$reportLines.Add("- 5 validation sequences")
    [void]$reportLines.Add("- 8 family-holdout sequences: complete CCRs-2, CCRs-3 and CPNAO")
    [void]$reportLines.Add("- 11 preregistered ablation arms")
    [void]$reportLines.Add("- 3 paired seeds per arm")
    [void]$reportLines.Add("- 33 JEPA pretrainings + 33 supervised fine-tunings")
    [void]$reportLines.Add("- Holdout opened only for BASE and selected best arm")
    [void]$reportLines.Add("- NOT the official ten-sequence benchmark")
    [void]$reportLines.Add("- Slider-750 / Slider-1000 not evaluated")

    $reportLines | Set-Content -LiteralPath $FinalReport -Encoding UTF8
    $reportLines | ForEach-Object { Write-Host $_ }

    Write-Status `
        -Status "completed" `
        -Stage "finished" `
        -Message "Article-v6 completo. Revisa FINAL_REPORT y los resúmenes."

    "SUCCESS" | Set-Content -LiteralPath $SuccessMarker -Encoding UTF8
    Write-Log "ARTICLE-V5 COMPLETADO." "PASS"
}
catch {
    $HadFatalError = $true
    $message = $_.Exception.Message
    Write-Log $message "FATAL"

    $failureText = @(
        "FAILED"
        "Time: $((Get-Date).ToString('o'))"
        "Message: $message"
        "Log: $MasterLog"
        ""
        "Stack:"
        $_.ScriptStackTrace
    )

    $failureText | Set-Content -LiteralPath $FailureMarker -Encoding UTF8

    Write-Status `
        -Status "failed" `
        -Stage "fatal_error" `
        -Message $message

    Write-Host ""
    Write-Host "PIPELINE FAILED" -ForegroundColor Red
    Write-Host "Motivo: $message"
    Write-Host "Log: $MasterLog"
    Write-Host "Marcador: $FailureMarker"
}
finally {
    Reset-AwakeMode

    try {
        Stop-Transcript | Out-Null
    }
    catch {
    }

    if ($HadFatalError) {
        exit 1
    }
}
