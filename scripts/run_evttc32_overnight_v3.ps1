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
# E-JEPA-TTC EvTTC-32 overnight pipeline v3 (Windows PowerShell 5.1)
# Compatible with Windows PowerShell 5.1.
#
# It performs:
#   1. Fail-fast preflight.
#   2. Validation of 32 EvTTC sequences.
#   3. Reversible promotion of evttc_complete_staging -> evttc.
#   4. Manifest, diagnostic 19/5/8 split, temporal index.
#   5. Train/validation and diagnostic-holdout voxel caches.
#   6. Protocol/config regeneration and protocol freeze.
#   7. One-epoch E0/E1 smoke tests.
#   8. Local git commit required by the frozen runner.
#   9. Full E0/E1 seeds 7/13/21.
#  10. Optional validation robustness.
#  11. One-time diagnostic holdout evaluation on 8 sequences.
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
        "scripts/run_evttc32_overnight.ps1",
        "configs/recovery_v3_protocol.yaml",
        "configs/experiment/flowmimic_e0_e1_evttc32_multiseed.yaml",
        "data/manifests/evttc_all32_local.yaml",
        "data/splits/evttc_all32_flowmimic_diagnostic.yaml"
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
$SplitPath = Join-Path $RepoRoot "data\splits\evttc_all32_flowmimic_diagnostic.yaml"
$IndexPath = Join-Path $RepoRoot "data\cache\evttc_all32_index.json"

$TrainValDir = Join-Path $RepoRoot "artifacts\features\evttc32_trainval"
$TrainValCache = Join-Path $TrainValDir "cache.npz"

$HoldoutDir = Join-Path $RepoRoot "artifacts\features\evttc32_diagnostic_holdout"
$HoldoutCache = Join-Path $HoldoutDir "cache.npz"

$ProtocolPath = Join-Path $RepoRoot "configs\recovery_v3_protocol.yaml"
$SourceConfigPath = Join-Path $RepoRoot "configs\experiment\flowmimic_e0_e1_multiseed.yaml"
$ConfigPath = Join-Path $RepoRoot "configs\experiment\flowmimic_e0_e1_evttc32_multiseed.yaml"
$FrozenProtocolPath = Join-Path $RepoRoot "artifacts\audit\recovery_v3\frozen_protocol.json"

$RunRootRelative = "artifacts/runs/flowmimic_evttc32"
$RunRoot = Join-Path $RepoRoot ($RunRootRelative -replace "/", "\")
$SummaryRelative = "artifacts/metrics/flowmimic_evttc32_multiseed_summary.json"
$SummaryPath = Join-Path $RepoRoot ($SummaryRelative -replace "/", "\")
$RobustnessRelative = "artifacts/metrics/flowmimic_evttc32_robustness.json"
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

    if (Test-PendingReboot) {
        throw "Windows tiene un reinicio pendiente. Reinicia antes de lanzar el run nocturno."
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
split_path = Path("data/splits/evttc_all32_flowmimic_diagnostic.yaml")

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

# Diagnostic holdout. It is NOT pristine/final:
# some configurations were already used or inspected in previous work.
test_canonical = {
    "CCRs-1-low-100",
    "CCRs-1-medium-100",
    "CCRs-1-high-100",
    "CCRs-2-low-100",
    "CCRs-2-medium-100",
    "CCRs-2-high-100",
    "CCRm-low-100",
    "CCRm-medium-100",
}

validation_canonical = {
    "CCRs-3-medium-100",
    "CCRs-side-high",
    "CPLA-high",
    "CPNA-high",
    "CPNAO-high",
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
    "protocol": "evttc-32-flowmimic-diagnostic-2026-07-26",
    "status": "reused_test_diagnostic",
    "evaluation_role": "diagnostic",
    "allowed_claim_levels": ["development", "diagnostic"],
    "test_was_previously_inspected": True,
    "notes": (
        "Diagnostic 19/5/8 sequence split. All 32 sequences participate in the "
        "protocol, but only train+validation enter model fitting/selection. "
        "The eight test configurations are excluded from the gate cache and "
        "opened once after the six checkpoints are fixed. This is not an "
        "official/final holdout because prior work inspected or trained on "
        "some overlapping configurations."
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

    Invoke-NativeStreaming `
        -FilePath $Python `
        -Arguments @(
            "-m", "e_jepa_ttc",
            "cache", "voxel",
            "--manifest", "data\manifests\evttc_all32_local.yaml",
            "--split", "data\splits\evttc_all32_flowmimic_diagnostic.yaml",
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

    Invoke-NativeStreaming `
        -FilePath $Python `
        -Arguments @(
            "-m", "e_jepa_ttc",
            "cache", "voxel",
            "--manifest", "data\manifests\evttc_all32_local.yaml",
            "--split", "data\splits\evttc_all32_flowmimic_diagnostic.yaml",
            "--index", "data\cache\evttc_all32_index.json",
            "--output", "artifacts\features\evttc32_diagnostic_holdout\cache.npz",
            "--width", "160",
            "--height", "90",
            "--bins", "5",
            "--no-normalize",
            "--metadata-channels",
            "--navigation-channels",
            "--include-split", "test"
        ) `
        -Label "Construir caché holdout diagnóstico" | Out-Null

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
    "artifacts/features/evttc32_diagnostic_holdout/cache.npz",
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
    # PROTOCOL + CONFIG
    # -----------------------------------------------------------------------
    Write-Step "ACTUALIZAR PROTOCOLO, CONFIG E0/E1 Y CONGELAR"

    Copy-Item `
        -LiteralPath $ProtocolPath `
        -Destination (Join-Path $RunAuditDir "recovery_v3_protocol.before.yaml") `
        -Force

    $PrepareHelper = Join-Path $RunAuditDir "prepare_protocol_and_config.py"
    @'
from hashlib import sha256
from pathlib import Path
import yaml

protocol_path = Path("configs/recovery_v3_protocol.yaml")
source_config_path = Path("configs/experiment/flowmimic_e0_e1_multiseed.yaml")
config_path = Path("configs/experiment/flowmimic_e0_e1_evttc32_multiseed.yaml")
manifest_path = Path("data/manifests/evttc_all32_local.yaml")
split_path = Path("data/splits/evttc_all32_flowmimic_diagnostic.yaml")
index_path = Path("data/cache/evttc_all32_index.json")
cache_path = Path("artifacts/features/evttc32_trainval/cache.npz")

def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

split_document = yaml.safe_load(split_path.read_text(encoding="utf-8"))
splits = split_document["splits"]

protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
protocol["cache_format_version"] = 2
protocol["resources"]["evttc_dataset_manifest"] = manifest_path.as_posix()
protocol["resources"]["evttc_split_manifest"] = split_path.as_posix()
protocol["resources"]["evttc_cache"] = cache_path.as_posix()
protocol["requirements"]["forbidden_ordinary_splits"] = ["test"]
protocol["requirements"]["dataset"]["name"] = "evttc"
protocol["requirements"]["dataset"]["version"] = "EvTTC-32-local-2026-07-26"
protocol["requirements"]["dataset"]["manifest_hash"] = file_hash(manifest_path)

protocol_path.write_text(
    yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)

protocol_sha256 = file_hash(protocol_path)
cache_sha256 = file_hash(cache_path)

config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
config["experiment"]["id"] = "flowmimic-e0-e1-evttc32-multiseed-2026-07-26"
config["experiment"]["claim_level"] = "validation"

config["protocol"]["version"] = "3.0"
config["protocol"]["sha256"] = protocol_sha256
config["protocol"]["final_test_opened"] = False
config["protocol"]["selection_split"] = "validation"
config["protocol"]["validation_sequences"] = splits["validation"]
config["protocol"]["physically_excluded_splits"] = ["test"]
# The robustness script accepts one sentinel closed sequence; the cache itself
# physically excludes all eight test sequences.
config["protocol"]["closed_sequence"] = splits["test"][0]

config["data"]["cache"] = cache_path.as_posix()
config["data"]["cache_sha256"] = cache_sha256
config["data"]["manifest"] = manifest_path.as_posix()
config["data"]["split"] = split_path.as_posix()
config["data"]["index"] = index_path.as_posix()
config["data"]["train_splits"] = ["train"]
config["data"]["validation_splits"] = ["validation"]
config["data"]["evaluation_splits"] = ["train", "validation"]

config["outputs"]["run_root"] = "artifacts/runs/flowmimic_evttc32"
config["outputs"]["summary"] = (
    "artifacts/metrics/flowmimic_evttc32_multiseed_summary.json"
)
config["outputs"]["robustness"] = (
    "artifacts/metrics/flowmimic_evttc32_robustness.json"
)

config_path.write_text(
    yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)

print("protocol_sha256=" + protocol_sha256)
print("cache_sha256=" + cache_sha256)
print("validation_sequences=" + ",".join(splits["validation"]))
print("test_sequences=" + ",".join(splits["test"]))
print("CONFIG_PREPARED")
'@ | Set-Content -LiteralPath $PrepareHelper -Encoding UTF8

    $prepareOutput = Invoke-NativeCapture `
        -FilePath $Python `
        -Arguments @($PrepareHelper) `
        -Label "Preparar protocolo y config"

    if ($prepareOutput -notmatch "CONFIG_PREPARED") {
        throw "No se completó la preparación del protocolo/config."
    }

    Invoke-NativeStreaming `
        -FilePath $Python `
        -Arguments @("scripts\freeze_protocol.py") `
        -Label "Congelar protocolo EvTTC-32" | Out-Null

    Assert-Path -Path $FrozenProtocolPath -Description "artefacto de protocolo congelado"

    # -----------------------------------------------------------------------
    # SMOKE
    # -----------------------------------------------------------------------
    if (-not $SkipSmoke) {
        Write-Step "SMOKE REAL DE UNA ÉPOCA PARA E0 Y E1"

        $SmokeRoot = Join-Path $RunAuditDir "smoke"
        New-Item -ItemType Directory -Force -Path $SmokeRoot | Out-Null

        foreach ($variant in @("E0", "E1")) {
            $alignmentWeight = "0.0"
            if ($variant -eq "E1") {
                $alignmentWeight = "0.25"
            }

            $sslDir = Join-Path $SmokeRoot ("{0}_ssl" -f $variant.ToLowerInvariant())
            $ftDir = Join-Path $SmokeRoot ("{0}_ft" -f $variant.ToLowerInvariant())

            Invoke-NativeStreaming `
                -FilePath $Python `
                -Arguments @(
                    "-m", "e_jepa_ttc",
                    "pretrain", "jepa",
                    "--cache", "artifacts\features\evttc32_trainval\cache.npz",
                    "--output-dir", $sslDir,
                    "--epochs", "1",
                    "--batch-size", "12",
                    "--learning-rate", "0.0003",
                    "--seed", "7",
                    "--device", "auto",
                    "--model", "event-tubelet-transformer",
                    "--navigation-mode", "enabled",
                    "--pretrain-splits", "train",
                    "--validation-splits", "validation",
                    "--temporal-horizons-ms", "20", "60", "100", "240", "500",
                    "--max-target-slop-ms", "10",
                    "--mask-ratio", "0.45",
                    "--block-count", "4",
                    "--mask-mode", "tubelet",
                    "--ema-momentum", "0.99",
                    "--regularizer", "variance",
                    "--variance-weight", "1.0",
                    "--min-std", "0.05",
                    "--dense-predictor", "transformer",
                    "--flowmimic-alignment-weight", $alignmentWeight,
                    "--flowmimic-inverse-ttc-weight", "0.0",
                    "--flowmimic-minimum-ttc-s", "0.8",
                    "--flowmimic-maximum-ttc-s", "4.0"
                ) `
                -Label "$variant smoke SSL" | Out-Null

            $sslMetricsPath = Join-Path $sslDir "metrics.json"
            Assert-Path -Path $sslMetricsPath -Description "$variant smoke SSL metrics"

            $sslMetrics = Get-Content -LiteralPath $sslMetricsPath -Raw | ConvertFrom-Json
            $checkpoint = [string]$sslMetrics.best_checkpoint
            Assert-Path -Path (Join-Path $RepoRoot ($checkpoint -replace "/", "\")) -Description "$variant smoke checkpoint"

            Invoke-NativeStreaming `
                -FilePath $Python `
                -Arguments @(
                    "-m", "e_jepa_ttc",
                    "train", "tiny-cnn",
                    "--cache", "artifacts\features\evttc32_trainval\cache.npz",
                    "--output-dir", $ftDir,
                    "--epochs", "1",
                    "--batch-size", "24",
                    "--learning-rate", "0.00003",
                    "--seed", "7",
                    "--device", "auto",
                    "--model", "event-tubelet-transformer",
                    "--navigation-mode", "enabled",
                    "--pretrained-encoder", $checkpoint,
                    "--train-splits", "train",
                    "--validation-splits", "validation",
                    "--evaluation-splits", "train", "validation"
                ) `
                -Label "$variant smoke downstream" | Out-Null

            Assert-Path -Path (Join-Path $ftDir "metrics.json") -Description "$variant smoke downstream metrics"
        }

        Write-Log "PASS: smoke E0/E1 completo." "PASS"
    }
    else {
        Write-Log "Smoke omitido por -SkipSmoke." "WARN"
    }

    # -----------------------------------------------------------------------
    # LOCAL COMMIT
    # -----------------------------------------------------------------------
    Write-Step "COMMIT LOCAL REQUERIDO POR EL RUNNER"

    $pathsToAdd = @(
        "configs/recovery_v3_protocol.yaml",
        "configs/experiment/flowmimic_e0_e1_evttc32_multiseed.yaml",
        "data/manifests/evttc_all32_local.yaml",
        "data/splits/evttc_all32_flowmimic_diagnostic.yaml"
    )

    $scriptFullPath = $MyInvocation.MyCommand.Path
    if ($scriptFullPath -and $scriptFullPath.StartsWith($RepoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
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

    # Add frozen artifact only if Git already tracks it. Ignored artifacts remain
    # usable locally without forcing large/ephemeral artifacts into the repository.
    & git ls-files --error-unmatch "artifacts/audit/recovery_v3/frozen_protocol.json" 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        & git add -- "artifacts/audit/recovery_v3/frozen_protocol.json"
        if ($LASTEXITCODE -ne 0) {
            throw "git add falló para frozen_protocol.json"
        }
    }

    $staged = (& git diff --cached --name-only)
    if ($staged) {
        Invoke-NativeStreaming `
            -FilePath "git" `
            -Arguments @(
                "commit",
                "-m",
                "Add EvTTC-32 FlowMimic overnight diagnostic gate"
            ) `
            -Label "Commit local EvTTC-32" | Out-Null
    }
    else {
        Write-Log "No había cambios nuevos para confirmar."
    }

    $dirtyAfterCommit = @(& git status --porcelain --untracked-files=normal)
    if ($dirtyAfterCommit.Count -gt 0) {
        $dirtyAfterCommit | ForEach-Object { Write-Host $_ }
        throw "El worktree no quedó limpio; el runner congelado abortaría."
    }

    $runCommit = (& git rev-parse HEAD).Trim()
    Write-Log "Commit del run: $runCommit"

    # -----------------------------------------------------------------------
    # FULL E0/E1 GATE
    # -----------------------------------------------------------------------
    Write-Step "RUN COMPLETO E0/E1: SEEDS 7, 13 Y 21"

    Write-Status -Status "running" -Stage "full_gate" -Message "Ejecutando E0/E1 multisemilla."

    Invoke-NativeStreaming `
        -FilePath $Python `
        -Arguments @(
            "scripts\run_flowmimic_multiseed.py",
            "--config",
            "configs\experiment\flowmimic_e0_e1_evttc32_multiseed.yaml"
        ) `
        -Label "FlowMimic EvTTC-32 E0/E1 multiseed" | Out-Null

    Assert-Path -Path $SummaryPath -Description "resumen multisemilla"

    # -----------------------------------------------------------------------
    # OPTIONAL ROBUSTNESS
    # -----------------------------------------------------------------------
    if ($WithRobustness) {
        Write-Step "ROBUSTEZ DE VALIDACIÓN (OPCIONAL)"

        Write-Log "Nota: el script upstream contiene textos de limitaciones heredados del protocolo antiguo; las métricas siguen calculándose sobre las 5 secuencias de validation definidas en el nuevo caché." "WARN"

        $robustOk = Invoke-NativeStreaming `
            -FilePath $Python `
            -Arguments @(
                "scripts\evaluate_flowmimic_robustness.py",
                "--config",
                "configs\experiment\flowmimic_e0_e1_evttc32_multiseed.yaml",
                "--output",
                $RobustnessRelative
            ) `
            -Label "Robustez raw-event EvTTC-32" `
            -ContinueOnError

        if ($robustOk) {
            Invoke-NativeStreaming `
                -FilePath $Python `
                -Arguments @(
                    "scripts\summarize_flowmimic_multiseed.py",
                    "--config",
                    "configs\experiment\flowmimic_e0_e1_evttc32_multiseed.yaml",
                    "--output",
                    $SummaryRelative,
                    "--robustness",
                    $RobustnessRelative
                ) `
                -Label "Regenerar resumen con robustez" | Out-Null
        }
        else {
            $RobustnessFailed = $true
            Write-Log "La robustez falló, pero el gate E0/E1 está completo. Se continúa con el holdout." "WARN"
        }
    }

    # -----------------------------------------------------------------------
    # ONE-TIME DIAGNOSTIC HOLDOUT
    # -----------------------------------------------------------------------
    if (-not $SkipHoldoutEvaluation) {
        Write-Step "APERTURA ÚNICA DEL HOLDOUT DIAGNÓSTICO DE 8 SECUENCIAS"

        $HoldoutMetricsDir = Join-Path $RepoRoot "artifacts\metrics\evttc32_diagnostic_holdout"
        New-Item -ItemType Directory -Force -Path $HoldoutMetricsDir | Out-Null

        foreach ($variant in @("e0", "e1")) {
            foreach ($seed in @(7, 13, 21)) {
                $checkpointRelative = "artifacts/runs/flowmimic_evttc32/flowmimic_full_{0}_seed{1}_ft30/tiny_cnn_best.pt" -f $variant, $seed
                $checkpointPath = Join-Path $RepoRoot ($checkpointRelative -replace "/", "\")
                $outputRelative = "artifacts/metrics/evttc32_diagnostic_holdout/{0}_seed{1}.json" -f $variant, $seed

                try {
                    Assert-Path -Path $checkpointPath -Description "$variant seed $seed checkpoint"

                    Invoke-NativeStreaming `
                        -FilePath $Python `
                        -Arguments @(
                            "-m", "e_jepa_ttc",
                            "train", "evaluate",
                            "--cache", "artifacts\features\evttc32_diagnostic_holdout\cache.npz",
                            "--checkpoint", $checkpointRelative,
                            "--output", $outputRelative,
                            "--batch-size", "64",
                            "--device", "auto",
                            "--model", "event-tubelet-transformer",
                            "--evaluation-splits", "test",
                            "--allow-final-test-evaluation"
                        ) `
                        -Label "Holdout $variant seed $seed" | Out-Null
                }
                catch {
                    [void]$HoldoutFailures.Add("$variant seed $seed`: $($_.Exception.Message)")
                    Write-Log "Falló holdout $variant seed $seed; se continúa con los demás." "ERROR"
                }
            }
        }

        $HoldoutSummaryHelper = Join-Path $RunAuditDir "summarize_holdout.py"
        @'
from pathlib import Path
import json
import statistics

root = Path("artifacts/metrics/evttc32_diagnostic_holdout")
rows = []

for variant in ("e0", "e1"):
    for seed in (7, 13, 21):
        path = root / f"{variant}_seed{seed}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["splits"]["test"]["metrics"]
        rows.append(
            {
                "variant": variant.upper(),
                "seed": seed,
                "mae_s": float(metrics["mae_s"]),
                "rmse_s": float(metrics["rmse_s"]),
                "median_abs_error_s": float(metrics["median_abs_error_s"]),
                "mean_abs_relative_error_pct": float(
                    metrics["mean_abs_relative_error_pct"]
                ),
            }
        )

aggregates = {}
for variant in ("E0", "E1"):
    selected = [row for row in rows if row["variant"] == variant]
    if selected:
        aggregates[variant] = {
            "seed_count": len(selected),
            "mae_s_mean": statistics.mean(row["mae_s"] for row in selected),
            "rmse_s_mean": statistics.mean(row["rmse_s"] for row in selected),
            "median_abs_error_s_mean": statistics.mean(
                row["median_abs_error_s"] for row in selected
            ),
            "mean_abs_relative_error_pct_mean": statistics.mean(
                row["mean_abs_relative_error_pct"] for row in selected
            ),
        }

paired = []
for seed in (7, 13, 21):
    e0 = next(
        (row for row in rows if row["variant"] == "E0" and row["seed"] == seed),
        None,
    )
    e1 = next(
        (row for row in rows if row["variant"] == "E1" and row["seed"] == seed),
        None,
    )
    if e0 and e1:
        paired.append(
            {
                "seed": seed,
                "e0_mae_s": e0["mae_s"],
                "e1_mae_s": e1["mae_s"],
                "e1_minus_e0_mae_s": e1["mae_s"] - e0["mae_s"],
                "e1_improvement_pct": (
                    100.0 * (e0["mae_s"] - e1["mae_s"]) / e0["mae_s"]
                ),
            }
        )

payload = {
    "artifact_type": "evttc32_diagnostic_holdout_summary",
    "claim_level": "diagnostic",
    "final_or_official": False,
    "limitations": [
        "This is not the complete official 10-sequence benchmark.",
        "Slider-750 and Slider-1000 are not evaluated.",
        "Some overlapping configurations were used or inspected in prior work.",
    ],
    "rows": rows,
    "aggregates": aggregates,
    "paired": paired,
}

output = root / "summary.json"
output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

text_lines = [
    "EvTTC-32 diagnostic holdout summary",
    "===================================",
    "",
]
for variant in ("E0", "E1"):
    if variant in aggregates:
        item = aggregates[variant]
        text_lines.append(
            f"{variant}: seeds={item['seed_count']} "
            f"MAE={item['mae_s_mean']:.6f}s "
            f"RMSE={item['rmse_s_mean']:.6f}s"
        )
text_lines.append("")
for item in paired:
    text_lines.append(
        f"seed {item['seed']}: E0={item['e0_mae_s']:.6f}s "
        f"E1={item['e1_mae_s']:.6f}s "
        f"improvement={item['e1_improvement_pct']:.3f}%"
    )

(root / "summary.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
print("\n".join(text_lines))
'@ | Set-Content -LiteralPath $HoldoutSummaryHelper -Encoding UTF8

        Invoke-NativeStreaming `
            -FilePath $Python `
            -Arguments @($HoldoutSummaryHelper) `
            -Label "Resumir holdout diagnóstico" | Out-Null
    }
    else {
        Write-Log "Holdout no abierto por -SkipHoldoutEvaluation." "WARN"
    }

    # -----------------------------------------------------------------------
    # FINAL REPORT
    # -----------------------------------------------------------------------
    Write-Step "INFORME FINAL"

    $FinalReport = Join-Path $RunAuditDir "FINAL_REPORT.txt"
    $reportLines = New-Object System.Collections.ArrayList

    [void]$reportLines.Add("E-JEPA-TTC EvTTC-32 overnight pipeline")
    [void]$reportLines.Add("=======================================")
    [void]$reportLines.Add("")
    [void]$reportLines.Add("Status: COMPLETED")
    [void]$reportLines.Add("Started: $($ScriptStarted.ToString('o'))")
    [void]$reportLines.Add("Completed: $((Get-Date).ToString('o'))")
    [void]$reportLines.Add("Commit: $runCommit")
    [void]$reportLines.Add("Config: configs/experiment/flowmimic_e0_e1_evttc32_multiseed.yaml")
    [void]$reportLines.Add("Summary: $SummaryRelative")
    [void]$reportLines.Add("Robustness requested: $([bool]$WithRobustness)")
    [void]$reportLines.Add("Robustness failed: $RobustnessFailed")
    [void]$reportLines.Add("Holdout skipped: $([bool]$SkipHoldoutEvaluation)")
    [void]$reportLines.Add("Holdout failures: $($HoldoutFailures.Count)")
    [void]$reportLines.Add("Log: $MasterLog")
    [void]$reportLines.Add("")
    [void]$reportLines.Add("Scientific scope:")
    [void]$reportLines.Add("- 19 train sequences")
    [void]$reportLines.Add("- 5 validation sequences")
    [void]$reportLines.Add("- 8 diagnostic holdout sequences")
    [void]$reportLines.Add("- NOT an official/final result")
    [void]$reportLines.Add("- Slider-750 / Slider-1000 not evaluated")

    if ($HoldoutFailures.Count -gt 0) {
        [void]$reportLines.Add("")
        [void]$reportLines.Add("Holdout failures:")
        foreach ($failure in $HoldoutFailures) {
            [void]$reportLines.Add("- $failure")
        }
    }

    $reportLines | Set-Content -LiteralPath $FinalReport -Encoding UTF8
    $reportLines | ForEach-Object { Write-Host $_ }

    Write-Status `
        -Status "completed" `
        -Stage "finished" `
        -Message "Pipeline completo. Revisa FINAL_REPORT.txt y los resúmenes."

    "SUCCESS" | Set-Content -LiteralPath $SuccessMarker -Encoding UTF8
    Write-Log "PIPELINE COMPLETADO." "PASS"
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
