param(
    [string]$Device = "cuda:0",
    [string]$GarlRepo = "E:\Garl-TTC",
    [string]$GarlDataset = "E:\GarlTTC_dataset",
    [string]$EapDataset = "E:\eAP_dataset",
    [int]$NumWorkers = 0,
    [int]$PrefetchFactor = 2,
    [int]$BootstrapResamples = 5000,
    [int]$HardwarePollSeconds = 5,
    [switch]$Force,
    [switch]$SkipTests,
    [switch]$SkipBudgetMatchedGarl
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Force Unicode-safe native-process I/O on Windows. This repository path may
# contain non-ASCII characters, so never rely on the active OEM code page.
$script:Utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $script:Utf8NoBom
[Console]::InputEncoding = $script:Utf8NoBom
$OutputEncoding = $script:Utf8NoBom
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

function Invoke-NativeExitCode {
    param(
        [Parameter(Mandatory=$true)][string]$FilePath,
        [Parameter(Mandatory=$false)][string[]]$Arguments = @(),
        [switch]$Quiet
    )
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $FilePath
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = [bool]$Quiet
    $psi.RedirectStandardError = [bool]$Quiet
    if ($Quiet) {
        $psi.StandardOutputEncoding = $script:Utf8NoBom
        $psi.StandardErrorEncoding = $script:Utf8NoBom
    }
    foreach ($arg in $Arguments) { [void]$psi.ArgumentList.Add([string]$arg) }
    $proc = [System.Diagnostics.Process]::new()
    $proc.StartInfo = $psi
    [void]$proc.Start()
    if ($Quiet) {
        $null = $proc.StandardOutput.ReadToEnd()
        $null = $proc.StandardError.ReadToEnd()
    }
    $proc.WaitForExit()
    return [int]$proc.ExitCode
}

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
if($NumWorkers -ne 0){
    throw "Scientific Recovery V3 requires -NumWorkers 0 with the current ~670 MiB torch cache shards. Re-shard before increasing worker count."
}

$ExpectedBase = "954fa52"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Master = Join-Path $Root "artifacts\scientific_recovery_master_v3"
$Logs = Join-Path $Master "logs"
$Audit = Join-Path $Master "audit"
$Configs = Join-Path $Master "configs"
$StatusPath = Join-Path $Master "master_status.json"
$HardwareCsv = Join-Path $Master "hardware\hardware_timeseries.csv"
$HardwareStop = Join-Path $Master "hardware\STOP"
$OutputZip = Join-Path $Root "artifacts\audit\E_JEPA_SCIENTIFIC_RECOVERY_MASTER_V3_RESULTS_$Timestamp.zip"

$A4Base2k = Join-Path $Root "artifacts\runs\causal_scale_eap_screen_a4_dinov3_relational_rgb_v2_seed7\summary.json"
$A4L8RunBase = Join-Path $Root "artifacts\runs\causal_scale_eap_screen_a4_s1_train8192_lambda8_seed7"
$A4L8Summary = Join-Path $A4L8RunBase "summary.json"
$A4L8Checkpoint = Join-Path $A4L8RunBase "model_best.pt"
$TrainManifest = Join-Path $Root "artifacts\cache\garl_object_event_common_roi_train8192_v1\manifest.json"
$ValManifest = Join-Path $Root "artifacts\cache\garl_object_event_common_roi_screen_v4\manifest.json"
$TeacherManifest = Join-Path $Root "artifacts\cache\dinov3_convnext_large_relational_a4_train8192_rgb_v1\manifest.json"
$PreflightV2 = Join-Path $Root "artifacts\metrics\a5_transport_preflight_v2\a5_transport_preflight_v2.json"
$PreflightV3 = Join-Path $Root "artifacts\metrics\a5_transport_preflight_v3_confirm\a5_transport_preflight_v3_confirm.json"

$script:Rows = @()
$script:HardwareProcess = $null
$script:LegacyWinner = $null
$script:LegacyWinnerRun = $null
$script:CausalWinner = $null
$script:FinalCandidateRun = $null
$script:FinalCandidateMode = "legacy"
$script:ClaimsBlocked = $false
$script:TransportBlocked = $false
$script:TransportStopReason = $null
$script:CausalBlocked = $false
$script:NvidiaSmiExe = $null
$script:MasterExitCode = 0

function Resolve-NvidiaSmi {
    # NVIDIA telemetry is observational only. It must never block scientific
    # training when torch CUDA is otherwise available. Resolve common Windows
    # install locations and expose the directory to child pwsh monitor processes.
    $resolved = $null
    try {
        $cmd = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
        if ($cmd -and $cmd.Source -and (Test-Path -LiteralPath $cmd.Source -PathType Leaf)) {
            $resolved = $cmd.Source
        }
    } catch {}

    if (-not $resolved) {
        $candidates = @(
            (Join-Path $env:WINDIR "System32\nvidia-smi.exe"),
            (Join-Path $env:ProgramFiles "NVIDIA Corporation\NVSMI\nvidia-smi.exe")
        )
        if ($env:ProgramW6432) {
            $candidates += (Join-Path $env:ProgramW6432 "NVIDIA Corporation\NVSMI\nvidia-smi.exe")
        }
        foreach ($candidate in $candidates) {
            if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
                $resolved = (Resolve-Path -LiteralPath $candidate).Path
                break
            }
        }
    }

    $script:NvidiaSmiExe = $resolved
    if ($resolved) {
        $dir = Split-Path -Parent $resolved
        $pathParts = @($env:PATH -split [IO.Path]::PathSeparator)
        if ($pathParts -notcontains $dir) {
            $env:PATH = $dir + [IO.Path]::PathSeparator + $env:PATH
        }
        Write-Host "NVIDIA telemetry: $resolved" -ForegroundColor DarkGray
    } else {
        Write-Warning "nvidia-smi.exe was not found. GPU telemetry will be unavailable, but this does not block training."
    }
}

function Write-Status {
    # Snapshot into a true PowerShell Object[] rather than wrapping a generic
    # List[object]. PowerShell 7.x can throw "Argument types do not match"
    # when @($genericList) is embedded in an ordered dictionary.
    $stepsSnapshot = [object[]]$script:Rows
    $trackedLines = @(& git -c core.quotepath=false status --porcelain=v1 --untracked-files=no)
    $trackedDirtyPaths = @($trackedLines | ForEach-Object {
        $payload = if($_.Length -ge 4){$_.Substring(3)}else{$_}
        if($payload -like "* -> *"){$payload = ($payload -split " -> ",2)[1]}
        $payload.Replace("\\","/").Trim()
    })
    $ignoredOperationalDirty = @($trackedDirtyPaths | Where-Object {
        $_ -like "scripts/monitor_scientific_recovery_v*.ps1"
    })
    $blockingTrackedDirty = @($trackedDirtyPaths | Where-Object { $_ -notin $ignoredOperationalDirty })
    $trackedDirtyCount = [int]$trackedDirtyPaths.Count
    $untrackedCount = [int]((& git ls-files --others --exclude-standard | Measure-Object).Count)
    $payload = [ordered]@{
        artifact_type = "scientific_recovery_master_status_v3"
        updated_at = (Get-Date).ToString("o")
        git_head = ((& git rev-parse HEAD) | Select-Object -First 1).Trim()
        git_dirty = [bool]($blockingTrackedDirty.Count -gt 0)
        git_tracked_dirty_count = $trackedDirtyCount
        git_science_dirty_count = [int]$blockingTrackedDirty.Count
        blocking_tracked_dirty_paths = [object[]]$blockingTrackedDirty
        ignored_operational_tracked_dirty_paths = [object[]]$ignoredOperationalDirty
        git_untracked_count = $untrackedCount
        legacy_winner = $script:LegacyWinner
        legacy_winner_run = $script:LegacyWinnerRun
        causal_winner = $script:CausalWinner
        final_candidate_run = $script:FinalCandidateRun
        final_candidate_mode = $script:FinalCandidateMode
        claims_blocked = $script:ClaimsBlocked
        sota_comparison_blocked = $script:ClaimsBlocked
        transport_blocked = $script:TransportBlocked
        transport_stop_reason = $script:TransportStopReason
        causal_blocked = $script:CausalBlocked
        private_test_opened = $false
        steps = $stepsSnapshot
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $StatusPath) | Out-Null
    $payload | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $StatusPath -Encoding utf8
}

function Add-StepStatus {
    param([string]$Name,[string]$Category,[string]$Status,[int]$ExitCode,[string]$Message,[string]$LogDir)
    $entry = [pscustomobject][ordered]@{
        name=$Name; category=$Category; status=$Status; exit_code=$ExitCode; message=$Message;
        log_dir=$LogDir; at=(Get-Date).ToString("o")
    }
    $script:Rows += $entry
    Write-Status
}

function Quote-DisplayArg {
    param([string]$Value)
    if ($Value -match '[\s"]') { return '"' + ($Value -replace '"','\"') + '"' }
    return $Value
}

function Invoke-LoggedProcess {
    param(
        [string]$Name,
        [string]$Category,
        [string]$FilePath,
        [string[]]$Arguments,
        [switch]$AllowFailure
    )
    $stepDir = Join-Path $Logs $Name
    New-Item -ItemType Directory -Force -Path $stepDir | Out-Null
    $stdout = Join-Path $stepDir "stdout.log"
    $stderr = Join-Path $stepDir "stderr.log"
    $command = ((Quote-DisplayArg $FilePath) + " " + (($Arguments | ForEach-Object { Quote-DisplayArg $_ }) -join " ")).Trim()
    $command | Set-Content -LiteralPath (Join-Path $stepDir "command.txt") -Encoding utf8
    (Get-Date).ToString("o") | Set-Content -LiteralPath (Join-Path $stepDir "started_at.txt") -Encoding utf8
    (& git rev-parse HEAD) | Set-Content -LiteralPath (Join-Path $stepDir "git_head.txt") -Encoding utf8
    (& git status --short) | Set-Content -LiteralPath (Join-Path $stepDir "git_status.txt") -Encoding utf8
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    Write-Host $command
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $FilePath
    $psi.WorkingDirectory = $Root
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.StandardOutputEncoding = $script:Utf8NoBom
    $psi.StandardErrorEncoding = $script:Utf8NoBom
    foreach ($arg in $Arguments) { [void]$psi.ArgumentList.Add([string]$arg) }
    $proc = [System.Diagnostics.Process]::new(); $proc.StartInfo=$psi
    [void]$proc.Start()
    $outTask=$proc.StandardOutput.ReadToEndAsync(); $errTask=$proc.StandardError.ReadToEndAsync()
    $proc.WaitForExit(); $out=$outTask.Result; $err=$errTask.Result
    $out | Set-Content -LiteralPath $stdout -Encoding utf8
    $err | Set-Content -LiteralPath $stderr -Encoding utf8
    if ($out) { Write-Host $out.TrimEnd() }
    if ($err) { Write-Host $err.TrimEnd() -ForegroundColor DarkYellow }
    $code=[int]$proc.ExitCode
    $code | Set-Content -LiteralPath (Join-Path $stepDir "exit_code.txt") -Encoding ascii
    (Get-Date).ToString("o") | Set-Content -LiteralPath (Join-Path $stepDir "finished_at.txt") -Encoding utf8
    if ($code -eq 0) { Add-StepStatus $Name $Category "PASS" $code "completed" $stepDir }
    else { Add-StepStatus $Name $Category "FAIL" $code "process exit $code" $stepDir; if (-not $AllowFailure) { throw "$Name failed (exit $code)" } }
    return $code
}

function Invoke-Python {
    param([string]$Name,[string]$Category,[string[]]$Arguments,[switch]$AllowFailure)
    return Invoke-LoggedProcess -Name $Name -Category $Category -FilePath ((Get-Command python).Source) -Arguments $Arguments -AllowFailure:$AllowFailure
}

function Test-JsonDecision {
    param([string]$Path,[string]$Expected)
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    try { return ((Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json).decision -eq $Expected) } catch { return $false }
}
function Test-JsonStatus {
    param([string]$Path,[string]$Expected)
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    try { return ((Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json).status -eq $Expected) } catch { return $false }
}

function Test-CompleteRun {
    param([string]$RunDir)
    $summary = Join-Path $RunDir "summary.json"
    $checkpoint = Join-Path $RunDir "model_best.pt"
    if (-not (Test-Path -LiteralPath $summary) -or -not (Test-Path -LiteralPath $checkpoint)) {
        return $false
    }
    try {
        $j = Get-Content -Raw -LiteralPath $summary | ConvertFrom-Json
        return ($null -ne $j.validation_metrics -and $null -ne $j.checkpoint)
    } catch {
        return $false
    }
}

function Archive-IncompleteRun {
    param([string]$RunDir)
    if (-not (Test-Path -LiteralPath $RunDir)) { return }
    if (Test-CompleteRun $RunDir) { return }
    $archiveRoot = Join-Path $Root "artifacts\runs\_incomplete_archive"
    New-Item -ItemType Directory -Force -Path $archiveRoot | Out-Null
    $leaf = Split-Path -Leaf $RunDir
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $dest = Join-Path $archiveRoot "${leaf}_${stamp}"
    Move-Item -LiteralPath $RunDir -Destination $dest
    Add-StepStatus "archive_${leaf}_${stamp}" "infrastructure" "ARCHIVED" 0 "incomplete/OOM run archived before clean retry" $dest
}

function Invoke-Train {
    param([string]$Name,[string]$Config,[string]$RunDir,[switch]$AllowFailure)
    if ((Test-CompleteRun $RunDir) -and -not $Force) {
        Add-StepStatus $Name "training" "REUSED" 0 "existing complete summary+checkpoint reused" $RunDir
        return 0
    }
    if (Test-Path -LiteralPath $RunDir) {
        if ($Force) { Remove-Item -Recurse -Force -LiteralPath $RunDir }
        else { Archive-IncompleteRun $RunDir }
    }
    New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
    return Invoke-Python $Name "training" @("scripts/train_causal_scale_eap_screen.py","--config",$Config,"--output-dir",$RunDir,"--device",$Device) -AllowFailure:$AllowFailure
}

function Test-StepCudaOom {
    param([string]$Name)
    $stepDir = Join-Path $Logs $Name
    foreach ($file in @("stdout.log","stderr.log")) {
        $path = Join-Path $stepDir $file
        if (Test-Path -LiteralPath $path) {
            $text = Get-Content -Raw -LiteralPath $path
            if ($text -match '(?i)(CUDA error:\s*out of memory|cudaErrorMemoryAllocation|CUDA out of memory)') {
                return $true
            }
        }
    }
    return $false
}

function Get-FreeVramMiB {
    if (-not $script:NvidiaSmiExe) { return -1 }
    try {
        $raw = (& $script:NvidiaSmiExe --query-gpu=memory.free --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
        if ($raw) { return [int]([double]$raw.Trim()) }
    } catch {}
    return -1
}

function Invoke-SequentialTrainPair {
    param(
        [string]$Prefix,
        [string]$ConfigA,[string]$RunA,
        [string]$ConfigB,[string]$RunB
    )
    Add-StepStatus "${Prefix}_execution_policy" "hardware" "SEQUENTIAL" 0 "V3 forbids concurrent CUDA training on 12GB WDDM after V1 OOM" $Logs
    $a=Invoke-Train "${Prefix}_seed13" $ConfigA $RunA -AllowFailure
    $b=Invoke-Train "${Prefix}_seed23" $ConfigB $RunB -AllowFailure
    return @($a,$b)
}

function Start-HardwareMonitor {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $HardwareCsv) | Out-Null
    Remove-Item -Force -ErrorAction SilentlyContinue $HardwareStop
    $args=@("-NoProfile","-ExecutionPolicy","Bypass","-File",(Join-Path $Root "scripts\monitor_hardware.ps1"),"-OutputCsv",$HardwareCsv,"-StopFile",$HardwareStop,"-IntervalSeconds",[string]$HardwarePollSeconds)
    try { $script:HardwareProcess=Start-Process -FilePath "pwsh" -ArgumentList $args -PassThru -WindowStyle Hidden } catch { Write-Warning "hardware monitor could not start: $_" }
}
function Stop-HardwareMonitor {
    try { New-Item -ItemType File -Force -Path $HardwareStop | Out-Null } catch {}
    if($script:HardwareProcess){try{$script:HardwareProcess.WaitForExit(15000)|Out-Null}catch{}}
}

function Read-GateDecision([string]$Path) {
    try { return [string](Get-Content -Raw $Path | ConvertFrom-Json).decision } catch { return "INVALID" }
}

function Package-Results {
    $runDirs = @(
        "artifacts/runs/causal_scale_eap_screen_a4_s1_train8192_lambda4_seed7",
        "artifacts/runs/causal_scale_eap_screen_a4_s1_train8192_lambda8_seed7",
        "artifacts/runs/causal_scale_eap_screen_a4_s1_train8192_lambda8_seed13",
        "artifacts/runs/causal_scale_eap_screen_a4_s1_train8192_lambda8_seed23",
        "artifacts/runs/scientific_recovery_a5_s1_seed7","artifacts/runs/scientific_recovery_a5_s1_seed13","artifacts/runs/scientific_recovery_a5_s1_seed23",
        "artifacts/runs/scientific_recovery_a6_s1_seed7","artifacts/runs/scientific_recovery_a6_s1_seed13","artifacts/runs/scientific_recovery_a6_s1_seed23",
        "artifacts/runs/scientific_recovery_a7_s1_seed7","artifacts/runs/scientific_recovery_a7_s1_seed13","artifacts/runs/scientific_recovery_a7_s1_seed23",
        "artifacts/runs/scientific_recovery_a4_causal_left_seed7","artifacts/runs/scientific_recovery_a4_causal_left_seed13","artifacts/runs/scientific_recovery_a4_causal_left_seed23",
        "artifacts/runs/scientific_recovery_a5_causal_left_seed7","artifacts/runs/scientific_recovery_a5_causal_left_seed13","artifacts/runs/scientific_recovery_a5_causal_left_seed23",
        "artifacts/runs/scientific_recovery_a6_causal_left_seed7","artifacts/runs/scientific_recovery_a6_causal_left_seed13","artifacts/runs/scientific_recovery_a6_causal_left_seed23",
        "artifacts/runs/scientific_recovery_a7_causal_left_seed7","artifacts/runs/scientific_recovery_a7_causal_left_seed13","artifacts/runs/scientific_recovery_a7_causal_left_seed23",
        "artifacts/runs/garl_budget_matched_s1_8192_seed7","artifacts/runs/garl_budget_matched_s1_8192_v2_seed7"
    ) | ForEach-Object { Join-Path $Root $_ } | Where-Object { Test-Path $_ }
    $dynamicRuns = Get-ChildItem (Join-Path $Root "artifacts\runs") -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "scientific_recovery_*_mb*_seed*" -or $_.Name -like "scientific_recovery_a4_s1_mb*_seed*" } |
        Select-Object -ExpandProperty FullName
    $runDirs = @($runDirs + $dynamicRuns | Sort-Object -Unique)
    $args=@("scripts/package_scientific_recovery_results.py","--master-root",$Master,"--repo-root",$Root,"--output-zip",$OutputZip)
    foreach($r in $runDirs){$args += @("--run-dir",$r)}
    $packageCode = Invoke-Python "99_package_results" "packaging" $args -AllowFailure
    if ($packageCode -ne 0) { throw "result packaging failed (exit $packageCode)" }
}

# Conservative numerics: no TF32/torch.compile changes. Do not request expandable_segments on Windows/WDDM.
$env:PYTHONUNBUFFERED = "1"
# Loader safety is more important than process count for ~670 MiB torch shards.
# Leave intra-process CPU parallelism available while keeping one cache reader.
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:OPENBLAS_NUM_THREADS = "4"
$env:NUMEXPR_NUM_THREADS = "4"

if ($Force -and (Test-Path $Master)) { Remove-Item -Recurse -Force $Master }
New-Item -ItemType Directory -Force -Path $Master,$Logs,$Audit,$Configs | Out-Null

# Fail-fast runtime compatibility probe: status serialization must work before
# any expensive training or evaluation process is started.
Write-Status
Resolve-NvidiaSmi
Start-HardwareMonitor

try {
    $head=(& git rev-parse HEAD).Trim()
    $mergeBaseCode = Invoke-NativeExitCode -FilePath ((Get-Command git).Source) -Arguments @("merge-base","--is-ancestor",$ExpectedBase,$head) -Quiet
    if($mergeBaseCode -ne 0){throw "HEAD $head is not a descendant of expected hardening base $ExpectedBase"}
    (& git status --short) | Set-Content (Join-Path $Master "git_status_start.txt") -Encoding utf8
    (& git log -30 --oneline --decorate) | Set-Content (Join-Path $Master "git_log_30.txt") -Encoding utf8
    if ($script:NvidiaSmiExe) {
        (& $script:NvidiaSmiExe) | Set-Content (Join-Path $Master "nvidia_smi_start.txt") -Encoding utf8
    } else {
        "nvidia-smi unavailable; GPU telemetry omitted. Scientific training is not blocked by telemetry availability." |
            Set-Content (Join-Path $Master "nvidia_smi_start.txt") -Encoding utf8
    }
    (python --version) | Set-Content (Join-Path $Master "python_version.txt") -Encoding utf8
    $cpuInfo = Get-CimInstance Win32_ComputerSystem -ErrorAction SilentlyContinue
    $osInfo = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
    $gpuInfo = $null
    if ($script:NvidiaSmiExe) {
        $gpuInfo = (& $script:NvidiaSmiExe --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
    }
    [ordered]@{
        logical_processors = if($cpuInfo){$cpuInfo.NumberOfLogicalProcessors}else{$null}
        physical_memory_gb = if($cpuInfo){[math]::Round([double]$cpuInfo.TotalPhysicalMemory/1GB,2)}else{$null}
        free_memory_gb = if($osInfo){[math]::Round(([double]$osInfo.FreePhysicalMemory*1KB)/1GB,2)}else{$null}
        gpu_snapshot = $gpuInfo
        nvidia_smi_available = [bool]$script:NvidiaSmiExe
        nvidia_smi_path = $script:NvidiaSmiExe
        num_workers_per_training = $NumWorkers
        prefetch_factor = $PrefetchFactor
        parallel_replications = $false
        parallel_policy = "disabled_on_12GB_WDDM_after_V1_dual_process_OOM"
        min_free_vram_for_parallel_mib = $null
        omp_threads_per_process = 4
        private_test_opened = $false
    } | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $Master "hardware_profile.json") -Encoding utf8

    # 00: code integrity. These tests do not open validation/test data.
    $compileArgs=@("-m","py_compile","src/e_jepa_ttc/models/causal_scale_ttc.py","src/e_jepa_ttc/training/causal_scale_eap.py","scripts/train_causal_scale_eap_screen.py","scripts/freeze_a4_s1_runtime_configs.py","scripts/freeze_a5_suite_configs.py","scripts/freeze_scientific_recovery_s1_configs.py","scripts/freeze_causal_hardening_configs.py","scripts/audit_scientific_recovery_contracts.py","scripts/audit_prefix_causality.py","scripts/evaluate_oracle_roi_stress.py","scripts/classify_scientific_recovery_gate.py","scripts/summarize_scientific_recovery_replication.py","scripts/paired_cluster_bootstrap.py","scripts/build_garl_budget_matched_subset.py","scripts/build_scientific_claim_readiness.py","scripts/package_scientific_recovery_results.py","scripts/freeze_hardware_rescue_config.py","scripts/audit_oracle_roi_future_invariance.py","scripts/summarize_a4_s1_replication.py")
    if((Invoke-Python "00_py_compile" "integrity" $compileArgs -AllowFailure) -ne 0){ throw "scientific-recovery code does not compile" }
    if(-not $SkipTests){
        $testArgs=@("-m","pytest","-q","tests/unit/test_scientific_recovery_causality.py","tests/unit/test_a4_s1_lambda8_contract.py","tests/unit/test_a5_local_transport.py","tests/unit/test_a6_transport_adapter.py","tests/unit/test_freeze_scientific_recovery_s1_configs.py","tests/unit/test_causal_scale_ttc.py","tests/unit/test_garl_matched_cached_training.py","tests/unit/test_scientific_recovery_v2.py","tests/unit/test_scientific_recovery_v3.py")
        if((Invoke-Python "01_pytest" "integrity" $testArgs -AllowFailure) -ne 0){ throw "integrity tests failed; training blocked" }
    }

    # 01: claim/casuality audits are diagnostic. Garl audit failure blocks claims, not E-JEPA training.
    $contractOut=Join-Path $Audit "scientific_contracts.json"
    if((Invoke-Python "02_contract_audit" "audit" @("scripts/audit_scientific_recovery_contracts.py","--garl-repo",$GarlRepo,"--output",$contractOut) -AllowFailure) -ne 0){$script:ClaimsBlocked=$true}
    $prefixOut=Join-Path $Audit "prefix_causality.json"
    if((Invoke-Python "03_prefix_audit" "audit" @("scripts/audit_prefix_causality.py","--output",$prefixOut) -AllowFailure) -ne 0){$script:CausalBlocked=$true}
    $futureRoiOut=Join-Path $Audit "oracle_roi_future_invariance.json"
    if((Invoke-Python "03b_oracle_roi_future_invariance" "audit" @("scripts/audit_oracle_roi_future_invariance.py","--output",$futureRoiOut) -AllowFailure) -ne 0){$script:ClaimsBlocked=$true}

    # 02: immutable A4-S1 data/teacher contract. Failure blocks S1-family training.
    foreach($p in @($TrainManifest,$ValManifest,$TeacherManifest)){if(-not(Test-Path $p)){throw "required S1 artifact missing: $p"}}
    if((Invoke-Python "04_a4_s1_prerequisites" "integrity" @("scripts/verify_a4_s1_lambda8_prerequisites.py","--config","configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a4_s1_train8192_lambda8_v1.yaml") -AllowFailure) -ne 0){throw "A4-S1 preregistration/data/teacher gate failed"}
    $a4CfgDir=Join-Path $Configs "a4_runtime"
    if((Invoke-Python "05_freeze_a4_runtime" "freeze" @("scripts/freeze_a4_s1_runtime_configs.py","--output-dir",$a4CfgDir,"--num-workers",[string]$NumWorkers,"--prefetch-factor",[string]$PrefetchFactor) -AllowFailure) -ne 0){throw "A4 runtime freeze failed"}

    # 10: execute the preregistered lambda=4 attribution control. Its failure does not block A5.
    $a4l4run=Join-Path $Root "artifacts\runs\causal_scale_eap_screen_a4_s1_train8192_lambda4_seed7"
    [void](Invoke-Train "10_a4_s1_lambda4_seed7" (Join-Path $a4CfgDir "a4_s1_lambda4_seed7.yaml") $a4l4run -AllowFailure)

    # Ensure primary A4-S1 lambda8 seed7 exists; usually reused from the completed branch.
    if(-not(Test-Path $A4L8Summary) -or -not(Test-Path $A4L8Checkpoint)){
        [void](Invoke-Train "11_a4_s1_lambda8_seed7" (Join-Path $a4CfgDir "a4_s1_lambda8_seed7.yaml") $A4L8RunBase -AllowFailure)
    } else { Add-StepStatus "11_a4_s1_lambda8_seed7" "training" "REUSED" 0 "existing A4-S1 seed7 result/checkpoint reused" $A4L8RunBase }
    if(-not(Test-Path $A4L8Summary) -or -not(Test-Path $A4L8Checkpoint)){throw "A4-S1 lambda8 seed7 comparator/checkpoint unavailable"}
    $A4GateSummary = $A4L8Summary
    $A4GateCheckpoint = $A4L8Checkpoint
    $A4GateConfig = Join-Path $a4CfgDir "a4_s1_lambda8_seed7.yaml"
    $HardwareRescueLabel = $null

    # Replicate A4-S1 lambda8. Failures are recorded but do not stop seed7 transport discovery.
    [void](Invoke-SequentialTrainPair "12_a4_s1_lambda8" (Join-Path $a4CfgDir "a4_s1_lambda8_seed13.yaml") (Join-Path $Root "artifacts\runs\causal_scale_eap_screen_a4_s1_train8192_lambda8_seed13") (Join-Path $a4CfgDir "a4_s1_lambda8_seed23.yaml") (Join-Path $Root "artifacts\runs\causal_scale_eap_screen_a4_s1_train8192_lambda8_seed23"))

    if((Test-Path $A4Base2k) -and (Test-Path (Join-Path $a4l4run "summary.json"))){
        [void](Invoke-Python "13_a4_attribution" "analysis" @("scripts/summarize_a4_s1_attribution.py","--a4-2k-lambda4",$A4Base2k,"--a4-8k-lambda4",(Join-Path $a4l4run "summary.json"),"--a4-8k-lambda8",$A4L8Summary,"--output",(Join-Path $Audit "a4_s1_attribution.json")) -AllowFailure)
    } else { Add-StepStatus "13_a4_attribution" "analysis" "SKIPPED" 0 "one attribution summary missing" $Audit }
    $a4rep13=Join-Path $Root "artifacts\runs\causal_scale_eap_screen_a4_s1_train8192_lambda8_seed13\summary.json"
    $a4rep23=Join-Path $Root "artifacts\runs\causal_scale_eap_screen_a4_s1_train8192_lambda8_seed23\summary.json"
    if((Test-Path $A4L8Summary) -and (Test-Path $a4rep13) -and (Test-Path $a4rep23)){
        [void](Invoke-Python "14_a4_s1_replication" "analysis" @("scripts/summarize_a4_s1_replication.py","--summary",$A4L8Summary,"--summary",$a4rep13,"--summary",$a4rep23,"--output",(Join-Path $Audit "a4_s1_lambda8_replication.json")) -AllowFailure)
    } else { Add-StepStatus "14_a4_s1_replication" "analysis" "SKIPPED" 0 "A4-S1 seed13/23 result missing" $Audit }

    # 20: train-only A5 transport selection. Reuse V3 if valid; otherwise rerun V2/V3 only (zero optimizer steps).
    $selection=$null
    if(Test-Path $PreflightV3){
        try{$j=Get-Content -Raw $PreflightV3|ConvertFrom-Json;if($j.decision.a5_corr_authorized -eq $true){$selection=$PreflightV3}}catch{}
    }
    if(-not $selection){
        [void](Invoke-Python "20_a5_preflight_v2" "train_only_preflight" @("scripts/diagnose_a5_transport_preflight_v2.py","--output-dir","artifacts/metrics/a5_transport_preflight_v2","--device",$Device,"--samples","512","--batch-size","8") -AllowFailure)
        if(Test-Path $PreflightV2){
            $v2=Get-Content -Raw $PreflightV2|ConvertFrom-Json
            if($v2.decision.a5_corr_authorized -eq $true){$selection=$PreflightV2}
            else {
                [void](Invoke-Python "21_a5_preflight_v3" "train_only_preflight" @("scripts/diagnose_a5_transport_preflight_v3_confirm.py","--v2-artifact",$PreflightV2,"--output-dir","artifacts/metrics/a5_transport_preflight_v3_confirm","--device",$Device,"--batch-size","8") -AllowFailure)
                if(Test-Path $PreflightV3){$v3=Get-Content -Raw $PreflightV3|ConvertFrom-Json;if($v3.decision.a5_corr_authorized -eq $true){$selection=$PreflightV3}}
            }
        }
    } else { Add-StepStatus "20_a5_preflight" "train_only_preflight" "REUSED" 0 "authorized train-only selection reused" $selection }

    if(-not $selection){
        $script:TransportBlocked=$true
        $script:TransportStopReason="train_only_preflight_not_authorized"
        Add-StepStatus "22_transport_branch" "decision" "STOP" 0 "A5 train-only transport preflight did not authorize a candidate" $Audit
    } else {
        $a5CfgDir=Join-Path $Configs "a5_s1"
        $freezeCode=Invoke-Python "22_freeze_a5_s1" "freeze" @("scripts/freeze_a5_suite_configs.py","--output-dir",$a5CfgDir,"--preflight-selection",$selection,"--include-scale","--scale-dino-lambda","8","--num-workers",[string]$NumWorkers,"--prefetch-factor",[string]$PrefetchFactor) -AllowFailure
        $a5cfg7=Join-Path $a5CfgDir "scale_8192_seed7.yaml"
        $a5cfg13=Join-Path $a5CfgDir "scale_8192_seed13.yaml"
        $a5cfg23=Join-Path $a5CfgDir "scale_8192_seed23.yaml"
        if($freezeCode -ne 0 -or -not(Test-Path $a5cfg7)){
            $script:TransportBlocked=$true
            $script:TransportStopReason="a5_config_freeze_failed"
            Add-StepStatus "23_a5_s1" "decision" "STOP_INFRASTRUCTURE" 0 "A5-S1 config could not be frozen; diagnostics continue" $Audit
        } else {
            $a5runs=@{}
            foreach($seed in @(7,13,23)){$a5runs[$seed]=Join-Path $Root "artifacts\runs\scientific_recovery_a5_s1_seed$seed"}
            $a5code=Invoke-Train "23_a5_s1_seed7" $a5cfg7 $a5runs[7] -AllowFailure

            # Hardware-only CUDA rescue. Never select by validation metrics.
            # Preserve effective batch 32 and create a matched A4 comparator under
            # the same microbatch/accumulation schedule before applying any gate.
            if($a5code -ne 0 -and (Test-StepCudaOom "23_a5_s1_seed7")){
                Add-StepStatus "23b_a5_hardware_rescue_trigger" "infrastructure" "OOM_RESCUE" 0 "batch32 CUDA OOM; trying preregistered feasibility sequence 16x2 then 8x4" $Audit
                $rescueRoot=Join-Path $Configs "hardware_rescue"
                New-Item -ItemType Directory -Force -Path $rescueRoot | Out-Null
                foreach($rescue in @(@(16,2),@(8,4))){
                    $mb=[int]$rescue[0]; $acc=[int]$rescue[1]; $tag="mb${mb}_ga${acc}"
                    $a4RescueCfg=Join-Path $rescueRoot "a4_s1_lambda8_${tag}_seed7.yaml"
                    [void](Invoke-Python "23c_freeze_a4_${tag}" "hardware_rescue" @("scripts/freeze_hardware_rescue_config.py","--source-config",$A4GateConfig,"--output-config",$a4RescueCfg,"--seed","7","--batch-size",[string]$mb,"--gradient-accumulation-steps",[string]$acc,"--num-workers","0","--prefetch-factor","2") -AllowFailure)
                    $a4RescueRun=Join-Path $Root "artifacts\runs\scientific_recovery_a4_s1_${tag}_seed7"
                    $a4RescueCode=Invoke-Train "23d_a4_${tag}_seed7" $a4RescueCfg $a4RescueRun -AllowFailure
                    if($a4RescueCode -ne 0){ continue }

                    $candidateCfg7=Join-Path $rescueRoot "a5_s1_${tag}_seed7.yaml"
                    $candidateCfg13=Join-Path $rescueRoot "a5_s1_${tag}_seed13.yaml"
                    $candidateCfg23=Join-Path $rescueRoot "a5_s1_${tag}_seed23.yaml"
                    foreach($spec in @(@(7,$candidateCfg7),@(13,$candidateCfg13),@(23,$candidateCfg23))){
                        [void](Invoke-Python "23e_freeze_a5_${tag}_seed$($spec[0])" "hardware_rescue" @("scripts/freeze_hardware_rescue_config.py","--source-config",$a5cfg7,"--output-config",[string]$spec[1],"--seed",[string]$spec[0],"--batch-size",[string]$mb,"--gradient-accumulation-steps",[string]$acc,"--num-workers","0","--prefetch-factor","2") -AllowFailure)
                    }
                    $candidateRuns=@{}
                    foreach($seed in @(7,13,23)){$candidateRuns[$seed]=Join-Path $Root "artifacts\runs\scientific_recovery_a5_s1_${tag}_seed$seed"}
                    $candidateCode=Invoke-Train "23f_a5_${tag}_seed7" $candidateCfg7 $candidateRuns[7] -AllowFailure
                    if($candidateCode -eq 0 -and (Test-CompleteRun $candidateRuns[7])){
                        $a5code=0
                        $a5cfg7=$candidateCfg7; $a5cfg13=$candidateCfg13; $a5cfg23=$candidateCfg23
                        $a5runs=$candidateRuns
                        $A4GateConfig=$a4RescueCfg
                        $A4GateSummary=Join-Path $a4RescueRun "summary.json"
                        $A4GateCheckpoint=Join-Path $a4RescueRun "model_best.pt"
                        $HardwareRescueLabel=$tag
                        Add-StepStatus "23g_a5_hardware_rescue_selected" "hardware_rescue" "PASS" 0 "selected by CUDA feasibility only: $tag; effective batch remains 32" $candidateRuns[7]
                        break
                    }
                    if(-not(Test-StepCudaOom "23f_a5_${tag}_seed7")){ break }
                }
            }

            if($a5code -eq 0 -and (Test-CompleteRun $a5runs[7])){
                $a5Gate=Join-Path $Audit "a5_s1_seed7_gate.json"
                [void](Invoke-Python "24_gate_a5_s1" "gate" @("scripts/classify_scientific_recovery_gate.py","--stage","a5","--base-summary",$A4GateSummary,"--candidate-summary",(Join-Path $a5runs[7] "summary.json"),"--output",$a5Gate) -AllowFailure)
                $decision=Read-GateDecision $a5Gate
                if($decision -eq "REPLICATE_A5"){
                    [void](Invoke-SequentialTrainPair "25_a5_s1" $a5cfg13 $a5runs[13] $a5cfg23 $a5runs[23])
                    if((Test-Path (Join-Path $a5runs[13] "summary.json")) -and (Test-Path (Join-Path $a5runs[23] "summary.json"))){
                        $rep=Join-Path $Audit "a5_s1_replication.json"
                        [void](Invoke-Python "26_replicate_a5" "gate" @("scripts/summarize_scientific_recovery_replication.py","--stage","a5","--base-summary",$A4GateSummary,"--summary",(Join-Path $a5runs[7] "summary.json"),"--summary",(Join-Path $a5runs[13] "summary.json"),"--summary",(Join-Path $a5runs[23] "summary.json"),"--required-passes","2","--output",$rep) -AllowFailure)
                        if(Test-JsonStatus $rep "PASS"){$script:LegacyWinner="a5";$script:LegacyWinnerRun=$a5runs[7]}
                    }
                } elseif($decision -eq "RUN_A6") {
                    $transportCfgDir=Join-Path $Configs "a6_a7_s1"
                    $fc=Invoke-Python "27_freeze_a6_a7_s1" "freeze" @("scripts/freeze_scientific_recovery_s1_configs.py","--a5-s1-config",$a5cfg7,"--a4-s1-checkpoint",$A4GateCheckpoint,"--output-dir",$transportCfgDir,"--num-workers",[string]$NumWorkers,"--prefetch-factor",[string]$PrefetchFactor) -AllowFailure
                    if($fc -eq 0){
                        $a6runs=@{};foreach($seed in @(7,13,23)){$a6runs[$seed]=Join-Path $Root "artifacts\runs\scientific_recovery_a6_s1_seed$seed"}
                        $a6code=Invoke-Train "30_a6_s1_seed7" (Join-Path $transportCfgDir "a6_s1_seed7.yaml") $a6runs[7] -AllowFailure
                        if($a6code -eq 0 -and (Test-Path (Join-Path $a6runs[7] "summary.json"))){
                            $a6Gate=Join-Path $Audit "a6_s1_seed7_gate.json"
                            [void](Invoke-Python "31_gate_a6_s1" "gate" @("scripts/classify_scientific_recovery_gate.py","--stage","a6","--base-summary",$A4GateSummary,"--candidate-summary",(Join-Path $a6runs[7] "summary.json"),"--a5-summary",(Join-Path $a5runs[7] "summary.json"),"--output",$a6Gate) -AllowFailure)
                            $d6=Read-GateDecision $a6Gate
                            if($d6 -eq "REPLICATE_A6"){
                                [void](Invoke-SequentialTrainPair "32_a6_s1" (Join-Path $transportCfgDir "a6_s1_seed13.yaml") $a6runs[13] (Join-Path $transportCfgDir "a6_s1_seed23.yaml") $a6runs[23])
                                if((Test-Path (Join-Path $a6runs[13] "summary.json")) -and (Test-Path (Join-Path $a6runs[23] "summary.json"))){
                                    $rep6=Join-Path $Audit "a6_s1_replication.json"; [void](Invoke-Python "33_replicate_a6" "gate" @("scripts/summarize_scientific_recovery_replication.py","--stage","a6","--base-summary",$A4GateSummary,"--summary",(Join-Path $a6runs[7] "summary.json"),"--summary",(Join-Path $a6runs[13] "summary.json"),"--summary",(Join-Path $a6runs[23] "summary.json"),"--a5-summary",(Join-Path $a5runs[7] "summary.json"),"--required-passes","2","--output",$rep6) -AllowFailure); if(Test-JsonStatus $rep6 "PASS"){$script:LegacyWinner="a6";$script:LegacyWinnerRun=$a6runs[7]}
                                }
                            } elseif($d6 -eq "RUN_A7") {
                                $a7runs=@{};foreach($seed in @(7,13,23)){$a7runs[$seed]=Join-Path $Root "artifacts\runs\scientific_recovery_a7_s1_seed$seed"}
                                $a7code=Invoke-Train "34_a7_s1_seed7" (Join-Path $transportCfgDir "a7_s1_seed7.yaml") $a7runs[7] -AllowFailure
                                if($a7code -eq 0 -and (Test-Path (Join-Path $a7runs[7] "summary.json"))){
                                    $a7Gate=Join-Path $Audit "a7_s1_seed7_gate.json"; [void](Invoke-Python "35_gate_a7_s1" "gate" @("scripts/classify_scientific_recovery_gate.py","--stage","a7","--base-summary",$A4GateSummary,"--candidate-summary",(Join-Path $a7runs[7] "summary.json"),"--a5-summary",(Join-Path $a5runs[7] "summary.json"),"--output",$a7Gate) -AllowFailure)
                                    if((Read-GateDecision $a7Gate) -eq "REPLICATE_A7"){
                                        [void](Invoke-SequentialTrainPair "36_a7_s1" (Join-Path $transportCfgDir "a7_s1_seed13.yaml") $a7runs[13] (Join-Path $transportCfgDir "a7_s1_seed23.yaml") $a7runs[23])
                                        if((Test-Path (Join-Path $a7runs[13] "summary.json")) -and (Test-Path (Join-Path $a7runs[23] "summary.json"))){
                                            $rep7=Join-Path $Audit "a7_s1_replication.json"; [void](Invoke-Python "37_replicate_a7" "gate" @("scripts/summarize_scientific_recovery_replication.py","--stage","a7","--base-summary",$A4GateSummary,"--summary",(Join-Path $a7runs[7] "summary.json"),"--summary",(Join-Path $a7runs[13] "summary.json"),"--summary",(Join-Path $a7runs[23] "summary.json"),"--a5-summary",(Join-Path $a5runs[7] "summary.json"),"--required-passes","2","--output",$rep7) -AllowFailure); if(Test-JsonStatus $rep7 "PASS"){$script:LegacyWinner="a7";$script:LegacyWinnerRun=$a7runs[7]}
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    $script:TransportBlocked=$true
                    $script:TransportStopReason="a5_gate_${decision}"
                    Add-StepStatus "27_transport_decision" "decision" "STOP" 0 "A5 gate=$decision; A6/A7 not scientifically justified" $Audit
                }
            } else {
                $script:TransportBlocked=$true
                $script:TransportStopReason="a5_seed7_infrastructure_failure"
                Add-StepStatus "24_transport_decision" "decision" "STOP_INFRASTRUCTURE" 0 "A5 seed7 failed; dependent A6/A7 blocked, independent diagnostics continue" $Audit
            }
        }
    }

    # 40: strict model-prefix causal hardening. Only a replicated legacy winner is promoted.
    if($script:LegacyWinner -and -not $script:CausalBlocked){
        $stage=[string]$script:LegacyWinner
        $legacyCfgDir = if($stage -eq "a5") { Join-Path $Configs "a5_s1" } else { Join-Path $Configs "a6_a7_s1" }
        $winnerCfg = if($stage -eq "a5") { $a5cfg7 } else { Join-Path $legacyCfgDir "${stage}_s1_seed7.yaml" }
        $causalDir=Join-Path $Configs "causal_$stage"
        [void](Invoke-Python "40_freeze_causal_${stage}_pre" "freeze" @("scripts/freeze_causal_hardening_configs.py","--a4-source-config",$A4GateConfig,"--winner-source-config",$winnerCfg,"--winner-stage",$stage,"--output-dir",$causalDir,"--num-workers",[string]$NumWorkers,"--prefetch-factor",[string]$PrefetchFactor) -AllowFailure)
        # A4 causal seeds are matched comparators for strict-causal replication.
        $causalHardwareSuffix = if($HardwareRescueLabel){"_$HardwareRescueLabel"}else{""}
        $a4cr=@{};foreach($seed in @(7,13,23)){$a4cr[$seed]=Join-Path $Root "artifacts\runs\scientific_recovery_a4_causal_left${causalHardwareSuffix}_seed$seed"}
        $c4=Invoke-Train "41_a4_causal_seed7" (Join-Path $causalDir "a4_s1_lambda8_causal_left_seed7.yaml") $a4cr[7] -AllowFailure
        if($c4 -eq 0 -and (Test-Path (Join-Path $a4cr[7] "model_best.pt"))){
            [void](Invoke-SequentialTrainPair "42_a4_causal" (Join-Path $causalDir "a4_s1_lambda8_causal_left_seed13.yaml") $a4cr[13] (Join-Path $causalDir "a4_s1_lambda8_causal_left_seed23.yaml") $a4cr[23])
            # Re-freeze to cryptographically bind A6/A7 initialization to causal A4 seed7.
            [void](Invoke-Python "43_refreeze_causal_${stage}" "freeze" @("scripts/freeze_causal_hardening_configs.py","--a4-source-config",$A4GateConfig,"--winner-source-config",$winnerCfg,"--winner-stage",$stage,"--causal-a4-checkpoint",(Join-Path $a4cr[7] "model_best.pt"),"--output-dir",$causalDir,"--num-workers",[string]$NumWorkers,"--prefetch-factor",[string]$PrefetchFactor) -AllowFailure)
            # For A6/A7 recovery fractions, train the causal A5 reference on matched seeds.
            $a5CausalDir=$null; $a5cr=@{}
            if($stage -in @("a6","a7")){
                $a5CausalDir=Join-Path $Configs "causal_a5_reference"
                $a5source=$a5cfg7
                [void](Invoke-Python "44_freeze_causal_a5_reference" "freeze" @("scripts/freeze_causal_hardening_configs.py","--a4-source-config",$A4GateConfig,"--winner-source-config",$a5source,"--winner-stage","a5","--output-dir",$a5CausalDir,"--num-workers",[string]$NumWorkers,"--prefetch-factor",[string]$PrefetchFactor) -AllowFailure)
                foreach($seed in @(7,13,23)){$a5cr[$seed]=Join-Path $Root "artifacts\runs\scientific_recovery_a5_causal_left${causalHardwareSuffix}_seed$seed"}
                [void](Invoke-Train "45_a5_causal_seed7" (Join-Path $a5CausalDir "a5_s1_causal_left_seed7.yaml") $a5cr[7] -AllowFailure)
                [void](Invoke-SequentialTrainPair "46_a5_causal" (Join-Path $a5CausalDir "a5_s1_causal_left_seed13.yaml") $a5cr[13] (Join-Path $a5CausalDir "a5_s1_causal_left_seed23.yaml") $a5cr[23])
            }
            $wcr=@{};foreach($seed in @(7,13,23)){$wcr[$seed]=Join-Path $Root "artifacts\runs\scientific_recovery_${stage}_causal_left${causalHardwareSuffix}_seed$seed"}
            $wc=Invoke-Train "47_${stage}_causal_seed7" (Join-Path $causalDir "${stage}_s1_causal_left_seed7.yaml") $wcr[7] -AllowFailure
            if($wc -eq 0){[void](Invoke-SequentialTrainPair "48_${stage}_causal" (Join-Path $causalDir "${stage}_s1_causal_left_seed13.yaml") $wcr[13] (Join-Path $causalDir "${stage}_s1_causal_left_seed23.yaml") $wcr[23])}
            $allCausal = @(7,13,23) | ForEach-Object { Test-Path (Join-Path $wcr[$_] "summary.json") }
            $allBase = @(7,13,23) | ForEach-Object { Test-Path (Join-Path $a4cr[$_] "summary.json") }
            if(($allCausal -notcontains $false) -and ($allBase -notcontains $false)){
                $repArgs=@("scripts/summarize_scientific_recovery_replication.py","--stage",$stage)
                foreach($seed in @(7,13,23)){$repArgs+=@("--base-summary",(Join-Path $a4cr[$seed] "summary.json"))}
                foreach($seed in @(7,13,23)){$repArgs+=@("--summary",(Join-Path $wcr[$seed] "summary.json"))}
                if($stage -in @("a6","a7")){foreach($seed in @(7,13,23)){$repArgs+=@("--a5-summary",(Join-Path $a5cr[$seed] "summary.json"))}}
                $causalRep=Join-Path $Audit "${stage}_causal_replication.json"; $repArgs+=@("--required-passes","2","--output",$causalRep)
                [void](Invoke-Python "49_${stage}_causal_replication" "gate" $repArgs -AllowFailure)
                if(Test-JsonStatus $causalRep "PASS"){$script:CausalWinner=$stage;$script:FinalCandidateRun=$wcr[7];$script:FinalCandidateMode="causal_left"}
            }
        }
    }

    # If no strict-causal transport winner exists, retain the best honest public-only diagnostic candidate.
    if(-not $script:FinalCandidateRun){
        if($script:LegacyWinner){
            $script:FinalCandidateRun=$script:LegacyWinnerRun; $script:FinalCandidateMode="legacy"
        } else {$script:FinalCandidateRun=$A4L8RunBase;$script:FinalCandidateMode="legacy"}
    }

    # 60: ROI stress is independent of transport gates and always useful when a checkpoint exists.
    $candidateCheckpoint=Join-Path $script:FinalCandidateRun "model_best.pt"
    $candidateSummary=Join-Path $script:FinalCandidateRun "summary.json"
    $candidatePred=Join-Path $script:FinalCandidateRun "validation_predictions.csv"
    if((Test-Path $candidateCheckpoint) -and (Test-Path $ValManifest)){
        [void](Invoke-Python "60_oracle_roi_stress" "audit" @("scripts/evaluate_oracle_roi_stress.py","--checkpoint",$candidateCheckpoint,"--validation-manifest",$ValManifest,"--output",(Join-Path $Audit "oracle_roi_stress.json"),"--device",$Device,"--batch-size","16","--num-workers","0","--prefetch-factor",[string]$PrefetchFactor) -AllowFailure)
    }

    # 70: budget-matched Garl 8192 comparator. Build the exact public subset
    # metadata on every V3 run (cheap) so paired bootstrap can cluster by
    # sequence+track. Reuse the expensive completed V2 comparator when present.
    $garlRun=Join-Path $Root "artifacts\runs\garl_budget_matched_s1_8192_v2_seed7"
    $garlSummary=Join-Path $garlRun "summary.json"
    $garlCheckpoint=Join-Path $garlRun "model_best.pt"
    $garlPred=Join-Path $garlRun "validation_predictions.parquet"
    $paired=$null
    $garlComplete=(Test-Path $garlSummary) -and (Test-Path $garlCheckpoint) -and (Test-Path $garlPred)
    if(-not $SkipBudgetMatchedGarl){
        $publicData=Join-Path $GarlDataset "data\train.parquet"; $publicLabels=Join-Path $GarlDataset "annotations\train.parquet"
        $subsetDir=Join-Path $Master "garl_budget_subset"
        $clusterMetadata=Join-Path $subsetDir "validation_data.parquet"
        $sub=-1
        if((Test-Path $publicData) -and (Test-Path $publicLabels) -and (Test-Path $TrainManifest) -and (Test-Path $ValManifest)){
            $sub=Invoke-Python "70_build_garl_8192_subset" "garl_compare" @("scripts/build_garl_budget_matched_subset.py","--train-cache-manifest",$TrainManifest,"--validation-cache-manifest",$ValManifest,"--public-data-parquet",$publicData,"--public-labels-parquet",$publicLabels,"--output-dir",$subsetDir) -AllowFailure
        } else {
            Add-StepStatus "70_build_garl_8192_subset" "garl_compare" "SKIPPED" 0 "public Garl train/labels or E-JEPA manifests missing" $Audit
        }
        if($garlComplete -and -not $Force){
            Add-StepStatus "72_train_garl_8192" "garl_compare" "REUSED" 0 "existing budget-matched Garl 8192/2048 comparator reused" $garlRun
        } elseif($sub -eq 0){
            $cacheDir=Join-Path $Root "artifacts\cache\garl_budget_matched_s1_8192_v2"
            $cacheManifest=Join-Path $cacheDir "manifest.json"
            $bc=0
            if((Test-Path $cacheManifest) -and -not $Force){
                Add-StepStatus "71_build_garl_8192_cache" "garl_compare" "REUSED" 0 "existing Garl 8192 preprocessing cache reused" $cacheDir
            } else {
                $bc=Invoke-Python "71_build_garl_8192_cache" "garl_compare" @("scripts/build_garl_matched_preprocessing_cache.py","--release-root",$GarlRepo,"--subset-manifest",(Join-Path $subsetDir "manifest.json"),"--eap-root",$EapDataset,"--output-dir",$cacheDir,"--batch-size","32","--num-workers","0","--shard-size","64","--seed","7") -AllowFailure
            }
            if($bc -eq 0){
                $gt=Invoke-Python "72_train_garl_8192" "garl_compare" @("scripts/train_garl_matched_from_cache.py","--release-root",$GarlRepo,"--cache-manifest",$cacheManifest,"--output-dir",$garlRun,"--device",$Device,"--seed","7","--epochs","18","--batch-size","32","--num-workers","0","--prefetch-factor","2","--expected-train-rows","8192","--expected-validation-rows","2048","--minimum-epochs","8","--early-stopping-patience","5","--maximum-runtime-hours","8") -AllowFailure
                $garlComplete=($gt -eq 0) -and (Test-Path $garlSummary) -and (Test-Path $garlCheckpoint) -and (Test-Path $garlPred)
            }
        }
        if($garlComplete -and (Test-Path $candidatePred)){
            $paired=Join-Path $Audit "paired_ejepa_vs_garl_8192.json"
            $pbArgs=@("scripts/paired_cluster_bootstrap.py","--ejepa-predictions",$candidatePred,"--garl-predictions",$garlPred,"--resamples",[string]$BootstrapResamples,"--seed","20260811","--output",$paired)
            if(Test-Path $clusterMetadata){$pbArgs+=@("--cluster-metadata",$clusterMetadata)}
            $pb=Invoke-Python "73_paired_bootstrap" "garl_compare" $pbArgs -AllowFailure
            if($pb -ne 0 -or -not(Test-Path $paired)){$script:ClaimsBlocked=$true}
        } else {
            $script:ClaimsBlocked=$true
            Add-StepStatus "73_paired_bootstrap" "garl_compare" "SKIPPED" 0 "Garl comparator or E-JEPA candidate predictions unavailable" $Audit
        }

        # Diagnostic-only paired comparison for the already-observed A5 signal.
        # This is deliberately outside every gate and may never alter A6/A7 selection.
        $a5DiagnosticPred=Join-Path $Root "artifacts\runs\scientific_recovery_a5_s1_seed7\validation_predictions.csv"
        if($garlComplete -and (Test-Path $a5DiagnosticPred)){
            $a5Diagnostic=Join-Path $Audit "paired_a5_vs_garl_8192_diagnostic_only.json"
            $a5PbArgs=@("scripts/paired_cluster_bootstrap.py","--ejepa-predictions",$a5DiagnosticPred,"--garl-predictions",$garlPred,"--resamples",[string]$BootstrapResamples,"--seed","20260811","--output",$a5Diagnostic)
            if(Test-Path $clusterMetadata){$a5PbArgs+=@("--cluster-metadata",$clusterMetadata)}
            [void](Invoke-Python "74_paired_a5_vs_garl_diagnostic_only" "diagnostic" $a5PbArgs -AllowFailure)
        } else {
            Add-StepStatus "74_paired_a5_vs_garl_diagnostic_only" "diagnostic" "SKIPPED" 0 "A5 or Garl predictions unavailable; no gate affected" $Audit
        }
    } else {
        $script:ClaimsBlocked=$true
        Add-StepStatus "70_garl_budget_matched" "garl_compare" "SKIPPED" 0 "-SkipBudgetMatchedGarl requested; SOTA comparison remains blocked" $Audit
    }

    # 80: claim boundary. This can only authorize readiness for a future one-shot sealed test; never SOTA itself.
    if((Test-Path $candidateSummary) -and (Test-Path $contractOut) -and (Test-Path $prefixOut)){
        $claimArgs=@("scripts/build_scientific_claim_readiness.py","--contract-audit",$contractOut,"--prefix-audit",$prefixOut,"--candidate-summary",$candidateSummary,"--candidate-mode",$script:FinalCandidateMode,"--output",(Join-Path $Audit "claim_readiness.json"))
        if($garlSummary -and (Test-Path $garlSummary)){$claimArgs+=@("--garl-budget-summary",$garlSummary)}
        if($paired -and (Test-Path $paired)){$claimArgs+=@("--paired-bootstrap",$paired)}
        [void](Invoke-Python "80_claim_readiness" "claim" $claimArgs -AllowFailure)
    }
}
catch {
    $script:MasterExitCode = 1
    Add-StepStatus "MASTER_EXCEPTION" "master" "FAIL" 1 $_.Exception.Message $Master
    Write-Warning "Master encountered a global integrity/infrastructure stop: $($_.Exception.Message). Independent evidence collected so far will still be packaged."
}
finally {
    Stop-HardwareMonitor
    Write-Status
    try {
        Package-Results
    } catch {
        $script:MasterExitCode = 1
        Write-Warning "Packaging failed: $_"
    }
    Write-Host "`n=== SCIENTIFIC RECOVERY MASTER COMPLETE ===" -ForegroundColor Green
    Write-Host "Master evidence: $Master"
    if(Test-Path $OutputZip){
        Write-Host "ZIP: $OutputZip" -ForegroundColor Green
        Write-Host "SHA256: $((Get-FileHash $OutputZip -Algorithm SHA256).Hash.ToLower())" -ForegroundColor Green
    }
    Write-Host "Private/test evaluation was NOT invoked by this script."
}

if ($script:MasterExitCode -ne 0) {
    exit $script:MasterExitCode
}
