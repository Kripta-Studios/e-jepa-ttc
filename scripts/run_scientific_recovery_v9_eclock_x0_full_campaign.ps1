[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$AuthorizedCommit,
    [string]$RepoRoot = 'C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\EVOCON_JEPA_Codex_Handoff\e-jepa-ttc-v9-eclock-x0',
    [string]$ReferenceRoot = 'C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\EVOCON_JEPA_Codex_Handoff\e-jepa-ttc',
    [string]$CacheRoot = 'C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\EVOCON_JEPA_Codex_Handoff\e-jepa-ttc\artifacts\cache\garl_object_event_common_roi_train8192_v1',
    [string]$OutputBase = 'artifacts\scientific_recovery_v9_eclock\campaigns',
    [ValidateSet('auto', 'fold_ram', 'shard_lru')]
    [string]$CacheMode = 'auto',
    [int]$TelemetryIntervalSeconds = 5,
    [switch]$ResumeCampaign
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$ExpectedBranch = 'scientific-recovery-v9-eclock-x0'
$RequiredStartingAncestor = 'af66f2c8ca2017059d7765b5f171e1cda866ab07'
$ParentV8 = '718e0bf7ca9950fbc0fc2a3537e4b0e0e25a72a2'
$Protocol = Join-Path $RepoRoot 'configs\protocol\scientific_recovery_v9_eclock_x0.json'
$Reference = Join-Path $RepoRoot 'configs\protocol\scientific_recovery_v9_eclock_x0_reference.json'
$ConfigRoot = Join-Path $RepoRoot 'configs\experiment\scientific_recovery_v9_eclock'
$PythonExecutable = Join-Path $ReferenceRoot '.venv\Scripts\python.exe'

$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONPATH = Join-Path $RepoRoot 'src'
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:OMP_NUM_THREADS = '16'
$env:MKL_NUM_THREADS = '16'
$env:OPENBLAS_NUM_THREADS = '16'
$env:NUMEXPR_MAX_THREADS = '16'
$env:TOKENIZERS_PARALLELISM = 'false'

function Invoke-GitText {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$GitArgs)
    $value = & git -C $RepoRoot @GitArgs 2>&1
    if ($LASTEXITCODE -ne 0) { throw "git failed: $($GitArgs -join ' ')" }
    return (($value | Out-String).Trim())
}

foreach ($path in @($RepoRoot, $ReferenceRoot, $CacheRoot, $PythonExecutable)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required path does not exist: $path" }
}
$Branch = Invoke-GitText branch --show-current
$Head = Invoke-GitText rev-parse HEAD
if ($Branch -ne $ExpectedBranch) { throw "Wrong branch: $Branch" }
if ($Head -ne $AuthorizedCommit) { throw "HEAD $Head differs from AuthorizedCommit $AuthorizedCommit" }
& git -C $RepoRoot merge-base --is-ancestor $RequiredStartingAncestor $Head
if ($LASTEXITCODE -ne 0) { throw 'Authorized commit is not descended from audited X0 HEAD' }
& git -C $RepoRoot merge-base --is-ancestor $ParentV8 $Head
if ($LASTEXITCODE -ne 0) { throw 'Authorized commit is not descended from V8 parent' }
if (Invoke-GitText status --porcelain=v1 --untracked-files=no) {
    throw 'Versioned worktree is not clean'
}

$ShortSha = $AuthorizedCommit.Substring(0, 12)
$OutputBaseResolved = if ([IO.Path]::IsPathRooted($OutputBase)) { $OutputBase } else { Join-Path $RepoRoot $OutputBase }
$CampaignRoot = Join-Path $OutputBaseResolved "x0-seed7-$ShortSha"
$LogsRoot = Join-Path $CampaignRoot 'master_logs'
$TelemetryRoot = Join-Path $CampaignRoot 'telemetry'
$StateRoot = Join-Path $CampaignRoot 'state'
if ((Test-Path -LiteralPath $CampaignRoot) -and -not $ResumeCampaign) {
    throw "Campaign root exists; explicit -ResumeCampaign is required: $CampaignRoot"
}
New-Item -ItemType Directory -Force -Path $LogsRoot, $TelemetryRoot, $StateRoot | Out-Null
$MasterLog = Join-Path $LogsRoot 'master.log'
$StateLog = Join-Path $StateRoot 'master_state.jsonl'
$StateSnapshot = Join-Path $StateRoot 'master_state.json'

function Write-State {
    param([string]$Stage, [string]$Status, [hashtable]$Extra = @{})
    $record = [ordered]@{
        utc = [DateTime]::UtcNow.ToString('o')
        stage = $Stage
        status = $Status
        branch = $Branch
        git_commit = $AuthorizedCommit
        campaign_root = $CampaignRoot
    }
    foreach ($key in $Extra.Keys) { $record[$key] = $Extra[$key] }
    ($record | ConvertTo-Json -Compress -Depth 10) | Add-Content -LiteralPath $StateLog -Encoding utf8
    $temporary = $StateSnapshot + '.tmp'
    $record | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $StateSnapshot -Force
}

function Invoke-LoggedPython {
    param([string]$Name, [string[]]$Arguments)
    $log = Join-Path $LogsRoot ($Name + '.log')
    "[$([DateTime]::UtcNow.ToString('o'))] START $Name" | Tee-Object -FilePath $MasterLog -Append
    Write-State -Stage $Name -Status 'running'
    & $PythonExecutable @Arguments 2>&1 | Tee-Object -FilePath $log -Append
    $code = $LASTEXITCODE
    "[$([DateTime]::UtcNow.ToString('o'))] END $Name exit=$code" | Tee-Object -FilePath $MasterLog -Append
    if ($code -ne 0) {
        Write-State -Stage $Name -Status 'fatal' -Extra @{ exit_code = $code; log = $log }
        throw "$Name failed with exit code $code"
    }
    Write-State -Stage $Name -Status 'complete' -Extra @{ exit_code = 0; log = $log }
}

function Get-ConfigPath {
    param([string]$Arm)
    $name = switch ($Arm) {
        'X0-A5-REPLAY' { 'x0_a5_replay.yaml' }
        'X0-PAIR-U' { 'x0_pair_u.yaml' }
        'X0-BASE-U' { 'x0_base_u.yaml' }
        'X0-DYN-U' { 'x0_dyn_u.yaml' }
        default { throw "Unknown arm: $Arm" }
    }
    return Join-Path $ConfigRoot $name
}

function Run-ArmOOF {
    param([string]$Arm, [string]$SelectedCacheMode)
    $arguments = @(
        (Join-Path $RepoRoot 'scripts\train_scientific_recovery_v9_eclock.py'),
        '--config', (Get-ConfigPath $Arm), '--mode', 'oof',
        '--cache-root', $CacheRoot, '--reference-root', $ReferenceRoot,
        '--output-root', (Join-Path $CampaignRoot $Arm), '--device', 'cuda',
        '--cache-mode', $SelectedCacheMode, '--resume-checkpoint-every', '100',
        '--milestone-updates', '250', '500', '1000', '2000', '4000', '6840',
        '--progress-log-every', '1', '--rich-log-every', '25', '--execute-authorized-oof'
    )
    if ($ResumeCampaign) { $arguments += '--resume-campaign' }
    Invoke-LoggedPython -Name ("oof-" + $Arm) -Arguments $arguments
}

function Aggregate-Arm {
    param([string]$Arm)
    $root = Join-Path $CampaignRoot $Arm
    $output = Join-Path $root 'aggregate.json'
    Invoke-LoggedPython -Name ("aggregate-" + $Arm) -Arguments @(
        (Join-Path $RepoRoot 'scripts\aggregate_scientific_recovery_v9_eclock.py'),
        '--config', (Get-ConfigPath $Arm), '--protocol', $Protocol, '--reference', $Reference,
        '--run-root', $root, '--reference-root', $ReferenceRoot, '--output', $output
    )
    Invoke-LoggedPython -Name ("verify-aggregate-" + $Arm) -Arguments @(
        (Join-Path $RepoRoot 'scripts\verify_scientific_recovery_v9_eclock.py'), $output
    )
}

function Start-GpuTelemetry {
    $command = Get-Command 'nvidia-smi.exe' -ErrorAction Stop
    return Start-Process -FilePath $command.Source -ArgumentList @(
        '--query-gpu=timestamp,index,name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.current.sm,clocks.current.memory,pstate',
        '--format=csv,noheader,nounits', "--loop=$TelemetryIntervalSeconds"
    ) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $TelemetryRoot 'gpu.csv') -RedirectStandardError (Join-Path $TelemetryRoot 'gpu.stderr.log')
}

function Start-HostTelemetry {
    return Start-Job -ScriptBlock {
        param($Destination, $Interval, $CampaignPid, $OutputPath)
        while ($true) {
            try {
                $os = Get-CimInstance Win32_OperatingSystem
                $cpu = Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average
                $proc = Get-Process -Id $CampaignPid -ErrorAction SilentlyContinue
                $drive = Get-PSDrive -Name ([IO.Path]::GetPathRoot($OutputPath).Substring(0,1))
                [ordered]@{
                    utc = [DateTime]::UtcNow.ToString('o'); cpu_load_percent = $cpu.Average
                    ram_total_bytes = [int64]$os.TotalVisibleMemorySize * 1024
                    ram_free_bytes = [int64]$os.FreePhysicalMemory * 1024
                    campaign_process_rss_bytes = if ($proc) { [int64]$proc.WorkingSet64 } else { $null }
                    output_drive_free_bytes = [int64]$drive.Free
                } | ConvertTo-Json -Compress | Add-Content -LiteralPath $Destination -Encoding utf8
            } catch {
                @{ utc=[DateTime]::UtcNow.ToString('o'); telemetry_error=$_.Exception.Message } | ConvertTo-Json -Compress | Add-Content -LiteralPath $Destination -Encoding utf8
            }
            Start-Sleep -Seconds $Interval
        }
    } -ArgumentList (Join-Path $TelemetryRoot 'host.jsonl'), $TelemetryIntervalSeconds, $PID, $CampaignRoot
}

function Stop-Telemetry {
    if ($script:GpuMonitor -and -not $script:GpuMonitor.HasExited) { Stop-Process -Id $script:GpuMonitor.Id -Force -ErrorAction SilentlyContinue }
    if ($script:HostMonitor) {
        Stop-Job $script:HostMonitor -ErrorAction SilentlyContinue
        Receive-Job $script:HostMonitor -ErrorAction SilentlyContinue | Out-Null
        Remove-Job $script:HostMonitor -Force -ErrorAction SilentlyContinue
    }
    $script:GpuMonitor = $null
    $script:HostMonitor = $null
}

$script:GpuMonitor = $null
$script:HostMonitor = $null
try {
    Write-State -Stage 'campaign' -Status 'started'
    $script:GpuMonitor = Start-GpuTelemetry
    $script:HostMonitor = Start-HostTelemetry

    $testPaths = @()
    $testPaths += Get-ChildItem -LiteralPath (Join-Path $RepoRoot 'tests\unit') -Filter 'test_collision_clock*.py' | ForEach-Object { $_.FullName }
    $testPaths += Join-Path $RepoRoot 'tests\unit\test_a5_local_transport.py'
    $testPaths += Join-Path $RepoRoot 'tests\integration\test_collision_clock_resume.py'
    $testPaths += Join-Path $RepoRoot 'tests\scientific\test_collision_clock_no_leakage.py'
    $testPaths += Join-Path $RepoRoot 'tests\regression\test_scientific_recovery_v8_immutable.py'
    Invoke-LoggedPython -Name 'qa-pytest-x0' -Arguments (@('-m', 'pytest', '-q') + $testPaths)

    $qualityFiles = @(
        'src/e_jepa_ttc/data/collision_clock_cache.py', 'src/e_jepa_ttc/training/collision_clock_eap.py',
        'src/e_jepa_ttc/evaluation/collision_clock_bootstrap.py', 'src/e_jepa_ttc/evaluation/collision_clock_gates.py',
        'src/e_jepa_ttc/evaluation/collision_clock_cross_arm.py', 'src/e_jepa_ttc/evaluation/collision_clock_runner.py',
        'scripts/train_scientific_recovery_v9_eclock.py', 'scripts/compare_scientific_recovery_v9_eclock_x0.py',
        'scripts/preflight_scientific_recovery_v9_eclock_x0.py', 'scripts/smoke_scientific_recovery_v9_eclock_x0.py',
        'scripts/report_scientific_recovery_v9_eclock_environment.py', 'scripts/analyze_scientific_recovery_v9_eclock_x0.py',
        'scripts/package_scientific_recovery_v9_eclock_x0_results.py',
        'tests/unit/test_collision_clock_cache.py', 'tests/unit/test_collision_clock_bootstrap.py',
        'tests/unit/test_collision_clock_gates.py', 'tests/integration/test_collision_clock_resume.py'
    ) | ForEach-Object { Join-Path $RepoRoot $_ }
    Invoke-LoggedPython -Name 'qa-ruff-check' -Arguments (@('-m', 'ruff', 'check') + $qualityFiles)
    Invoke-LoggedPython -Name 'qa-ruff-format' -Arguments (@('-m', 'ruff', 'format', '--check') + $qualityFiles)
    Invoke-LoggedPython -Name 'qa-pyright' -Arguments (@('-m', 'pyright', '--venvpath', $ReferenceRoot) + $qualityFiles)

    $parseErrors = $null
    $parseTokens = $null
    [System.Management.Automation.Language.Parser]::ParseFile($PSCommandPath, [ref]$parseTokens, [ref]$parseErrors) | Out-Null
    if ($parseErrors.Count -ne 0) { throw "PowerShell AST parse failed: $parseErrors" }
    Write-State -Stage 'qa-powershell-ast' -Status 'complete'
    & git -C $RepoRoot diff --check
    if ($LASTEXITCODE -ne 0) { throw 'git diff --check failed' }
    Write-State -Stage 'qa-git-diff-check' -Status 'complete'

    foreach ($arm in @('X0-A5-REPLAY','X0-BASE-U','X0-DYN-U','X0-PAIR-U')) {
        Invoke-LoggedPython -Name ("dry-run-" + $arm) -Arguments @(
            (Join-Path $RepoRoot 'scripts\train_scientific_recovery_v9_eclock.py'),
            '--config', (Get-ConfigPath $arm), '--mode', 'dry-run', '--cache-root', $CacheRoot,
            '--reference-root', $ReferenceRoot, '--output-root', (Join-Path $CampaignRoot ("dry-run-" + $arm))
        )
    }

    Invoke-LoggedPython -Name 'environment-report' -Arguments @(
        (Join-Path $RepoRoot 'scripts\report_scientific_recovery_v9_eclock_environment.py'),
        '--output', (Join-Path $CampaignRoot 'environment.json'), '--cache-root', $CacheRoot,
        '--reference-root', $ReferenceRoot, '--authorized-commit', $AuthorizedCommit
    )
    Invoke-LoggedPython -Name 'preflight' -Arguments @(
        (Join-Path $RepoRoot 'scripts\preflight_scientific_recovery_v9_eclock_x0.py'),
        '--repo-root', $RepoRoot, '--cache-root', $CacheRoot, '--reference-root', $ReferenceRoot,
        '--authorized-commit', $AuthorizedCommit, '--cache-mode', $CacheMode,
        '--output-root', (Join-Path $CampaignRoot 'preflight')
    )
    $decision = Get-Content -Raw -LiteralPath (Join-Path $CampaignRoot 'preflight\cache_engineering_decision.json') | ConvertFrom-Json
    $SelectedCacheMode = [string]$decision.selected_mode
    if ($SelectedCacheMode -notin @('fold_ram','shard_lru')) { throw "Invalid selected cache mode: $SelectedCacheMode" }

    Invoke-LoggedPython -Name 'real-smoke' -Arguments @(
        (Join-Path $RepoRoot 'scripts\smoke_scientific_recovery_v9_eclock_x0.py'),
        '--cache-root', $CacheRoot, '--reference-root', $ReferenceRoot,
        '--output-root', (Join-Path $CampaignRoot 'smoke'), '--device', 'cuda',
        '--cache-mode', $SelectedCacheMode, '--fold', '0', '--max-rows', '32',
        '--execute-authorized-outer-train-smoke'
    )

    Run-ArmOOF -Arm 'X0-A5-REPLAY' -SelectedCacheMode $SelectedCacheMode
    Aggregate-Arm 'X0-A5-REPLAY'
    Run-ArmOOF -Arm 'X0-BASE-U' -SelectedCacheMode $SelectedCacheMode
    Run-ArmOOF -Arm 'X0-DYN-U' -SelectedCacheMode $SelectedCacheMode
    Aggregate-Arm 'X0-BASE-U'
    Aggregate-Arm 'X0-DYN-U'
    $comparisonRoot = Join-Path $CampaignRoot 'comparisons'
    New-Item -ItemType Directory -Force -Path $comparisonRoot | Out-Null
    Invoke-LoggedPython -Name 'compare-DYN-vs-BASE' -Arguments @(
        (Join-Path $RepoRoot 'scripts\compare_scientific_recovery_v9_eclock_x0.py'),
        '--base-run-root', (Join-Path $CampaignRoot 'X0-BASE-U'),
        '--dyn-run-root', (Join-Path $CampaignRoot 'X0-DYN-U'), '--protocol', $Protocol,
        '--reference', $Reference, '--output', (Join-Path $comparisonRoot 'x0_dyn_vs_base.json'),
        '--gate-output', (Join-Path $comparisonRoot 'x0_dyn_vs_base_gate.json')
    )
    Run-ArmOOF -Arm 'X0-PAIR-U' -SelectedCacheMode $SelectedCacheMode
    Aggregate-Arm 'X0-PAIR-U'
    Stop-Telemetry
    Invoke-LoggedPython -Name 'final-analysis' -Arguments @(
        (Join-Path $RepoRoot 'scripts\analyze_scientific_recovery_v9_eclock_x0.py'),
        '--campaign-root', $CampaignRoot, '--reference-root', $ReferenceRoot,
        '--output', (Join-Path $CampaignRoot 'CODEX_X0_FINAL_CAMPAIGN_REPORT.md')
    )
    Invoke-LoggedPython -Name 'package-results' -Arguments @(
        (Join-Path $RepoRoot 'scripts\package_scientific_recovery_v9_eclock_x0_results.py'),
        '--campaign-root', $CampaignRoot, '--git-commit', $AuthorizedCommit, '--output-root', $CampaignRoot
    )
    Write-State -Stage 'campaign' -Status 'complete'
    Write-Host "CAMPAIGN COMPLETE: $CampaignRoot"
} catch {
    Stop-Telemetry
    Write-State -Stage 'campaign' -Status 'fatal' -Extra @{ message = $_.Exception.Message }
    $failure = [ordered]@{
        status = 'incomplete'; classification = 'integrity_or_runtime'; utc = [DateTime]::UtcNow.ToString('o')
        message = $_.Exception.Message; stack = $_.ScriptStackTrace; git_commit = $AuthorizedCommit
    }
    $failureRoot = Join-Path $CampaignRoot 'failure'
    New-Item -ItemType Directory -Force -Path $failureRoot | Out-Null
    $failure | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $failureRoot 'fatal.json') -Encoding utf8
    try {
        Invoke-LoggedPython -Name 'fatal-analysis' -Arguments @(
            (Join-Path $RepoRoot 'scripts\analyze_scientific_recovery_v9_eclock_x0.py'),
            '--campaign-root', $CampaignRoot, '--reference-root', $ReferenceRoot,
            '--output', (Join-Path $CampaignRoot 'CODEX_X0_FINAL_CAMPAIGN_REPORT.md')
        )
        Invoke-LoggedPython -Name 'fatal-package' -Arguments @(
            (Join-Path $RepoRoot 'scripts\package_scientific_recovery_v9_eclock_x0_results.py'),
            '--campaign-root', $CampaignRoot, '--git-commit', $AuthorizedCommit, '--output-root', $CampaignRoot
        )
    } catch {
        Write-State -Stage 'fatal-handoff' -Status 'fatal' -Extra @{ message = $_.Exception.Message }
    }
    throw
} finally {
    Stop-Telemetry
}
