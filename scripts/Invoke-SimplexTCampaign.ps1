<#
.SYNOPSIS
Runs a pinned SIMPLEX-T orchestration plan synchronously, retaining attempt logs.
.DESCRIPTION
Requires a resolved plan with run, resume and verify commands for every step.
This shell is not a resolved T0-T6 campaign plan or a scientific freeze.
Completed steps are skipped only after their verifier succeeds. Resource pauses
are retried; other errors stop execution. Regeneration requires distinct outputs
and a new pinned plan; existing caches and checkpoints are never reset here.
.PARAMETER Resume
Allows an existing RunRoot with the same plan hash and selects resume commands.
.PARAMETER ValidateOnly
Checks command and plan pins without launching workers or creating RunRoot.
.PARAMETER MaxResourceRetries
Zero retries without a numerical limit while this foreground process remains
alive. A positive limit returns exit 3 when reached; use -Resume to continue.
.NOTES
Workers must enforce RAM, VRAM, disk reservations and scientific prerequisites.
The shell does not grant resource leases or authorize protected data access.
Worker commands may declare attempt_report=true to append --report with a fresh
JSON path alongside their log. Do not also supply --report in arguments. This
changes only the receipt destination, never a scientific launch configuration.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Plan,
    [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$PlanSha256,
    [Parameter(Mandatory)][string]$PythonExecutable,
    [Parameter(Mandatory)][string]$RunRoot,
    [switch]$Resume,
    [switch]$ValidateOnly,
    [ValidateRange(1, 60)][int]$ResourceRetrySeconds = 30,
    [ValidateRange(0, 100000)][int]$MaxResourceRetries = 0
)

# This is the synchronous orchestration shell, not scientific authority. The
# campaign's resolved plan must call real admission, freeze and output verifiers.
# Verifier exits: 0 complete, 10 incomplete, 3 resource pause; others stop.
# Exit 3 from a worker means resource pause. Other errors are never auto-retried.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$work = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$python = (Resolve-Path -LiteralPath $PythonExecutable).Path
$planPath = (Resolve-Path -LiteralPath $Plan).Path
$destination = [IO.Path]::GetFullPath($RunRoot)
if (-not $destination.StartsWith($work + [IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase)) { throw 'RunRoot must be inside this companion worktree.' }
if ((Get-FileHash -LiteralPath $planPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $PlanSha256) {
    throw 'Plan SHA256 mismatch.'
}
$spec = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
if ($spec.schema -ne 'simplex_t_orchestration_plan_v1' -or @($spec.steps).Count -eq 0) {
    throw 'A resolved, nonempty orchestration plan is required.'
}
$ids = @{}
function Resolve-WorkerExitCode([int]$Code, [string]$ScriptPath, [string]$LogPath) {
    # The original immutable replay uses exit1 for these two safe boundaries.
    # Do not match arbitrary occurrences of RESOURCE_PAUSE inside a traceback.
    if ($Code -eq 1 -and [IO.Path]::GetFileName($ScriptPath) -eq 'build_simplex_t_context_features.py') {
        $lastLine = Get-Content -LiteralPath $LogPath -Tail 1
        if ($lastLine -ceq 'RuntimeError: RESOURCE_PAUSE at completed query boundary' -or
            $lastLine -ceq 'RuntimeError: RESOURCE_PAUSE before model load') { return 3 }
    }
    return $Code
}
function Assert-Command($command) {
    $scriptPath = [IO.Path]::GetFullPath((Join-Path $work $command.script))
    if (-not $scriptPath.StartsWith($PSScriptRoot + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase) -or [IO.Path]::GetExtension($scriptPath) -ne '.py') {
        throw 'Only pinned companion Python entry points are allowed.'
    }
    if ((Get-FileHash -LiteralPath $scriptPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $command.sha256) {
        throw "Entry point changed: $scriptPath"
    }
    foreach ($argument in @($command.arguments)) {
        if ($argument -isnot [string]) { throw 'Arguments must be explicit strings.' }
    }
    if ($command.PSObject.Properties.Name -contains 'attempt_report') {
        if ($command.attempt_report -isnot [bool] -or -not $command.attempt_report) {
            throw 'attempt_report, when declared, must be true.'
        }
        if (@($command.arguments | Where-Object { $_ -eq '--report' -or $_ -like '--report=*' }).Count -gt 0) {
            throw 'attempt_report cannot override an explicit report argument.'
        }
    }
    return $scriptPath
}
foreach ($step in $spec.steps) {
    if ($step.id -notmatch '^[a-zA-Z0-9_-]+$' -or $ids.ContainsKey($step.id)) { throw 'Invalid or duplicate step ID.' }
    $ids[$step.id] = $true
    foreach ($field in @('run', 'resume', 'verify')) { $null = Assert-Command $step.$field }
    if ($step.verify.PSObject.Properties.Name -contains 'attempt_report') {
        throw 'attempt_report is for workers, not completion verifiers.'
    }
}
if ($ValidateOnly) { Write-Output "Validated $(@($spec.steps).Count) command triples; no campaign executed."; exit 0 }
if ((Test-Path -LiteralPath $destination) -and -not $Resume) {
    throw 'Existing RunRoot requires -Resume; use a new plan and output root for regeneration.'
}
$null = New-Item -ItemType Directory -Path $destination -Force
$leasePath = Join-Path $destination 'ORCHESTRATOR.lock'
$lease = [IO.File]::Open($leasePath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
$previousPythonPath = $env:PYTHONPATH
$previousThreads = @{}
foreach ($name in @('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')) {
    $previousThreads[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
$previousLocation = Get-Location
$identityPath = Join-Path $destination 'PLAN_SHA256.txt'
try {
    if (Test-Path -LiteralPath $identityPath) {
        if ((Get-Content -LiteralPath $identityPath -Raw).Trim() -ne $PlanSha256) { throw 'Resume plan changed.' }
    }
    else { [IO.File]::WriteAllText($identityPath, $PlanSha256) }
    $env:PYTHONPATH = Join-Path $work 'src'
    foreach ($name in $previousThreads.Keys) { [Environment]::SetEnvironmentVariable($name, '4', 'Process') }
    Set-Location -LiteralPath $work
    function Invoke-Logged($command, [string]$logPath, [switch]$Worker) {
        if ((Get-FileHash -LiteralPath $planPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $PlanSha256) {
            throw 'Plan changed during execution.'
        }
        $scriptPath = Assert-Command $command
        $arguments = @('-B', $scriptPath) + @($command.arguments)
        if ($Worker -and $command.PSObject.Properties.Name -contains 'attempt_report') {
            $arguments += @('--report', "$logPath.report.json")
        }
        $invocation = @{
            schema = 'simplex_t_orchestrator_invocation_v1'
            plan_sha256 = $PlanSha256
            script_sha256 = $command.sha256
            python = $python
            arguments = $arguments
            worker = [bool]$Worker
        } | ConvertTo-Json -Depth 5
        [IO.File]::WriteAllText("$logPath.command.json", $invocation)
        Write-Host "Running $([IO.Path]::GetFileName($scriptPath)); log: $logPath"
        & $python @arguments *> $logPath
        $code = $LASTEXITCODE
        if ($Worker) { return (Resolve-WorkerExitCode $code $scriptPath $logPath) }
        return $code
    }
    foreach ($step in $spec.steps) {
        $attempt = 0
        $resourceRetries = 0
        while ($true) {
            $stamp = [Guid]::NewGuid().ToString('N')
            $prefix = Join-Path $destination "$($step.id)_$stamp"
            $verified = Invoke-Logged $step.verify "$prefix.verify.log"
            if ($verified -eq 0) { Write-Host "Verified: $($step.id)"; break }
            if ($verified -eq 3) {
                $resourceRetries += 1
                if ($MaxResourceRetries -gt 0 -and $resourceRetries -ge $MaxResourceRetries) { exit 3 }
                Write-Host "Verifier resource pause at $($step.id); retry in $ResourceRetrySeconds seconds."
                Start-Sleep -Seconds $ResourceRetrySeconds
                continue
            }
            if ($verified -ne 10) { throw "Verifier failed for $($step.id), exit $verified. See $prefix.verify.log" }
            $command = if ($Resume -or $attempt -gt 0) { $step.resume } else { $step.run }
            $code = Invoke-Logged $command "$prefix.run.log" -Worker
            $attempt += 1
            if ($code -eq 3) {
                $resourceRetries += 1
                if ($MaxResourceRetries -gt 0 -and $resourceRetries -ge $MaxResourceRetries) {
                    Write-Host "Resource pause retained at $($step.id); rerun with -Resume."
                    exit 3
                }
                Write-Host "Resource pause at $($step.id); retry in $ResourceRetrySeconds seconds."
                Start-Sleep -Seconds $ResourceRetrySeconds
            }
            elseif ($code -ne 0) {
                throw "Worker failed for $($step.id), exit $code. No automatic rescue. See $prefix.run.log"
            }
            # A bounded successful cache slice is not completion: verify again.
        }
    }
    Write-Output 'All declared steps verified. Scientific completion is established by the final delivery verifier, not this message.'
}
finally {
    $env:PYTHONPATH = $previousPythonPath
    foreach ($name in $previousThreads.Keys) {
        [Environment]::SetEnvironmentVariable($name, $previousThreads[$name], 'Process')
    }
    Set-Location -LiteralPath $previousLocation.Path
    $lease.Dispose()
    Remove-Item -LiteralPath $leasePath
}
