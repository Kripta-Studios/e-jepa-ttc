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
# Exit 10 from a verifier means incomplete; every other failure stops the graph.
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
    return $scriptPath
}
foreach ($step in $spec.steps) {
    if ($step.id -notmatch '^[a-zA-Z0-9_-]+$' -or $ids.ContainsKey($step.id)) { throw 'Invalid or duplicate step ID.' }
    $ids[$step.id] = $true
    foreach ($field in @('run', 'resume', 'verify')) { $null = Assert-Command $step.$field }
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
    function Invoke-Logged($command, [string]$logPath) {
        if ((Get-FileHash -LiteralPath $planPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $PlanSha256) {
            throw 'Plan changed during execution.'
        }
        $scriptPath = Assert-Command $command
        $arguments = @('-B', $scriptPath) + @($command.arguments)
        Write-Host "Running $([IO.Path]::GetFileName($scriptPath)); log: $logPath"
        & $python @arguments *> $logPath
        return $LASTEXITCODE
    }
    foreach ($step in $spec.steps) {
        $attempt = 0
        $resourceRetries = 0
        while ($true) {
            $stamp = [Guid]::NewGuid().ToString('N')
            $prefix = Join-Path $destination "$($step.id)_$stamp"
            $verified = Invoke-Logged $step.verify "$prefix.verify.log"
            if ($verified -eq 0) { Write-Host "Verified: $($step.id)"; break }
            if ($verified -ne 10) { throw "Verifier failed for $($step.id), exit $verified. See $prefix.verify.log" }
            $command = if ($Resume -or $attempt -gt 0) { $step.resume } else { $step.run }
            $code = Invoke-Logged $command "$prefix.run.log"
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
