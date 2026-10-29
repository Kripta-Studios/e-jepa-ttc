param(
    [Parameter(Mandatory = $true)][string]$Repository,
    [Parameter(Mandatory = $true)][string]$PythonExecutable,
    [Parameter(Mandatory = $true)][string]$Module,
    [Parameter(Mandatory = $true)][string]$Run,
    [Parameter(Mandatory = $true)][string]$StageId,
    [Parameter(Mandatory = $true)][string]$ResultPath,
    [Parameter(Mandatory = $true)][string]$IntentPath,
    [Parameter(Mandatory = $true)][string]$IntentNonce,
    [Parameter(Mandatory = $true)][string]$StageLockPath,
    [int]$Rotation = -1
)

$ErrorActionPreference = 'Stop'
if ($PythonExecutable.Contains('"') -or $Run.Contains('"') -or $Repository.Contains('"')) {
    throw 'Quoted executable/run paths are not supported.'
}
$repositoryPath = [System.IO.Path]::GetFullPath($Repository)
$env:PYTHONUTF8 = '1'
$existingPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = ([System.IO.Path]::Combine($repositoryPath, 'src') +
    [System.IO.Path]::PathSeparator + $repositoryPath)
if ($existingPythonPath) {
    $env:PYTHONPATH += [System.IO.Path]::PathSeparator + $existingPythonPath
}
$lockTarget = [System.IO.Path]::GetFullPath($StageLockPath)
[System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($lockTarget)) | Out-Null
$lockStream = [System.IO.FileStream]::new(
    $lockTarget,
    [System.IO.FileMode]::CreateNew,
    [System.IO.FileAccess]::ReadWrite,
    [System.IO.FileShare]::None,
    4096,
    [System.IO.FileOptions]::DeleteOnClose
)
try {
$launcherProcess = Get-Process -Id $PID
$launcherCreateTime = ([double]$launcherProcess.StartTime.ToUniversalTime().ToFileTimeUtc() /
    10000000.0) - 11644473600.0
$handshake = [ordered]@{
    schema = 'final_refit_scientific_oof_launch_intent_v1'
    stage_id = $StageId
    intent_nonce = $IntentNonce
    launcher_pid = [int]$PID
    launcher_create_time = $launcherCreateTime
    repository = $repositoryPath
    run = [System.IO.Path]::GetFullPath($Run)
}
$intentTarget = [System.IO.Path]::GetFullPath($IntentPath)
[System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($intentTarget)) | Out-Null
$intentTemporary = $intentTarget + '.' + [Guid]::NewGuid().ToString('N') + '.tmp'
[System.IO.File]::WriteAllText(
    $intentTemporary,
    (($handshake | ConvertTo-Json -Depth 4 -Compress) + [Environment]::NewLine),
    [System.Text.UTF8Encoding]::new($false)
)
Move-Item -LiteralPath $intentTemporary -Destination $intentTarget
$arguments = @('-m', $Module, '--run', $Run)
if ($Rotation -ge 0) {
    $arguments += @('--rotation', $Rotation.ToString())
}
$started = [DateTimeOffset]::UtcNow.ToString('o')
$target = [System.IO.Path]::GetFullPath($ResultPath)
$directory = [System.IO.Path]::GetDirectoryName($target)
[System.IO.Directory]::CreateDirectory($directory) | Out-Null
$stdout = $target + '.stdout.log'
$stderr = $target + '.stderr.log'
Push-Location -LiteralPath $repositoryPath
$previousErrorActionPreference = $ErrorActionPreference
try {
    # Windows PowerShell 5.1 converts native stderr into ErrorRecord objects.
    # Continue is required here so a nonzero scientific child can still be
    # recorded atomically in RESULT.json.
    $ErrorActionPreference = 'Continue'
    & $PythonExecutable @arguments 1> $stdout 2> $stderr
    $finalExitCode = [int]$LASTEXITCODE
}
finally {
    $ErrorActionPreference = $previousErrorActionPreference
    Pop-Location
}
$result = [ordered]@{
    schema = 'final_refit_scientific_oof_launcher_result_v1'
    stage_id = $StageId
    intent_nonce = $IntentNonce
    exit_code = $finalExitCode
    child_pid = $null
    launcher_pid = [int]$PID
    started_utc = $started
    finished_utc = [DateTimeOffset]::UtcNow.ToString('o')
}
$temporary = $target + '.' + [Guid]::NewGuid().ToString('N') + '.tmp'
[System.IO.File]::WriteAllText(
    $temporary,
    (($result | ConvertTo-Json -Depth 4 -Compress) + [Environment]::NewLine),
    [System.Text.UTF8Encoding]::new($false)
)
Move-Item -LiteralPath $temporary -Destination $target -Force
}
finally {
    $lockStream.Dispose()
    if ([System.IO.File]::Exists($lockTarget)) {
        [System.IO.File]::Delete($lockTarget)
    }
}
exit $finalExitCode
