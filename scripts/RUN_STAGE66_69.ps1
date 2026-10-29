param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$HandoffRoot,
    [Parameter(Mandatory=$true)][string]$LocalInputs,
    [Parameter(Mandatory=$true)][string]$OutputRoot,
    [ValidateSet('preflight','full','analyze','package')][string]$Mode='full',
    [switch]$Resume
)
$ErrorActionPreference='Stop'
if (!(Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Python executable unavailable' }
if (!(Test-Path -LiteralPath $LocalInputs -PathType Leaf)) { throw 'LOCAL_INPUTS unavailable' }
$runnerArgs=@((Join-Path $PSScriptRoot 'run_scientific_recovery_v9_stage66_69.py'),'--handoff-root',$HandoffRoot,'--local-inputs',$LocalInputs,'--output-root',$OutputRoot,'--mode',$Mode)
if ($Resume) { $runnerArgs += '--resume' }
$logPath=Join-Path $OutputRoot ('foreground_'+[DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss_ffff')+'.log')
Start-Transcript -LiteralPath $logPath | Out-Null
try {
    & $Python @runnerArgs
    $result=$LASTEXITCODE
} finally {
    Stop-Transcript | Out-Null
}
Write-Output "STAGE66_69_FOREGROUND_EXIT=$result"
exit $result
