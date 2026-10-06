param(
    [Parameter(Mandatory = $true)][string]$ConfigPath,
    [Parameter(Mandatory = $true)][string]$ConfigSha256
)
$ErrorActionPreference = 'Stop'
if ((Get-FileHash -LiteralPath $ConfigPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ConfigSha256) {
    throw 'Hourly supervisor configuration changed'
}
$campaignSettings = Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
$campaignTaskName = 'E-JEPA TRAIN40 hourly 20261007'
$campaignRunner = Join-Path $PSScriptRoot 'run_hourly_supervisor.ps1'
$campaignPowerShell = Join-Path $env:SystemRoot 'System32/WindowsPowerShell/v1.0/powershell.exe'
$campaignArguments = '-NoProfile -WindowStyle Hidden -File "' + $campaignRunner + '" -ConfigPath "' + $ConfigPath + '" -ConfigSha256 ' + $ConfigSha256
$campaignNow = Get-Date
$campaignFirstCheck = $campaignNow.Date.AddHours($campaignNow.Hour + 1)
$campaignDeadline = if ($campaignSettings.deadline_utc -is [DateTime]) {
    $campaignSettings.deadline_utc.ToLocalTime()
} else {
    [DateTimeOffset]::Parse($campaignSettings.deadline_utc, [Globalization.CultureInfo]::InvariantCulture).LocalDateTime
}
$campaignDuration = $campaignDeadline - $campaignFirstCheck
if ($campaignDuration.TotalSeconds -le 0) { throw 'Campaign deadline already reached' }
$campaignTrigger = New-ScheduledTaskTrigger -Once -At $campaignFirstCheck -RepetitionInterval (New-TimeSpan -Hours 1) -RepetitionDuration $campaignDuration
$campaignTrigger.EndBoundary = $campaignDeadline.ToString('s')
$campaignAction = New-ScheduledTaskAction -Execute $campaignPowerShell -Argument $campaignArguments -WorkingDirectory $campaignSettings.root
$campaignSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$campaignPrincipal = New-ScheduledTaskPrincipal -UserId $campaignSid -LogonType Interactive -RunLevel Limited
$campaignTaskSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Seconds 0)
$campaignExisting = Get-ScheduledTask -TaskName $campaignTaskName -ErrorAction SilentlyContinue
if ($campaignExisting) {
    $campaignOwnedPrefix = $campaignArguments.Substring(0, $campaignArguments.LastIndexOf(' -ConfigSha256 ') + 15)
    if (-not $campaignExisting.Actions.Arguments.StartsWith($campaignOwnedPrefix)) {
        throw 'Existing task has different arguments; preserve it'
    }
    Set-ScheduledTask -TaskName $campaignTaskName -Action $campaignAction -Trigger $campaignTrigger -Settings $campaignTaskSettings -Principal $campaignPrincipal | Out-Null
} else {
    Register-ScheduledTask -TaskName $campaignTaskName -Action $campaignAction -Trigger $campaignTrigger -Settings $campaignTaskSettings -Principal $campaignPrincipal -Description 'Hourly TRAIN40 supervisor recovery; admitted queue only; no direct GPU trainer launches.' | Out-Null
}
$campaignTask = Get-ScheduledTask -TaskName $campaignTaskName
$campaignDirectory = Join-Path $campaignSettings.output 'hourly_supervisor'
Export-ScheduledTask -TaskName $campaignTaskName | Set-Content -LiteralPath (Join-Path $campaignDirectory 'WINDOWS_TASK.xml') -Encoding Unicode
[pscustomobject]@{
    status = 'REGISTERED'
    task_name = $campaignTaskName
    config_sha256 = $ConfigSha256
    first_check_local = $campaignFirstCheck.ToString('o')
    deadline_local = $campaignDeadline.ToString('o')
    interval = 'PT1H'
    principal_logon_type = 'Interactive'
    multiple_instances = 'IgnoreNew'
    state = [string]$campaignTask.State
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $campaignDirectory 'INSTALL_RECEIPT.json') -Encoding UTF8
Start-ScheduledTask -TaskName $campaignTaskName
Get-Content -LiteralPath (Join-Path $campaignDirectory 'INSTALL_RECEIPT.json') -Raw -Encoding UTF8
