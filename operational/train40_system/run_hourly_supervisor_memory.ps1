param(
    [Parameter(Mandatory = $true)][string]$ConfigPath,
    [Parameter(Mandatory = $true)][string]$ConfigSha256
)
$ErrorActionPreference = 'Stop'
if ((Get-FileHash -LiteralPath $ConfigPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ConfigSha256) {
    throw 'Hourly H8 memory supervisor configuration changed'
}
$campaignSettings = Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
Set-Location -LiteralPath $campaignSettings.root
$env:PYTHONUTF8 = '1'
$campaignPython = $campaignSettings.commands.controller[0]
& $campaignPython '-m' 'operational.train40_system.hourly_supervisor_transport_memory' '--config' $ConfigPath '--config-sha256' $ConfigSha256
exit $LASTEXITCODE
