param(
    [string]$MasterRoot = ".\artifacts\scientific_recovery_master_v2",
    [int]$IntervalSeconds = 2
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "SilentlyContinue"
$root = (Resolve-Path $MasterRoot).Path
$start = (Get-Item $root).CreationTime
$ramTotalGB = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB

while ($true) {
    Clear-Host
    $statusFile = Join-Path $root "master_status.json"
    $status = if (Test-Path $statusFile) {
        try { Get-Content $statusFile -Raw | ConvertFrom-Json } catch { $null }
    } else { $null }

    $completed = @()
    if ($status -and $status.steps) { $completed = @($status.steps | ForEach-Object { $_.name }) }

    $logRoot = Join-Path $root "logs"
    $logDirs = Get-ChildItem $logRoot -Directory -ErrorAction SilentlyContinue | Sort-Object CreationTime
    $active = $logDirs | Where-Object { $_.Name -notin $completed } | Select-Object -Last 1
    if ($active) {
        $stage = $active.Name
        $stageElapsed = (Get-Date) - $active.CreationTime
    } elseif ($completed.Count -gt 0) {
        $stage = "Entre etapas / último: $($completed[-1])"
        $stageElapsed = $null
    } else {
        $stage = "Inicializando"
        $stageElapsed = $null
    }

    $cpu = [math]::Round(((Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average),1)
    $os = Get-CimInstance Win32_OperatingSystem
    $ramFreeGB = $os.FreePhysicalMemory * 1KB / 1GB
    $ramUsedGB = $ramTotalGB - $ramFreeGB
    $ramPct = 100 * $ramUsedGB / $ramTotalGB

    $gpuRaw = nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>$null
    $g = @($gpuRaw -split ',' | ForEach-Object { $_.Trim() })
    $gpuPct=0.0;$vramUsed=0.0;$vramTotal=0.0;$gpuTemp="?";$gpuPower="?"
    if($g.Count -ge 5){
        [double]::TryParse($g[0],[ref]$gpuPct)|Out-Null
        [double]::TryParse($g[1],[ref]$vramUsed)|Out-Null
        [double]::TryParse($g[2],[ref]$vramTotal)|Out-Null
        $gpuTemp=$g[3];$gpuPower=$g[4]
    }
    $vramPct = if($vramTotal -gt 0){100*$vramUsed/$vramTotal}else{0}
    $elapsed=(Get-Date)-$start

    Write-Host "============================================================"
    Write-Host " E-JEPA SCIENTIFIC RECOVERY MASTER V2"
    Write-Host "============================================================"
    Write-Host (" Hora:          {0:HH:mm:ss}" -f (Get-Date))
    Write-Host (" Tiempo total:  {0:hh\:mm\:ss}" -f $elapsed)
    Write-Host ""
    Write-Host " ETAPA ACTUAL:"
    Write-Host " $stage" -ForegroundColor Cyan
    if($stageElapsed){Write-Host (" Tiempo etapa:  {0:hh\:mm\:ss}" -f $stageElapsed)}
    Write-Host ""
    Write-Host (" CPU:   {0,5:N1}%   | RAM:  {1:N1}/{2:N1} GB ({3:N1}%)" -f $cpu,$ramUsedGB,$ramTotalGB,$ramPct)
    Write-Host (" GPU:   {0,5:N1}%   | VRAM: {1:N0}/{2:N0} MiB ({3:N1}%)" -f $gpuPct,$vramUsed,$vramTotal,$vramPct)
    Write-Host (" TEMP:  {0} C     | POWER: {1} W" -f $gpuTemp,$gpuPower)
    if($status){
        Write-Host ""
        Write-Host (" Pasos terminados: {0}" -f @($status.steps).Count)
        Write-Host (" Legacy winner:     {0}" -f $status.legacy_winner)
        Write-Host (" Causal winner:     {0}" -f $status.causal_winner)
        Write-Host (" Transport blocked: {0}" -f $status.transport_blocked)
        Write-Host (" SOTA blocked:      {0}" -f $status.sota_comparison_blocked)
    }
    Write-Host ""
    Write-Host " Ctrl+C cierra SOLO este monitor."
    Start-Sleep -Seconds $IntervalSeconds
}
