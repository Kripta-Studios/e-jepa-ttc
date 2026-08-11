param(
    [Parameter(Mandatory=$true)][string]$OutputCsv,
    [Parameter(Mandatory=$true)][string]$StopFile,
    [int]$IntervalSeconds = 5
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"
if ($IntervalSeconds -lt 1) { throw "IntervalSeconds must be >=1" }
$parent = Split-Path -Parent $OutputCsv
if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
"timestamp,cpu_pct,ram_used_gb,ram_available_gb,gpu_util_pct,vram_used_mb,vram_total_mb,gpu_temp_c,gpu_power_w" | Set-Content -LiteralPath $OutputCsv -Encoding utf8
while (-not (Test-Path -LiteralPath $StopFile)) {
    $ts = (Get-Date).ToString("o")
    $cpu = ""
    $ramUsed = ""
    $ramAvail = ""
    try {
        $os = Get-CimInstance Win32_OperatingSystem
        $total = [double]$os.TotalVisibleMemorySize * 1KB
        $free = [double]$os.FreePhysicalMemory * 1KB
        $ramUsed = [math]::Round(($total-$free)/1GB,3)
        $ramAvail = [math]::Round($free/1GB,3)
        $cpu = [math]::Round((Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average,1)
    } catch {}
    $gpuUtil=""; $vramUsed=""; $vramTotal=""; $temp=""; $power=""
    try {
        $gpu = (& nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
        if ($gpu) {
            $parts = $gpu -split ',' | ForEach-Object { $_.Trim() }
            if ($parts.Count -ge 5) { $gpuUtil=$parts[0]; $vramUsed=$parts[1]; $vramTotal=$parts[2]; $temp=$parts[3]; $power=$parts[4] }
        }
    } catch {}
    "$ts,$cpu,$ramUsed,$ramAvail,$gpuUtil,$vramUsed,$vramTotal,$temp,$power" | Add-Content -LiteralPath $OutputCsv -Encoding utf8
    Start-Sleep -Seconds $IntervalSeconds
}
