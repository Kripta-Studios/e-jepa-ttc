param([switch]$Loop)
$ErrorActionPreference = 'Stop'
$repoPath = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$runPath = Join-Path $repoPath 'artifacts/decision_recovery_20261017'
$monitorPath = Join-Path $runPath 'hourly'
[void](New-Item -ItemType Directory -Path $monitorPath -Force)

function Read-Record([string]$Relative) {
    $path = Join-Path $runPath $Relative
    if (Test-Path -LiteralPath $path) { return Get-Content -LiteralPath $path -Raw | ConvertFrom-Json }
    return $null
}
function Save-Record([string]$Path, $Value) {
    $temporary = "$Path.$PID.tmp"
    $Value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}
function Observe {
    $now = [DateTimeOffset]::Now
    $pipeline = Read-Record 'PIPELINE_STATE.json'
    $previous = Read-Record 'hourly/LATEST.json'
    $process = if ($pipeline) { Get-Process -Id $pipeline.pid -ErrorAction SilentlyContinue } else { $null }
    $alive = $null -ne $process
    $fits = @()
    $remaining = 0
    $rate = $null
    foreach ($arm in @('L0','LW')) {
        $folder = "training/fits/FULL-$arm-S7"
        $journal = Read-Record "$folder/UPDATE_JOURNAL.json"
        $runtime = Read-Record "$folder/PERSISTENT_RUNTIME.json"
        $completed = if ($journal) { [int]$journal.completed_updates } else { 0 }
        $remaining += [Math]::Max(0, 12000-$completed)
        $seconds = if ($runtime -and $runtime.recent_updates.Count) {
            ($runtime.recent_updates.total_s | Measure-Object -Average).Average
        } else { $null }
        if ($seconds -gt 0 -and $completed -lt 12000) { $rate = $seconds }
        $fits += [ordered]@{
            arm=$arm; completed=$completed; endpoint=12000
            durable=if($journal){$journal.durable_updates}else{0}
            pending=if($journal){$journal.pending_update_upper}else{0}
            recovery=if($journal){$journal.recovery_upper}else{0}
            seconds_per_update=$seconds
            checkpoint_sha256=(Read-Record "$folder/CHECKPOINT_RECEIPT.json").checkpoint_sha256
        }
    }
    $total = ($fits.completed | Measure-Object -Sum).Sum
    $alerts = @()
    $terminal = $pipeline.status -in @('INITIAL_SCREEN_COMPLETE','STOPPED_ON_ERROR')
    if (!$alive -and !$terminal) { $alerts += 'Proceso ausente sin cierre del pipeline' }
    if ($pipeline.status -eq 'STOPPED_ON_ERROR') { $alerts += $pipeline.error }
    if ($previous -and $alive -and $pipeline.status -eq 'SCIENTIFIC_L0_LW' -and
        $previous.total_updates -eq $total) { $alerts += 'Sin nuevos updates desde la observacion anterior' }
    if (($fits.recovery | Measure-Object -Sum).Sum -gt 0) { $alerts += 'Hay updates de recuperacion contabilizados' }
    $eta = if ($alive -and $rate -gt 0 -and $remaining -gt 0) {
        $now.AddSeconds($remaining*$rate).ToString('o')
    } else { $null }
    $evaluation = @()
    foreach ($arm in @('L0','LW')) {
        foreach ($role in @('P','H')) {
            $folder = Join-Path $runPath "evaluation/predictions/$role/$arm"
            $chunks = @(Get-ChildItem -LiteralPath $folder -Filter 'chunk_*.json' -ErrorAction SilentlyContinue)
            $rows = 0
            foreach ($chunk in $chunks) {
                $receipt = Get-Content -LiteralPath $chunk.FullName -Raw | ConvertFrom-Json
                $rows += $receipt.stop - $receipt.start
            }
            $evaluation += @{arm=$arm;role=$role;completed_rows=$rows}
        }
    }
    $report = Read-Record 'evaluation/REPORT.json'
    $continuation = Read-Record 'CONTINUATION_STATE.json'
    $value = [ordered]@{
        observed_local=$now.ToString('o'); observed_utc=$now.UtcDateTime.ToString('o')
        timezone='Europe/Madrid'; pipeline_status=$pipeline.status
        pipeline_pid=$pipeline.pid; process_alive=$alive
        process_started_utc=if($alive){$process.StartTime.ToUniversalTime().ToString('o')}else{$null}
        total_updates=$total; training_updates_remaining=$remaining; fits=$fits
        training_eta_local=$eta; eta_excludes_evaluation=$true
        evaluation=$evaluation; alerts=$alerts; terminal=$terminal
        replication_decision=$report.replication_decision
        continuation=$continuation
        resource_guard=(Read-Record 'training/RESOURCE_LATEST.json')
        writes_training=$false; automatic_chat_notification=$false
    }
    $stamp=$now.ToString('yyyyMMdd_HHmmss_zzz').Replace(':','')
    Save-Record (Join-Path $monitorPath "$stamp.json") $value
    Save-Record (Join-Path $monitorPath 'LATEST.json') $value
    $lines=@("# Revision V19: $($now.ToString('yyyy-MM-dd HH:mm:ss zzz'))",'',
        "Estado: $($pipeline.status). Proceso vivo: $alive.",
        "Updates: $total / 24000. ETA de entrenamiento: $eta.",
        'La ETA no incluye evaluacion ni exports.','','| Brazo | Updates | Checkpoint durable | s/update |',
        '|---|---:|---:|---:|')
    foreach($fit in $fits){$lines += "| $($fit.arm) | $($fit.completed) | $($fit.durable) | $($fit.seconds_per_update) |"}
    $lines += @('',"Alertas: $(if($alerts.Count){$alerts -join '; '}else{'ninguna'}).",'',
        'Observador local de solo lectura; no reinicia ni modifica entrenamiento.')
    if ($continuation) { $lines += @('',"Continuacion: $($continuation.status). $($continuation.description)") }
    $lines | Set-Content -LiteralPath (Join-Path $monitorPath 'LATEST.md') -Encoding utf8
    return $value
}

if (!$Loop) { Observe | ConvertTo-Json -Depth 12; exit }
Save-Record (Join-Path $monitorPath 'OBSERVER.json') @{
    pid=$PID; started=[DateTimeOffset]::Now.ToString('o'); schedule='Cada hora local en punto'
    script=$PSCommandPath; timezone='Europe/Madrid'; stops_on_pipeline_terminal=$false
}
while ($true) {
    $now=Get-Date
    $next=$now.Date.AddHours($now.Hour+1)
    while ((Get-Date) -lt $next) {
        Start-Sleep -Milliseconds ([int][Math]::Min(30000,[Math]::Max(50,($next-(Get-Date)).TotalMilliseconds)))
    }
    try {
        $result=Observe
    } catch {
        Save-Record (Join-Path $monitorPath 'MONITOR_ERROR.json') @{
            utc=[DateTimeOffset]::UtcNow.ToString('o'); error=$_.Exception.Message
        }
    }
}
