param([string]$Repo = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
$Run = Join-Path $Repo 'artifacts/joint_measurement_20261012/cost/garl_routes'
$Source = Join-Path $Run 'RESULT.json'
$Result = Get-Content -LiteralPath $Source -Raw | ConvertFrom-Json
if ($Result.status -ne 'COMPLETE' -or $Result.unique_queries -ne 64 -or $Result.route_observations -ne 256) {
    throw 'Complete registered Garl comparison required.'
}
$Hash = (Get-FileHash -LiteralPath $Source -Algorithm SHA256).Hash.ToLowerInvariant()
$Lines = [System.Collections.Generic.List[string]]::new()
$Lines.Add('## Resultados medidos el 16 de octubre de 2026')
$Lines.Add('')
$Lines.Add('64 consultas P, dos pasadas y dos rutas: 128 pares / 256 observaciones. Todos los pares conservan inputs y predicciones exactamente iguales, incluidos los TTC no finitos representados explícitamente. Modelo público completo, CPU FP32 y un hilo PyTorch; cero updates.')
$Lines.Add('')
$Lines.Add('| Ruta | Componente | Media ms | p50 ms | p95 ms | p99 ms |')
$Lines.Add('|---|---|---:|---:|---:|---:|')
foreach ($Route in @('UPSTREAM', 'LOCAL')) {
    foreach ($Field in @('preprocess_ms', 'model_emission_ms', 'total_ms')) {
        $Value = $Result.summary.$Route.$Field
        if ($Value.n -ne 128) { throw 'Incomplete timing population.' }
        $Lines.Add(('| {0} | {1} | {2:F3} | {3:F3} | {4:F3} | {5:F3} |' -f
            $Route, $Field, $Value.mean_ms, $Value.p50_ms, $Value.p95_ms, $Value.p99_ms))
    }
}
$Lines.Add('')
$Lines.Add('LOCAL mejora ligeramente la media total y empeora la mediana frente a UPSTREAM. Esta medición no establece una aceleración uniforme ni un ganador de precisión. El original ya estaba vectorizado. El estado de la caché de disco del sistema operativo no se controló; los dos órdenes se alternaron según el registro.')
$Lines.Add('')
$Lines.Add('| Sesión | Carga ms | Primer forward ms | Preparado p50 ms | Preparado p95 ms |')
$Lines.Add('|---|---:|---:|---:|---:|')
foreach ($File in Get-ChildItem -Path (Join-Path $Run 'sessions/*/PREPARED.json')) {
    $Value = Get-Content -LiteralPath $File.FullName -Raw | ConvertFrom-Json
    $Lines.Add(('| {0} | {1:F3} | {2:F3} | {3:F3} | {4:F3} |' -f
        $File.Directory.Name, $Value.load_ms, $Value.first_forward_ms,
        $Value.warm_summary.p50_ms, $Value.warm_summary.p95_ms))
}
$Lines.Add('')
$Lines.Add('La medición preparada usa un único tensor residente antes de las rutas raw; sus tiempos difieren de los forwards posteriores. Se conservan ambos ámbitos y su orden temporal, sin tratarlos como observaciones intercambiables ni atribuir una causa no medida a esa diferencia. Primer forward no significa disco frío. La memoria registrada es RSS y pico de vida del proceso; no un pico aislado por ruta.')
$Lines.Add('')
$Lines.Add('Las cifras CPU no son comparables directamente con los costes CUDA de V18 ni con las métricas del benchmark oficial. Esta comparación de implementación no accede a etiquetas para seleccionar consultas o modelos.')
$Lines.Add('')
$Lines.Add(('Fuente RESULT.json, SHA-256: `{0}`.' -f $Hash))
$Lines.Add('')
$Lines.Add('Regenerar: `pwsh -NoProfile -File scripts/Build-JointMeasurementGarlCostReport.ps1`. Los JSON por par y MEASUREMENTS.csv permanecen en cost/garl_routes y se incluyen en el bundle de resultados.')
$Lines.Add('')
$Report = Join-Path $Repo 'docs/joint_measurement_20261012/GARL_ROUTE_COST.md'
$Existing = Get-Content -LiteralPath $Report -Raw
$Existing = ($Existing -split '(?m)^## Resultados medidos el 16 de octubre de 2026', 2)[0].TrimEnd()
$Existing = $Existing.Replace('# Coste de las dos rutas Garl: registro previo', '# Coste medido de las dos rutas Garl')
$Existing = $Existing.Replace(
    "El medidor está implementado y registrado; **la medición real está pendiente**.`r`nB1 conserva prioridad y no se ha cargado otro modelo junto a su entrenamiento.",
    'La medición registrada está completa: 64 consultas y 128 pares. Se ejecutó después de B1, sin entrenamiento concurrente.')
$Existing = $Existing.Replace(
    "El medidor está implementado y registrado; **la medición real está pendiente**.`nB1 conserva prioridad y no se ha cargado otro modelo junto a su entrenamiento.",
    'La medición registrada está completa: 64 consultas y 128 pares. Se ejecutó después de B1, sin entrenamiento concurrente.')
[System.IO.File]::WriteAllText($Report, $Existing + "`n`n" + ($Lines -join "`n"), [System.Text.UTF8Encoding]::new($false))
$Evidence = Join-Path $Repo 'docs/joint_measurement_20261012/evidence_garl_route_cost'
New-Item -ItemType Directory -Path $Evidence -Force | Out-Null
Copy-Item -LiteralPath $Source -Destination (Join-Path $Evidence 'RESULT.json')
Write-Output "Garl cost report regenerated from $Hash"
