[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$SourceRepo,
    [Parameter(Mandatory=$true)][string]$Destination,
    [switch]$AllowFetch
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Base = '396abe922537d9dac19d6db3451e59d283636e20'
$Branch = 'scientific-recovery-v18-joint-measurement'
$PackageRoot = Split-Path -Parent $PSScriptRoot
$Manifest = Get-Content -Raw -LiteralPath (Join-Path $PackageRoot 'MANIFEST.json') | ConvertFrom-Json
foreach ($Entry in $Manifest.files) {
    $File = Join-Path $PackageRoot $Entry.path
    if (-not (Test-Path -LiteralPath $File -PathType Leaf)) { throw "Falta $($Entry.path)" }
    if ((Get-Item -LiteralPath $File).Length -ne $Entry.bytes) { throw "Tamaño cambiado: $($Entry.path)" }
    $Hash = (Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Hash -ne $Entry.sha256) { throw "Hash cambiado: $($Entry.path)" }
}
if (Test-Path -LiteralPath $Destination) { throw 'El destino existe. Usa el estado de resume; no sobrescribas.' }
$Top = & git -C $SourceRepo rev-parse --show-toplevel
if ($LASTEXITCODE -ne 0) { throw 'SourceRepo no es un checkout Git.' }
& git -C $SourceRepo cat-file -e "$Base^{commit}" 2>$null
if ($LASTEXITCODE -ne 0) {
    if (-not $AllowFetch) { throw 'Falta el commit; vuelve a ejecutar con -AllowFetch para recuperar la rama.' }
    & git -C $SourceRepo fetch origin scientific-recovery-v17-representation-audit
    if ($LASTEXITCODE -ne 0) { throw 'No se pudo recuperar la rama.' }
    & git -C $SourceRepo cat-file -e "$Base^{commit}"
    if ($LASTEXITCODE -ne 0) { throw 'El commit fijado sigue ausente; no uses otro HEAD silenciosamente.' }
}
& git -C $SourceRepo show-ref --verify --quiet "refs/heads/$Branch"
if ($LASTEXITCODE -eq 0) { throw 'La rama V18 ya existe. Localiza su worktree antes de crear otro escritor.' }
& git -C $SourceRepo worktree add -b $Branch $Destination $Base
if ($LASTEXITCODE -ne 0) { throw 'git worktree add falló; no hagas reset ni clean.' }
$Docs = Join-Path $Destination 'docs/joint_measurement_handoff_20261012'
New-Item -ItemType Directory -Path $Docs -Force | Out-Null
Get-ChildItem -LiteralPath $PackageRoot -Force | Copy-Item -Destination $Docs -Recurse
Write-Host "Worktree preparado: $Destination"
Write-Host 'No se han aplicado patches científicos, entrenado modelos, hecho commits ni push.'
Write-Host "Arranque: codex --cd `"$Destination`""
