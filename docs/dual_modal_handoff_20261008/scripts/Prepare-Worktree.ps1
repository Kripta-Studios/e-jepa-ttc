[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$SourceRepo,
    [Parameter(Mandatory=$true)][string]$Destination,
    [switch]$AllowFetch
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Base = 'b46d40bf1f38ae3e3ea201d58b79391fa475ca3c'
$Branch = 'scientific-recovery-v13-dual-modal-ttc'
$Package = Split-Path -Parent $PSScriptRoot
$Manifest = Join-Path $Package 'SHA256SUMS.txt'
if (!(Test-Path -LiteralPath $Manifest -PathType Leaf)) { throw 'Falta SHA256SUMS.txt' }
$Seen = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
foreach ($Line in [IO.File]::ReadAllLines($Manifest)) {
    if ($Line -notmatch '^([0-9a-f]{64})  (.+)$') { throw 'Manifiesto no valido' }
    $Expected = $Matches[1]; $Relative = $Matches[2]
    if ([IO.Path]::IsPathRooted($Relative) -or $Relative.Contains('..') -or $Relative.Contains('\') -or !($Seen.Add($Relative))) { throw 'Ruta insegura o repetida en manifiesto' }
    $File = Join-Path $Package $Relative
    if (!(Test-Path -LiteralPath $File -PathType Leaf)) { throw "Falta payload: $Relative" }
    $Actual = (Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Actual -ne $Expected) { throw "SHA256 distinto: $Relative" }
}
if (!(Get-Command git -ErrorAction SilentlyContinue)) { throw 'git no disponible' }
$Source = (Resolve-Path -LiteralPath $SourceRepo).Path
$Top = & git -C $Source rev-parse --show-toplevel
if ($LASTEXITCODE -ne 0) { throw 'SourceRepo no es un repositorio git' }
$Source = $Top.Trim()
& git -C $Source cat-file -e "$Base^{commit}" 2>$null
if ($LASTEXITCODE -ne 0) {
    if (!$AllowFetch) { throw 'El commit base no esta local. Usa -AllowFetch para recuperar la rama, sin merge.' }
    & git -C $Source fetch --no-tags origin scientific-recovery-v12-efficient-context
    if ($LASTEXITCODE -ne 0) { throw 'Fallo fetch. No reset/pull sustitutivo.' }
    & git -C $Source cat-file -e "$Base^{commit}"
    if ($LASTEXITCODE -ne 0) { throw 'El commit base sigue ausente' }
}
if (Test-Path -LiteralPath $Destination) { throw 'Destino existente: no sobrescribir. Usa prompt de resume en el worktree existente.' }
& git -C $Source show-ref --verify --quiet "refs/heads/$Branch"
if ($LASTEXITCODE -eq 0) { throw 'La rama V13 ya existe. Localizala con git worktree list y reanuda; no la recrees.' }
& git -C $Source worktree add -b $Branch $Destination $Base
if ($LASTEXITCODE -ne 0) { throw 'No se pudo crear worktree' }
$Dest = (Resolve-Path -LiteralPath $Destination).Path
$Handoff = Join-Path $Dest 'docs/dual_modal_handoff_20261008'
if (Test-Path -LiteralPath $Handoff) { throw 'Destino handoff ya existe' }
New-Item -ItemType Directory -Path $Handoff | Out-Null
Get-ChildItem -LiteralPath $Package -Force | Copy-Item -Destination $Handoff -Recurse
$Binding = [ordered]@{source_repo=$Source; worktree=$Dest; source_role='read_only_historical_artifacts'; base_commit=$Base; branch=$Branch; handoff=$Handoff; note='Do not reuse historical output roots; resolve current processes and data through manifests.'}
$Binding | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Dest 'LOCAL_V13_BINDING.json') -Encoding utf8
Write-Output "Worktree listo: $Dest"
Write-Output "Abrir: codex --cd `"$Dest`""
Write-Output 'Pegar PROMPT_CODEX_START.md. No se ha ejecutado entrenamiento, reset, clean, merge o push.'
