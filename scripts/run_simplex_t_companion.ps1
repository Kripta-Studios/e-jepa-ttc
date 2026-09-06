param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('audit', 'prepare', 'run', 'status', 'package')]
    [string]$Command,
    [Parameter(Mandatory = $true)]
    [string]$LocalPaths,
    [Parameter(Mandatory = $true)]
    [string]$PythonExecutable,
    [string]$OutputPath,
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$resolvedPython = (Resolve-Path -LiteralPath $PythonExecutable).Path
$resolvedPaths = (Resolve-Path -LiteralPath $LocalPaths).Path
$companionRoot = Split-Path -Parent $PSScriptRoot
$previousPythonPath = $env:PYTHONPATH
$previousOmp = $env:OMP_NUM_THREADS
$previousMkl = $env:MKL_NUM_THREADS
$previousBlas = $env:OPENBLAS_NUM_THREADS
$arguments = @('-B', "$PSScriptRoot\run_simplex_t_companion.py", $Command,
    '--local-paths', $resolvedPaths)
if ($OutputPath) { $arguments += @('--output', $OutputPath) }
if ($Resume) { $arguments += '--resume' }
try {
    $env:PYTHONPATH = "$companionRoot\src"
    $env:OMP_NUM_THREADS = '4'
    $env:MKL_NUM_THREADS = '4'
    $env:OPENBLAS_NUM_THREADS = '4'
    & $resolvedPython @arguments
    $resultCode = $LASTEXITCODE
}
finally {
    $env:PYTHONPATH = $previousPythonPath
    $env:OMP_NUM_THREADS = $previousOmp
    $env:MKL_NUM_THREADS = $previousMkl
    $env:OPENBLAS_NUM_THREADS = $previousBlas
}
exit $resultCode
