# Comandos existentes y comandos que Codex debe implementar

## Verificar este handoff

Desde su carpeta extraída, con Python del entorno existente:

```powershell
python -B scripts/verify_package.py --root .
python -B -m unittest discover -s tests -v
python -B scripts/inspect_source_zip.py
```

## Crear un worktree nuevo

Desde PowerShell 7 en la carpeta V12 del proyecto (fuente sólo lectura):

```powershell
$ErrorActionPreference = 'Stop'
$Handoff = Join-Path $HOME 'Downloads/E_JEPA_TTC_RGB_PORT_20261008'
$SourceRepo = (& git rev-parse --show-toplevel)
if ($LASTEXITCODE -ne 0) { throw 'No estás dentro del repo existente.' }
$SourceRepo = $SourceRepo.Trim()
$Worktree = Join-Path (Split-Path -Parent $SourceRepo) 'e-jepa-ttc-v13-rgb-port'
pwsh -NoProfile -File "$Handoff/scripts/Prepare-RGBPort.ps1" -SourceRepo $SourceRepo -Destination $Worktree -AllowFetch
if ($LASTEXITCODE -ne 0) { throw 'Preparación fallida; no usar reset/clean.' }
codex --cd "$Worktree"
```

Si la rama/directorio ya existe, el preparador se detiene: inspeccionar y reanudar su trabajo en vez de eliminarlo. No crear dos sesiones escritoras de los mismos artefactos. Leer COORDINATE_R1_AND_V13.md antes de solapar tareas.

## Reproducir esta revisión numérica, opcional

Estos scripts ya existen. Es suficiente verificar los recibos suministrados salvo que sea necesaria una nueva comprobación. No repetirlos en bucles de entrenamiento.

```powershell
python -B scripts/inspect_source_zip.py --extract ../sota_source_verified
python -B scripts/audit_predictions.py --source-root ../sota_source_verified --output ../sota_descriptive_new
python -B scripts/replay_published_scores.py --source-root ../sota_source_verified --output ../sota_replay_new
```

Los directorios de salida deben ser nuevos y externos a la evidencia original. Los últimos dos requieren NumPy. El replay ejecuta el scorer publicado fijado por SHA, no inferencia ni optimización. Su comparación distingue bytes de valores y no borra diferencias de saltos de línea.

## Ejecución de la campaña

Los siguientes módulos NO están integrados por este ZIP. Codex debe implementarlos sobre el repo, probarlos y publicar sus comandos efectivos:

```powershell
& $Python -m operational.rgb_port.run preflight --config configs/rgb_port/execution.json
& $Python -m operational.rgb_port.run execute --config configs/rgb_port/execution.json
& $Python -m operational.rgb_port.run resume --run artifacts/rgb_port_20261008
& $Python -m operational.rgb_port.run package --run artifacts/rgb_port_20261008
```

$Python debe apuntar al entorno local validado del proyecto. Este documento no instala PyTorch, no actualiza drivers y no crea un entorno alternativo incompatible por defecto.
