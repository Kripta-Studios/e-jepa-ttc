# V17: reproducción y continuidad

Raíz de código y evidencia:

```text
C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\EVOCON_JEPA_Codex_Handoff\e-jepa-ttc-v17-representation-audit
```

La campaña está en `artifacts/representation_audit_20261012`. Usa una rama y un worktree independientes. No modifica entrenamientos ni admisiones de V13/R1/V16. No se han realizado pushes, merges, submissions ni cambios globales del equipo.

## Reanudar en este equipo

Desde PowerShell:

```powershell
pwsh -NoProfile -File 'C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\EVOCON_JEPA_Codex_Handoff\e-jepa-ttc-v17-representation-audit\scripts\Resume-ResearchV17.ps1'
```

`-Queue A`, `B`, `D`, `E`, `F`, `G`, `H` o `I` limita la reanudación a una cola. I exige el rechazo de G y H conforme a su protocolo independiente. El script rechaza un worker vivo y no crea un supervisor permanente. Cada trainer conserva checkpoint, optimizador, scheduler, RNG CPU/CUDA, sampler y diario de updates. Los checkpoints completos se verifican y se reutilizan. Una dependencia ausente o un gate fallido no autoriza un experimento sustituto no registrado.

Los contadores globales están en `ACCOUNTING.json` y `GPU_BUDGET.json`. El presupuesto es de un millón de updates físicos y 64.800 segundos dentro de scopes GPU; ambos son límites, no una obligación de consumo. Las reservas pendientes se cargan conservadoramente. Los ajustes cerrados se contabilizan aparte, sin ocultarlos como ausencia de entrenamiento.

## Dependencias externas al bundle de resultados

- V13: padres finales matched A5/C2F/PAIR, manifests P/H/V y sus metadatos; historiales PHASE17 H/V y normalizador H. Se valida su identidad antes de entrenar.
- V15: admisión y cachés vectoriales P/H/V con receipts por shard. P contiene targets separados; la inferencia rechaza claves `target_` y `dino_`.
- V14: métricas del control histórico M0/L0 para contrastes registrados.
- Entorno Python/PyTorch, dependencias del repositorio y GPU compatibles. La GPU original es una RTX5070Ti Laptop; la semilla no garantiza identidad bit a bit en otro hardware o versión de librerías.

Los datos crudos y las cachés completas no se publican como parte de los pesos. El bundle debe declarar expresamente sus exclusiones y hashes de dependencias; no se presenta como un paquete autónomo de reproducción de productores desde datos crudos.

## Pesos de inferencia

Cada `exports/<fit>/WEIGHTS.pt` contiene pesos, configuración y la identidad del operador. Los readouts geométricos incluyen coeficientes y normalización de altura como buffers. El loader no necesita el checkpoint padre ni el JSON original de calibración:

```python
from pathlib import Path
from operational.representation_audit.export_weights import load_export

model = load_export(Path('artifacts/representation_audit_20261012/exports/Q1-S7/WEIGHTS.pt'))
```

El modelo recibe los vectores originales de A5 preparados por la ruta admitida. Las cabezas reciben PHASE17, tiempos, máscaras, expertos y la entrada adicional normalizada según `MODEL_METADATA.json`. No se debe cargar un residual equal-slope como `C*tanh(s)` ni omitir la calibración de altura. Un checkpoint nativo que ignore esos operadores no es equivalente.

`RECEIPT.json` documenta la recarga exacta en CPU; `checkpoint_last.pt` conserva además el estado de entrenamiento para reanudar. Las pruebas técnicas de continuidad y los exports no se presentan como nuevos resultados científicos.

## Regenerar informes

En el worktree, con el entorno compartido y `PYTHONPATH=src;.`:

```powershell
& ../e-jepa-ttc/.venv/Scripts/python.exe scripts/report_v17_queues.py
& ../e-jepa-ttc/.venv/Scripts/python.exe scripts/plot_v17.py
& ../e-jepa-ttc/.venv/Scripts/python.exe -m operational.representation_audit.export_weights
```

El primer comando reproduce las métricas desde CSV, conserva resultados por grupo y calcula intervalos mediante remuestreo de grupos. Las figuras distinguen población, semillas y scope. Los resultados negativos no se eliminan al regenerar.
