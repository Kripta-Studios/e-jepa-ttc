# 08 — Reproducción, archivos y entrega al agente externo

## Primera verificación, sin GPU ni datasets

Desde la raíz V13, Python 3.11 o posterior y biblioteca estándar:

```powershell
python -B docs/agent_handoff_20261012/replay.py --output E:/EJEPA_results/handoff_replay_20261012
```

El script verifica SHA256 de fuentes, recalcula MAE, RTE, sesgo, cobertura y
MiD por banda de los CSV FCWD/Dev32, los contrasta con DIAGNOSTICS y genera
`04_RESULTADOS_GENERADOS.md` y `REPLAY_RESULT.json`. No modifica fuentes ni
entrena. Las tablas V y de costes se regeneran de sus JSON; reproducir su
medición original exige datos/modelos/hardware. El replay V12 independiente:

```powershell
python -B docs/agent_handoff_20261010/replay_public.py --output E:/EJEPA_results/v12_handoff_replay
```

Ejecutar este segundo comando desde el árbol **V12**, no desde V13.

El manifiesto distingue los bytes originales de los bytes Git cuando difieren
solo por finales CRLF/LF. Se comprobó esa equivalencia y se registraron ambos
SHA256 para `REAL_INPUT_HISTORY_QA.json`. El replay acepta exclusivamente esas
identidades enumeradas; no normaliza o ignora cambios arbitrarios.

## Mapa de evidencia pública V13

| Pregunta | Archivos |
|---|---|
| ¿Terminaron los fits? | `docs/agent_handoff_20261012/evidence/*ENDPOINT_FREEZE.json`, `HEADS_V_ROUND_FREEZE.json`, `TASK_STATUS.json` |
| ¿Qué mejoró en V? | `evidence/V_RESULTS.json`, tabla 04 y gates/CI contenidos en ese JSON |
| ¿Qué predijo cada consulta FCWD? | `docs/fcwd_rgb_registration_20261012/SCORED_PREDICTIONS.csv` |
| ¿Qué predijo cada consulta Dev32? | `docs/dev32_transfer_20261012/SCORED_PREDICTIONS.csv` |
| ¿Dónde falla y con qué cobertura? | `DIAGNOSTICS.json` de cada cohorte; `SUMMARY_COMPARISONS.json` FCWD |
| ¿Se alteraron pesos durante la reparación? | `docs/fcwd_rgb_registration_20261012/FINAL_QA.json`, recibos de hashes/updates |
| ¿Qué transformación RGB se usó? | `REGISTRATION_SEALED.json`, `PUBLISHED_INTRINSICS.json`, `REAL_INPUT_HISTORY_QA.json` |
| ¿Qué tests pasaron? | Recibos QA de cada intervención; no sumar suites solapadas |
| ¿Cuánto cuesta cada ruta? | `docs/external_mask_20261012/COST_SUMMARY.json` y `evidence/ROUTE_COSTS.json` |
| ¿Cómo se calculó MiD? | `operational/external_transfer_report.py`, `operational/rgb_port/evaluate.py`, replay de esta carpeta |

## Código para empezar a investigar

```text
src/e_jepa_ttc/models/causal_scale_ttc.py   productores y router
src/e_jepa_ttc/rgb_port/                   datos, cabezas, fusión
operational/rgb_port/prepare.py           masa y preparación de roles
operational/rgb_port/transfer.py          inferencia externa congelada
operational/rgb_port_revision/            pérdidas T2/T3, caché, intervalos, historia
operational/rgb_port_rgb_loader/          DINO, collate y guardas
operational/rgb_port_rgb_overlap/         decode/prefetch y ensayo geométrico
operational/rgb_port_signature_fix/       reparación H_R_RAW
operational/rgb_port_fusion_fix/          mediana CUDA determinista
operational/rgb_port_external_fix/        contrato de máscara nativa
operational/fcwd_rgb_registration/        transformación RGB y campaña
operational/fcwd_rgb_score_fix.py         lector GT faltante FCWD
operational/dev32_missing_truth_score.py  scoring Dev32 con GT faltante
operational/external_missing_truth_queue.py recuperación de cola final
operational/external_transfer_report.py   validación de sellado y reporte
tests/test_fcwd_rgb_registration.py
tests/test_fcwd_missing_truth.py
tests/test_rgb_port_external_fix.py
```

Para tests con el entorno del proyecto: `PYTHONPATH=src;.` en Windows o
`PYTHONPATH=src:.` en Linux. Evitar imports accidentales de otros worktrees.
La calibración utilizó mat-io 1.0.1 y opencv-python-headless 5.0.0.93 en
`E:/EJEPA_results/fcwd_rgb_calibration_20261012/deps`; esos paquetes no están
incluidos en el ZIP. Consultar pyproject/lock y los recibos antes de instalarlos.

## Para reproducir inferencia o entrenamiento

Se necesitan los datasets con sus licencias, índices/splits, cachés, targets
DINO si aplica, normalizadores y todos los checkpoints padre enumerados por
hash. Un CSV y un hash prueban la identidad del artefacto referido; no contienen
los pesos. La entrega pública permite auditar código y estadísticas, pero no
garantiza una reanudación completa en una máquina que carece de esos activos.

El run canónico está en `V13/artifacts/rgb_port_20261008/` localmente, donde
V13 significa la raíz real del repositorio, no una carpeta interna del checkout.
La campaña FCWD está en `E:/EJEPA_results/fcwd_rgb_calibration_20261012/campaign`.
Las raíces absolutas de los recibos describen la ejecución histórica; no deben
hardcodearse para un entorno nuevo.

`python -m operational.external_missing_truth_queue resume --run artifacts/rgb_port_20261008`
es el wrapper final de continuidad. No hace falta ejecutarlo para analizar
esta entrega: los fits y evaluaciones descritos ya terminaron y el contrato
FCWD oficial sigue bloqueado por una razón distinta. No borrar freezes,
checkpoints o journals para forzar tareas a completarse.

## Construcción y comprobación del ZIP

```powershell
python -B docs/agent_handoff_20261012/build_bundle.py --v12 ../e-jepa-ttc-v12-efficient-context --output E:/EJEPA_results/agent_handoff_20261012/EJEPA_V12_V13_HANDOFF_20261012.zip
python -B docs/agent_handoff_20261012/build_bundle.py --verify E:/EJEPA_results/agent_handoff_20261012/EJEPA_V12_V13_HANDOFF_20261012.zip
```

El constructor lee los objetos Git de los commits indicados por HEAD, no mezcla
ediciones locales sin commit. Incluye archivos textuales versionados de código,
tests, configs, scripts y docs en `V12/` y `V13/`. Excluye datos crudos, pesos,
artefactos binarios, `.git`, entornos y binding privado. Los documentos que
referencian imágenes/ZIP/pesos excluidos conservan sus rutas; no se afirma que
todos esos activos estén dentro del bundle.

`BUNDLE_MANIFEST.json` registra commits y SHA256 de cada entrada. El verificador
exige coincidencia del inventario, ausencia de entradas duplicadas, CRC válido
y hashes exactos. Se genera además un recibo y SHA256 externo del ZIP. El
replay debe funcionar también sobre una extracción limpia del bundle. El ZIP
se entrega en E: y no se añade como binario grande al historial Git.
