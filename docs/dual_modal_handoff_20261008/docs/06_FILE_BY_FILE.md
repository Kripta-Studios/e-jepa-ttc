# 6. Integración concreta (rutas nuevas propuestas)

No son ficheros que ya existan. El agente debe crearlos e integrarlos, probando los
componentes heredados antes de usarlos. No editar la identidad de source snapshots.

| Nuevo archivo | Responsabilidad y dependencia real |
|---|---|
| `src/e_jepa_ttc/dual_ttc/contracts.py` | Tipos de inputs/modalidad, roles/grupos, anchors/disponibilidad. |
| `src/e_jepa_ttc/dual_ttc/inputs.py` | Par nativo+context4; reutilizar preprocessingGarl y lectoresbounded. |
| `src/e_jepa_ttc/dual_ttc/rgb_sync.py` | Extraer selección causal de inputs.py de la transferencia sin sus ROOThardcoded. |
| `src/e_jepa_ttc/dual_ttc/cache.py` | Shards e inputs, manifest separado, no genericCSV como cache de embeddings mutables. |
| `src/e_jepa_ttc/dual_ttc/encoder.py` | AdaptadorR50/FPN; pesosgenéricoslocales pinned, BNstatseval. |
| `src/e_jepa_ttc/dual_ttc/correlation.py` | Similaridades locales/masks retenidos; puede adaptar local_transport sin alterar viejo. |
| `src/e_jepa_ttc/dual_ttc/model.py` | Direct/corr/eventE, ramaRGB, fallbackE, salida fase. |
| `src/e_jepa_ttc/dual_ttc/bridge.py` | H8frozen+RGBpair, helperhidden con parity. |
| `src/e_jepa_ttc/dual_ttc/loss.py` | Masas y punto/cuántiles; KDTRAIN y RGBaux. Reutilizar phase.py real. |
| `src/e_jepa_ttc/dual_ttc/checkpoint.py` | Reutilizar DurableState/atomicIO con rootsV13 y estadointegral. |
| `src/e_jepa_ttc/dual_ttc/train.py` | Dataset-role aware, progressdurable, endpointsfijos, encodergradients reales. |
| `src/e_jepa_ttc/dual_ttc/infer.py` | infer_event y infer_rgb_event separados; sin targets, nocallsRGB enE. |
| `src/e_jepa_ttc/dual_ttc/evaluate.py` | Scoringnativo/soportecomún,grupos,CIs,colas,fallos. |
| `operational/dual_ttc/queue.py` | DAG acotado, no workreassignment de presupuestos. |
| `operational/dual_ttc/prepare.py` | Inventario,roles,cachés,QA,freeze; R1actualno duplicado. |
| `operational/dual_ttc/compare_garl.py` | Modelosnativospúblicos y calibrador diagnosticotransparente; noE3restart. |
| `operational/dual_ttc/export.py` | Torchweights,modelcard,predicciones,submissionLOCALschema,hashes. |
| `tests/test_dual_ttc_*.py` | Sintéticos+paridadrealTRAIN+resume; pruebasreference no sustituyen integración. |

## Comandos que debe implementar el agente

Los siguientes son **interfaces a implementar**, no comandos ya funcionales del repo:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
& <python_local> -m operational.dual_ttc.prepare --config configs/dual_ttc/campaign.json --output artifacts/dual_ttc_20261008
& <python_local> -m operational.dual_ttc.queue run --config configs/dual_ttc/campaign.json --output artifacts/dual_ttc_20261008
& <python_local> -m operational.dual_ttc.queue status --output artifacts/dual_ttc_20261008
& <python_local> -m operational.dual_ttc.queue resume --output artifacts/dual_ttc_20261008
& <python_local> -m operational.dual_ttc.export --output artifacts/dual_ttc_20261008 --verify
```

El usuario NO debe lanzarlos antes de que Codex los integre y pruebe `--help` y una
invocación real limitada. El agente entrega al final `RUN_COMMANDS.ps1` con rutas
reales descubiertas y comandos ejecutados, no placeholders. Los scripts existentes
de preparación del worktree y auditor offline del paquete sí son ejecutables ahora.

## Entrega obligatoria

`CODEX_DUAL_TTC_FINAL_REPORT.md`, `NEXT_DECISION_DUAL_TTC.json`, `DATA_ROLES.json`,
`MODEL_REGISTRY.json`, `INPUT_CONTRACTS.json`, `SCIENTIFIC_PROTOCOL.json`, freezes,
`ACCOUNTING.json`, curvas completas y endpoints/sha, comparadores nativos intactos,
resultados por query/grupo/modo, sidecarnumericos, logs y tests, `SOTA_CLAIM_MATRIX.md`,
`RUN_COMMANDS.ps1`, modelcardsE/ER/KD si existe, bundleZIP y .sha256.

Publicar negativos y tareas bloqueadas con dependencia real. El bundle debe permitir
al menos regenerar tablas y replay desde inputs incluidos; si no incluye datos/pesos
externos, inventariarlos y no prometer reproducción raw sólo con el ZIP.
