# 14 — Contratos y comandos para implementar el plan

## Convención de este documento

**Existente** significa que la ruta y sus opciones se comprobaron en el commit revisado. **Propuesto** significa que debe implementarse: copiar su comando ahora no ejecutará esa función. Los ejemplos no acreditan haber corrido una prueba ni autorizado una nueva evaluación oficial.

Ejecutar desde la raíz del repositorio, con rutas de datos recibidas por variables/argumentos. No introducir rutas personales en código o manifests públicos. Antes de trabajar, conservar el checkout original y crear una rama de investigación; no modificar paquetes sellados ni certificados históricos.

## 1. Mapa de responsabilidades

| Ruta | Estado | Uso |
|---|---|---|
| `operational/evttc_competition/readers.py` | Existente | Lectura, clocks y bbox |
| `operational/evttc_competition/geometry.py` | Existente | Calibración y proyección declaradas |
| `operational/evttc_competition/prepare.py` | Existente | Preparación del baseline |
| `operational/evttc_competition/runtime.py` | Existente | Ejecución del candidato congelado |
| `operational/final_refit/training.py` | Existente | Arquitectura `PhaseOnlyFinal` |
| `scripts/run_evttc_competition.py` | Existente | Run, preparación y empaquetado según política original |
| `scripts/verify_evttc_competition_run.py` | Existente | Auditoría de integridad/cobertura |
| `scripts/verify_garl_evttc_competition_run.py` | Existente | Verificación del comparador congelado |
| `operational/evttc_transfer_audit/identity.py` | Propuesto | Recortes, clocks e igualdad RGB |
| `operational/evttc_transfer_audit/protocol.py` | Propuesto | Exclusiones, grupos y roles |
| `operational/evttc_transfer_audit/targets.py` | Propuesto | Semántica y alineamiento GT |
| `operational/evttc_transfer_audit/scoring.py` | Propuesto | Métricas y población explícita |
| `operational/evttc_transfer_audit/diagnostics.py` | Propuesto | Trazas y fixtures físicos |
| `scripts/audit_evttc_transfer.py` | Propuesto | CLI única de las auditorías |
| `scripts/train_evttc_adaptation.py` | Propuesto | Piloto F1/F2/Fcap o P0 validado |
| `scripts/benchmark_evttc_runtime.py` | Propuesto | R00 con frontera de tiempo explícita |

Las rutas de auditoría prolongan el mapa del [plan original](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/docs/superpowers/plans/2026-10-30-evttc-transfer-diagnosis.md). Mantener módulos pequeños y reutilizar lectores; no crear un segundo parser HDF5 para toda la campaña.

## 2. Comandos existentes comprobados

### Bash

Variables requeridas: `EVTTC_COMPETITION_ROOT`, `EVTTC_CANDIDATE_ROOT` y `EVTTC_RUN_ROOT`. Cada una debe apuntar a un recurso real; el verificador no reconstruye archivos que falten.

```bash
python scripts/verify_evttc_competition_run.py \
  --run "$EVTTC_RUN_ROOT" \
  --data-root "$EVTTC_COMPETITION_ROOT" \
  --output output/evttc_review/independent_audit.json
```

Una preparación nueva, sin forward del modelo, se solicita explícitamente así:

```bash
python scripts/run_evttc_competition.py \
  --data-root "$EVTTC_COMPETITION_ROOT" \
  --candidate "$EVTTC_CANDIDATE_ROOT" \
  --output output/evttc_review/prepare_only \
  --device cpu \
  --prepare-only \
  --limit 2
```

El segundo comando crea un artefacto parcial y no una evaluación de precisión. `--prepare-only` no puede resumirse como inferencia. Para una ejecución nueva sin ese modo, usar otro directorio. `--resume` solo conserva una ejecución compatible y sus hashes; no solventa un ledger de error eliminando la fila. [Runner](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/scripts/run_evttc_competition.py), [verificador](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/scripts/verify_evttc_competition_run.py).

### PowerShell

PowerShell usa `$env:NOMBRE`, no `$NOMBRE` de Bash para variables de entorno. Para evitar errores con continuaciones de línea, esta forma utiliza una sola línea:

```powershell
python scripts/verify_evttc_competition_run.py --run "$env:EVTTC_RUN_ROOT" --data-root "$env:EVTTC_COMPETITION_ROOT" --output output/evttc_review/independent_audit.json
```

No copiar `$env:` dentro de un comando Bash. Los paths con espacios van entre comillas en ambos shells.

### Regresión focalizada existente

```bash
python -m pytest tests/unit/test_evttc_competition_readers.py tests/unit/test_evttc_competition_geometry.py tests/unit/test_evttc_competition_prepare.py tests/unit/test_evttc_competition_runtime.py tests/unit/test_evttc_competition_submission.py tests/unit/test_evttc_competition_audit.py -q
```

La prueba de paridad CUDA sellada necesita `EVTTC_CANDIDATE_ROOT`; reportar skips y recursos faltantes. No afirmar que pasar tests de fixtures valida los datos reales. [Tests de runtime](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/tests/unit/test_evttc_competition_runtime.py).

## 3. API propuesta de auditoría

| Interfaz propuesta | Resultado y condición |
|---|---|
| `match_recording(competition_hdf5, source_hdf5) -> IdentityResult` | Offset entero µs, índices, hashes y estado `unique/ambiguous/no_match` |
| `audit_lineage(candidate, inventory, exclusions) -> LineageReport` | Ancestros por componente y exposición conocida/desconocida |
| `validate_protocol(policy, inventory, identities, lineage) -> ProtocolReport` | Violaciones, relaciones sin resolver y roles sellables |
| `align_targets(queries, labels, clock_map, target_contract) -> TargetRows` | Valor, validez, razón y procedencia por consulta |
| `score_rows(predictions, targets, policy) -> ScoreReport` | Población, cobertura, macro/micro y alcance |
| `trace_query(sensor, query, calibration, model) -> TraceRecord` | Contratos, productores, contexto y residual, sin GT |

No hacer que una lista vacía de infracciones equivalga a independencia certificada si quedan relaciones desconocidas. `target_contract` debe describir instante, geometría, signo, unidades, interpolación y ceros; el offset no suple esos campos.

## 4. CLI propuesta y argumentos completos

Todos los siguientes subcomandos pertenecen al **script todavía no implementado** `scripts/audit_evttc_transfer.py`.

| Subcomando | Argumentos obligatorios |
|---|---|
| `identity` | `--manifest`, `--data-root`, `--competition-root`, `--reference-evidence`, `--output` |
| `lineage` | `--candidate`, `--component-inventory`, `--exclusions`, `--output` |
| `protocol` | `--inventory`, `--identities`, `--lineage`, `--policy`, `--output` |
| `targets` | `--queries`, `--labels-manifest`, `--clock-map`, `--target-contract`, `--output` |
| `score` | `--predictions`, `--targets`, `--policy`, `--output` |
| `diagnose` | `--data-root`, `--candidate`, `--query-manifest`, `--protocol`, `--diagnostic-config`, `--output`, `--device` |

Ejemplo propuesto de D00 en Bash:

```bash
python scripts/audit_evttc_transfer.py identity \
  --manifest data/manifests/evttc_all32_local.yaml \
  --data-root "$EVTTC_DATA_ROOT" \
  --competition-root "$EVTTC_COMPETITION_ROOT" \
  --reference-evidence docs/superpowers/plans/evidence/2026-10-30-evttc-audit.json \
  --output output/evttc_transfer_v2/D00_identity
```

Equivalente propuesto en PowerShell:

```powershell
python scripts/audit_evttc_transfer.py identity --manifest data/manifests/evttc_all32_local.yaml --data-root "$env:EVTTC_DATA_ROOT" --competition-root "$env:EVTTC_COMPETITION_ROOT" --reference-evidence docs/superpowers/plans/evidence/2026-10-30-evttc-audit.json --output output/evttc_transfer_v2/D00_identity
```

Ningún subcomando de auditoría entrena, descarga datos o envía resultados implícitamente. `protocol` valida una asignación conforme a la política; no vuelve a sortear grupos hasta encontrar una puntuación favorable.

## 5. Esquemas propuestos

Ejemplo de registro de población; los marcadores deben resolverse, no aceptarse como valores reales:

```yaml
schema: evttc_transfer_population_v1
protocol_id: evttc_transfer_v2
source_commit: 30b28b5c14eb9ed60067ed2f1dc262d7b60f2654
records:
  - sequence_id: <id_real>
    recording_group: <grupo_verificado>
    acquisition_group: <sesion_verificada_o_unknown>
    role: <fit_selection_holdout_exposed_excluded>
    identity_evidence_sha256: <hash_real>
    lineage_status: <verified_exposed_unknown>
    review_status: <approved_unresolved>
```

Un rol sin resolución de independencia no puede recibir `approved` por defecto. Distinguir grupo de grabación y grupo de adquisición evita tratar ambas cámaras o recortes como ejemplos independientes.

Campos mínimos del reporte de scoring:

```yaml
schema: evttc_transfer_score_v1
scope: <diagnostic_partial_development_equivalent_official>
protocol_sha256: <hash_real>
population_sha256: <hash_real>
prediction_sha256: <hash_real>
target_contract_sha256: <hash_real>
counts:
  requested: <int>
  finite_prediction: <int>
  compatible_target: <int>
  scored: <int>
missing_reasons: <conteos_por_razon>
metrics:
  per_group: <tabla>
  macro: <valores>
  micro: <valores>
official_equivalence: <verified_unverified>
```

El código debe impedir que un alcance parcial se etiquete como oficial. Una ejecución correcta puede producir un diagnóstico parcial con razones explícitas; completar ese comando no acredita equivalencia.

## 6. Entrenamiento y runtime propuestos

```bash
python scripts/train_evttc_adaptation.py \
  --config configs/experiment/evttc_f1_pilot_v1.yaml \
  --protocol configs/protocol/evttc_transfer_v2.yaml \
  --features output/evttc_transfer_v2/features/manifest.json \
  --output output/evttc_transfer_v2/F1_seed7 \
  --seed 7 \
  --device cuda
```

Este script/configuración también son **propuestos**. Validar roles, genealogía, hashes, presupuesto y disponibilidad antes de cargar GPU. Rechazar features con targets de holdout en entrenamiento y cambios de configuración al resumir. El recibo guarda el criterio exacto de selección y qué datos fueron consultados.

Para R00 se propone `scripts/benchmark_evttc_runtime.py` con argumentos obligatorios `--candidate`, `--query-manifest`, `--protocol`, `--runtime-config`, `--output` y `--device`. La configuración declara fronteras temporales, warm-up técnico, repeticiones, orden, tolerancias y cachés. No heredar silenciosamente esos valores del entrenamiento.

## 7. Estados de salida y aceptación

Propuesta: código 0 significa tarea completada dentro de su alcance declarado; 1, violación de contrato; 2, argumentos inválidos; 3, prerrequisito o recurso ausente. El JSON conserva el estado científico separado (`verified`, `unresolved`, `diagnostic_partial`, `rejected`). Así no se confunde éxito de un proceso con éxito de una hipótesis.

Las pruebas nuevas deben cubrir timestamp periódico con imágenes distintas, hash que cambia por un píxel, candidatos ambiguos, roles cruzados, labels fuera de cobertura, extremos temporales distintos y futuro que no altera el pasado. Los nombres propuestos solo pasan a «existente» cuando haya implementación, prueba y commit verificables.
