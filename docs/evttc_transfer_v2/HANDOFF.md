# PHASE-TTC: ejecución del paquete del 30 de octubre

## Resultado y alcance

Se extrajo el ZIP y se verificaron CRC, tamaño y SHA-256 de los veinte documentos
enumerados por su manifiesto. La ejecución partió de
`30b28b5c14eb9ed60067ed2f1dc262d7b60f2654`, en la rama
`research/evttc-transfer-v2`. El directorio extraído conserva los 21 Markdown.

La primera iteración ya tiene herramientas ejecutables y evidencia real de identidad,
alineación de targets, comparación de predicciones guardadas e instrumentación.
La investigación completa sigue pendiente: no se han superado los contratos de
independencia y evaluación necesarios para seleccionar una intervención aprendida.
La decisión de piloto está en [PILOT_DECISION.md](PILOT_DECISION.md).

| Ticket | Resultado | Límite pendiente |
|---|---|---|
| T00 | Entorno aislado, inventario real, candidato identificado, run histórico verificado | El checkout actual tiene una diferencia de formato respecto al código sellado; se conserva la distinción |
| T01 | Ocho correspondencias únicas y 365 frames con igualdad de píxeles y hashes | La identidad RGB no certifica independencia de adquisiciones |
| T02 | Nueve componentes y normalizadores identificados; genealogía documentada | Selección histórica y ancestros de teachers con estado desconocido |
| T03 | Ocho exclusiones y 24 candidatas registradas como desarrollo expuesto | Relaciones de adquisición sin aprobar; no hay grupos autorizados para entrenamiento |
| T04 | Evaluador diagnóstico con 434 filas y 354 targets válidos | Geometría/endpoint y política completa del scorer oficial sin verificar |
| T05 | Seis baselines guardados y cinco comparaciones pareadas regenerables | Referencia geométrica real y evaluación de desarrollo pendientes de contrato |
| T06 | Fixtures físicos y trazas del candidato con control de paridad | Diagnóstico en desarrollo independiente, overlays cuantificados y causalidad del fallo real pendientes |
| T07 | Decisión explícita de no abrir entrenamiento todavía | Resolver contratos antes de F1; JEPA permanece condicionado |

## Resultados que se pueden afirmar

- El CLI reproduce las ocho correspondencias del anexo contra las 32 grabaciones.
- El evaluador conserva como ausentes `CCRs1-medium:32` a `CCRs1-medium:42`.
  Las 69 consultas de Slider permanecen sin target compatible verificado.
- La comparación usa predicciones originales firmadas, conserva negativos y muestra
  ausencias. [BASELINES.md](BASELINES.md) se genera desde ledgers, no desde cifras copiadas.
- Los pesos, normalizadores, ZIP y ledger originales no se han modificado.
- La verificación histórica de 434 filas pasa al usar los bytes del archivo preservado.
  El verificador sobre el checkout actual detectó primero una diferencia en
  `local_transport.py`. El commit `820e6a6f` cambió el formato; la comparación AST
  entre ambas versiones coincide. Se conservaron ambos recibos.

Las métricas de [evidence/iteration_summary.json](evidence/iteration_summary.json)
pertenecen a un diagnóstico local expuesto. No reproducen el score oficial ni
demuestran SOTA. La rama RGB tiene mejor error medio que la fusión en el subconjunto
compartido de esta comparación, pero esa observación no autoriza elegir un modelo
usando las consultas de competición.

## Genealogía y roles

[evidence/lineage.json](evidence/lineage.json) identifica los nueve checkpoints,
ambos normalizadores y sus recibos. La receta declara TRAIN40 y distingue ancestros
OOF de dependencias de despliegue. El inventario mantiene `unresolved` donde los
recibos consultados no prueban el historial de selección o los ancestros recursivos.
No se convierte `frozen` en una afirmación de transferencia limpia.

[El manifest de población](../../data/splits/evttc_transfer_v2.yaml) conserva las
32 grabaciones. Los ocho solapamientos quedan excluidos. Las otras 24 reciben el
rol `development_exposed` y `acquisition_group: unknown`; ninguna recibe por defecto
el rol de test independiente. El validador propaga exclusiones por aliases,
grabación y adquisición, impide roles cruzados y bloquea entrenamiento con relaciones
sin resolver o una política que no lo permita.

## Slider adicional

La búsqueda encontró [Slider-500 en la página oficial de Event-Aided TTC](https://nail-hnu.github.io/EventAidedTTC/).
Se descargaron su bag, cajas, TTC y calibración a un directorio nuevo. No se
inspeccionaron los valores TTC ni se ejecutó inferencia sobre esta secuencia.
El bag usa `/dvs/events` y `/aps/image_raw`; el lector congelado exige
`/capture_node/events` y `/image_raw`. El rechazo queda registrado en
`output/evttc_transfer_v2/external/slider500_receipt.json`, junto con hashes.
La relación con las adquisiciones de competición y los términos de reutilización
siguen sin verificar. Descargarla no la convierte en confirmación independiente.

La [página de formato EvTTC](https://nail-hnu.github.io/EvTTC/download/data_format/)
documenta índice, timestamp, distancia, velocidad y TTC del CSV. No resuelve por sí
sola la equivalencia con targets y postprocesamiento del organizador.

## Reproducción

PowerShell, desde la raíz. El entorno original tenía un `.pth` incompatible con la
codificación local y una instalación de torch incompleta. Se creó uno nuevo:

```powershell
$env:UV_PROJECT_ENVIRONMENT = '.venv-transfer'
uv sync --locked --no-editable --python 3.11
$env:UV_PROJECT_ENVIRONMENT = (Resolve-Path .venv-transfer).Path
$env:PYTHONPATH = 'src;.'
$env:PYTHONUTF8 = '1'
```

Si la caché local de torch vuelve a contener archivos incompletos, el comando usado
para repararla fue `uv --no-cache sync --locked --no-editable --python 3.11 --reinstall-package torch`.
El forward/backward CUDA y la prueba de paridad del candidato verificaron el entorno.

Las variables siguientes deben apuntar a recursos locales existentes:
`EVTTC_DATA_ROOT` contiene las rutas `datasets/evttc/...` del manifest,
`EVTTC_COMPETITION_ROOT` las diez secuencias, `EVTTC_CANDIDATE_ROOT` el paquete
`FINAL_CANDIDATE`, `EVTTC_RUN_ROOT` el run original y `EVTTC_GARL_RUN_ROOT` el de Garl.
Usar una salida nueva al cambiar código o configuración; los CLIs rechazan
sobrescribir recibos distintos.

```powershell
.venv-transfer/Scripts/python.exe -X utf8 scripts/audit_evttc_transfer.py identity --manifest data/manifests/evttc_all32_local.yaml --data-root "$env:EVTTC_DATA_ROOT" --competition-root "$env:EVTTC_COMPETITION_ROOT" --reference-evidence docs/superpowers/plans/evidence/2026-10-30-evttc-audit.json --output output/reproduction/D00
.venv-transfer/Scripts/python.exe -X utf8 scripts/prepare_evttc_transfer_iteration.py --manifest data/manifests/evttc_all32_local.yaml --identities output/reproduction/D00/report.json --data-root "$env:EVTTC_DATA_ROOT" --candidate "$env:EVTTC_CANDIDATE_ROOT" --run "$env:EVTTC_RUN_ROOT" --package PHASE_TTC_Ruta_SOTA_2026-10-30 --package-zip PHASE_TTC_Plan_SOTA_2026-10-30.zip --output output/reproduction/inputs
.venv-transfer/Scripts/python.exe -X utf8 scripts/audit_evttc_transfer.py lineage --candidate "$env:EVTTC_CANDIDATE_ROOT" --component-inventory output/reproduction/inputs/components.json --exclusions output/reproduction/inputs/exclusions.json --output output/reproduction/D01
.venv-transfer/Scripts/python.exe -X utf8 scripts/audit_evttc_transfer.py protocol --inventory output/reproduction/inputs/population.json --identities output/reproduction/D00/report.json --lineage output/reproduction/D01/report.json --policy configs/protocol/evttc_transfer_v2.yaml --output output/reproduction/D02
.venv-transfer/Scripts/python.exe -X utf8 scripts/audit_evttc_transfer.py targets --queries output/reproduction/inputs/queries.json --labels-manifest output/reproduction/inputs/labels.json --clock-map output/reproduction/inputs/clock_map.json --target-contract configs/protocol/evttc_transfer_targets_v1.yaml --output output/reproduction/E00
.venv-transfer/Scripts/python.exe -X utf8 scripts/report_evttc_transfer.py --phase-run "$env:EVTTC_RUN_ROOT" --garl-run "$env:EVTTC_GARL_RUN_ROOT" --targets output/reproduction/E00/report.json --policy configs/protocol/evttc_transfer_v2.yaml --output output/reproduction/E01
.venv-transfer/Scripts/python.exe -X utf8 scripts/audit_evttc_transfer.py diagnose --data-root "$env:EVTTC_COMPETITION_ROOT" --candidate "$env:EVTTC_CANDIDATE_ROOT" --query-manifest output/reproduction/inputs/queries.json --protocol output/reproduction/D02/report.json --diagnostic-config configs/protocol/evttc_transfer_diagnostics_v1.yaml --output output/reproduction/A00 --device cuda
```

`verify_evttc_historical_snapshot.py --help` documenta la reconstrucción aislada
del código histórico y reutiliza el verificador existente. No cambia el checkout.

Los subcomandos de auditoría usan códigos 0 (completado dentro del alcance),
1 (contrato violado), 2 (argumentos inválidos), 3 (recurso ausente). Un proceso
con salida 0 y estado científico `unresolved` no autoriza entrenamiento.

Ver [EXECUTION_LEDGER.md](EXECUTION_LEDGER.md) para pruebas, decisiones y fallos conservados.
