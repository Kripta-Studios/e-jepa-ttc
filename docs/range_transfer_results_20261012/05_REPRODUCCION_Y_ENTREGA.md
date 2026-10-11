# 05. Reproducción, rutas y entrega

## Rutas del host original

```text
V14:
C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\EVOCON_JEPA_Codex_Handoff\e-jepa-ttc-v14-range-transfer

Resultados V14, relativos a esa raíz:
artifacts/range_transfer_20261010/

Informes de esta entrega:
docs/range_transfer_results_20261012/

V13, dependencia de sólo lectura:
../e-jepa-ttc-v13-rgb-port/artifacts/rgb_port_20261008/

Evidencia V12 de sólo lectura:
../e-jepa-ttc-v12-efficient-context/
```

## Qué contiene cada nivel

| Ruta relativa a la campaña V14 | Contenido |
|---|---|
| fits/RT-M*-L*-S7/checkpoint_last.pt | Seis pesos nuevos, estado de optimizador y continuidad |
| fits/*/CHECKPOINT_RECEIPT.json | Endpoint, contador y hash de checkpoint |
| fits/*/CURVE.csv | 2.500 updates por fit: loss, norma previa a clipping y LR |
| evaluation/screen/*/PREDICTIONS.csv | 16.432 predicciones por celda sobre la misma V |
| evaluation/screen/*/LOSSES_PER_QUERY.csv | MiD, error absoluto/relativo, sesgo y cobertura por consulta |
| evaluation/screen/*/METRICS.json | Métricas usadas en los gates |
| evaluation/screen/*/GROUP_METRICS.json | Todas las métricas por grupo |
| evaluation/screen/*/SUBRANGE*.csv | Desglose descriptivo por subrango y grupo |
| bootstrap/screen/ | Cinco contrastes, fragmentos y sus intervalos |
| MASS_AUDIT.csv, H_EXPOSURE.csv, GRADIENT_AUDIT.csv | Masa, draws y presión del objetivo |
| matched_H_controls/, matched_H_raw_adapted/ | Trazas de cabezas y productores H |
| v12_cached_attribution/, v12_raw_attribution/ | Diagnóstico histórico; pesos TRAIN40 fuera del factorial |
| qa/ | Pruebas, scripts auxiliares, errores y checkpoints sintéticos |
| NEXT_DECISION.json, SELECTION_FREEZE.json | Cierre NO_CANDIDATE y gates |

GitHub contiene la implementación, el handoff, informes, figuras y evidencia ligera por banda/grupo. El ZIP añade pesos, predicciones y pérdidas por consulta, trazas y logs. No subo datos sensoriales crudos al repositorio ni incluyo las cachés H/V completas de V13 en este bundle.

## Regenerar estos informes sin entrenar

Desde la raíz V14, con el Python existente:

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B scripts/build_range_transfer_reports.py
```

El generador verifica fuentes científicas, pesos, recibos y predicciones, copia la evidencia ligera y genera las tablas. Requiere la campaña completa del ZIP en `artifacts/range_transfer_20261010`. No ejecuta modelos ni recalcula la selección.

Para crear un nuevo ZIP con esos informes y la campaña:

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B scripts/build_range_transfer_reports.py --bundle
```

El generador verifica los hashes del ZIP y escribe un `.sha256` y un `.verification.json` externos. El manifiesto interno identifica cada archivo incluido. Los informes históricos en artifacts describen el momento de ejecución anterior al push; los informes de esta carpeta documentan la publicación autorizada después.

## Verificar el ZIP sin Torch ni datos

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B scripts/build_range_transfer_reports.py --verify-zip 'artifacts/RANGE_TRANSFER_V14_REPORTS_RESULTS_20261012.zip'
Get-FileHash -Algorithm SHA256 -LiteralPath 'artifacts/RANGE_TRANSFER_V14_REPORTS_RESULTS_20261012.zip'
```

Compare el hash externo con el entregado y compruebe el manifiesto interno. Una coincidencia de hashes acredita integridad de los archivos, no validez científica por sí sola.

## Continuidad de entrenamiento: campaña cerrada

El comando real usado fue `pwsh -NoProfile -File scripts/Resume-RangeTransfer.ps1`. Conserva el checkpoint y no acorta endpoints. No hace falta invocarlo para leer o regenerar informes. La cola terminó NO_CANDIDATE y no debe ampliarse con el presupuesto sobrante.

Un replay completo del entrenamiento requiere las fuentes históricas, los tres padres matched, los manifiestos y las cachés H/V de V13 con los hashes admitidos. Los paths absolutos de los recibos pertenecen al host original; moverlos a otra máquina exige resolver esas dependencias sin alterar la identidad científica. El ZIP no pretende sustituir los datasets ni esos padres.

## Reproducción de diagnósticos y pruebas

Los scripts de `artifacts/range_transfer_20261010/qa/` conservan los comandos auxiliares. TRACE_MATCHED_H_ADAPTED.py usa matched_pair_adapter.py y requiere los padres/inputs locales; no necesita V ni optimizer.step. VERIFY_SCIENTIFIC_DELIVERY.py verifica resultados y genera desglose por consulta/subrango. Los scripts antiguos de reconstrucción tienen nombres históricos: use el generador de informes de esta entrega para empaquetar el cierre final.

Las pruebas de integración publicadas se ejecutan así:

```powershell
$env:PYTHONPATH='src;.'
$env:OMP_NUM_THREADS='1'
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B -m pytest tests/test_range_transfer.py tests/test_range_transfer_integration.py -q --override-ini addopts=
```

## Decisión y trabajo no ejecutado

El usuario puede revisar qué rango empeora y dónde empieza el sesgo antes de autorizar otro estudio. Esta entrega no elige nuevos coeficientes, encoders, semillas, calibraciones o splits. No hay resultado de réplica, transferencia nueva, test oficial, robustez global ni latencia E2E V14. Una investigación posterior necesita definir su propia hipótesis y sus gates antes de ejecutarla.
