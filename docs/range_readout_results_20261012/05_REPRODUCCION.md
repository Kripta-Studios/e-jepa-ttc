# Reproducción y bundle

La raíz del worktree es `e-jepa-ttc-v15-range-readout`, junto a V13 y V14. Los resultados viven en `artifacts/range_readout_20261012/`.

```powershell
pwsh -NoProfile -File scripts/Resume-RangeReadout.ps1 -Help
pwsh -NoProfile -File scripts/Resume-RangeReadout.ps1
& '../e-jepa-ttc/.venv/Scripts/python.exe' scripts/build_range_readout_reports.py --bundle
```

No lance otra continuación mientras exista el coordinador activo. Los endpoints completos se restauran sin añadir updates. Un fallo requiere diagnosticar el log indicado antes de reanudar; no use reset, clean ni borre checkpoints.

El bundle contiene informes, código nuevo, protocolo recibido, pesos y estados nuevos, curvas, predicciones, diagnósticos, fragmentos vectoriales, features de las familias, pruebas y logs. Los archivos están relacionados mediante BUNDLE_MANIFEST.json y el SHA-256 externo. El verificador comprueba también duplicados y CRC.

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' scripts/build_range_readout_reports.py --verify-zip 'artifacts/V15_RANGE_READOUT_REPORTS_RESULTS_20261012.zip'
Get-FileHash -Algorithm SHA256 -LiteralPath 'artifacts/V15_RANGE_READOUT_REPORTS_RESULTS_20261012.zip'
```

El bundle no sustituye los datos sensoriales originales ni los tres padres matched V13. ADMISSION.json registra sus rutas y hashes; los manifiestos conservan los paths del host original. Repetir desde sensores requiere esas dependencias. La reproducción de tablas desde predicciones y la verificación de hashes no equivale a repetir inferencia ni entrenamiento.

Un checkpoint R requiere `load_endpoint(..., ResidualSpec(bound))`; su atributo `.model` conserva el forward raw del productor con el operador explícito. C20 no puede interpretarse como `0.20*tanh(s)` nativo. Los encoders son los mismos padres compartidos, no nuevas representaciones entrenadas con tres semillas.

No se hace push en esta campaña. Un resultado FCWD/Dev32, si la rama de transferencia llega a habilitarse, seguirá siendo desarrollo externo expuesto.
