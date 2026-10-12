# Entrega y reproducción de V18

Este procedimiento prepara los entregables; no establece que la campaña esté
terminada. La conclusión científica se deriva de endpoints y cohortes completas,
las decisiones registradas y el presupuesto real. Un paquete parcial debe
conservar `remaining_work` y nunca convertirse por su nombre en una victoria.

## Evidencia primaria

Después de los cuatro endpoints y de las cinco cohortes completas de cada rol
P/H, ejecutar desde el worktree y con el entorno compartido:

```powershell
$env:PYTHONPATH = 'src;.'
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B -m operational.joint_measurement.primary_evidence
```

El comando completa coste de forward emparejado, selección prospectiva, recálculo
NumPy, bootstrap por grupos y exports. Puede reanudarse si el coste se fragmenta.
La selección primaria no termina la investigación adaptativa ni la rama RGB.
El coste preparado excluye preprocesado sensorial y no sustituye la medida E2E.

## ZIP esencial

Antes de empaquetar, guardar el estado completo de los fits, cerrar escritores
propios, actualizar el informe y escribir `CAMPAIGN_CONCLUSION.json` en la carpeta
de la campaña con `research_status`, `remaining_work` y las referencias concretas
a los resultados/decisiones que sostienen la conclusión. Este documento no
autoriza escribir una conclusión sin esa evidencia. Confirmar el código en Git
local antes de fijar el paquete.

```powershell
$Bundle = Join-Path $HOME 'Downloads/E_JEPA_TTC_V18_RESEARCH_RESULTS.zip'
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B -m operational.joint_measurement.bundle build $Bundle
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B -m operational.joint_measurement.bundle verify $Bundle
```

La ruta debe ser nueva. El empaquetador conserva código versionado, evidencia
V18 positiva y negativa, checkpoints, predicciones, curvas y recibos sensoriales.
Excluye locks, archivos temporales y bytecode. No incluye las cachés sensoriales
masivas ni los padres históricos externos. Los exports permiten inferencia
portable sobre sus ejemplos; reentrenar requiere esas dependencias admitidas.

La verificación compara el conjunto exacto de miembros, tamaños, SHA-256 y CRC.
Publica un `.sha256` y un `.verification.json` locales. No realiza ningún push,
publicación, submission ni envío a un servidor.

## Extracción independiente

Usar una carpeta nueva y guardar el resultado de verificación fuera de ella.
El ejemplo siguiente supone que `$Bundle` apunta al ZIP creado y que la consola
está en el worktree V18:

```powershell
$Python = (Resolve-Path '../e-jepa-ttc/.venv/Scripts/python.exe').Path
$Extracted = Join-Path $HOME 'Downloads/E_JEPA_TTC_V18_INDEPENDENT'
$Receipt = Join-Path $HOME 'Downloads/E_JEPA_TTC_V18_INDEPENDENT.json'
if (Test-Path -LiteralPath $Extracted) { throw 'Usa una carpeta de extracción nueva.' }
Expand-Archive -LiteralPath $Bundle -DestinationPath $Extracted
Push-Location -LiteralPath (Join-Path $Extracted 'source')
try {
    $env:PYTHONPATH = 'src;.'
    & $Python -B -m operational.joint_measurement.replay $Extracted --output $Receipt
    if ($LASTEXITCODE -ne 0) { throw 'La reproducción independiente ha fallado.' }
}
finally { Pop-Location }
```

`-B` evita introducir bytecode en la extracción intacta. Se recalculan los campos
numéricos documentados de las tablas y se cargan los TorchScript en CPU con sus
ejemplos sin etiquetas. Se preservan las tolerancias incluidas en cada binding;
las diferencias no se ocultan ni se convierten en igualdad bit a bit. Esta
verificación no reproduce eventos crudos, no reentrena productores y no confirma
SOTA. También se recalculan los testigos de cambio del encoder de DIRECT y CORR,
cuando estén incluidos, comparando los checkpoints conservados de 0 y 100 updates.
Se verifican hashes, identidad, pesos finitos, diferencias por tensor y continuidad
del contador. Esto acredita adaptación sobre P; no demuestra generalización.
Un paquete sin tablas, exports ni estos testigos informa explícitamente que no ha
realizado reproducción científica.

En este host Windows se usan streams de Python para guardar y cargar TorchScript:
la ruta C++ directa falló con el nombre de usuario Unicode. La prueba de extracción
independiente cubre el guardado, carga y comparación tras esa corrección.
