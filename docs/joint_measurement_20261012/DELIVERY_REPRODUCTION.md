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

## Reanudar la rama actual

```powershell
./scripts/Resume-JointMeasurement.ps1 -ShowCommand
./scripts/Resume-JointMeasurement.ps1
```

El primer comando sólo muestra la invocación. El segundo rechaza procesos V18
activos y, cuando B1 está registrado, continúa sus checkpoints completos con
`correspondence_isolated`. Si ya están congelados los cuatro endpoints, selecciona
`correspondence_evaluate`. Tras la evidencia B1 completa exige interpretar los
resultados; no decide una nueva receta ni repite la matriz primaria. Los runtimes
conservan sus propios locks y verificaciones de admisión. `-Stage Primary` permite
seleccionar explícitamente el runner histórico; `-Once` sólo se admite para él.

El script no programa la vigilancia. El observador existente se arma con el PID
real del driver y comprueba su identidad cada hora; no debe duplicarse. La
reanudación del 16 de octubre a las 12:21 y sus nueve comprobaciones de continuidad
se documentan en [RESUME_20261016.md](RESUME_20261016.md).

## ZIP esencial

La fase primaria completada se describe en `primary_milestone/README.md` dentro
de la campaña. Para regenerar las tablas y la conclusión desde su evidencia:

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B -m operational.joint_measurement.primary_analysis
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B -m operational.joint_measurement.primary_closure
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B -m operational.joint_measurement.report
```

`primary_closure` documenta el resultado negativo de esta fase, no termina el
objetivo. Conserva la selección previa H y añade una decisión posterior a V,
sin sobrescribir `NEXT_DECISION.json`. El snapshot primario registra la espera
de autorización para ampliar a RGB el alcance del presupuesto de 48 horas.
Si cambia esa autorización o se ejecuta otra rama, hay que interpretar y
actualizar la conclusión de campaña antes de crear una entrega posterior.

Antes de empaquetar, guardar el estado completo de los fits, cerrar escritores
propios, actualizar el informe y escribir `CAMPAIGN_CONCLUSION.json` en la carpeta
de la campaña con `research_status`, `remaining_work` y las referencias concretas
a los resultados/decisiones que sostienen la conclusión. Este documento no
autoriza escribir una conclusión sin esa evidencia. Confirmar el código en Git
local antes de fijar el paquete.

```powershell
$Bundle = Join-Path $HOME 'Downloads/E_JEPA_TTC_V18_RESULTS_COMPACT.zip'
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B -m operational.joint_measurement.compact_delivery $Bundle
& '../e-jepa-ttc/.venv/Scripts/python.exe' -B -m operational.joint_measurement.bundle verify $Bundle
```

La ruta debe ser nueva. El empaquetador compacto publica únicamente un ZIP de
como máximo **100.000.000 bytes**, verificado antes de publicarlo. Conserva código
versionado, evidencia positiva y negativa, pesos portables y sus ejemplos,
predicciones consolidadas, curvas y recibos. Omite estados de optimizador, sensores,
cachés, fragmentos, locks, temporales y padres históricos externos. El manifiesto
enumera las omisiones. Reentrenar requiere las dependencias admitidas que permanecen
locales. El antiguo comando `bundle build` no aplica este límite y no debe usarse
para esta entrega.

La verificación compara el conjunto exacto de miembros, tamaños, SHA-256 y CRC.
Publica un `.sha256` y un `.verification.json` locales. No realiza ningún push,
publicación, submission ni envío a un servidor.

## Extracción independiente

Usar una carpeta nueva y guardar el resultado de verificación fuera de ella.
La extracción debe quedar fuera de `artifacts/joint_measurement_20261012`: el
contador vivo recorre los journals de ese directorio y contaría otra vez los
journals copiados. El ZIP y los recibos sí pueden conservarse en la campaña.
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
El testigo de CORR-FROZEN, si está incluido, comprueba por separado que los bytes
del encoder no cambian entre 0 y 100, que el predictor sí cambia y que los pasos
del optimizador y la curva cubren ese intervalo. No acredita por sí solo la
invariancia durante los 12.000 updates: esa comprobación requiere el endpoint.
Si se incluye `runtime/optimization_20261013/RUNTIME_BINDING.json`, el replay
exige también la evidencia de reanudación optimizada: compara los nueve estados
de graph6 frente al prefijo compartido4 más restored2, verifica su contabilidad
física y los hashes de las fuentes del snapshot extraído. Son ocho updates
históricos retenidos y cero nuevos. La igualdad exacta se refiere a esa
reanudación; no a la comparación numérica entre eager y CUDA Graphs. Se preservan
las admisiones estrictas fallidas, la política numérica y sus bindings. El replay
de estados no vuelve a ejecutar kernels CUDA ni mide de nuevo el rendimiento.

Un paquete sin tablas, exports ni testigos de encoder o runtime informa
explícitamente que no ha realizado reproducción científica.

En este host Windows se usan streams de Python para guardar y cargar TorchScript:
la ruta C++ directa falló con el nombre de usuario Unicode. La prueba de extracción
independiente cubre el guardado, carga y comparación tras esa corrección.
