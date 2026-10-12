# Comandos y propietarios

Directorio de trabajo de esta entrega:

```powershell
Set-Location -LiteralPath 'C:/Users/Álvaro Schwiedop/Desktop/KriptaStudios/EVOCON_JEPA_Codex_Handoff'
$ContinuidadPython = './e-jepa-ttc/.venv/Scripts/python.exe'
$ContinuidadScripts = './output/continuidad_v16_v18_20261012/scripts'
```

Actualizar el estado V18 y los diagnósticos exclusivamente desde predicciones
completas ya selladas, sin lanzar inferencia ni tocar V18:

```powershell
& $ContinuidadPython -B -c "import sys;sys.path.insert(0,'output/continuidad_v16_v18_20261012/scripts');from prepare_delivery import snapshot;snapshot()"
& $ContinuidadPython -B "$ContinuidadScripts/diagnose_v18_predictions.py"
```

El segundo comando no produce resultados P/H/V mientras no existan sus tablas
completas. No abre H adelantadamente ni solicita evaluar un checkpoint parcial.
Para NATIVE, la columna raw_phase publicada se reconstruye desde el TTC; la
observación de la fase interna verdadera queda pendiente del propietario V18.

El mensaje para esa sesión está en
`E_JEPA_SHARED_COORDINATION/V16_V18_CONTINUIDAD_20261012_MENSAJE.md`, acompañado
de `V16_V18_CONTINUIDAD_20261012_REQUEST.json`. Es un traspaso pasivo: no existe
acuse de lectura ni se ha inyectado un mensaje en su sesión interactiva.

Antes de cualquier recuperación V18, su propietario debe comprobar los
procesos, el directorio real, journal, recibo y presupuesto. Un lanzador del
venv y su hijo Python constituyen un solo supervisor. Mientras siga vivo,
**no ejecutar otro Resume-JointMeasurement.ps1**.

Comprobación de procesos, de sólo lectura:

```powershell
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match 'operational\.joint_measurement\.run' } |
    Select-Object ProcessId, ParentProcessId, CreationDate, ExecutablePath, CommandLine
```

El comando real de recuperación del propietario, sólo una vez acreditado que
no queda otro supervisor y siguen admitidos sus recibos y presupuesto, es:

```powershell
& './e-jepa-ttc-v18-joint-measurement/scripts/Resume-JointMeasurement.ps1'
```

No se ha ejecutado durante esta continuación. No reemplaza su autorización
existente ni amplía presupuesto. El orden actualmente registrado prioriza
DIRECT, CORR y CORR-FROZEN, y después reanuda NATIVE: sigue exigiendo los cuatro
endpoints de 12.000 updates antes de H.

Los comandos realmente ejecutados para reproducir V16 están guardados, con
directorios, salida y tiempos, en `verification/EXECUTION.json`. No es necesario
repetirlos para enviar la submission. La reproducción usa inputs V cacheados,
sin reentrenar ni reconstruir los encoders completos.

Los scripts de coste son experimentos CPU aislados. Sus carpetas de evidencia
contienen registros de fuentes y población; una segunda ejecución no debe
sobrescribir una medición ya cerrada. No son un servicio ni un trabajo programado.
Las seis pruebas del adaptador se pueden comprobar así:

```powershell
& $ContinuidadPython -B "$ContinuidadScripts/test_rgb_exact_cache.py" -v
```

El envío oficial es manual del usuario, siguiendo `EVALUACION_OFICIAL.md`.
No se han creado comandos automáticos de subida.
