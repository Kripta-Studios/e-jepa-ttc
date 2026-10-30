# Verificación final

Batería completa con el entorno Windows corregido: **7297 aprobadas,
46 omitidas, 0 fallos y 0 errores**,
además de **17 subtests aprobados**, en 709.93
segundos. Fuente: `output/phase_campaign_full_utf8.xml` y resumen pytest.

Las pruebas específicas de campaña, lectura, caché y fixtures físicos:
**31 aprobadas**. Ruff: limpio. Pyright: cero errores y avisos.
El verificador científico recalculó scores y comprobó sellos, población y controles
causales. La revisión independiente aceptó el alcance limitado; sus dos detalles
de auditoría se cerraron con pruebas RED→GREEN y un suplemento inmutable.

La primera ejecución general tuvo siete fallos por codificación de subprocesos
Windows al atravesar la ruta con «Álvaro». Pasaron los siete al heredar UTF-8, y
después se repitió la batería completa bajo el entorno corregido. Se conservan
el log inicial, el reintento y la ejecución final; no se atribuyen esos fallos al
entrenamiento ni se ocultan. `PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8` y el entorno
`.venv-transfer` quedan explícitos en los comandos de reproducción.

La rama local se conserva para continuar aquí. Los datos, ZIPs y otros archivos
preexistentes del usuario permanecen en el workspace. No hay promoción científica
implícita en un resultado de tests correcto: prevalecen los límites y regresiones
del informe de resultados.
