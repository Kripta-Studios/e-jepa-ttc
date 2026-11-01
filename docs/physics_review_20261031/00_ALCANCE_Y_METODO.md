# Alcance y método de la revisión

## Corte y jerarquía de fuentes

Ventana solicitada: seis meses hasta el 31 de octubre de 2026. Se usó el intervalo desde `2026-04-30T00:00:00+02:00` hasta antes de `2026-11-01T00:00:00+01:00`. Git selecciona por fecha de commit; las tablas mensuales agrupan por fecha de autor. Se conservan ambas en el JSON. El primer commit disponible tiene fecha de autor 30 de junio de 2026. La copia no es shallow.

HEAD fijado: `f8131a031f61d9270323fe3961dad88016b582cd`. Las referencias son las disponibles localmente, incluidas ramas remotas ya conocidas y otras refs. No se hizo fetch ni se afirma que no existan commits remotos adicionales. Al inicio el árbol no mostraba cambios. Los archivos de esta revisión se añaden en una carpeta aparte.

Al cerrar apareció trabajo concurrente hasta `e750e38130ae02ba2a362bee3270b4fc3cb8d063`. Sus tres commits y resultados se incorporan en la [actualización final](13_ACTUALIZACION_CONCURRENTE.md), conservando el inventario inicial y sus hashes. El dossier no modifica ese trabajo ni sus procesos. Los conteos de 1.516 commits y 493 Markdown corresponden al corte inicial; no se presentan como inventario del árbol posterior.

Se da prioridad a predicciones y recibos con identidad, después a resúmenes regenerables, código del commit usado y decisiones fechadas. Un plan prueba intención. Un commit prueba que se registró un cambio. Ninguno prueba que un entrenamiento terminara. Un hash permite reconocer bytes, pero no prueba que su contenido sea científicamente correcto.

## Cobertura efectiva

El inventario mecánico cubre los cuerpos de los 1.516 commits alcanzables en la ventana y sus entradas numstat, así como rutas, hashes y encabezados de los documentos actuales. Se leyeron las secciones del manuscrito modular y sus tablas relevantes, la arquitectura final, las métricas y los resultados tempranos, intermedios y tardíos. Se contrastaron las propuestas con código de Stage68, conversión temporal, controles de historia y documentación de los pilotos del 31 de octubre.

Se revisaron los hitos de integridad de julio/agosto con `git show`, incluido el diff que retiró los marcadores eAP ficticios del smoke. Se verificó que el fallback isotrópico ya existía en el commit de entrenamiento de Stage68; no se infirió su presencia histórica a partir del código actual solamente.

El análisis de los 493 Markdown combina inventario completo y lectura dirigida de informes, resultados, decisiones y pasajes pertinentes. **No se afirma lectura crítica línea por línea de todos los documentos, los 1.716 JSON o los 1.516 parches.** Tampoco se inspeccionaron todos los HDF5 ni se reprodujeron todos los runs. Las tablas escalares históricas se tratan como proyecciones publicadas cuando faltan los originales necesarios para recalcularlas.

## Verificaciones ejecutadas

El verificador original del artículo pasó: 13 hashes, ocho resúmenes de transferencia, doce campos oficiales, reducción ponderada de MiD, ausencia de consultas duplicadas entre los dos cohortes y tres grupos de secuencia compartidos. Se recalcularon además media, mediana, p95, máximo, RTE y excedencias de cuatro métodos sobre las 468 filas etiquetadas Extended-new. Se comprobaron ejemplos aritméticos del reloj firmado y la diferencia de anclaje entre endpoints.

Los intervalos bootstrap se leen del análisis archivado; no se afirma haber repetido sus 10.000 réplicas. El script nuevo no entrena, no carga checkpoints y no accede a sensores. El resultado queda en [verification.json](verification.json).

No se volvió a compilar el PDF: esta revisión evalúa contenido y evidencia, no maquetación. El recibo anterior documenta 52 páginas, 24 figuras y 27 tablas. No se atribuyen a esta revisión las pruebas pytest registradas en campañas anteriores ni se certifica el estado actual de toda la suite.

## Reproducción de esta revisión

Desde la raíz:

```powershell
python -S docs/physics_review_20261031/build_inventory.py
python -S docs/physics_review_20261031/verify_evidence.py
```

El inventario se rehace con las refs disponibles en ese momento; por ello su HEAD y conteos pueden cambiar. Los archivos originales inventariados tienen SHA-256. El directorio de esta revisión queda excluido del corpus para evitar contarse a sí mismo. El registro Git completo ocupa varios archivos porque conserva las rutas modificadas de cada commit.

## Vocabulario de certeza

En este dossier, «verificado» identifica un recálculo o inspección ejecutados en esta revisión. «Registrado» significa que el recibo o informe original comunica ese resultado. «Inferencia» es una interpretación que conecta evidencias. «Propuesta» es una prueba sin ejecutar. Esta separación evita convertir una ley correcta, un plan o un informe regenerable en evidencia de una mejora del modelo.
