# 01 — Revisión independiente y decisión

## Fuentes examinadas

Se han leído el índice, los ocho informes y las tablas de `docs/agent_handoff_20261010` en V12 commit `1a686e338a1145c6794bec0704bdae4cda95f06d`; su padre es `d1f561df7b173464de24e1b8e3ef09170081f7fe`. Se ha usado el snapshot V13 `da068ebdeb2d967ee10b2afa348e1317b8720ad0`. Las métricas V12 no se atribuyen a V13.

La clonación HTTPS falló por DNS. En cambio, el ZIP del usuario contiene los snapshots completos. Se comprobaron 6.755 entradas de manifiesto, CRC y seguridad de extracción; el handoff pasó 511 hashes y 87 enlaces. Se reconstruyó el árbol Git desde todos los blobs/modos: V12 coincide con `8da1e1b35816ab634e68c5b1146b7610624d9028` y V13 con `485b724e699a55815803abf583272ab523d294b3`, ambos contrastados con la API de GitHub. Esto verifica el contenido de los snapshots, no equivale a disponer de `.git`, datos crudos ni checkpoints grandes.

## Resultado vigente

| Población | H8 | Garl eventos | Garl RGB+eventos | Métrica |
|---|---:|---:|---:|---|
| eAP TRAIN40 | 72,5393 | 93,6824 | no usada aquí | overall_MiD; entrenamiento |
| Dev32 | 138,3771 | 165,9840 | 136,3360 | mean_MiD; desarrollo expuesto |
| FCWD | 79,9518 | 61,8165 | mapping no admitido | mean_MiD; tres secuencias expuestas |

H8 de esta tabla es la **mediana de predicciones TTC de tres cabezas** sobre productores comunes, no la media de sus pérdidas publicada en etapas anteriores. Dev32/FCWD no contienen negativos: no existe overall_MiD oficial completo. Una reducción de MAE no implica menor MiD.

## Déficit FCWD

Las 597 filas elegibles tienen bandas c/s/l/>10 de 201/189/108/99. H8 suma +27,904676 unidades al gap por c; las restantes restan 9,769396. El déficit neto es +18,135280 MiD. Las tres secuencias empeoran, por lo que no depende de una sola escena.

En c, H8 tiene MAE≈0,7004 s, sesgo +0,6565 s y 83,58% de sobreestimaciones, frente a MAE Garl≈0,3272 s y MiDc 95,0568 frente a 177,9379. H8 subestima los 99 targets >10 s. El sesgo global negativo mezcla esos comportamientos: aplicar una corrección constante no los resuelve a la vez.

La reproyección y el estudiante no cierran el déficit. Warp+student da mean_MiD≈74,5607 y MiDc≈162,9689; todavía pierde frente a Garl. Student sin warp mejora FCWD y perjudica Dev32. No promover por el dataset favorable.

## Cálculos adicionales de este paquete

`evidence/independent_analysis/` se obtiene del CSV nativo congelado FCWD, no de una inferencia nueva. Difiere a últimas cifras del replay CPU de referencia. Sus hallazgos:

- Targets: mínimo 1,194195 s y máximo 28,922619 s. No hay casos ≤1 s.
- Ninguna predicción H8 mediana satura ±60 s. El cap no explica la compresión de rango observada en FCWD.
- H8 gana a Garl en aproximadamente 55,28% de consultas por MiD, pero pierde la media por la magnitud de errores cruciales.
- Las veinte mayores desventajas suman 65,20% del gap **neto**; son casos cruciales próximos en el tiempo. No son veinte escenas independientes ni un porcentaje del error absoluto total.
- Si todas las otras bandas quedasen idénticas, MiDc≈124,07 bastaría algebraicamente para empatar el mean_MiD de Garl. Exigiría reducir c en un 30,27%. Es contabilidad descriptiva, no un objetivo ajustado para seleccionar sobre FCWD. Ganar también la banda c exige ir por debajo de ≈95,06.
- Para predicciones H8 en (2,3] s, los targets abarcan aproximadamente 1,19–3,49 s. No existe una corrección por rango de predicción evidentemente correcta para todos esos ejemplos.

## Hallazgo de código: masas diferentes

V12 `simplex_t/cache.py::training_mass` aplica masa por secuencia y banda TTC, con pesos TRAIN relativos 0,5/0,3/0,1/0,1, renormalizados sólo si falta una banda en TRAIN.

V13 `operational/rgb_port/prepare.py::_role_payload` asigna `1/(n_grupos*n_filas_grupo)` a filas RGB disponibles; esa masa pasa a las cabezas. Por tanto la receta V13 no es idéntica a la ponderación por bandas V12. Hay que comprobar el manifest efectivo: esto no prueba un bug, no explica causalmente los errores V12 y no permite modificar un fit activo.

Además, V12/H8 **ya optimiza L1 de fase**. «Añadir MiD» como si estuviera ausente sería repetir un objetivo existente. El nuevo experimento debe separar distribución de peso, pérdida, sampler y decodificador.

## Dictamen sobre las intervenciones ya ensayadas

Direct cambia salida, red, batch, scheduler y pérdida; no identifica causalmente el efecto de ninguno. H4/H2 son truncaciones en inferencia, no modelos dedicados. El estudiante imita features, no está optimizado directamente para TTC. Warp es una aproximación de ROI, no una caché exacta. Todos sus resultados se conservan.

La fase de V13 está en curso: los avances de E_A5/E_C2F no equivalen a endpoints RGB ni a cabezas/fusión terminadas. El snapshot no es telemetría viva. La prioridad es completar y evaluar lo autorizado sin cambiar sus fuentes, BF16, sampler o pérdidas.
