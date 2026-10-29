# V14 RANGE-TRANSFER: entrega detallada

Fecha de esta entrega: 12 de octubre de 2026, Europe/Madrid. Los recibos usan UTC; el cierre alrededor de las 22:08 UTC del día 11 corresponde al día 12 en Madrid.

**Resultado: NO_CANDIDATE.** Ejecuté seis cabezas event-only de 2.500 updates, congelé sus endpoints y evalué las mismas 16.432 consultas V. Las variantes que redistribuyen masa reducen el error crucial, pero incumplen las salvaguardas de otros rangos. Ninguna receta habilita las réplicas seeds13/23. Conservo los seis pesos y sus resultados.

## Informes

1. [Trabajo realizado, identidades y método](01_EJECUCION_Y_METODO.md).
2. [Diagnósticos: masa, gradientes y origen del sesgo](02_DIAGNOSTICOS.md).
3. [Resultados completos, gates, grupos y bootstrap](03_RESULTADOS_Y_DECISION.md).
4. [Pruebas, coste, contabilidad y fallos](04_PRUEBAS_COSTE_Y_CONTABILIDAD.md).
5. [Reproducción, rutas y contenido del bundle](05_REPRODUCCION_Y_ENTREGA.md).

## Resultado principal

| Receta seed7 | MiDc | MiDs | MiDl | MiDn | Overall MiD | mean_MiD | MAE s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RT-M0-L0-S7 | 156.832 | 54.776 | 52.217 | 61.382 | 106.208 | 70.367 | 2.512 |
| RT-M0-L1-S7 | 149.592 | 51.168 | 56.973 | 61.009 | 101.945 | 68.911 | 2.029 |
| RT-M1-L0-S7 | 134.470 | 63.112 | 76.397 | 84.208 | 102.229 | 85.291 | 2.979 |
| RT-M1-L1-S7 | 136.187 | 65.151 | 83.999 | 80.041 | 104.043 | 85.173 | 2.341 |
| RT-M2-L0-S7 | 134.871 | 67.812 | 77.768 | 87.619 | 104.318 | 88.570 | 3.369 |
| RT-M2-L1-S7 | 134.591 | 70.530 | 83.826 | 79.928 | 104.830 | 86.637 | 2.379 |

Menor es mejor en las columnas de error. Overall MiD y mean_MiD tienen agregaciones distintas; el informe 03 explica ambas. Estos resultados corresponden a validación interna V.

Ejecuté 15.000 updates científicos y 20 técnicos sintéticos. La recuperación fue cero. Mantengo el presupuesto no consumido sin reasignarlo. Dejé V13/R1, sus entrenamientos y sus resultados intactos.

El usuario autorizó el push de estos informes y del código en su petición del 12 de octubre. Esa autorización posterior permite publicar la rama V14; el handoff original conserva su prohibición histórica de push como evidencia del alcance anterior. La publicación no autoriza nuevos fits, cambios en V13 ni merge.

Las tablas se regeneran con `scripts/build_range_transfer_reports.py`. [REPORT_INPUTS.json](REPORT_INPUTS.json) registra los hashes de sus fuentes. Los JSON/CSV ligeros están en `evidence/`; las predicciones por consulta, checkpoints y trazas completas están en el ZIP.
