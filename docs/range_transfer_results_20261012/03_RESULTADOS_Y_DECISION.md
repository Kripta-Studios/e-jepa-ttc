# 03. Resultados, gates y decisión

## Población y métricas

Evalué 16.432 consultas V de ocho grupos, idénticas para las seis celdas. Todas produjeron predicciones finitas y MiD válido en el soporte evaluado. Los seis endpoints ya estaban congelados. V es desarrollo interno; estos números no son test oficial de GarlTTC.

Por consulta, con delta=0,1 segundos:

```text
MiD_i = 10000 * abs(log(1 - 0.1 / T_pred_i) - log(1 - 0.1 / T_true_i))
```

MiDc cubre (0,1;3], MiDs (3;6], MiDl (6;10] y MiDn negativos del protocolo. Primero calculo la media de MiD dentro de cada grupo y banda; después promedio grupos. El agregado nativo es `0.5*MiDc + 0.3*MiDs + 0.1*MiDl + 0.1*MiDn`. mean_MiD es la media por consulta de toda la población y da más peso a los rangos con más filas. MAE está en segundos y también agrupa consultas. No sustituyo una agregación por otra según favorezca una receta.

Si falta una banda no completo overall_MiD. En esta V están presentes las cuatro: 1.845 cruciales, 4.284 medias, 3.283 largas y 7.020 negativas.

## Factorial completo

| Receta seed7 | MiDc | MiDs | MiDl | MiDn | Overall MiD | mean_MiD | MAE s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RT-M0-L0-S7 | 156.832 | 54.776 | 52.217 | 61.382 | 106.208 | 70.367 | 2.512 |
| RT-M0-L1-S7 | 149.592 | 51.168 | 56.973 | 61.009 | 101.945 | 68.911 | 2.029 |
| RT-M1-L0-S7 | 134.470 | 63.112 | 76.397 | 84.208 | 102.229 | 85.291 | 2.979 |
| RT-M1-L1-S7 | 136.187 | 65.151 | 83.999 | 80.041 | 104.043 | 85.173 | 2.341 |
| RT-M2-L0-S7 | 134.871 | 67.812 | 77.768 | 87.619 | 104.318 | 88.570 | 3.369 |
| RT-M2-L1-S7 | 134.591 | 70.530 | 83.826 | 79.928 | 104.830 | 86.637 | 2.379 |

Las diferencias de últimas cifras entre tablas de subrangos y métricas originales proceden del CSV y de las formas numéricas log/log1p del recálculo descriptivo. La selección utiliza METRICS.json congelado, no las tablas redondeadas de este informe.

## Gates fijados antes de ver V

1. MiDc debe bajar al menos 10 % y mejorar en al menos 60 % de los grupos con cruciales.
2. El agregado nativo no puede empeorar.
3. Cada banda media/larga/negativa puede empeorar como máximo `max(2 MiD, 5 % del control)`; el MAE global no puede superar 1,02 veces el control.
4. Cobertura completa; error de signo como máximo +0,5 puntos porcentuales; p95 de la parte positiva del error crucial como máximo 1,05 veces el control.
5. Aviso tardío y falsa alerta pueden aumentar como máximo un punto porcentual cuando cada denominador tenga al menos 30 casos. El soporte insuficiente impide promover una receta.

Con el control observado, los máximos de MiDs/MiDl/MiDn son aproximadamente 57,514 / 54,828 / 64,451. Todas las variantes superan el límite largo. Los denominadores de aviso tienen soporte suficiente; NO_CANDIDATE se debe a incumplimientos, no a una banda ausente.

| Candidato | Mejora MiDc % | Gates incumplidos | Elegible |
| --- | --- | --- | --- |
| RT-M0-L1-S7 | 4.616 | crucial_10pct, large, late_warning | No |
| RT-M1-L0-S7 | 14.258 | large, mae, negative, sign, small | No |
| RT-M1-L1-S7 | 13.163 | large, negative, small | No |
| RT-M2-L0-S7 | 14.003 | large, mae, negative, sign, small | No |
| RT-M2-L1-S7 | 14.181 | large, negative, small | No |

## Interpretación de los factores

**M1-L0** obtiene el menor MiDc (134,470 frente a 156,832; mejora del 14,26 %). El rango largo pasa de 52,217 a 76,397 y el negativo de 61,382 a 84,208; MAE sube de 2,512 a 2,979 s. La redistribución mejora lo corto a costa de otras exigencias del protocolo.

**M0-L1** ofrece el menor agregado y MAE, pero sólo reduce MiDc un 4,62 %. Empeora el rango largo y el aviso tardío. No cumple el objetivo por elegir una métrica favorable.

**L1 con M1/M2** reduce MAE respecto a sus L0 correspondientes y mantiene parte de la mejora crucial. Aun así incumple las bandas media, larga y negativa. **M2** no establece una mejora global frente a M1 en esta semilla. No hay réplica que permita estimar la variación entre semillas.

Esta matriz aporta evidencia de un compromiso entre rangos dentro de la receta ensayada. No demuestra que una ponderación distinta, otro decoder o un productor nuevo resuelvan el problema; esas pruebas requerirían otro protocolo.

## Signo, sobreestimación y avisos

| Receta | Error signo % | p95 sobreestimación crucial s | Aviso tardío % (n=76) | Falsa alerta % (n=16356) |
| --- | --- | --- | --- | --- |
| RT-M0-L0-S7 | 4.710 | 3.585 | 94.737 | 0.006 |
| RT-M0-L1-S7 | 4.583 | 2.702 | 100.000 | 0.000 |
| RT-M1-L0-S7 | 5.599 | 1.574 | 89.474 | 0.116 |
| RT-M1-L1-S7 | 5.027 | 1.441 | 86.842 | 0.391 |
| RT-M2-L0-S7 | 5.818 | 1.562 | 82.895 | 0.324 |
| RT-M2-L1-S7 | 4.820 | 1.405 | 81.579 | 0.416 |

La frontera de aviso urgente es 1 s. Un TTC negativo predicho no cuenta como aviso. El denominador de aviso tardío son las 76 consultas con target (0,1;1]; el de falsa alerta reúne las restantes 16.356 consultas. Los porcentajes altos de aviso tardío describen una limitación seria de esta cohorte. No interpreto tasas bajas de falsa alerta como seguridad vehicular.

El p95 anterior calcula `quantile(max(pred-target,0),0.95)` sobre todos los cruciales. Es distinto de un percentil calculado sólo sobre los errores positivos.

## Bootstrap pareado por grupos

| Candidato menos control | Δ MiDc | IC95 % bootstrap | Grupos | Remuestreos |
| --- | --- | --- | --- | --- |
| RT-M0-L1-S7 | -7.239 | [-11.351; -3.480] | 8 | 1000 |
| RT-M1-L0-S7 | -22.361 | [-35.066; -9.507] | 8 | 1000 |
| RT-M1-L1-S7 | -20.644 | [-33.708; -8.536] | 8 | 1000 |
| RT-M2-L0-S7 | -21.961 | [-37.946; -5.642] | 8 | 1000 |
| RT-M2-L1-S7 | -22.241 | [-38.434; -6.950] | 8 | 1000 |

Cada contraste usa 1.000 remuestreos de ocho grupos completos, en diez fragmentos reproducibles de 100 draws. Los índices se comparten por construcción entre control/candidato dentro de cada contraste. No trato las ventanas correlacionadas como observaciones independientes. Los intervalos describen la diferencia MiDc de esta seed7; no incorporan variación de entrenamiento entre semillas ni convierten el análisis en test externo. Que un intervalo crucial sea favorable no elimina los otros gates.

## MiDc por grupo

| Grupo V | M0-L0 | M0-L1 | M1-L0 | M1-L1 | M2-L0 | M2-L1 |
| --- | --- | --- | --- | --- | --- | --- |
| 2cyv0Oedzg | 211.522 | 195.913 | 163.387 | 158.076 | 156.153 | 150.636 |
| 5ilM1PX2vz | 150.278 | 149.121 | 126.783 | 135.953 | 122.620 | 128.384 |
| 6h5yRW2LGc | 258.542 | 242.920 | 213.088 | 229.624 | 211.466 | 228.800 |
| WbCh1DRerJ | 158.755 | 148.897 | 155.158 | 146.119 | 163.301 | 155.601 |
| Z30i8WuL2N | 82.866 | 79.669 | 93.354 | 98.453 | 98.490 | 100.003 |
| pzJcg1OCfc | 104.223 | 104.053 | 87.467 | 86.514 | 98.484 | 95.812 |
| rhYxPcU29n | 168.796 | 162.587 | 131.264 | 129.559 | 128.802 | 119.431 |
| t79dBxj1WS | 119.673 | 113.580 | 105.262 | 105.199 | 99.652 | 98.061 |

Los JSON GROUP_METRICS contienen las demás bandas y diagnósticos por grupo. SUBRANGE_GROUP_RESULTS.csv conserva el detalle por grupo y subrango. Las celdas ausentes se mantienen ausentes.

## Subrangos descriptivos: MiD macro por grupos

| Subrango | Consultas | M0-L0 | M0-L1 | M1-L0 | M1-L1 | M2-L0 | M2-L1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1-1 | 76 | 547.906 | 574.362 | 444.119 | 439.398 | 399.726 | 390.748 |
| 1-2 | 658 | 226.037 | 210.255 | 183.076 | 184.522 | 176.474 | 173.041 |
| 2-3 | 1111 | 93.205 | 88.766 | 87.879 | 90.843 | 96.318 | 98.219 |
| 3-6 | 4284 | 54.776 | 51.168 | 63.112 | 65.151 | 67.812 | 70.530 |
| 6-8 | 2000 | 50.607 | 52.586 | 78.722 | 83.518 | 82.433 | 85.337 |
| 8-10 | 1283 | 54.175 | 63.346 | 73.994 | 85.649 | 71.543 | 82.478 |
| negative | 7020 | 61.382 | 61.009 | 84.208 | 80.041 | 87.619 | 79.928 |

Estas tablas permiten localizar el comportamiento dentro de la banda crucial. No añadí gates ni reajusté la selección después de obtenerlas.

## Decisión cerrada

`SELECTION_FREEZE.json` registra candidato null y NO_CANDIDATE. No ejecuté las cuatro réplicas candidato/control de seeds13/23 ni una evaluación de transferencia de endpoints nuevos. Conservo todos los resultados, incluidos los brazos que empeoran. Los 10.000 updates científicos no consumidos no financian otro experimento.

Fuentes: [resultados CSV](evidence/FACTORIAL_RESULTS.csv), [gates congelados](evidence/SELECTION_FREEZE.json), [decisión final](evidence/NEXT_DECISION.json), carpetas `evidence/evaluation/screen/` y `evidence/bootstrap/screen/`.
