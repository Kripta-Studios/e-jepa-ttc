# 04 — Diagnóstico FCWD y causas que todavía deben aislarse

## Hallazgo principal

El déficit de H8 frente a Garl event-only se concentra en **sobreestimar TTC positivo corto**. Su sesgo global negativo no lo revela porque la subestimación de TTC largo domina el promedio en segundos. Una corrección uniforme del sesgo probablemente movería uno de esos errores en la dirección equivocada; debe probarse por banda y secuencia.

El análisis nuevo de este handoff usa únicamente los CSV públicos, sin volver a inferir ni cambiar pesos. [`replay_public.py`](replay_public.py) ha recalculado las 2.388 filas (597 consultas × 4 variantes), comprobado emparejamiento y GT, y reproducido exactamente el MiD por fila. [REPLAY_RESULT.json](evidence/recomputed/REPLAY_RESULT.json) registra hashes y resultado. El CSV scored contiene solo población elegible; las 33 consultas excluidas se conservan en [INELIGIBLE_TARGETS.csv](../sota_evidence_20261010/evidence/streaming/fcwd_stream_cpu/INELIGIBLE_TARGETS.csv).

## Localización por rango

| Banda FCWD | n | H8 MAE s | H8 sesgo s | H8 sobreestima | H8 MiD | Garl MiD |
|---|---:|---:|---:|---:|---:|---:|
| (0,3] | 201 | 0,7004 | +0,6565 | 83,58% | 177,9379 | 95,0568 |
| (3,6] | 189 | 0,5460 | +0,2596 | 54,50% | 25,8348 | 36,6983 |
| (6,10] | 108 | 0,9811 | −0,7188 | 23,15% | 17,6379 | 34,2118 |
| >10 | 99 | 6,2640 | −6,2640 | 0% | 52,3035 | 72,3958 |

Fuente calculada: [FCWD_SUMMARY.csv](evidence/recomputed/FCWD_SUMMARY.csv). Los 99 casos largos se subestiman todos. En la banda crucial el p95 del error absoluto original es 2,441 s para H8 y 1,320 s para Garl, según [ERROR_BANDS.csv](../sota_evidence_20261010/evidence/streaming/fcwd_stream_cpu/ERROR_BANDS.csv).

La banda crucial tiene MAE Garl≈0,3272 s y sesgo≈+0,1659 s, mucho menores que H8. El hecho de que H8 gane en las otras bandas no compensa esa diferencia en mean_MiD. El overall oficial daría mayor peso a c, pero no puede calcularse aquí porque falta n; no fabricar una puntuación hipotética como si fuera resultado oficial.

## Descomposición exacta de la diferencia

Para cada banda b, la contribución al gap de **mean_MiD** es:

```text
(n_b / 597) * (MiD_H8,b - MiD_Garl,b)
```

| Banda | Contribución al gap H8−Garl |
|---|---:|
| c | +27,904676 |
| s | −3,439211 |
| l | −2,998297 |
| >10 | −3,331888 |
| Suma | +18,135280 |

Fuente: [FCWD_GAP_CONTRIBUTIONS.csv](evidence/recomputed/FCWD_GAP_CONTRIBUTIONS.csv). La banda crucial explica más del 100% de la desventaja neta: las otras bandas la amortiguan. Esta identidad algebraica localiza el problema; no identifica qué módulo lo causa.

El efecto aparece en las tres secuencias. Las diferencias H8−Garl de mean_MiD son aproximadamente +16,339 (FCWD1), +15,166 (FCWD2) y +23,171 (FCWD3). No depende únicamente de una secuencia con un outlier. Aun así, tres secuencias no representan todas las condiciones de circulación.

## Consultas que el agente debería inspeccionar primero

La lista completa está en [FCWD_PAIRED_QUERIES.csv](evidence/recomputed/FCWD_PAIRED_QUERIES.csv), ordenada por la desventaja MiD de H8. Ejemplos:

| Consulta | GT s | H8 s | Garl s | Warp s | Warp+student s |
|---|---:|---:|---:|---:|---:|
| FCWD1:0181 | 1,2070 | 2,5228 | 1,2685 | 2,3752 | 2,1814 |
| FCWD3:0169 | 1,1956 | 2,8396 | 1,3485 | 2,7790 | 2,3756 |
| FCWD3:0170 | 1,1949 | 2,4611 | 1,2711 | 2,3899 | 2,1483 |
| FCWD1:0182 | 1,2102 | 2,4480 | 1,2710 | 2,2882 | 2,1433 |

Los casos cercanos en el tiempo están correlacionados. No deben presentarse como cuatro fallos independientes. Para revisar la geometría y la señal visual harán falta HDF5, cajas y pesos; GitHub permite localizar las consultas y comparar sus salidas, pero no contiene todos esos activos.

## Qué enseña la reproyección

Warp reduce MiD crucial 177,938→167,485 y mean_MiD 79,952→77,453, pero empeora MAE global 1,625→1,666 s. Warp+student reduce a 162,969 y 74,561, respectivamente, con MAE 1,506 s. Ninguno alcanza Garl.

La fracción de sobreestimación crucial pasa de 83,58% a 85,07% con warp y a 86,57% con warp+student. A la vez baja la magnitud del error. **Frecuencia y magnitud son propiedades distintas**: no describir el estudiante como una corrección completa de la sobreestimación.

La reutilización media en el replay es aproximadamente 6,913 observaciones, con 1,087 cálculos nuevos por consulta. El mínimo IoU registrado de ROI usada es≈0,651 y el desplazamiento temporal aproximado máximo≈349 μs. No hay evidencia suficiente para atribuir la mejora de MiD a una causa única. Puede intervenir suavizado, cambio de distribución de entrada o diferencias en características; hace falta una ablation controlada.

## Señales upstream que ya están publicadas

La revisión previa ya contiene [feature_shift.csv](../ttc_revision_20261009/tables/feature_shift.csv) y [upstream_metrics.csv](../ttc_revision_20261009/tables/upstream_metrics.csv). No es necesario reconstruirlas para comenzar. En las 630 consultas FCWD, la mediana de `a5_transport` pasa de 0,011094 en TRAIN40 a 0,005669; `c2f_transport`, de 0,012238 a 0,006925. Las fases medianas, en cambio, aumentan: A5 0,011075→0,018586 y C2F 0,010616→0,018278. Son columnas distintas: no intercambiar transporte y fase.

La fracción fuera de la envolvente train [q0,005; q0,995] es 9,84% para log-count/log-rate, 9,68% para log-varianza A5 y 7,46% para log-varianza C2F. Que las fases no salgan de esa envolvente no demuestra igualdad de distribución. Estas estadísticas usan población de entradas; no atribuir automáticamente sus diferencias a error sobre las 597 filas GT elegibles.

Los expertos convertidos a TTC con límite ±60 s tienen MAE FCWD: A5≈3,1023 s, C2F≈3,5437 s y PAIR≈2,7020 s. La mediana de fases actuales da≈2,9873 s, frente a H8≈1,6249 s. Por tanto, quitar simplemente la cabeza temporal no está respaldado como mejora general por esos agregados. Falta separar el rango crucial y aislar qué corrección aprende la cabeza. El [CSV por consulta publicado ahora](evidence/comparison/FCWD_PREDICTIONS.csv) conserva transporte, confianza y log-varianza A5/C2F junto a las predicciones finales; no contiene todos los tensores internos de router/PAIR.

## Hipótesis priorizadas y pruebas que las distinguen

### 1. Desbalance de supervisión y calibración por rango

**Observación:** TRAIN40 tiene solo 336 consultas con TTC positivo≤1 s, alrededor del 0,38% del total. H8 ya sobreestima ese grupo en entrenamiento (sesgo +0,5181 s). La transferencia expone errores cortos similares, aunque FCWD no cubre≤1 s.

**Hipótesis:** la receta y la representación favorecen valores centrales y no aprenden suficiente sensibilidad a la aproximación rápida. No basta con contar muestras: los pesos efectivos y gradientes por término también importan.

**Prueba:** con particiones nuevas por secuencia, registrar masa de pérdida y norma de gradiente por banda, balancear únicamente usando train y comparar la misma cabeza, batch, scheduler y presupuesto. La validación debe medir c y avisos, sin sacrificar negativos ni largos. No estimar pesos mirando el error FCWD.

### 2. Pérdida/decodificación en fase y compresión del rango largo

**Observación:** para fase cercana a cero, TTC≈0,1/fase; una pequeña variación de fase produce una gran variación TTC. El decodificador H8 cambia de rama entre −60 y +60. Direct elimina esa discontinuidad, pero empeora FCWD y el sesgo largo.

**Hipótesis:** la parametrización contribuye a errores de signo/rango, pero no es la explicación completa. El resultado Direct refuta que cambiar solo el tipo de salida garantice resolver FCWD; además esa revisión cambió varios hiperparámetros.

**Prueba:** aislar loss y decodificación manteniendo todo lo demás fijo. Evaluar simultáneamente fase, TTC, MiD, signo y cobertura. Para TTC positivo válido, la sensibilidad de fase es proporcional a `0.1/[T*(T-0.1)]`; esta ponderación ayuda a entender la discrepancia con MAE, pero no decide por sí sola la pérdida óptima.

### 3. Información perdida por los productores o el cuello de 17 características

**Observación:** las cabezas solo reciben 17 características por observación. Los productores congelados no cambian durante el ajuste Direct. El estudiante A5 puede mejorar FCWD y empeorar Dev32.

**Hipótesis:** C2F/PAIR o el mecanismo de combinación introducen sesgo dependiente del dominio; también puede faltar información suficiente para recuperar el TTC verdadero. La destilación quizá regulariza una señal perjudicial, pero no se ha demostrado.

**Prueba:** guardar predicciones individuales A5/C2F/PAIR, discrepancias entre expertos, pesos de router y salida residual de cabeza. Comparar ablations A5 solo, A5+C2F, A5+PAIR y combinación completa con idéntica normalización y validación. Correlaciones de error no bastan: modificar un factor cada vez.

### 4. Geometría ROI, remapeo y causalidad

**Observación:** se usa ROI de consulta para reconstruir historia, y la ruta warp no puede recuperar eventos fuera de la ROI original. FCWD tiene sensores y calibración diferentes; no existe admisión full RGB+eventos.

**Hipótesis:** los errores de crop, escala, timestamps, altura observable o selección temporal pueden alterar la expansión aparente. El CSV actual no demuestra que haya un bug geométrico.

**Prueba:** para las consultas listadas, comparar índices de evento, unidades temporales, cajas antes/después del mapping, masa de voxels y deltas medidos con la referencia exacta. Registrar cuándo estaba disponible cada ROI. Ejecutar primero paridad exacta; estudiar warp solo después.

### 5. Memoria temporal, normalización y cambio de dominio

**Observación:** H4 ayuda ligeramente Dev32 y empeora FCWD; H2 empeora FCWD aún más. El sesgo largo ya existe en TRAIN40 (≥8 s: −2,1726 s H8), por lo que no puede explicarse exclusivamente por targets externos>10 s.

**Hipótesis:** historia, normalización y rango de supervisión interactúan. Recortar contexto reduce coste, pero puede eliminar señales necesarias de expansión.

**Prueba:** separar truncación en inferencia de entrenamiento dedicado H4/H2. Auditar edades reales, máscaras y normalización train-only. Comparar intervalos temporales físicos, no solo número de posiciones en un tensor.

## Límites del diagnóstico

No hay en este handoff trazas suficientes para concluir que un productor concreto sea el culpable, ni nuevas inferencias que certifiquen una corrección. El ranking de consultas es posterior a observar resultados. FCWD ya sirve de desarrollo; mejorar esas tres secuencias no cerrará generalización. El siguiente informe de experimentos debe conservar intentos fallidos, decisiones previas y una evaluación no usada para seleccionarlas.
