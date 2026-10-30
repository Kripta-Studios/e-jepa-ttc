# R0 frente a R2, semilla 7 — cierre y siguiente intervención

Fecha: 2026-10-31. Informe regenerado desde JSON verificados.

R2 completa 8.192 updates y 32.768 exposiciones. En selección completa, la AP pasa de 0.09400020 a 0.11645298: +0.02245278 absoluto, +23.89 % relativo.

Se conserva R2 como baseline de trabajo. Las semillas 13 y 23 de R0/R2 quedan aplazadas; no queda ninguna ejecución automática pendiente. La siguiente prueba propuesta es R2 con muestreo limitado por identidad para clases raras, aislado de cambios de arquitectura, pérdidas y augmentation fotométrica.

## Comparación con presupuesto fijo

Ambos modelos usan D4096, batch 4, FP32, R50, ocho clases, 50 queries, 11 grupos y la semilla 7. Se comparan checkpoints finales al mismo update. R0 usa LR constante; R2 combina warmup 5 %, cosine hasta 0,05× y LR del backbone 0,1×. Por tanto, la mejora no se puede atribuir exclusivamente al LR diferencial. R0 fue reanudado a 4.096 con linaje verificado; R2 comenzó frío.

Los logs confirman los mismos tokens en el mismo orden en los 8,192 updates: 32,768 exposiciones por run y 4,096 frames únicos. El recibo `consumed_train_parity.json` fija el hash de ese calendario.

| Update | AP R0 completa | AP R2 completa | Diferencia |
|---:|---:|---:|---:|
| 4096 | 0.07662383 | 0.08690119 | +0.01027736 |
| 8192 | 0.09400020 | 0.11645298 | +0.02245278 |

Población idéntica: 23,958 frames, 190,469 anotaciones raw, 107,308 evaluables y 83,161 ignoradas por profundidad. Se verificaron los hashes del contenido de cada GT, el protocolo y los checkpoints.

| Clase | GT evaluable completo | AP R0 | AP R2 | Diferencia |
|---|---:|---:|---:|---:|---:|
| car | 59074 | 0.49068602 | 0.53165455 | +0.04096853 |
| bicycle | 687 | 0.00685269 | 0.01363593 | +0.00678324 |
| truck | 14083 | 0.13180223 | 0.19315043 | +0.06134820 |
| pedestrian | 7460 | 0.00644289 | 0.01441345 | +0.00797056 |
| motorcycle | 22187 | 0.07187399 | 0.10524124 | +0.03336725 |
| bus | 1916 | 0.03722947 | 0.06019388 | +0.02296441 |
| tricycle | 1828 | 0.00711434 | 0.01333437 | +0.00622003 |
| other_vehicle | 73 | 0.00000000 | 0.00000000 | +0.00000000 |

| Umbral XY | Recall sin exigir clase R0 | R2 |
|---:|---:|---:|
| 0.5 m | 0.252246 | 0.280538 |
| 1.0 m | 0.427769 | 0.462118 |
| 2.0 m | 0.641900 | 0.662886 |
| 4.0 m | 0.858622 | 0.846013 |

R2 mejora AP en las siete clases con AP no nula, pero no recupera other_vehicle. El recall a 4 m disminuye; la ganancia de AP no implica mejora de todas las propiedades de las propuestas.

## Panel monitor y selección de checkpoint

| Update | AP R0, 1.024 frames | AP R2, 1.024 frames |
|---:|---:|---:|
| 1024 | 0.03727560 | 0.06220262 |
| 2048 | 0.07372468 | 0.07811640 |
| 3072 | 0.07051789 | 0.10292892 |
| 4096 | 0.07918912 | 0.09026599 |
| 5120 | 0.08981751 | 0.10384537 |
| 6144 | 0.10420094 | 0.11027994 |
| 7168 | 0.09704602 | 0.11491516 |
| 8192 | 0.09987541 | 0.11681927 |

R0 seleccionó update 6.144 y R2 update 8.192 por AP del panel. La tabla completa anterior compara ambos finales a 8.192; no es una comparación de los dos checkpoints seleccionados sobre selección completa.

El panel contiene solo 26 bicicletas. Su AP final cae en R2, mientras que la AP sobre las 687 anotaciones de bicicleta de selección completa aumenta. La impresión inicial de colapso global queda descartada. No se debe elegir una intervención exclusivamente por la curva rara de este panel.

## Diagnóstico de todas las anotaciones raras

Correspondencia greedy, una a una, sin exigir clase, a distancia XY < 2 m. Se incluyen todos los objetos y predicciones de cada frame para conservar la competencia por asignaciones; después se agregan las dos clases raras. Una propuesta cercana no demuestra identidad correcta.

| Run | Clase | GT | Propuestas asignadas a <2 m | Clase correcta | Confusiones |
|---|---|---:|---:|---:|---|
| r0 | bicycle | 687 | 285 | 54 | motorcycle: 146, truck: 66, pedestrian: 8, car: 9, tricycle: 1, bus: 1 |
| r0 | other_vehicle | 73 | 37 | 0 | truck: 10, car: 25, tricycle: 1, bus: 1 |
| r2 | bicycle | 687 | 386 | 35 | motorcycle: 267, pedestrian: 62, tricycle: 7, car: 5, truck: 10 |
| r2 | other_vehicle | 73 | 13 | 0 | truck: 4, car: 8, pedestrian: 1 |

## Oracles diagnósticos: solo panel de 1.024 frames

| Transformación GT | AP R0 | AP R2 |
|---|---:|---:|
| class | 0.21265406 | 0.19861932 |
| center | 0.13740615 | 0.16480059 |
| geometry | 0.09987541 | 0.11681927 |
| ranking | 0.11950755 | 0.14250240 |

Son intervenciones con GT sobre correspondencias limitadas a 2 m, con rematching posterior; no son rendimiento alcanzable ni cotas superiores globales. El oracle de ranking puede empeorar una clase y no demuestra qué ganaría un calibrador. El oracle de dimensiones/yaw no cambia la métrica si no altera la admisión geométrica: AP aquí se basa en distancia de centros. La reevaluación normal del panel tuvo paridad exacta con los resultados guardados.

## Decisión: R2 + muestreo raro limitado por identidad

La evidencia favorece conservar la receta R2 y estudiar la escasa supervisión de las clases raras. Esto es una hipótesis de intervención, no una demostración causal de que el desbalance explique todos los errores.

| Clase train | Anotaciones entrenables en D4096 | Pares (secuencia, track_id) |
|---|---:|---:|
| car | 10455 | 1548 |
| bicycle | 162 | 69 |
| other_vehicle | 7 | 6 |
| pedestrian | 245 | 130 |
| tricycle | 97 | 51 |
| bus | 110 | 24 |

Prueba propuesta `R2-rare-v1`: mismo D4096, 8.192 updates, batch 4, semilla 7, receta R2 y evaluación natural sin modificaciones. Único factor: sampler ponderado usando exclusivamente etiquetas de train, empezando por bicicleta y other_vehicle. Construir un calendario determinista con exposición por frame y por identidad rara limitada a 4× la correspondiente al calendario uniforme; registrar exposiciones efectivas, frames únicos e identidades. El límite se aplica al calendario materializado, no solo a pesos que luego podrían producir repeticiones excesivas. Mantener el presupuesto total y documentar qué exposiciones comunes se desplazan. Esto adelanta el factor de muestreo de R4 sin mezclarlo con augmentation R3.

Antes de lanzarlo: comprobar que el calendario cumple los límites, se reanuda exactamente y conserva el protocolo, split y geometría. Para la decisión de screening, comparar el final a 8.192 con este R2 sobre los 23.958 frames: promover solo si mejora AP macro y el promedio AP de las dos clases objetivo, y ninguna otra clase cae más de 0,01 AP absoluto. Este umbral es una regla operacional propuesta, no significación estadística. Informar todas las clases y los fallos aunque no pase. Las semillas adicionales se reservan para confirmar una intervención prometedora.

Si el sampler no aporta, conservar R2 y pasar a augmentation fotométrica R3 como factor aislado. Si other_vehicle sigue sin soporte eficaz, ampliar diversidad de datos antes de atribuir el fallo a arquitectura: repetir seis identidades anotadas no crea seis adquisiciones independientes ni nuevos objetos. La siguiente intervención queda determinada; este cierre no ha lanzado otro entrenamiento.

## Alcance y reproducibilidad

Resultado exploratorio de una semilla, sobre desarrollo ya expuesto. No es puntuación oficial eAP, SOTA, test independiente ni evidencia nueva de precisión TTC. No se han usado test ni calibración para esta decisión.

Evidencia: [comparison.json](evidence/r0_r2_seed7_20261031/comparison.json), [diagnostics.json](evidence/r0_r2_seed7_20261031/diagnostics.json), [rare_full.json](evidence/r0_r2_seed7_20261031/rare_full.json), [curvas](evidence/r0_r2_seed7_20261031/monitor_curves.json). Incluyen hashes de checkpoints, evaluaciones y fuentes. El script ejecutado para la verificación inicial se conserva como `analysis_executed.py`; `scripts/compare_detection_r0_r2.py` añade el diagnóstico raro completo y una anotación de tipo sin cambiar las reglas de verificación.

Regenerar este informe desde la raíz del worktree:

```text
python scripts/report_detection_r0_r2.py --evidence docs/detection_sota_20261029/evidence/r0_r2_seed7_20261031 --coverage docs/detection_sota_20261029/evidence/r0_r2_seed7_20261031/d4096_coverage.json --output docs/detection_sota_20261029/R0_R2_SEED7_DECISION.md
```

Comprobaciones: verificador congelado de summaries/checkpoints y cuatro evaluaciones completas; igualdad de contenido GT entre runs; paridad exacta de métricas al reevaluar los dos paneles; ocho tests de diagnósticos; Ruff y Pyright. Tras retirar el controlador original no se capturó el código de salida del hijo: el cierre se acredita por artefactos completos y verificados, sin inventar un exit code.
