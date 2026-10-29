# Estudio posterior SIMPLEX-T: modelos congelados

P0, P1, P2 y P4 completados. P3 bloqueado por ausencia de un slot exclusivo explícito GPU/lectura pesada. No se ejecutaron nuevos fits ni updates de optimizador. T6, H8 registrado y los 24 endpoints de la campaña nocturna permanecen intactos.

Se verificaron los hashes de los 24 endpoints (60.000 updates históricos guardados). Las 98.304 filas de 12 combinaciones modelo/seed conservan las 8.192 queries, nueve secuencias, targets y folds emparejados. La diferencia máxima de scores frente a las publicaciones físicas fue 1,9895196601282805e-13 MiD. Se exportaron 583 filas de contrastes y 90.079 filas de episodios/contrastes; los episodios son agrupaciones descriptivas por track/timestamp con salto fijo de 100 ms, no una demostración de tracking online.

H16 mejora frente a H8 en las tres seeds de cabeza. Delta medio de pérdidas de las dos nuevas: −2,423702276 MiD; CI jerárquico95 [−5,244259455, +0,041091485]. Con las tres seeds: −2,542665966; CI [−5,213397889, −0,111240531]. Seed7 fue exploratoria ya observada. Las pérdidas se promedian por query; nunca se promedian TTC para crear un ensemble. Esto acredita estabilidad local de optimización de las cabezas, con incertidumbre entre escenas y desarrollo reutilizado.

En OBneIVg4Cw los deltas H16−H8 son +4,799021788/+0,433603142/+3,279250808 para seeds7/13/23. Para las dos nuevas, el bucket positivo 0–3 s empeora +8,640189 MiD mientras los otros tres buckets mejoran; su delta secuencia es +1,856427. Es una localización descriptiva del fallo, no una causa demostrada. No se ha eliminado ninguna secuencia, retocado umbrales o elegido otro checkpoint. MODEL_DIAGNOSTICS.csv conserva signos, saturación, cap, cobertura, escapes y ganancias/daños por grupo. El hull completo de tres expertos es sólo diagnóstico para brazos reducidos; la cobertura de q10/q90 no demuestra calibración.

La revisión acotada de C2F oficial V7 y Garl verifica queries/folds coincidentes tras resolver aliases fold/outer_fold y target_ttc_s/target_ttc. Las diferencias de targets son de última cifra CSV: 1,7763568394002505e-15 y 8,881784197001252e-16 s. No justifican declarar otra población. La autoridad completa de productores, ROI/modalidad/availability, TRAIN y selección sigue sin quedar ligada por esos manifiestos; ambos permanecen como evidencia insuficiente para proclamarlos ganadores comparables. Los flags preliminares sin aliases quedan explícitamente sustituidos por COMPARATOR_REVIEW_RECONCILED.json.

Se midieron 13.500 inferencias batch1 en tres bloques, nueve cabezas, 500 medidas/modelo/bloque; 25 warmups/modelo/bloque y 540 fragmentos persistidos. Se conserva CPU FP32, cuatro threads, interop2 y emisión TTC canónica float64. El orden de modelos rota por fragmento, fijado antes de medir. Se publican todos los bloques, no la ronda más rápida. Los valores siguientes son mediana y rango de p95 entre bloques, no intervalos de confianza ni latencia del sistema completo.

| Cabeza seed7 | MiD OLD_DEV | p95 mediano ms | Rango p95 ms |
|---|---:|---:|---:|
| H1_SEED7 | 132.831619 | 9.278575 | 9.244095–9.442505 |
| H8_SEED7 | 121.646332 | 19.472580 | 19.379230–21.568570 |
| H16_SEED7 | 118.865739 | 31.450440 | 28.245235–31.787760 |
| FULL_C0 | 121.404706 | 21.074470 | 19.711580–21.102735 |
| A5_ONLY_C0 | 127.785376 | 21.420545 | 21.054685–21.446800 |
| C2F_ONLY_C0 | 126.704974 | 19.281625 | 19.190105–20.580805 |
| A5_PAIR_C0 | 124.668732 | 19.825955 | 19.764670–21.172155 |
| SET_AGE_C0 | 122.442604 | 9.995545 | 9.758150–10.540485 |
| SET_NOTIME_C0 | 123.686600 | 9.914290 | 9.869945–10.083915 |

Los costes corresponden exclusivamente a entradas TRAIN preparadas, fold0/seed7. No incluyen raw slice, ROI, contexto, expertos, detector, tracker ni decisión AEB. La carga del host, los intervalos entre llamadas y el backend pueden afectar los tiempos; esta ejecución registra backend/shapes/timer pero no aísla la causa de la discrepancia con los perfiles históricos. La precisión usa tres folds; el coste no equivale a medir esos tres sistemas en despliegue.

Las tres reducciones y los dos agregadores siguen fallando el cribado original de precisión: límite superior CI95 del exceso <2 MiD, signo ≤+0,5 puntos porcentuales, crucial ≤+5 MiD y cobertura completa. El requisito de ≥20% de ahorro p95 sólo se examina en su scope de cabeza. No altera los CI, no demuestra equivalencia por un p-valor no significativo y no autoriza sustitución del sistema. FULL_C0 controla el cambio lambda_cost=0; su comparación con H8 tampoco promociona una nueva receta.

Se implementó y probó una interfaz de tiempos segmentados condicionada a ROI externo, con sincronización explícita y rechazo de llamadas raw/productores sin admisión. Sus pruebas usan fixtures, no resultados eAP. La integración con productores reales y la medición integral quedan pendientes: falta el slot exclusivo del propietario Stage70–76; después deben validarse los bindings TRAIN raw, pesos de expertos y semántica de disponibilidad. No se obtuvo permiso de hardware o lectura pesada por observar procesos ausentes, ni se cambió Stage70–76.

Decisión: mantener H8 como candidato histórico. H16 es el challenger de precisión local; el coste integral y la generalización independiente siguen por establecer. No añadir seeds, arquitecturas, encoders, LATENT, datasets o entrenamiento de rescate. El siguiente paso dentro de este plan es únicamente el perfilado congelado integral al disponer de la dependencia externa. Una futura confirmación en escenas independientes requeriría protocolo y autorización distintos.

El bundle contiene predicciones canónicas de los 12 grupos modelo/seed, pesos e inputs preparados deduplicados para nueve cabezas de coste, tiempos por fragmento, recibos estadísticos históricos, código y pruebas. Permite regenerar scores, contrastes por estrato, cuantiles de tiempos archivados e inferencia de las nueve cabezas desde caché. No reconstruye eventos crudos, productores ni expertos; no puede reproducir físicamente tiempos de pared idénticos. La regeneración no equivale a reentrenar o a replicar todas las genealogías.

Reanudación exacta local de P2, reutilizando fragmentos ya completos: `..\e-jepa-ttc\.venv\Scripts\python.exe -B operational/simplex_t_post_campaign/run.py profile`. P2 completo no requiere repetir medidas. P3 no tiene comando de ejecución real admitido todavía; usar sólo la interfaz verificada una vez concedido y validado el slot. Regeneración del bundle extraído: `python -B regenerate.py --root . --output REGENERATION.json --heads`.

Alcance temporal del diagnóstico: 8189 grupos de anotación, 8186 con una sola query. Las anotaciones admitidas son mayoritariamente observaciones aisladas; este inventario no reconstruye trayectorias continuas ni tiempos de reacción. La regeneración verifica también estas particiones y todas las filas de diagnóstico exportadas.

![Precisión frente a coste de cabeza, con los tres bloques publicados](ACCURACY_HEAD_COST.png)
