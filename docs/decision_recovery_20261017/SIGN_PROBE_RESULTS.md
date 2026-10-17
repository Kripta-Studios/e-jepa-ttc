# V19: probe lineal de signo sobre FULL congelado

Estado: **COMPLETE_DIAGNOSTIC_NOT_CANDIDATE**, con gate interpretativo rechazado. El ajuste convergió, pero este lector lineal fijo empeora el signo emitido por FULL en H. Hay señal predictiva frente al control permutado; el resultado no demuestra que la representación carezca de información de signo.

S1 usa 194 descriptores: medias y máximos de los 96 canales de la torre espacial FULL, delta temporal nativa y su logaritmo. Raw phase y TTC se conservan solo para comprobar paridad y no entran como variables del probe. Se ajustan normalización y regresión logística únicamente sobre P completo: 53.890 filas, 31.493 positivas y 22.397 negativas, de 24 grupos.

El protocolo fija C=1, solver LBFGS, tolerancia 1e−10, umbral 0,5 y sin ponderación de clase. El control SHUFFLED permuta una vez las etiquetas P con semilla 7. H contiene 18.422 filas, 10.796 positivas y 7.626 negativas, de ocho grupos; sigue siendo desarrollo expuesto. No se actualiza el encoder ni se modifica ninguna predicción TTC del sistema.

| Lectura | Error global | Error negativos | Error positivos | Error balanceado | Brier | AUC ROC | ECE10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FULL_emitted_sign | 5.607% | 12.130% | 1.000% | 6.565% | — | — | — |
| S1 | 17.381% | 20.378% | 15.265% | 17.821% | 0.119749 | 0.910118 | 0.030340 |
| SHUFFLED | 41.412% | 99.921% | 0.083% | 50.002% | 0.243509 | 0.494558 | 0.024228 |

El error balanceado es la media de los errores condicionales positivo y negativo; no es una media macro por grupos. FULL aporta signos duros y no se le atribuyen métricas probabilísticas. La precisión media AP de S1 es 0,932304 frente a 0,584025 del control.

| Condición interpretativa registrada | Valor observado | Cumple |
|---|---|---|
| Error negativo S1 <= 0,75 × FULL | 20,378% frente a límite 9,097% | No |
| Error positivo S1 <= FULL + 0,005 | 15,265% frente a límite 1,500% | No |
| Límite superior IC95 del cambio de error balanceado < 0 | +16,123 puntos porcentuales | No |
| Brier menor que constante de prevalencia P | 0,119749 < 0,242600 | Sí |
| Error balanceado menor que SHUFFLED | 17,821% < 50,002% | Sí |

| Signo verdadero | Ambos correctos | Rescatados por S1 | Dañados por S1 | Ambos incorrectos |
|---|---:|---:|---:|---:|
| negative | 5615 | 457 | 1086 | 468 |
| positive | 9091 | 57 | 1597 | 51 |

S1 rescata 457 negativos y daña 1.086 que FULL resolvía; en positivos rescata 57 y daña 1.597. El cambio de error balanceado S1−FULL es +11,256 puntos porcentuales, con IC95 descriptivo [+7,446; +16,123]. Se remuestrean grupos 2.000 veces manteniendo las predicciones fijas, sin reajuste. H reutilizado y este bootstrap no acreditan confirmación independiente.

Esta comparación rechaza la hipótesis operativa de que el lector lineal registrado recupere el signo mejor que la salida FULL. No descarta otros lectores, ni permite elegir ahora umbrales usando H. Superar el control y la constante de prevalencia demuestra utilidad predictiva de estas variables para este diagnóstico, sin habilitar una cabeza TTC, promoción o SOTA. V no se abrió.

## Recuperaciones técnicas preservadas

La materialización P recuperó exactamente 211 chunks originales, que se conservaron con sus recibos. sample_token había quedado serializado como ndarray object. Tras verificar recibos y hashes, se reconstruyeron identificadores desde la porción registrada, sin deserializar ese miembro object, y el caché derivado se escribió en Unicode sin pickle. No cambiaron el endpoint, las consultas, su orden, los descriptores, raw phase, TTC, los umbrales de paridad ni el protocolo del probe.

Los recibos finales P/H contienen 53.890/18.422 filas y 194 variables sin targets en los cachés. Ambos declaran paridad EXACT de raw phase y TTC con máximo error absoluto 0 y cero cambios de signo, y cero updates de encoder. La enmienda TECHNICAL_RECOVERY conserva las fuentes registradas y las de recuperación.

El primer fit CPU con máximo 2.000 iteraciones terminó sin converger. Se conserva ORIGINAL_FAILURE, sin resultados ni scores H de ese intento: ese fallo numérico no se interpretó como ausencia de señal. La continuación se registró aparte en 7cedbfae4f1f12c872f7f870f785180cfb01cf94; solo amplió max_iter a 10.000, manteniendo objetivo, entradas, regularización, tolerancia, semilla y umbral. S1 convergió en 2.743 iteraciones y SHUFFLED en 2.958, antes de calcular los scores H. El módulo original del probe permanece preservado para los contratos de R1.

La fuente de las métricas es sign_probe_v2_continuation/RESULT.json, no un RESULT del intento original sign_probe_v2. El router R1 tiene registro, ajuste y rechazo propios; su evaluación no se sustituye por este diagnóstico.

## Evidencia

Directorio de resultados: artifacts/decision_recovery_20261017/sign_probe_v2_continuation. Extracción y enmienda de tokens: artifacts/decision_recovery_20261017/sign_probe_v2.

SHA256 de RESULT: a3657eea27900759b54af6eeaa2e8d781477c34d32c9f940336e9f5a234bc13f.
SHA256 del registro de continuación: 142a454e82c4233fe6542626f6ecb999c7eaf194469ab884fbd445f9beef8a2d.
SHA256 de ORIGINAL_FAILURE: af9108978a077764b4d7646ef41f1f330b748bc5c4603493e4e07d79216bdf7e.
SHA256 de H_SIGN_PREDICTIONS: 7c679b18545f9013d3ee7356024a3b635965861e1173ffde965d7b0e227d1527.
SHA256 de TECHNICAL_RECOVERY: b3f3311e973df5248cc2efb706bd534d9140c620ade1f70b3cbbda201f29a665.

Verificación documental: hashes de resultado, registro, fallo original, ambos modelos, predicciones y recibos P/H comprobados; tablas transcritas de RESULT. No se ejecutaron nuevos fits ni inferencias para este documento.
