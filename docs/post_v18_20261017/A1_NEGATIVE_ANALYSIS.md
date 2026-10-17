# A1-FULL: magnitud y saturación en P

Analizamos 53.890 predicciones P guardadas; 22.397 tienen TTC negativo. Reutilizamos el diagnóstico V19 y añadimos rangos, grupos y comprobación CPU de la conversión de fase. No ejecutamos el modelo ni calculamos gradientes.

| Modelo | MiDn macro por grupo | MiD negativo pooled | MAE negativo, s | Error de signo negativo | MAE con signo correcto, s |
|---|---:|---:|---:|---:|---:|
| FULL | 93.547 | 96.302 | 11.478 | 14.56% | 7.708 |
| A5 | 55.522 | 54.665 | 5.863 | 4.13% | 4.294 |

El error de signo de esta tabla se condiciona a targets negativos de P. No equivale a la tasa global en H. Cada modelo acierta el signo en consultas distintas; el JSON conserva también celdas emparejadas FULL/A5.

| A1-FULL, targets negativos | n | MAE, s | MiD pooled | Fracción del error absoluto negativo |
|---|---:|---:|---:|---:|
| sign_correct_saturated_False | 18269 | 5.568 | 67.859 | 39.57% |
| sign_correct_saturated_True | 868 | 52.752 | 132.304 | 17.81% |
| sign_wrong_saturated_False | 2639 | 25.666 | 265.472 | 26.35% |
| sign_wrong_saturated_True | 621 | 67.349 | 163.815 | 16.27% |

FULL acumula el 57.38% del error absoluto negativo en consultas con signo correcto. Hay un problema de magnitud además del error de dirección. La saturación afecta a 1489 consultas negativas (6.65%) y concentra el 34.08% del error absoluto negativo. Las cuatro celdas anteriores son disjuntas; no sumamos otra vez saturación y signo.

| Componente aditivo MiDn | FULL | A5 | Delta FULL - A5 |
|---|---:|---:|---:|
| correct_saturated | 4.799 | 2.047 | 2.752 |
| correct_unsaturated | 55.679 | 44.975 | 10.704 |
| wrong_saturated | 4.134 | 1.772 | 2.361 |
| wrong_unsaturated | 28.935 | 6.728 | 22.208 |

Estas contribuciones conservan los denominadores de la banda negativa completa y suman el MiDn original. El delta por celda combina cambios de error y de pertenencia: no constituye una atribución causal. La fracción del error absoluto de arriba mide otra cantidad.

El incremento total MiDn es 38.025. Las celdas saturadas aportan 5.114 (13.45% de ese delta); la celda de signo incorrecto sin saturación aporta 22.208 (58.40%). La saturación pesa en MAE, pero no concentra la mayor parte del deterioro MiDn. Quitar el límite de 60 s no queda justificado como solución por este análisis.

| Rango target P, s | n FULL | MAE FULL, s | MAE A5, s | Razón de magnitudes, mediana |
|---|---:|---:|---:|---:|
| [-10, -8) | 5010 | 18.067 | 11.344 | 1.600 |
| [-8, -4) | 13676 | 10.681 | 5.079 | 1.462 |
| [-4, -2) | 3374 | 5.495 | 1.387 | 1.429 |
| [-2, -1) | 334 | 5.786 | 1.043 | 1.702 |
| [-1, -0.5) | 3 | 3.657 | 0.608 | 3.447 |
| [-0.5, 0) | 0 | N/D | N/D | N/D |

Los rangos son descriptivos y se fijan para este desglose. El intervalo [-1,-0,5) tiene solo tres consultas y [-0,5,0) ninguna: no permiten concluir comportamiento cerca de TTC cero. Cerca de fase cero significa TTC de gran magnitud, no TTC próximo a cero.

En las mismas 18856 consultas donde ambos modelos aciertan el signo, el MiD pooled es 70.514 para FULL y 45.796 para A5. La regresión de magnitud persiste al fijar las consultas, por lo que no depende solo de comparar subconjuntos de signo diferentes.

| Grupo P | n negativos | MiD FULL | MiD A5 | Delta MiD |
|---|---:|---:|---:|---:|
| xkiVthsGwf | 1553 | 163.034 | 58.221 | 104.813 |
| 8mNnrXjpVB | 1534 | 141.099 | 62.244 | 78.855 |
| VLA3oBKFyX | 822 | 132.548 | 70.555 | 61.993 |
| pakWgyPNTL | 561 | 141.737 | 86.120 | 55.617 |
| OBneIVg4Cw | 596 | 118.537 | 65.550 | 52.987 |

Mostramos los cinco mayores incrementos de MiD negativo por grupo; el JSON conserva todos los grupos y el CSV todos los cruces. FULL empeora en 24 de 24 grupos con negativos. Este ranking no selecciona consultas de entrenamiento.

La reconversión CPU reproduce el TTC guardado con error máximo 7.62939453e-06 s. Hay 0 fases crudas negativas sujetas al clipping MIN_PHASE/MAX_PHASE. Por tanto, ese clipping no explica estos casos; el límite de magnitud de 60 s sí está presente.

La conversión nativa usa q = -expm1(-clamp(fase))/0,1 y TTC = sign(q)/max(|q|,1/60). Cerca de fase cero, el inverso amplifica discrepancias hasta alcanzar el límite de 60 s. Dentro del límite, la salida queda saturada; no afirmamos que su derivada siga creciendo. Fase cero emite +60 s. El clipping de fase y el límite de magnitud son operaciones distintas.

A5 puede guardar fase reconstruida desde TTC: su etiqueta de saturación describe el umbral de conversión, sin probar el estado crudo original del productor. FULL sí guarda raw_phase. No modificamos clipping ni salidas.

El CSV contiene todos los grupos, rangos y cruces signo/saturación, incluidos los vacíos. Los estadísticos condicionales son diagnósticos, no una nueva puntuación oficial. No derivamos conflictos de gradiente ni causalidad de estas tablas.

Único contraste pertinente: el L0/LW ya registrado en V19, que mantiene FULL, padre, sampler y presupuesto y cambia la masa negativa. No abrimos otra copia ni proponemos BCE como solución demostrada. El resultado de esa intervención pertenece a V19 y requiere su propia evidencia.

Reproducir: `python -B scripts/post_v18_review.py prepare` con el entorno del proyecto y `PYTHONPATH=src;.` en Windows. Fuentes y hashes: [A1_NEGATIVE_DIAGNOSTICS.json](A1_NEGATIVE_DIAGNOSTICS.json). Desglose: [A1_NEGATIVE_RANGE_GROUP.csv](A1_NEGATIVE_RANGE_GROUP.csv).
