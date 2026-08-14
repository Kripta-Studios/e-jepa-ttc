# Scientific Recovery V7: cierre final

Fecha de corte: 2026-08-14. Rama:
`scientific-recovery-v7-balanced-oof`. Los cuatro brazos iniciales y el único
control condicional completaron tres folds OOF con seed 7 sobre los mismos 8.192
tokens. Public validation, private test, EvTTC test y CodaBench permanecieron
cerrados.

## Resultado OOF firmado

A5 revaluado obtiene `158.449 MiD` a cobertura puntual completa y `155.374 MiD`
selectivo con `95.227%` de cobertura. Garl local obtiene `144.353 MiD`, cobertura
completa y `0%` de failure.

| Brazo | MiD puntual | Delta vs A5 | MiD selectivo | Cobertura selectiva | P(delta<0) |
|---|---:|---:|---:|---:|---:|
| V7-SOFT | 165.116 | +6.668 | 162.283 | 94.446% | 0.0004 |
| V7-C2F | 158.573 | +0.125 | 156.324 | 94.958% | 0.4460 |
| V7-T20 | 165.260 | +6.812 | 162.257 | 95.288% | 0.0020 |
| V7-CAP-S | 167.025 | +8.576 | 163.212 | 94.995% | 0.0002 |
| SOFT partial-freeze | 167.826 | +9.378 | 164.744 | 94.885% | 0.0000 |

Intervalos bootstrap emparejados, 422 clusters `sequence_id+track_id`, 5.000
réplicas y seed 20260813:

- SOFT: mediana `+6.703`, IC95% `[+3.084,+10.547]`.
- C2F: mediana `+0.215`, IC95% `[-3.025,+3.432]`.
- T20: mediana `+6.877`, IC95% `[+2.290,+11.514]`.
- CAP-S: mediana `+8.683`, IC95% `[+3.735,+13.630]`.
- SOFT partial-freeze: mediana `+9.427`, IC95% `[+5.359,+13.272]`.

Los cuatro brazos produjeron exactamente 8.192 predicciones OOF únicas, tres
folds, nueve secuencias, `100%` de puntos finitos y `0%` de failure puntual. Todos
fallaron `mechanism_positive`, `geometry_positive` y `confirmation_candidate`.
El control partial-freeze cumple la misma integridad y falla los tres gates.

Los deltas puntuales partial-freeze frente a A5 son `+8.037`, `+6.779` y
`+13.317 MiD` en F0/F1/F2. Seleccionó epochs 13/13/10; F2 terminó por early
stopping en epoch 15. Cada fold congeló 9.664 parámetros, dejó 414.610
entrenables y registró cero solapamiento entre parámetros congelados y optimizer.

## Geometría

La retención se calcula contra los parents A4 fold-locales. El gate exige al menos
60% y signo positivo en las cuatro medidas.

| Brazo | Slope bbox | Slope física | Std ratio bbox | Std ratio física |
|---|---:|---:|---:|---:|
| V7-SOFT | 20.6% | 19.8% | 28.6% | 28.6% |
| V7-C2F | 15.7% | 15.3% | 24.1% | 24.0% |
| V7-T20 | 12.8% | 11.3% | 24.4% | 24.3% |
| V7-CAP-S | 22.6% | 21.7% | 28.4% | 28.3% |
| SOFT partial-freeze | 18.9% | 18.8% | 28.6% | 29.2% |

SOFT mejora algo la geometría relativa a los otros brazos, pero queda lejos del
umbral. La distillation no compensa la destrucción de señal temporal con el encoder
completamente entrenable. Congelar `encoder.features[0:3]` tampoco cambia la
conclusión: conserva menos slope que SOFT y prácticamente el mismo std-ratio.

## Inferencia y decisión

- C2F es compatible con efecto nulo como modelo único: sus folds se cancelan y
  el intervalo cruza cero. Solo F1 mejora A5 (`−2.069 MiD`).
- T20 empeora en los tres folds. Más bins por polaridad no explica el gap con Garl
  en esta arquitectura; no autoriza ASTW.
- CAP-S empeora en los tres folds y no conserva geometría. Cap-M queda bloqueado.
- SOFT empeora TTC y falla geometría. Su único control congeló
  `encoder.features[0:3]` con teacher, losses y pesos sin cambios; también empeora
  TTC y falla geometría.

El control parcial está congelado antes de entrenar en
`configs/experiment/scientific_recovery_v7_fold_chain/soft_partial_freeze_manifest.json`,
artefacto `c41fabd0c4e12220d531bc39748aea40de078a316e14c9eca5b599bd93ecf174`.
El estudiante parte de cero; solo las tres primeras etapas dejan de recibir
gradientes. No se permite barrer capas ni pesos.

V7 se cierra negativo. No se ejecutaron seeds 13/23 ni la ablation JEPA porque no
existe candidato. El resultado no demuestra que la geometría densa mejore TTC;
solo demuestra que ninguno de estos mecanismos conserva el parent ni supera A5.
La retención geométrica sigue siendo necesaria para un claim mecanístico, no como
gate universal de un predictor TTC posterior.

## Diagnóstico post hoc A5/C2F

Tras cerrar el screen se midió complementariedad por muestra. Este análisis no fue
preregistrado y no es un resultado V7 promocionable.

- A5 y C2F ganan exactamente 50% de las muestras cada uno.
- Un oracle que usa el target para elegir predicción obtiene `133.074 MiD`; es un
  techo no desplegable.
- Un router logístico leave-one-fold-out con event count/rate, flow, guard margin
  y log-varianza de ambos expertos obtiene `153.519 MiD`, delta `−4.929` frente a
  A5.
- Deltas por fold: `−3.571`, `−7.957`, `−3.259 MiD`. AUC de routing:
  `.632/.595/.615`.
- Bootstrap de 5.000 réplicas y 422 clusters: mediana `−4.919`, IC95%
  `[−7.033,−2.910]`, `P(delta<0)=1.0`.

La señal justifica una hipótesis TTC-first prospectiva con stacking anidado. No
autoriza routing por fold, secuencia, track, bbox o bucket; tampoco resuelve
geometría, pues ambos expertos fallan ese gate. Una evaluación futura debe
preregistrarse y usar datos o seeds no empleados para diseñar el router.

## Fuentes auditables

- `artifacts/scientific_recovery_v7/results/{soft,c2f,t20,cap_s,soft_partial_freeze}_seed7_oof.json`
- `artifacts/scientific_recovery_v7/audit/{soft,c2f,t20,cap_s,soft_partial_freeze}_geometry.json`
- `artifacts/runs/scientific_recovery_v7_*_fold*_seed7/summary.json`
- `configs/protocol/scientific_recovery_v7_balanced_oof.json`
- `artifacts/scientific_recovery_v7/diagnostics/FINAL_TRAINING_RESULTS.md`

El agregado partial-freeze firmado tiene identidad
`98e8f6fde78ce18a5a4339418cc6ec7f4c423dd3d50b518f37ed5bc63691b0b9`.
Los directorios `artifacts/` están ignorados por Git. Sus JSON incluyen hashes de
artefacto y deben acompañar al commit en un paquete de evidencia. No se abrió
ningún test externo.
